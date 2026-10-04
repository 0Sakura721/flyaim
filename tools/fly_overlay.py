"""赛博果蝇悬浮窗(复刻 BV1fQYB6oECS 式"果蝇玩原神"直播悬浮窗效果)。

    & $py tools/fly_overlay.py                 # 演示模式(合成神经活动 + 动作流)
    & $py tools/fly_overlay.py --state .cache/ann2_state.json   # 挂真实训练状态
    & $py tools/fly_overlay.py --interactive   # 关闭点击穿透(可拖动调试)

效果(对照视频里的画面):
    - 无边框、**总在最前**、**点击穿透**(WS_EX_TRANSPARENT)、**不抢焦点**
      (WS_EX_NOACTIVATE)—— 覆盖在游戏画面上也不影响操作;
    - 深色圆角面板 + 3D 脑点云(4,178 个真实解剖分组的采样神经元,慢速自转),
      神经元按活动度由暗蓝渐变到亮橙;
    - KPI 行(神经元 / 突触 / 活动度 / FPS)+ 活动度火花线 + 当前动作大字
      (按住 W · 向前移动 / 空格 · 跳跃 / Shift · 冲刺 ...);
    - 动作切换瞬间全脑"激发"一下(视频里决策→脑区亮起→按键的观感)。

数据:
    - 默认演示:活动度 = 分组平滑随机游走 + 动作激发;动作流按权重循环。
    - `--state` 指向 ann_dashboard 的 StatePublisher 输出文件时,显示**真实**
      训练活动度/帧数/FPS(文件 >5s 未更新则自动回落演示并如实标注)。

实现:tkinter 无边框窗 + `-transparentcolor` 色键(圆角外全透明);
     WS_EX_* 由 ctypes 设置;每帧 numpy 画点 → PIL 写字 → ImageTk 贴图。
"""

from __future__ import annotations

import argparse
import ctypes
import json
import math
import random
import time
from collections import deque
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------- Win32 常量

GWL_EXSTYLE = -20
WS_EX_LAYERED = 0x00080000
WS_EX_TRANSPARENT = 0x00000020
WS_EX_TOPMOST = 0x00000008
WS_EX_TOOLWINDOW = 0x00000080
WS_EX_NOACTIVATE = 0x08000000

KEY = (1, 2, 3)            # 透明色键(几乎不可能出现在画面里)
PANEL = (16, 22, 29)       # 面板底色
BORDER = (42, 54, 68)
TITLE_C = (216, 226, 238)
DIM_C = (123, 139, 160)
ACCENT = (232, 150, 60)
CYAN = (53, 208, 192)
GREEN = (32, 192, 96)
WHITE = (234, 242, 255)

ACTIONS = [  # (键名, 中文描述, 权重, 是否算"大动作")
    ("W", "向前移动", 5.0, False), ("W", "向前移动", 5.0, False),
    ("A", "向左移动", 2.0, False), ("D", "向右移动", 2.0, False),
    ("S", "向后移动", 1.2, False),
    ("SPACE", "跳跃", 1.6, True), ("SHIFT", "冲刺", 1.4, True),
    ("MOUSE1", "普通攻击", 1.2, True), ("E", "元素战技", 0.8, True),
]


def _load_font(size: int, bold: bool = False):
    from PIL import ImageFont

    for name in (("msyhbd.ttc" if bold else "msyh.ttc"), "msyh.ttc", "arial.ttf"):
        try:
            return ImageFont.truetype(f"C:\\Windows\\Fonts\\{name}", size)
        except Exception:
            continue
    return ImageFont.load_default()


class DemoSource:
    """演示数据源:分组随机游走 + 动作激发 + 动作流。"""

    def __init__(self, groups, seed: int = 20261005):
        self.rng = np.random.default_rng(seed)
        self.groups = np.asarray(groups)
        self._ug, self._inv = np.unique(self.groups, return_inverse=True)
        self.g_n = len(self._ug)
        # 每组一个基础活动相位;组内共享趋势 + 个体噪声
        self.g_phase = self.rng.uniform(0, 2 * math.pi, self.g_n)
        self.g_speed = self.rng.uniform(0.4, 1.4, self.g_n)
        self.a = np.full(len(groups), 0.25, dtype=np.float32)
        self.burst = 0.0
        self.action = None          # (key, desc)
        self._act_left = 0.0
        self._gap = 0.5

    def tick(self, dt: float):
        t = time.perf_counter()
        drive = 0.27 + 0.18 * np.sin(self.g_phase + t * 0.35 * self.g_speed)
        self.a = 0.90 * self.a + 0.10 * (np.clip(drive, 0.03, 1.0)[self._inv]
                                         + 0.15 * self.rng.standard_normal(len(self.a)))
        self.a = np.clip(self.a + self.burst * 0.5
                         * (self.rng.random(len(self.a)) < 0.35), 0.02, 1.0)
        self.burst *= 0.90
        # 动作流
        self._act_left -= dt
        if self._act_left <= 0:
            if self.action is not None:          # 刚结束一个动作 → 间隔
                self.action = None
                self._act_left = self.rng.uniform(0.25, 0.9)
            else:                                # 开始下一个动作
                keys, descs, ws, big = zip(*ACTIONS)
                i = self.rng.choice(len(ACTIONS), p=np.asarray(ws) / sum(ws))
                self.action = (keys[i], descs[i])
                self._act_left = self.rng.uniform(0.6, 2.2)
                self.burst = 1.0 if big[i] else 0.55

    @property
    def mean(self) -> float:
        return float(self.a.mean())


class StateFileSource:
    """从 ann_dashboard 的状态文件读真实活动度;过期自动回落演示。"""

    def __init__(self, path: Path, fallback: DemoSource):
        self.path = Path(path)
        self.fb = fallback
        self._last_read = 0.0
        self._cache = None

    def _read(self):
        if time.perf_counter() - self._last_read < 0.5:
            return self._cache
        self._last_read = time.perf_counter()
        try:
            self._cache = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:
            self._cache = None
        return self._cache

    def fresh(self) -> bool:
        try:
            return time.time() - self.path.stat().st_mtime < 5.0
        except Exception:
            return False

    def activity(self):
        s = self._read()
        if self.fresh() and s and s.get("activity"):
            return np.asarray(s["activity"], dtype=np.float32), True
        self.fb.tick(1 / 30)
        return self.fb.a, False

    def header(self):
        s = self._read() or {}
        return (f"帧 {s.get('frame', 0):,}", f"{(s.get('fps') or 0):.0f} f/s",
                str(s.get("status") or "")[:44])


class Overlay:
    def __init__(self, args):
        import tkinter as tk
        from PIL import Image, ImageTk  # noqa: F401

        self.tk, self.Image, self.ImageTk = tk, Image, ImageTk
        self.scale = args.scale
        self.W, self.H = int(392 * args.scale), int(560 * args.scale)
        self.fps = args.fps
        self.f_real = _load_font(int(13 * args.scale))
        self.f_title = _load_font(int(13 * args.scale), True)
        self.f_lab = _load_font(int(10 * args.scale))
        self.f_val = _load_font(int(16 * args.scale), True)
        self.f_act = _load_font(int(17 * args.scale), True)
        self.f_foot = _load_font(int(9 * args.scale))

        # ---- 点云(真实解剖布局 + 合成 z 轴深度)
        lay = json.loads((ROOT / ".cache/ann_layout.json").read_text(encoding="utf-8"))
        self.lx = np.asarray(lay["x"], np.float32) - 0.5
        self.ly = np.asarray(lay["y"], np.float32) - 0.5
        groups = np.asarray(lay["group"])
        rng = np.random.default_rng(20261005)
        ug = {g: i for i, g in enumerate(sorted(set(groups)))}
        gz = np.array([((ug[g] * 0.618) % 1.0) - 0.5 for g in groups], np.float32)
        self.lz = np.clip(gz * 0.55 + rng.standard_normal(len(groups)) * 0.07,
                          -0.42, 0.42).astype(np.float32)
        self.demo = DemoSource(groups)
        self.src = (StateFileSource(ROOT / args.state, self.demo)
                    if args.state else None)
        self.spark = deque([0.2] * 90, maxlen=90)
        self.theta = 0.0
        self._last_action_desc = None

        # ---- 窗口
        self.root = tk.Tk()
        self.root.overrideredirect(True)
        self.root.config(bg="#%02x%02x%02x" % KEY)
        self.root.attributes("-transparentcolor", "#%02x%02x%02x" % KEY)
        self.root.geometry(f"{self.W}x{self.H}+{args.x}+{args.y}")
        self.label = tk.Label(self.root, bd=0, bg="#%02x%02x%02x" % KEY)
        self.label.pack()
        self.root.update_idletasks()
        self.root.update()
        self.root.attributes("-topmost", True)
        self._apply_exstyle(clickthrough=not args.interactive)
        # tk 在属性变更时会重写 EXSTYLE(实测会丢 WS_EX_TOPMOST),延迟再钉一次
        self.root.after(1000, lambda: self._apply_exstyle(clickthrough=not args.interactive))
        self._photo = None
        self.root.after(int(1000 / self.fps), self._tick)

    def _apply_exstyle(self, clickthrough: bool):
        user32 = ctypes.windll.user32
        hwnd = user32.GetParent(self.root.winfo_id())
        ex = user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
        ex |= WS_EX_LAYERED | WS_EX_TOOLWINDOW | WS_EX_TOPMOST | WS_EX_NOACTIVATE
        if clickthrough:
            ex |= WS_EX_TRANSPARENT
        user32.SetWindowLongW(hwnd, GWL_EXSTYLE, ex)
        user32.SetWindowPos(hwnd, -1, 0, 0, 0, 0, 0x0001 | 0x0002 | 0x0010)  # HWND_TOPMOST

    # ------------------------------------------------------------ 渲染

    def _panel(self, img):
        """PIL 侧:圆角面板底 + 边框 + 标题 + 分隔线。"""
        from PIL import ImageDraw

        d = ImageDraw.Draw(img)
        m = int(10 * self.scale)
        r = int(18 * self.scale)
        d.rounded_rectangle([m, m, self.W - m, self.H - m], radius=r,
                            fill=PANEL, outline=BORDER, width=2)
        # 标题行
        y = int(20 * self.scale)
        pulse = 120 + int(80 * math.sin(time.perf_counter() * 3))
        d.ellipse([m + 16 * self.scale, y + 3, m + 16 * self.scale + 8 * self.scale,
                   y + 3 + 8 * self.scale], fill=(pulse, GREEN[1], GREEN[2]))
        d.text((m + 30 * self.scale, y - 2), "CYBER-FLY · MaleCNS v1.0",
               font=self.f_title, fill=TITLE_C)
        d.text((self.W - m - 14 * self.scale, y - 2),
               time.strftime("%H:%M:%S"), font=self.f_foot, fill=DIM_C, anchor="ra")
        d.line([m + 12, int(42 * self.scale), self.W - m - 12, int(42 * self.scale)],
               fill=BORDER, width=1)
        return d

    def _brain(self, arr, act):
        """numpy 侧:3D 旋转 + 投影 + 按活动度上色画点。"""
        s = self.scale
        cx, cy = self.W / 2, int(196 * s)
        bw, bh = int(336 * s), int(252 * s)
        self.theta = (self.theta + 0.010) % (2 * math.pi)
        c, sn = math.cos(self.theta), math.sin(self.theta)
        x2 = self.lx * c + self.lz * sn
        z2 = -self.lx * sn + self.lz * c
        dpt = 1.0 / (1.0 - 0.38 * z2)
        px = (cx + x2 * bw * dpt).astype(np.int32)
        py = (cy + self.ly * bh * dpt).astype(np.int32)
        ok = (px > 12) & (px < self.W - 12) & (py > int(44 * s)) & (py < int(330 * s))
        px, py, a, dpt = px[ok], py[ok], act[ok], dpt[ok]
        # 色彩对比:静息暗蓝可见,高活动烧成亮橙
        hot = np.clip((a - 0.18) * 1.7, 0.0, 1.0) ** 0.8
        br = np.clip(0.38 + 0.62 * hot, 0, 1) * np.clip(0.55 + 0.45 * dpt, 0.4, 1.25)
        col = np.empty((len(px), 3), np.float32)
        for k in range(3):   # 暗蓝(60,100,160) -> 亮橙 ACCENT
            col[:, k] = (60 + (ACCENT[k] - 60) * hot) * br
        col = col.astype(np.uint8)
        big = a > 0.78
        mid = (a > 0.52) & ~big
        small = ~big & ~mid
        Hh, Ww = arr.shape[:2]
        for m_, sel in ((0, small), (1, mid), (2, big)):
            if not sel.any():
                continue
            xs, ys, cs = px[sel], py[sel], col[sel]
            for oy in (-1, 0, 1):
                for ox in (-1, 0, 1):
                    if m_ == 0 and (ox or oy):
                        continue
                    if m_ == 1 and abs(ox) + abs(oy) > 1:
                        continue
                    arr[np.clip(ys + oy, 0, Hh - 1), np.clip(xs + ox, 0, Ww - 1)] = cs

    def _spark_curve(self, d):
        """PIL 侧:活动度火花线。"""
        s = self.scale
        x0, x1 = int(24 * s), self.W - int(24 * s)
        y0, y1 = int(352 * s), int(392 * s)
        d.rectangle([x0, y0, x1, y1], fill=(10, 14, 19), outline=BORDER)
        vals = np.asarray(self.spark, np.float32)
        hi = max(0.35, float(vals.max()) * 1.15)
        pts = [(x0 + (x1 - x0) * i / (len(vals) - 1),
                y1 - (y1 - y0 - 3) * float(v) / hi - 2) for i, v in enumerate(vals)]
        d.line(pts, fill=ACCENT, width=2)
        d.text((x1 - 6, y0 + 3), "神经活动", font=self.f_foot, fill=DIM_C, anchor="ra")

    def _texts(self, d, act_mean, kpi, action, real_note, status=""):
        """PIL 侧:KPI + 动作大字 + 页脚。"""
        s = self.scale
        m = int(10 * s)
        y0 = int(404 * s)
        for i, (lab, val) in enumerate(kpi):
            xx = m + 16 * s + i * ((self.W - 2 * m) / 4)
            d.text((xx, y0), lab, font=self.f_lab, fill=DIM_C)
            d.text((xx, y0 + int(15 * s)), val, font=self.f_val, fill=WHITE)
        # 动作大字
        y1 = int(458 * s)
        d.line([m + 12, y1 - int(10 * s), self.W - m - 12, y1 - int(10 * s)],
               fill=BORDER, width=1)
        if action is None:
            d.text((self.W / 2, y1 + int(16 * s)), "· 待机 ·", font=self.f_act,
                   fill=DIM_C, anchor="mm")
        else:
            key, desc = action
            d.text((self.W / 2 - int(6 * s), y1 + int(16 * s)), f"按住 {key}",
                   font=self.f_act, fill=ACCENT, anchor="rm")
            d.text((self.W / 2 + int(6 * s), y1 + int(16 * s)), f"— {desc}",
                   font=self.f_act, fill=WHITE, anchor="lm")
        d.text((self.W / 2, self.H - int(22 * s)),
               real_note + (f" · {status}" if status else ""),
               font=self.f_foot, fill=DIM_C, anchor="mm")

    def _tick(self):
        from PIL import ImageDraw

        t0 = time.perf_counter()
        dt = 1.0 / self.fps
        real_note = "演示数据 · 合成神经活动"
        frame_s, status, kpi_fps = "—", "", f"{self.fps} f/s"
        action = None
        # ---- 数据:真实状态文件(新鲜) → 优先;否则演示源
        act = None
        if self.src is not None:
            act, real = self.src.activity()
            if real:
                real_note = "实时连接 · 训练状态"
                frame_s, fps_s, status = self.src.header()
                kpi_fps = fps_s if "f/s" in fps_s else f"{fps_s} f/s"
        if act is None:
            self.demo.tick(dt)
            act = self.demo.a
        action = self.demo.action
        if action is not None and action[1] != self._last_action_desc:
            self.demo.burst = max(self.demo.burst, 0.8)
        self._last_action_desc = action[1] if action else None
        self.spark.append(float(np.mean(act)))

        # ---- 画
        img = self.Image.new("RGB", (self.W, self.H), KEY)
        d = self._panel(img)
        arr = np.asarray(img).copy()
        self._brain(arr, act)
        img = self.Image.fromarray(arr)
        d = ImageDraw.Draw(img)
        self._spark_curve(d)
        mean = float(np.mean(act))
        kpi = [("神经元", "166,700"), ("突触", "25.58M"),
               ("活动度", f"{mean*100:.0f}%"), ("帧率", kpi_fps)]
        self._texts(d, mean, kpi, action, real_note, status)
        ph = self.ImageTk.PhotoImage(img)
        self._photo = ph
        self.label.config(image=ph)
        # ---- 帧率自适应
        el = time.perf_counter() - t0
        delay = max(4, int(1000 / self.fps - el * 1000))
        self.root.after(delay, self._tick)

    def run(self):
        self.root.mainloop()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--x", type=int, default=-1, help="左上角 x(默认右上区)")
    ap.add_argument("--y", type=int, default=120)
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--state", default="", help="ann_dashboard 状态文件(挂真实训练)")
    ap.add_argument("--interactive", action="store_true",
                    help="关闭点击穿透(允许鼠标点击/拖动;调试用)")
    args = ap.parse_args()
    import os

    print(f"fly_overlay pid={os.getpid()}(关闭: taskkill /F /PID {os.getpid()})",
          flush=True)
    if args.x < 0:
        try:
            import tkinter as tk

            r = tk.Tk()
            args.x = r.winfo_screenwidth() - int(392 * args.scale) - 48
            r.destroy()
        except Exception:
            args.x = 1400
    Overlay(args).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
