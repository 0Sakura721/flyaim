"""赛博果蝇悬浮窗(对齐 BV1fQYB6oECS 实机设计)。

    & $py tools/fly_overlay.py                 # 启动(配置存 .cache/fly_overlay_config.json)
    & $py tools/fly_overlay.py --state .cache/ann2_state.json   # 挂真实训练活动度
    & $py tools/fly_overlay.py --reset         # 重置配置

布局对照视频(2026-10-05 逐帧核对):
    标题栏 FLY | 神经活动 → 状态行"● 神经活动报告中" → **左脑右蝇**(琥珀色
    脑点云 + 果蝇插画与指示箭头) → **键帽行**(W A S D SHIFT SPACE,按下点亮)
    → 神经元/突触统计 → 3×2 分组数据格(带进度条) → 控制增量(鼠标 Δ) →
    累计统计 → 底部全宽波形条 → **JSON 事件控制台**(真实输入事件流)。
    与视频一致:脑云静置微旋、琥珀主色;与我此前版本不同:无大字动作行,
    动作信息由键帽 + 事件流承载。

交互(全部真数据):
    - 键帽/事件流 = GetAsyncKeyState 真实轮询;脑云分群随真实输入激发
      (移动→VNC 运动群,鼠标移动→视叶,攻击→cb_motor),页脚诚实标注。
    - WDA_EXCLUDEFROMCAPTURE:屏幕捕获(果蝇的眼睛)看不到本窗。
    - 全局热键:Ctrl+Alt+方向键 移动 | +- 缩放 | C 穿透 | X 捕获排除 |
      H 隐藏 | R 重载配置 | Q 退出;几何/开关持久化。
"""

from __future__ import annotations

import argparse
import ctypes
import ctypes.wintypes as wt
import json
import math
import os
import queue
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / ".cache" / "fly_overlay_config.json"

# ---------------------------------------------------------------- Win32

GWL_EXSTYLE = -20
WS_EX_LAYERED = 0x00080000
WS_EX_TRANSPARENT = 0x00000020
WS_EX_TOPMOST = 0x00000008
WS_EX_TOOLWINDOW = 0x00000080
WS_EX_NOACTIVATE = 0x08000000
WDA_NONE = 0x00
WDA_EXCLUDEFROMCAPTURE = 0x11
MOD_ALT = 0x1
MOD_CONTROL = 0x2
WM_HOTKEY = 0x0312
VK_OEM_PLUS = 0xBB
VK_OEM_MINUS = 0xBD

KEY = (1, 2, 3)            # 透明色键
PANEL = (13, 18, 26)       # 面板底(视频取色近似 #0d1219)
PANEL2 = (7, 10, 15)       # 控制台底
BORDER = (38, 48, 62)
TITLE_C = (210, 220, 232)
DIM_C = (110, 124, 140)
ACCENT = (232, 150, 60)
AMBER_HI = (255, 196, 110)
CYAN = (80, 200, 190)
RED_C = (224, 96, 96)
GREEN = (60, 200, 110)
WHITE = (232, 240, 250)
CHIP_BG = (34, 44, 58)

DEFAULT_CONFIG = {
    "x": None, "y": 100, "scale": 1.0, "clickthrough": True,
    "hide": False, "rotate_deg_per_frame": 0.04,   # 视频里脑云近乎静置
    "show": {"brain": True, "stats": True, "spark": True, "console": True},
    "exclude_from_capture": True,
    # 自定义监视键:vk=虚拟键码,label=键帽,desc=动作,groups=激发分群
    "watch_keys": [
        {"vk": 0x57, "label": "W", "desc": "向前移动",
         "groups": ["vnc_motor", "descending_neuron"]},
        {"vk": 0x41, "label": "A", "desc": "向左移动", "groups": ["vnc_motor"]},
        {"vk": 0x53, "label": "S", "desc": "向后移动", "groups": ["vnc_motor"]},
        {"vk": 0x44, "label": "D", "desc": "向右移动", "groups": ["vnc_motor"]},
        {"vk": 0xA0, "label": "SHIFT", "desc": "冲刺",
         "groups": ["vnc_motor", "vnc_efferent"]},
        {"vk": 0x20, "label": "SPACE", "desc": "跳跃",
         "groups": ["vnc_efferent", "vnc_motor"]},
        {"vk": 0x01, "label": "M1", "desc": "普通攻击", "groups": ["cb_motor"]},
        {"vk": 0x02, "label": "M2", "desc": "瞄准",
         "groups": ["cb_sensory", "visual_projection"]},
    ],
    "cap_keys": ["W", "A", "S", "D", "SHIFT", "SPACE"],   # 键帽行显示顺序
}


def load_config() -> dict:
    if CONFIG_PATH.exists():
        try:
            return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        except Exception:
            pass
    cfg = json.loads(json.dumps(DEFAULT_CONFIG))
    save_config(cfg)
    return cfg


def save_config(cfg: dict) -> None:
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(5):
        try:
            CONFIG_PATH.write_text(json.dumps(cfg, ensure_ascii=False, indent=1),
                                   encoding="utf-8")
            return
        except PermissionError:
            time.sleep(0.02 * (attempt + 1))


def _load_font(size: int, bold: bool = False):
    from PIL import ImageFont

    for name in (("msyhbd.ttc" if bold else "msyh.ttc"), "msyh.ttc", "arial.ttf"):
        try:
            return ImageFont.truetype(f"C:\\Windows\\Fonts\\{name}", size)
        except Exception:
            continue
    return ImageFont.load_default()


class InputMirror:
    """真实输入镜像:按键状态 + 鼠标速度 + 事件流(供控制台)。"""

    def __init__(self, watch: list[dict]):
        self.user32 = ctypes.windll.user32
        self.watch = watch
        self._prev_cur = None
        self.events: deque[str] = deque(maxlen=9)
        self._last_mouse_log = 0.0
        # 每秒窗统计
        self.win_t0 = time.perf_counter()
        self.win_events = 0
        self.win_dx = 0
        self.win_dy = 0
        self.rate = 0.0          # 事件/秒(上一秒)
        self.speed = 0.0         # 鼠标 px/s(上一秒)

    def poll(self):
        held = [k for k in self.watch
                if self.user32.GetAsyncKeyState(int(k["vk"])) & 0x8000]

        class POINT(ctypes.Structure):
            _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]

        p = POINT()
        dx = dy = 0
        if self.user32.GetCursorPos(ctypes.byref(p)):
            cur = (p.x, p.y)
            if self._prev_cur is not None:
                dx, dy = cur[0] - self._prev_cur[0], cur[1] - self._prev_cur[1]
            self._prev_cur = cur
        self.win_dx += dx
        self.win_dy += dy
        # 事件流:mouse_move 节流 160ms
        now = time.perf_counter()
        if (dx or dy) and now - self._last_mouse_log > 0.16:
            self._last_mouse_log = now
            self.log_event("mouse_move", f'"dx":{dx},"dy":{dy}')
            self.win_events += 1
        # 每秒结算
        if now - self.win_t0 >= 1.0:
            self.rate = self.win_events / (now - self.win_t0)
            self.speed = math.hypot(self.win_dx, self.win_dy) / (now - self.win_t0)
            self.win_t0, self.win_events = now, 0
            self.win_dx, self.win_dy = 0, 0
        return held, (dx, dy)

    def log_event(self, kind: str, payload: str = ""):
        t = time.strftime("%H:%M:%S", time.localtime()) + f".{int(time.time()*1000)%1000:03d}"
        line = f'{{"ts":"{t}","type":"{kind}"'
        if payload:
            line += f",{payload}"
        self.events.append(line + "}")

    def key_down(self, k):
        self.log_event("key_down", f'"key":"{k["label"]}","desc":"{k["desc"]}"')
        self.win_events += 1

    def key_up(self, k):
        self.log_event("key_up", f'"key":"{k["label"]}"')
        self.win_events += 1


class HotkeyThread(threading.Thread):
    def __init__(self, cmd_q: "queue.Queue"):
        super().__init__(daemon=True, name="overlay-hotkey")
        self.q = cmd_q
        self.keys = [
            (MOD_CONTROL | MOD_ALT, 0x25),          # 0 ←
            (MOD_CONTROL | MOD_ALT, 0x27),          # 1 →
            (MOD_CONTROL | MOD_ALT, 0x26),          # 2 ↑
            (MOD_CONTROL | MOD_ALT, 0x28),          # 3 ↓
            (MOD_CONTROL | MOD_ALT, VK_OEM_PLUS),   # 4 放大
            (MOD_CONTROL | MOD_ALT, VK_OEM_MINUS),  # 5 缩小
            (MOD_CONTROL | MOD_ALT, 0x43),          # 6 C 穿透
            (MOD_CONTROL | MOD_ALT, 0x42),          # 7 B 脑图
            (MOD_CONTROL | MOD_ALT, 0x54),          # 8 T 统计格
            (MOD_CONTROL | MOD_ALT, 0x4B),          # 9 K 波形
            (MOD_CONTROL | MOD_ALT, 0x4A),          # 10 J 控制台
            (MOD_CONTROL | MOD_ALT, 0x58),          # 11 X 捕获排除
            (MOD_CONTROL | MOD_ALT, 0x48),          # 12 H 隐藏
            (MOD_CONTROL | MOD_ALT, 0x52),          # 13 R 重载配置
            (MOD_CONTROL | MOD_ALT, 0x51),          # 14 Q 退出
        ]

    def run(self):
        user32 = ctypes.windll.user32
        for i, (mod, vk) in enumerate(self.keys, 1):
            user32.RegisterHotKey(None, i, mod | 0x4000, vk)
        self.tid = ctypes.windll.kernel32.GetCurrentThreadId()
        msg = wt.MSG()
        while user32.GetMessageW(ctypes.byref(msg), None, 0, 0) > 0:
            if msg.message == WM_HOTKEY:
                self.q.put(int(msg.wParam) - 1)
        for i in range(1, len(self.keys) + 1):
            user32.UnregisterHotKey(None, i)

    def stop(self):
        ctypes.windll.kernel32.PostThreadMessageW(getattr(self, "tid", 0), 0x0012, 0, 0)


class Overlay:
    def __init__(self, args):
        import tkinter as tk
        from PIL import Image, ImageTk

        self.tk, self.Image, self.ImageTk = tk, Image, ImageTk
        self.cfg = load_config()
        if args.reset:
            self.cfg = json.loads(json.dumps(DEFAULT_CONFIG))
        self.scale = float(args.scale) if args.scale else float(self.cfg.get("scale", 1.0))
        self.show = self.cfg.get("show", dict(DEFAULT_CONFIG["show"]))
        self.clickthrough = bool(self.cfg.get("clickthrough", True))
        self.exclude = bool(self.cfg.get("exclude_from_capture", True))
        self.hide = bool(self.cfg.get("hide", False))
        self.rot = float(self.cfg.get("rotate_deg_per_frame", 0.25)) * math.pi / 180.0
        self.fps = args.fps
        self.state_path = args.state

        self.f_title = _load_font(int(12 * self.scale), True)
        self.f_lab = _load_font(int(9 * self.scale))
        self.f_val = _load_font(int(12 * self.scale), True)
        self.f_sm = _load_font(int(10 * self.scale))
        self.f_con = _load_font(int(9 * self.scale))
        self.f_cap = _load_font(int(10 * self.scale), True)
        self.f_foot = _load_font(int(9 * self.scale))

        # ---- 点云
        lay = json.loads((ROOT / ".cache/ann_layout.json").read_text(encoding="utf-8"))
        self.lx = np.asarray(lay["x"], np.float32) - 0.5
        self.ly = np.asarray(lay["y"], np.float32) - 0.5
        self.groups = np.asarray(lay["group"])
        rng = np.random.default_rng(20261005)
        ug = {g: i for i, g in enumerate(sorted(set(self.groups)))}
        self.g_idx = {g: np.flatnonzero(self.groups == g) for g in ug}
        gz = np.array([((ug[g] * 0.618) % 1.0) - 0.5 for g in self.groups], np.float32)
        self.lz = np.clip(gz * 0.55 + rng.standard_normal(len(self.groups)) * 0.07,
                          -0.42, 0.42).astype(np.float32)
        self.a = np.full(len(self.groups), 0.22, np.float32)
        self.theta = 0.0
        self.spark = deque([0.2] * 110, maxlen=110)

        # ---- 输入 / 统计
        self._reload_watch()
        self.mirror = InputMirror(self.watch)
        self.held_prev: set[str] = set()
        self.total_events = 0
        self.frames = 0
        self.t_start = time.perf_counter()

        # ---- 热键
        self.cmd_q: "queue.Queue" = queue.Queue()
        self.hotkeys = HotkeyThread(self.cmd_q)
        self.hotkeys.start()

        # ---- 窗口
        self.W, self.H = int(424 * self.scale), int(636 * self.scale)
        self.root = tk.Tk()
        self.root.overrideredirect(True)
        self.root.config(bg="#%02x%02x%02x" % KEY)
        self.root.attributes("-transparentcolor", "#%02x%02x%02x" % KEY)
        x = self.cfg.get("x")
        y = self.cfg.get("y", 100)
        if x is None:
            x = self.root.winfo_screenwidth() - self.W - 48
        self.root.geometry(f"{self.W}x{self.H}+{int(x)}+{int(y)}")
        self.label = tk.Label(self.root, bd=0, bg="#%02x%02x%02x" % KEY)
        self.label.pack()
        self.root.update_idletasks()
        self.root.update()
        self.root.attributes("-topmost", True)
        self._hwnd = ctypes.windll.user32.GetParent(self.root.winfo_id())
        self._apply_exstyle()
        self._apply_capture_affinity()
        self.root.after(1000, self._reassert_style)
        self.root.after(int(1000 / self.fps), self._tick)
        if self.hide:
            self.root.withdraw()

    # ------------------------------------------------------------ 窗口属性

    def _apply_exstyle(self):
        user32 = ctypes.windll.user32
        ex = user32.GetWindowLongW(self._hwnd, GWL_EXSTYLE)
        ex |= WS_EX_LAYERED | WS_EX_TOOLWINDOW | WS_EX_NOACTIVATE | WS_EX_TOPMOST
        ex = (ex | WS_EX_TRANSPARENT) if self.clickthrough else (ex & ~WS_EX_TRANSPARENT)
        user32.SetWindowLongW(self._hwnd, GWL_EXSTYLE, ex)
        user32.SetWindowPos(self._hwnd, -1, 0, 0, 0, 0, 0x0001 | 0x0002 | 0x0010)

    def _apply_capture_affinity(self):
        user32 = ctypes.windll.user32
        aff = WDA_EXCLUDEFROMCAPTURE if self.exclude else WDA_NONE
        if not user32.SetWindowDisplayAffinity(self._hwnd, aff):
            print("[overlay] SetWindowDisplayAffinity 失败(需 Win10 2004+)", flush=True)

    def _reassert_style(self):
        self._apply_exstyle()
        self._apply_capture_affinity()

    def _save_geometry(self):
        self.cfg.update(x=self.root.winfo_x(), y=self.root.winfo_y(),
                        scale=self.scale, clickthrough=self.clickthrough,
                        show=self.show, exclude_from_capture=self.exclude,
                        hide=self.hide)
        save_config(self.cfg)

    def _reload_watch(self):
        cfg = load_config()
        self.watch = cfg.get("watch_keys", DEFAULT_CONFIG["watch_keys"])
        self.show = cfg.get("show", self.show)
        self.cap_keys = cfg.get("cap_keys", DEFAULT_CONFIG["cap_keys"])
        if getattr(self, "mirror", None) is not None:
            self.mirror.watch = self.watch

    # ------------------------------------------------------------ 热键

    def _handle_cmd(self, idx: int):
        s = 24
        act = {0: lambda: self._move(-s, 0), 1: lambda: self._move(s, 0),
               2: lambda: self._move(0, -s), 3: lambda: self._move(0, s),
               4: lambda: self._rescale(1.1), 5: lambda: self._rescale(0.9),
               6: self._toggle_click, 7: lambda: self._tog("brain"),
               8: lambda: self._tog("stats"), 9: lambda: self._tog("spark"),
               10: lambda: self._tog("console"), 11: self._toggle_excl,
               12: self._toggle_hide, 13: self._reload_watch,
               14: self._quit}
        fn = act.get(idx)
        if fn:
            fn()
        if idx != 14:
            self._save_geometry()

    def _toggle_click(self):
        self.clickthrough = not self.clickthrough
        self._apply_exstyle()

    def _toggle_excl(self):
        self.exclude = not self.exclude
        self._apply_capture_affinity()

    def _toggle_hide(self):
        self.hide = not self.hide
        self.root.withdraw() if self.hide else self.root.deiconify()

    def _tog(self, key):
        self.show[key] = not self.show.get(key, True)

    def _move(self, dx, dy):
        self.root.geometry(f"+{self.root.winfo_x() + dx}+{self.root.winfo_y() + dy}")

    def _rescale(self, factor):
        self.scale = float(np.clip(self.scale * factor, 0.6, 2.2))
        w, h = int(424 * self.scale), int(636 * self.scale)
        self.root.geometry(f"{w}x{h}")
        self.f_title = _load_font(int(12 * self.scale), True)
        self.f_lab = _load_font(int(9 * self.scale))
        self.f_val = _load_font(int(12 * self.scale), True)
        self.f_sm = _load_font(int(10 * self.scale))
        self.f_con = _load_font(int(9 * self.scale))
        self.f_cap = _load_font(int(10 * self.scale), True)
        self.f_foot = _load_font(int(9 * self.scale))

    def _quit(self):
        self._save_geometry()
        self.hotkeys.stop()
        self.root.destroy()

    # ------------------------------------------------------------ 脑活动

    def _brain_activity(self):
        """真实输入 → 分群激发;--state 新鲜时优先真实训练活动度。

        返回 (活动度, held, 说明, (mdx, mdy))。
        """
        held, (mdx, mdy) = self.mirror.poll()
        excl = " · 已排除捕获" if self.exclude else ""
        if self.state_path:
            p = ROOT / self.state_path
            try:
                s = json.loads(p.read_text(encoding="utf-8"))
                if time.time() - p.stat().st_mtime < 5.0 and s.get("activity"):
                    a = np.asarray(s["activity"], np.float32)
                    if a.size == len(self.a):
                        return a, held, "实时训练状态" + excl, (mdx, mdy)
            except Exception:
                pass
        target = 0.20 + 0.06 * np.sin(
            np.arange(len(self.a)) * 0.013 + time.perf_counter() * 0.4)
        for k in held:
            for g in k.get("groups", []):
                idx = self.g_idx.get(g)
                if idx is not None:
                    target[idx] = np.minimum(1.0, target[idx] + 0.45)
        mspeed = abs(mdx) + abs(mdy)
        if mspeed:
            for g in ("ol_sensory", "ol_intrinsic", "visual_projection"):
                idx = self.g_idx.get(g)
                if idx is not None:
                    target[idx] = np.minimum(1.0, target[idx] + min(0.5, mspeed * 0.02))
        self.a = 0.80 * self.a + 0.20 * (target + 0.05 * np.random.default_rng(
            1).standard_normal(len(self.a))).astype(np.float32)
        return self.a, held, "输入映射 · 非真实仿真" + excl, (mdx, mdy)

    # ------------------------------------------------------------ 渲染部件

    def _frame_pil(self):
        """PIL 侧:面板底 + 标题栏 + 状态行。"""
        from PIL import ImageDraw

        img = self.Image.new("RGB", (self.W, self.H), KEY)
        d = ImageDraw.Draw(img)
        s = self.scale
        m = int(8 * s)
        d.rectangle([m, m, self.W - m, self.H - m], fill=PANEL, outline=BORDER,
                    width=1)
        d.line([m, int(26 * s), self.W - m, int(26 * s)], fill=BORDER, width=1)
        d.text((m + 10 * s, int(6 * s)), "FLY / 神经活动", font=self.f_title,
               fill=TITLE_C)
        right = "已排除捕获" if self.exclude else "捕获可见"
        d.text((self.W - m - 10 * s, int(7 * s)), right, font=self.f_foot,
               fill=(GREEN if self.exclude else RED_C), anchor="ra")
        pulse = 120 + int(80 * math.sin(time.perf_counter() * 3))
        d.ellipse([m + 10 * s, int(32 * s), m + 17 * s, int(39 * s)],
                  fill=(pulse, GREEN[1], GREEN[2]))
        d.text((m + 22 * s, int(31 * s)), "神经活动报告中", font=self.f_sm,
               fill=TITLE_C)
        return img, d, m, s

    def _brain(self, arr, act, box):
        """numpy 侧:脑点云(琥珀主色,静置微旋)。box=(x0,y0,x1,y1)。"""
        s = self.scale
        x0, y0, x1, y1 = box
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2 + int(6 * s)
        bw, bh = int((x1 - x0) * 0.94), int((y1 - y0) * 0.88)
        self.theta = (self.theta + self.rot) % (2 * math.pi)
        c, sn = math.cos(self.theta), math.sin(self.theta)
        x2 = self.lx * c + self.lz * sn
        z2 = -self.lx * sn + self.lz * c
        dpt = 1.0 / (1.0 - 0.38 * z2)
        px = (cx + x2 * bw * dpt).astype(np.int32)
        py = (cy + self.ly * bh * dpt).astype(np.int32)
        ok = (px >= x0 + 2) & (px < x1 - 2) & (py >= y0 + 2) & (py < y1 - 2)
        px, py, a, dpt = px[ok], py[ok], act[ok], dpt[ok]
        # 视频观感:静息也有清晰的琥珀色,活跃处烧成亮琥珀
        hot = np.clip((a - 0.10) * 1.45, 0.0, 1.0) ** 0.8
        br = np.clip(0.62 + 0.38 * hot, 0, 1) * np.clip(0.55 + 0.45 * dpt, 0.4, 1.25)
        col = np.empty((len(px), 3), np.float32)
        for k in range(3):   # 暗琥珀 (128,98,60) -> 亮琥珀 AMBER_HI
            col[:, k] = (128 + (AMBER_HI[k] - 128) * hot) * br
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
                    arr[np.clip(ys + oy, 0, Hh - 1),
                        np.clip(xs + ox, 0, Ww - 1)] = cs

    def _fly(self, d, cx, cy, size, last_dx):
        """PIL 侧:果蝇插画(头朝上、红复眼、翅后掠)+ 指示箭头。"""
        s = self.scale
        u = size / 100.0
        # 翅膀(半透明浅灰蓝,从胸向后上方掠出,先画在底层)
        for sx in (-1, 1):
            d.polygon([(cx + sx * 6 * u, cy - 16 * u),
                       (cx + sx * 58 * u, cy - 54 * u),
                       (cx + sx * 62 * u, cy - 34 * u),
                       (cx + sx * 22 * u, cy + 4 * u)],
                      fill=(148, 172, 198), outline=(196, 214, 232))
        # 腿(每侧 3 条,细)
        for sx in (-1, 1):
            for i, (x2, y2) in enumerate(((26, 2), (30, 16), (24, 30))):
                d.line([cx + sx * 8 * u, cy - 12 * u + i * 12 * u,
                        cx + sx * x2 * u, cy + y2 * u],
                       fill=(96, 66, 28), width=max(1, int(1.5 * u)))
        # 腹部(深琥珀带条纹)
        d.ellipse([cx - 9 * u, cy + 0 * u, cx + 9 * u, cy + 46 * u],
                  fill=(168, 120, 54), outline=(104, 72, 30))
        for yy in (10, 20, 30):
            d.line([cx - 8 * u, cy + yy * u, cx + 8 * u, cy + yy * u],
                   fill=(112, 78, 34), width=max(1, int(2 * u)))
        # 胸
        d.ellipse([cx - 11 * u, cy - 24 * u, cx + 11 * u, cy + 4 * u],
                  fill=(198, 148, 72), outline=(104, 72, 30))
        # 头 + 大红复眼(占头大半,视频里最醒目的特征)
        d.ellipse([cx - 9 * u, cy - 38 * u, cx + 9 * u, cy - 22 * u],
                  fill=(190, 142, 66), outline=(104, 72, 30))
        d.ellipse([cx - 8 * u, cy - 36 * u, cx - 0.5 * u, cy - 24 * u],
                  fill=(200, 42, 40))
        d.ellipse([cx + 0.5 * u, cy - 36 * u, cx + 8 * u, cy - 24 * u],
                  fill=(218, 56, 46))
        # 指示箭头(随最近鼠标水平方向,视频里的橙色弧箭头)
        ax = cx + int(66 * u)
        ay = cy - int(30 * u)
        if last_dx != 0:
            sx = 1 if last_dx > 0 else -1
            d.line([ax - sx * 18 * u, ay + 8 * u, ax + sx * 12 * u, ay - 4 * u],
                   fill=ACCENT, width=max(2, int(3.4 * u)))
            d.polygon([(ax + sx * 20 * u, ay - 8 * u),
                       (ax + sx * 8 * u, ay - 14 * u),
                       (ax + sx * 14 * u, ay + 2 * u)], fill=ACCENT)

    def _keycaps(self, d, x0, y0, held_labels, width):
        """PIL 侧:键帽行(按下的点亮),撑满给定宽度。"""
        s = self.scale
        n = max(1, len(self.cap_keys))
        gap = int(5 * s)
        cw = (width - gap * (n - 1)) / n
        ch = int(28 * s)
        x = x0
        for name in self.cap_keys:
            on = name in held_labels
            d.rounded_rectangle([x, y0, x + cw, y0 + ch], radius=int(4 * s),
                                fill=(ACCENT if on else CHIP_BG),
                                outline=(ACCENT if on else BORDER), width=1)
            d.text((x + cw / 2, y0 + ch / 2), name, font=self.f_cap,
                   fill=(30, 20, 8) if on else DIM_C, anchor="mm")
            x += cw + gap

    def _stat_cell(self, d, x0, y0, w, label, val, frac, color):
        s = self.scale
        d.text((x0, y0), label, font=self.f_lab, fill=DIM_C)
        d.text((x0 + w - 4 * s, y0 - int(1 * s)), val, font=self.f_sm,
               fill=TITLE_C, anchor="ra")
        bx0, bx1 = x0, x0 + w - int(6 * s)
        by = y0 + int(15 * s)
        d.rectangle([bx0, by, bx1, by + int(4 * s)], fill=(24, 32, 42))
        fw = int((bx1 - bx0) * float(np.clip(frac, 0, 1)))
        if fw > 0:
            d.rectangle([bx0, by, bx0 + fw, by + int(4 * s)], fill=color)

    def _stats_grid(self, d, x0, y0, act, held, note):
        """PIL 侧:3×2 分组数据格(视频里的表格)。"""
        s = self.scale
        w = (self.W - 2 * x0 - int(16 * s)) / 3
        groups = {
            "视叶": ("ol_sensory", "ol_intrinsic", "visual_projection"),
            "中央脑": ("cb_intrinsic", "cb_sensory", "cb_motor"),
            "腹神经索": ("vnc_intrinsic", "vnc_motor", "vnc_efferent"),
        }
        cells = []
        for i, (lab, gs) in enumerate(groups.items()):
            idx = np.concatenate([self.g_idx[g] for g in gs if g in self.g_idx])
            m = float(np.mean(act[idx])) if idx.size else 0.0
            cells.append((lab, f"{m*100:.0f}%", m, ACCENT))
        cells += [("鼠标", f"{self.mirror.speed:.0f}px/s",
                   min(1.0, self.mirror.speed / 800), CYAN),
                  ("按键", f"{len(held)}", min(1.0, len(held) / 3), GREEN),
                  ("事件", f"{self.mirror.rate:.0f}/s",
                   min(1.0, self.mirror.rate / 20), TITLE_C)]
        for i, (lab, val, frac, col) in enumerate(cells):
            cx = x0 + (i % 3) * (w + int(8 * s))
            cy = y0 + (i // 3) * int(36 * s)
            self._stat_cell(d, cx, cy, w, lab, val, frac, col)

    def _spark(self, d, x0, y0, x1, y1):
        """PIL 侧:底部全宽波形条(活动度,青色细线)。"""
        d.rectangle([x0, y0, x1, y1], fill=PANEL2, outline=BORDER)
        vals = np.asarray(self.spark, np.float32)
        hi = max(0.35, float(vals.max()) * 1.15)
        pts = [(x0 + (x1 - x0) * i / (len(vals) - 1),
                y1 - (y1 - y0 - 2) * float(v) / hi - 1) for i, v in enumerate(vals)]
        d.line(pts, fill=CYAN, width=1)
        d.text((x1 - 5, y0 + 2), "60s", font=self.f_lab, fill=DIM_C, anchor="ra")

    def _console(self, d, x0, y0, x1, y1):
        """PIL 侧:JSON 事件控制台(真实输入事件流)。"""
        s = self.scale
        d.rectangle([x0, y0, x1, y1], fill=PANEL2, outline=BORDER)
        d.text((x0 + 6, y0 + 2), "input_events", font=self.f_lab, fill=DIM_C)
        n = max(3, int((y1 - y0 - int(18 * s)) / (11 * s)))
        evs = list(self.mirror.events)[-n:]
        yy = y0 + int(16 * s)
        for line in evs:
            if yy > y1 - int(12 * s):
                break
            d.text((x0 + 6, yy), line, font=self.f_con, fill=(150, 165, 182))
            yy += int(11 * s)

    # ------------------------------------------------------------ 主循环

    def _tick(self):
        from PIL import ImageDraw

        t0 = time.perf_counter()
        while True:
            try:
                idx = self.cmd_q.get_nowait()
            except queue.Empty:
                break
            self._handle_cmd(idx)
            if not self.root.winfo_exists():
                return
        act, held, note, (mdx, mdy) = self._brain_activity()
        # 按键 down/up 事件(边缘检测)
        labels = {k["label"]: k for k in held}
        for k in held:
            if k["label"] not in self.held_prev:
                self.mirror.key_down(k)
        for lab in self.held_prev - set(labels):
            self.mirror.key_up({"label": lab})
        self.held_prev = set(labels)
        self.total_events = self.frames
        self.spark.append(float(np.mean(act)))
        self.frames += 1

        s = self.scale
        img, d, m, _ = self._frame_pil()
        # 布局行
        brain_box = (m, int(46 * s), m + int(258 * s), int(322 * s))
        arr = np.asarray(img).copy()
        if self.show.get("brain", True):
            self._brain(arr, act, brain_box)
            img = self.Image.fromarray(arr)
            d = ImageDraw.Draw(img)
        else:
            d.rectangle(brain_box, fill=PANEL2, outline=BORDER)
            d.text(((brain_box[0] + brain_box[2]) / 2,
                    (brain_box[1] + brain_box[3]) / 2), "脑图已隐藏",
                   font=self.f_sm, fill=DIM_C, anchor="mm")
        # 右列:果蝇 + 神经元统计 + 当前动作
        rx0 = m + int(262 * s)
        self._fly(d, rx0 + int(56 * s), int(100 * s), int(64 * s),
                  0 if not mdx else (1 if mdx > 0 else -1))
        d.text((rx0 + int(6 * s), int(158 * s)), "166,700 个神经元",
               font=self.f_lab, fill=DIM_C)
        d.text((rx0 + int(6 * s), int(172 * s)), "25.58M 个突触",
               font=self.f_lab, fill=DIM_C)
        ol_idx = np.concatenate([self.g_idx[g] for g in
                                 ("ol_sensory", "ol_intrinsic", "visual_projection")
                                 if g in self.g_idx])
        d.text((rx0 + int(6 * s), int(196 * s)),
               f"视叶活动 {float(np.mean(act[ol_idx]))*100:.0f}%",
               font=self.f_lab, fill=DIM_C)
        if held:
            k = held[0]
            d.text((rx0 + int(6 * s), int(216 * s)),
                   f"当前: {k['label']}" +
                   (f" +{len(held)-1}" if len(held) > 1 else ""),
                   font=self.f_sm, fill=ACCENT)
            d.text((rx0 + int(6 * s), int(232 * s)), k["desc"],
                   font=self.f_sm, fill=TITLE_C)
        else:
            d.text((rx0 + int(6 * s), int(216 * s)), "当前: 待机",
                   font=self.f_sm, fill=DIM_C)
        # 全宽键帽行(视频同款,按下的点亮)
        self._keycaps(d, m + int(4 * s), int(330 * s), set(labels),
                      self.W - 2 * m - int(8 * s))
        d.text((m + int(6 * s), int(366 * s)),
               f"本次运行 {self.frames} 帧 · 用时 {time.perf_counter()-self.t_start:.0f}s",
               font=self.f_sm, fill=TITLE_C)
        if self.show.get("stats", True):
            self._stats_grid(d, m + int(6 * s), int(390 * s), act, held, note)
        # 控制增量(鼠标)
        d.text((m + int(6 * s), int(468 * s)),
               f"鼠标增量 Δx {self.mirror.win_dx:+d} · Δy {self.mirror.win_dy:+d} px/s"
               f"   事件 {self.mirror.rate:.1f}/s",
               font=self.f_sm, fill=CYAN)
        # 波形
        if self.show.get("spark", True):
            self._spark(d, m, int(492 * s), self.W - m, int(530 * s))
        # 控制台
        if self.show.get("console", True):
            self._console(d, m, int(536 * s), self.W - m, self.H - m - int(22 * s))
        d.text((self.W / 2, self.H - m - int(11 * s)), note,
               font=self.f_foot, fill=DIM_C, anchor="mm")
        ph = self.ImageTk.PhotoImage(img)
        self._photo = ph
        self.label.config(image=ph)
        el = time.perf_counter() - t0
        self.root.after(max(4, int(1000 / self.fps - el * 1000)), self._tick)

    def run(self):
        print(f"fly_overlay pid={os.getpid()}", flush=True)
        print("热键: Ctrl+Alt+方向键 移动 | +- 缩放 | C 穿透 | B 脑图 | T 统计 | "
              "K 波形 | J 控制台 | X 捕获排除 | H 隐藏 | R 重载配置 | Q 退出",
              flush=True)
        self.root.mainloop()
        self.hotkeys.stop()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--x", type=int, default=-1)
    ap.add_argument("--y", type=int, default=100)
    ap.add_argument("--scale", type=float, default=0.0, help="覆盖配置的缩放")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--state", default="", help="ann_dashboard 状态文件")
    ap.add_argument("--reset", action="store_true", help="忽略并重置配置")
    args = ap.parse_args()
    Overlay(args).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
