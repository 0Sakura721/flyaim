"""赛博果蝇悬浮窗(交互版):真实按键镜像 + 捕获排除 + 热键调整。

    & $py tools/fly_overlay.py                 # 启动(配置存 .cache/fly_overlay_config.json)
    & $py tools/fly_overlay.py --state .cache/ann2_state.json   # 挂真实训练活动度
    & $py tools/fly_overlay.py --reset         # 重置配置

与初版的差别(应用户要求):
    1. **不再有演示数据**:动作行实时镜像你真实按下的键(GetAsyncKeyState);
       脑点云按**真实输入**激发对应解剖分群(移动键→VNC 运动群,鼠标移动→
       视叶光流,攻击→中央脑运动群)。诚实标注:这是输入映射,不是真实仿真。
    2. **自定义按键**:编辑配置文件 watch_keys(VK 码/标签/描述/映射分群),
       Ctrl+Alt+R 热重载;**开关功能**:Ctrl+Alt+B/K/S/A 分别开关脑图/KPI/
       火花线/动作区。
    3. **果蝇忽略浮窗**:SetWindowDisplayAffinity(WDA_EXCLUDEFROMCAPTURE)
       把窗口从屏幕捕获中排除(果蝇的 bettercam/DXGI 捕获看不到浮窗;
       Win10 2004+)。Ctrl+Alt+X 切换。
    4. **浮窗调整**:全局热键(浮窗穿透时鼠标点不到,必须用系统热键):
       Ctrl+Alt+方向键移动,+- 缩放,C 切换穿透,H 隐藏/显示,Q 退出;
       位置/尺寸/开关状态全部持久化。

脑活动数据优先级:--state 真实训练活动度(新鲜时) > 输入映射 > 静息噪声。
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

KEY = (1, 2, 3)
PANEL = (16, 22, 29)
BORDER = (42, 54, 68)
TITLE_C = (216, 226, 238)
DIM_C = (123, 139, 160)
ACCENT = (232, 150, 60)
GREEN = (32, 192, 96)
WHITE = (234, 242, 255)

DEFAULT_CONFIG = {
    "x": None, "y": 120, "scale": 1.0, "clickthrough": True,
    "hide": False,
    "show": {"brain": True, "kpi": True, "spark": True, "action": True},
    "exclude_from_capture": True,
    # 自定义监视键:vk=虚拟键码,label=键名,desc=中文动作,groups=激发的解剖分群
    "watch_keys": [
        {"vk": 0x57, "label": "W", "desc": "向前移动",
         "groups": ["vnc_motor", "descending_neuron"]},
        {"vk": 0x41, "label": "A", "desc": "向左移动",
         "groups": ["vnc_motor"]},
        {"vk": 0x53, "label": "S", "desc": "向后移动",
         "groups": ["vnc_motor"]},
        {"vk": 0x44, "label": "D", "desc": "向右移动",
         "groups": ["vnc_motor"]},
        {"vk": 0x20, "label": "SPACE", "desc": "跳跃",
         "groups": ["vnc_efferent", "vnc_motor"]},
        {"vk": 0xA0, "label": "SHIFT", "desc": "冲刺",
         "groups": ["vnc_motor", "vnc_efferent"]},
        {"vk": 0x01, "label": "MOUSE1", "desc": "普通攻击",
         "groups": ["cb_motor"]},
        {"vk": 0x02, "label": "MOUSE2", "desc": "瞄准",
         "groups": ["cb_sensory", "visual_projection"]},
    ],
}


def load_config() -> dict:
    if CONFIG_PATH.exists():
        try:
            return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        except Exception:
            pass
    cfg = json.loads(json.dumps(DEFAULT_CONFIG))   # deep copy
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
    """真实按键镜像:GetAsyncKeyState 轮询 + 鼠标速度(→视叶光流)。"""

    def __init__(self, watch: list[dict]):
        self.user32 = ctypes.windll.user32
        self.watch = watch            # [{vk,label,desc,groups}]
        self._prev_cur = None

    def poll(self):
        """返回 (held 列表[{label,desc,groups}], 鼠标速度 px/tick)。"""
        held = []
        for k in self.watch:
            if self.user32.GetAsyncKeyState(int(k["vk"])) & 0x8000:
                held.append(k)

        class POINT(ctypes.Structure):
            _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]

        p = POINT()
        speed = 0
        if self.user32.GetCursorPos(ctypes.byref(p)):
            cur = (p.x, p.y)
            if self._prev_cur is not None:
                speed = abs(cur[0] - self._prev_cur[0]) + abs(cur[1] - self._prev_cur[1])
            self._prev_cur = cur
        return held, speed


class HotkeyThread(threading.Thread):
    """RegisterHotKey 后台线程;命令入队(避免跨线程摸 tkinter)。"""

    # (modifier, vk) — id 为数组下标
    def __init__(self, cmd_q: "queue.Queue"):
        super().__init__(daemon=True, name="overlay-hotkey")
        self.q = cmd_q
        self.keys = [
            (MOD_CONTROL | MOD_ALT, 0x25),   # 0 ← 移动
            (MOD_CONTROL | MOD_ALT, 0x27),   # 1 →
            (MOD_CONTROL | MOD_ALT, 0x26),   # 2 ↑
            (MOD_CONTROL | MOD_ALT, 0x28),   # 3 ↓
            (MOD_CONTROL | MOD_ALT, VK_OEM_PLUS),   # 4 放大
            (MOD_CONTROL | MOD_ALT, VK_OEM_MINUS),  # 5 缩小
            (MOD_CONTROL | MOD_ALT, 0x43),   # 6 C 穿透开关
            (MOD_CONTROL | MOD_ALT, 0x42),   # 7 B 脑图
            (MOD_CONTROL | MOD_ALT, 0x4B),   # 8 K KPI
            (MOD_CONTROL | MOD_ALT, 0x54),   # 9 T 火花线
            (MOD_CONTROL | MOD_ALT, 0x41),   # 10 A 动作区
            (MOD_CONTROL | MOD_ALT, 0x58),   # 11 X 捕获排除
            (MOD_CONTROL | MOD_ALT, 0x48),   # 12 H 隐藏/显示
            (MOD_CONTROL | MOD_ALT, 0x52),   # 13 R 重载配置
            (MOD_CONTROL | MOD_ALT, 0x51),   # 14 Q 退出
        ]

    def run(self):
        user32 = ctypes.windll.user32
        for i, (mod, vk) in enumerate(self.keys, 1):
            user32.RegisterHotKey(None, i, mod | 0x4000, vk)   # 0x4000 NOREPEAT
        self.tid = ctypes.windll.kernel32.GetCurrentThreadId()
        import ctypes.wintypes as wt

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
            self.cfg["x"] = args.x if args.x >= 0 else None
        self.scale = float(self.cfg.get("scale", 1.0))
        if args.scale and args.scale != 1.0:
            self.scale = args.scale
        self.show = self.cfg.get("show", dict(DEFAULT_CONFIG["show"]))
        self.clickthrough = bool(self.cfg.get("clickthrough", True))
        self.exclude = bool(self.cfg.get("exclude_from_capture", True))
        self.hide = bool(self.cfg.get("hide", False))
        self.fps = args.fps
        self.state_path = args.state

        self.f_title = _load_font(int(13 * self.scale), True)
        self.f_lab = _load_font(int(10 * self.scale))
        self.f_val = _load_font(int(16 * self.scale), True)
        self.f_act = _load_font(int(17 * self.scale), True)
        self.f_foot = _load_font(int(9 * self.scale))

        # ---- 点云(真实解剖布局 + 合成深度)
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
        self._prev_cur = None
        self.a = np.full(len(self.groups), 0.22, np.float32)
        self.spark = deque([0.2] * 90, maxlen=90)
        self.theta = 0.0

        # ---- 输入镜像
        self._reload_watch()
        self.mirror = InputMirror(self.watch)

        # ---- 热键线程
        self.cmd_q: "queue.Queue" = queue.Queue()
        self.hotkeys = HotkeyThread(self.cmd_q)
        self.hotkeys.start()

        # ---- 窗口
        self.W, self.H = int(392 * self.scale), int(560 * self.scale)
        self.root = tk.Tk()
        self.root.overrideredirect(True)
        self.root.config(bg="#%02x%02x%02x" % KEY)
        self.root.attributes("-transparentcolor", "#%02x%02x%02x" % KEY)
        x = self.cfg.get("x")
        y = self.cfg.get("y", 120)
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
        if self.clickthrough:
            ex |= WS_EX_TRANSPARENT
        else:
            ex &= ~WS_EX_TRANSPARENT
        user32.SetWindowLongW(self._hwnd, GWL_EXSTYLE, ex)
        user32.SetWindowPos(self._hwnd, -1, 0, 0, 0, 0, 0x0001 | 0x0002 | 0x0010)

    def _apply_capture_affinity(self):
        """WDA_EXCLUDEFROMCAPTURE:屏幕捕获(果蝇的眼睛)看不到本窗。"""
        user32 = ctypes.windll.user32
        aff = WDA_EXCLUDEFROMCAPTURE if self.exclude else WDA_NONE
        if not user32.SetWindowDisplayAffinity(self._hwnd, aff):
            print("[overlay] SetWindowDisplayAffinity 失败(需 Win10 2004+),"
                  "浮窗会被捕获看到", flush=True)

    def _reassert_style(self):
        self._apply_exstyle()
        self._apply_capture_affinity()

    def _save_geometry(self):
        self.cfg.update(x=self.root.winfo_x(), y=self.root.winfo_y(),
                        scale=self.scale, clickthrough=self.clickthrough,
                        show=self.show, exclude_from_capture=self.exclude,
                        hide=self.hide)
        save_config(self.cfg)

    # ------------------------------------------------------------ 配置

    def _reload_watch(self):
        cfg = load_config()
        self.watch = cfg.get("watch_keys", DEFAULT_CONFIG["watch_keys"])
        self.show = cfg.get("show", self.show)

    # ------------------------------------------------------------ 热键

    def _handle_cmd(self, idx: int):
        s = 20
        if idx == 0:
            self._move(-s, 0)
        elif idx == 1:
            self._move(s, 0)
        elif idx == 2:
            self._move(0, -s)
        elif idx == 3:
            self._move(0, s)
        elif idx == 4:
            self._rescale(1.1)
        elif idx == 5:
            self._rescale(0.9)
        elif idx == 6:
            self.clickthrough = not self.clickthrough
            self._apply_exstyle()
        elif idx == 7:
            self.show["brain"] = not self.show.get("brain", True)
        elif idx == 8:
            self.show["kpi"] = not self.show.get("kpi", True)
        elif idx == 9:
            self.show["spark"] = not self.show.get("spark", True)
        elif idx == 10:
            self.show["action"] = not self.show.get("action", True)
        elif idx == 11:
            self.exclude = not self.exclude
            self._apply_capture_affinity()
        elif idx == 12:
            self.hide = not self.hide
            self.root.withdraw() if self.hide else self.root.deiconify()
        elif idx == 13:
            self._reload_watch()
        elif idx == 14:
            self._quit()
            return
        self._save_geometry()

    def _move(self, dx, dy):
        self.root.geometry(f"+{self.root.winfo_x() + dx}+{self.root.winfo_y() + dy}")

    def _rescale(self, factor):
        self.scale = float(np.clip(self.scale * factor, 0.6, 2.5))
        w, h = int(392 * self.scale), int(560 * self.scale)
        self.root.geometry(f"{w}x{h}")
        self.f_title = _load_font(int(13 * self.scale), True)
        self.f_lab = _load_font(int(10 * self.scale))
        self.f_val = _load_font(int(16 * self.scale), True)
        self.f_act = _load_font(int(17 * self.scale), True)
        self.f_foot = _load_font(int(9 * self.scale))

    def _quit(self):
        self._save_geometry()
        self.hotkeys.stop()
        self.root.destroy()

    # ------------------------------------------------------------ 脑活动

    def _brain_activity(self, dt):
        """真实输入 → 分群激发;--state 新鲜时优先真实训练活动度。

        返回 (活动度, held 列表, 页脚说明)。
        """
        held, mspeed = self.mirror.poll()
        excl = " · 已排除捕获" if self.exclude else ""
        if self.state_path:
            p = ROOT / self.state_path
            try:
                s = json.loads(p.read_text(encoding="utf-8"))
                if time.time() - p.stat().st_mtime < 5.0 and s.get("activity"):
                    a = np.asarray(s["activity"], np.float32)
                    if a.size == len(self.a):
                        return a, held, "实时训练状态" + excl
            except Exception:
                pass
        target = 0.20 + 0.06 * np.sin(
            np.arange(len(self.a)) * 0.013 + time.perf_counter() * 0.4)
        for k in held:                      # 按键 → 映射分群
            for g in k.get("groups", []):
                idx = self.g_idx.get(g)
                if idx is not None:
                    target[idx] = np.minimum(1.0, target[idx] + 0.45)
        if mspeed:                          # 鼠标移动 → 视叶光流
            for g in ("ol_sensory", "ol_intrinsic", "visual_projection"):
                idx = self.g_idx.get(g)
                if idx is not None:
                    target[idx] = np.minimum(1.0, target[idx] + min(0.5, mspeed * 0.02))
        self.a = 0.80 * self.a + 0.20 * (target + 0.05 * np.random.default_rng(
            1).standard_normal(len(self.a))).astype(np.float32)
        return self.a, held, "输入映射 · 非真实仿真" + excl

    # ------------------------------------------------------------ 渲染

    def _panel(self, img):
        from PIL import ImageDraw

        d = ImageDraw.Draw(img)
        s = self.scale
        m, r = int(10 * s), int(18 * s)
        d.rounded_rectangle([m, m, self.W - m, self.H - m], radius=r,
                            fill=PANEL, outline=BORDER, width=2)
        y = int(20 * s)
        pulse = 120 + int(80 * math.sin(time.perf_counter() * 3))
        d.ellipse([m + 16 * s, y + 3, m + 24 * s, y + 11 * s],
                  fill=(pulse, GREEN[1], GREEN[2]))
        d.text((m + 30 * s, y - 2), "CYBER-FLY · MaleCNS v1.0",
               font=self.f_title, fill=TITLE_C)
        d.text((self.W - m - 14 * s, y - 2), time.strftime("%H:%M:%S"),
               font=self.f_foot, fill=DIM_C, anchor="ra")
        d.line([m + 12, int(42 * s), self.W - m - 12, int(42 * s)],
               fill=BORDER, width=1)
        return d

    def _brain(self, arr, act):
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
        top = int(44 * s) if self.show.get("spark", True) else int(44 * s)
        ok = (px > 12) & (px < self.W - 12) & (py > top) & (py < int(330 * s))
        px, py, a, dpt = px[ok], py[ok], act[ok], dpt[ok]
        hot = np.clip((a - 0.18) * 1.7, 0.0, 1.0) ** 0.8
        br = np.clip(0.38 + 0.62 * hot, 0, 1) * np.clip(0.55 + 0.45 * dpt, 0.4, 1.25)
        col = np.empty((len(px), 3), np.float32)
        for k in range(3):
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
                    arr[np.clip(ys + oy, 0, Hh - 1),
                        np.clip(xs + ox, 0, Ww - 1)] = cs

    def _spark_curve(self, d):
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

    def _texts(self, d, kpi, held, note):
        s = self.scale
        m = int(10 * s)
        y0 = int(404 * s)
        for i, (lab, val) in enumerate(kpi):
            xx = m + 16 * s + i * ((self.W - 2 * m) / 4)
            d.text((xx, y0), lab, font=self.f_lab, fill=DIM_C)
            d.text((xx, y0 + int(15 * s)), val, font=self.f_val, fill=WHITE)
        y1 = int(458 * s)
        d.line([m + 12, y1 - int(10 * s), self.W - m - 12, y1 - int(10 * s)],
               fill=BORDER, width=1)
        if not held:
            d.text((self.W / 2, y1 + int(16 * s)), "· 待机 ·", font=self.f_act,
                   fill=DIM_C, anchor="mm")
        elif len(held) == 1:
            k = held[0]
            d.text((self.W / 2 - int(6 * s), y1 + int(16 * s)), f"按住 {k['label']}",
                   font=self.f_act, fill=ACCENT, anchor="rm")
            d.text((self.W / 2 + int(6 * s), y1 + int(16 * s)), f"— {k['desc']}",
                   font=self.f_act, fill=WHITE, anchor="lm")
        else:   # 多键同按:紧凑列出
            keys = " + ".join(k["label"] for k in held)
            descs = " / ".join(k["desc"] for k in held[:3])
            d.text((self.W / 2, y1 + int(16 * s)), f"{keys} — {descs}",
                   font=self.f_act, fill=WHITE, anchor="mm")
        d.text((self.W / 2, self.H - int(22 * s)), note,
               font=self.f_foot, fill=DIM_C, anchor="mm")

    # ------------------------------------------------------------ 主循环

    def _tick(self):
        from PIL import ImageDraw

        t0 = time.perf_counter()
        # 热键命令(非阻塞)
        while True:
            try:
                idx = self.cmd_q.get_nowait()
            except queue.Empty:
                break
            self._handle_cmd(idx)
            if not self.root.winfo_exists():
                return
        out = self._brain_activity(1.0 / self.fps)
        act, held, note = out
        self.spark.append(float(np.mean(act)))

        img = self.Image.new("RGB", (self.W, self.H), KEY)
        d = self._panel(img)
        arr = np.asarray(img).copy()
        if self.show.get("brain", True):
            self._brain(arr, act)
        img = self.Image.fromarray(arr)
        d = ImageDraw.Draw(img)
        if self.show.get("spark", True):
            self._spark_curve(d)
        mean = float(np.mean(act))
        kpi = ([("神经元", "166,700"), ("突触", "25.58M"),
                ("活动度", f"{mean*100:.0f}%"),
                ("输入", f"{len(held)} 键" if held else "—")]
               if self.show.get("kpi", True) else [])
        self._texts(d, kpi, held, note)
        ph = self.ImageTk.PhotoImage(img)
        self._photo = ph
        self.label.config(image=ph)
        el = time.perf_counter() - t0
        self.root.after(max(4, int(1000 / self.fps - el * 1000)), self._tick)

    def run(self):
        print(f"fly_overlay pid={os.getpid()}", flush=True)
        print("热键: Ctrl+Alt+方向键 移动 | +- 缩放 | C 穿透 | B 脑图 | K KPI | "
              "T 火花线 | A 动作区 | X 捕获排除 | H 隐藏 | R 重载配置 | Q 退出",
              flush=True)
        self.root.mainloop()
        self.hotkeys.stop()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--x", type=int, default=-1)
    ap.add_argument("--y", type=int, default=120)
    ap.add_argument("--scale", type=float, default=0.0, help="覆盖配置的缩放")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--state", default="", help="ann_dashboard 状态文件")
    ap.add_argument("--reset", action="store_true", help="忽略并重置配置")
    args = ap.parse_args()
    Overlay(args).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
