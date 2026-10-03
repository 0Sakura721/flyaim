"""鼠标注入(ActionSink):把增益后的鼠标计数送进操作系统输入队列。

===============================================================================
设计
===============================================================================
`Arena.step()` 里「准星位移」的替代品。用 Win32 `SendInput` 注入**相对**
鼠标移动(`MOUSEEVENTF_MOVE`),这是 AutoHotkey 等标准自动化工具的同一条
路径 —— 不做驱动级注入(Interception/vHid),不做反作弊规避。

要点:
  - 单位是 **鼠标计数**(mickey),不是像素。最终准星转多少度由
    游戏灵敏度(GainConfig.counts_per_360)决定。
  - Windows「提高指针精确度」(指针加速度)对 Raw Input 消费者的注入
    增量语义有干扰风险:本模块提供 `pointer_accel_enabled()` 检测,
    CLI 会在开启时打印警告(建议在系统设置里关掉)。
  - `SendInputSink` 是唯一动真实输入的实现;测试一律用 `NullSink`。
"""

from __future__ import annotations

import ctypes
import logging
import time
from ctypes import wintypes

import numpy as np

logger = logging.getLogger(__name__)

_MOUSEEVENTF_MOVE = 0x0001
_MOUSEEVENTF_LEFTDOWN = 0x0002
_MOUSEEVENTF_LEFTUP = 0x0004
_MOUSEEVENTF_ABSOLUTE = 0x8000
_SENDINPUT_OK = 1


# ---------------------------------------------------------------- Win32 结构

ULONG_PTR = ctypes.c_size_t


class _MOUSEINPUT(ctypes.Structure):
    _fields_ = [
        ("dx", wintypes.LONG),
        ("dy", wintypes.LONG),
        ("mouseData", wintypes.DWORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ULONG_PTR),
    ]


class _KEYBDINPUT(ctypes.Structure):
    _fields_ = [
        ("wVk", wintypes.WORD),
        ("wScan", wintypes.WORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ULONG_PTR),
    ]


class _INPUTUNION(ctypes.Union):
    _fields_ = [("mi", _MOUSEINPUT), ("ki", _KEYBDINPUT)]


class _INPUT(ctypes.Structure):
    _anonymous_ = ("u",)
    _fields_ = [("type", wintypes.DWORD), ("u", _INPUTUNION)]


_INPUT_MOUSE = 0
_INPUT_KEYBOARD = 1
_KEYEVENTF_KEYUP = 0x0002


def pointer_accel_enabled() -> bool | None:
    """检测「提高指针精确度」(鼠标加速度)。True=开启;None=检测失败。

    SPI_GETMOUSE 取回 3 个参数;全 0 表示加速度关闭。
    """
    try:
        params = (wintypes.INT * 3)()
        ok = ctypes.windll.user32.SystemParametersInfoW(0x0003, 0, params, 0)
        if not ok:
            return None
        return bool(params[0] or params[1] or params[2])
    except Exception:
        return None


def cursor_position() -> tuple[int, int] | None:
    """当前光标位置(像素)。供注入冒烟测试移动后复位。"""
    try:
        pt = wintypes.POINT()
        if ctypes.windll.user32.GetCursorPos(ctypes.byref(pt)):
            return int(pt.x), int(pt.y)
    except Exception:
        pass
    return None


# ---------------------------------------------------------------- Sink 实现


class NullSink:
    """不注入任何输入,只记录。所有测试/彩排的默认选择。"""

    name = "null"

    def __init__(self) -> None:
        self.total_counts = np.zeros(2, dtype=np.int64)
        self.n_calls = 0
        self.last_counts = (0, 0)

    def send(self, dx: int, dy: int) -> bool:
        self.total_counts += (int(dx), int(dy))
        self.last_counts = (int(dx), int(dy))
        self.n_calls += 1
        return True

    def close(self) -> None:
        pass


class SendInputSink:
    """Win32 SendInput 相对鼠标移动。**会真实移动用户的鼠标**。"""

    name = "sendinput"

    def __init__(self, max_counts_per_tick: int | None = None) -> None:
        self.max_counts_per_tick = (
            None if max_counts_per_tick is None else int(max_counts_per_tick)
        )
        self.total_counts = np.zeros(2, dtype=np.int64)
        self.n_calls = 0
        self.last_counts = (0, 0)
        self.n_clicks = 0
        self._user32 = ctypes.windll.user32
        accel = pointer_accel_enabled()
        if accel:
            logger.warning(
                "SendInputSink: 检测到「提高指针精确度」已开启 —— 游戏若经由系统指针"
                "路径处理输入,注入增量会被非线性放大。建议关闭后再跑受控实验。"
            )

    def _send_raw(self, mi_flags: int, dx: int = 0, dy: int = 0) -> bool:
        inp = _INPUT()
        inp.type = _INPUT_MOUSE
        inp.mi.dx = dx
        inp.mi.dy = dy
        inp.mi.dwFlags = mi_flags
        n = self._user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(_INPUT))
        return n == _SENDINPUT_OK

    def send(self, dx: int, dy: int) -> bool:
        dx, dy = int(dx), int(dy)
        if self.max_counts_per_tick is not None:
            m = abs(self.max_counts_per_tick)
            dx = int(np.clip(dx, -m, m))
            dy = int(np.clip(dy, -m, m))
        if dx == 0 and dy == 0:
            self.last_counts = (0, 0)
            return True
        ok = self._send_raw(_MOUSEEVENTF_MOVE, dx, dy)
        self.n_calls += 1
        self.last_counts = (dx, dy)
        if ok:
            self.total_counts += (dx, dy)
            return True
        logger.error("SendInput 失败: delta=(%d, %d)", dx, dy)
        return False

    def click(self) -> bool:
        """左键单击(按下+抬起)。开火语义 = 靶场「每帧自动开火」的游戏版:
        由 TriggerOnTarget 在准星压住靶时调用,不是网络的输出。"""
        ok1 = self._send_raw(_MOUSEEVENTF_LEFTDOWN)
        ok2 = self._send_raw(_MOUSEEVENTF_LEFTUP)
        self.n_clicks += 1
        return ok1 and ok2

    def close(self) -> None:
        pass


def move_cursor_abs(x: int, y: int) -> bool:
    """绝对移动光标到屏幕像素 (x, y)(0-65535 归一化换算,主屏)。

    用于把光标摆到 UI 按钮上再 click()(如 Aim Lab 的「点击开始」)。
    """
    import ctypes

    try:
        ctypes.windll.user32.SetProcessDPIAware()
    except Exception:
        pass
    sw = int(ctypes.windll.user32.GetSystemMetrics(0))
    sh = int(ctypes.windll.user32.GetSystemMetrics(1))
    nx = int(np.clip(x, 0, sw - 1)) * 65535 // max(1, sw - 1)
    ny = int(np.clip(y, 0, sh - 1)) * 65535 // max(1, sh - 1)
    inp = _INPUT()
    inp.type = _INPUT_MOUSE
    inp.mi.dx = nx
    inp.mi.dy = ny
    inp.mi.dwFlags = _MOUSEEVENTF_MOVE | _MOUSEEVENTF_ABSOLUTE
    return ctypes.windll.user32.SendInput(
        1, ctypes.byref(inp), ctypes.sizeof(_INPUT)
    ) == _SENDINPUT_OK


def click_at(x: int, y: int) -> bool:
    """把光标移到屏幕 (x, y) 并左键单击(UI 自动化用,如开始任务)。"""
    if not move_cursor_abs(x, y):
        return False
    time.sleep(0.05)
    sink = SendInputSink()
    ok = sink.click()
    sink.close()
    return ok


def press_key(vk: int) -> bool:
    """注入一次按键按下+抬起(如 VK 0x52 = R,Aim Lab 的"再练一次"热键)。"""
    inp = _INPUT()
    inp.type = _INPUT_KEYBOARD
    inp.ki.wVk = int(vk)
    inp.ki.dwFlags = 0  # keydown
    ok1 = ctypes.windll.user32.SendInput(
        1, ctypes.byref(inp), ctypes.sizeof(_INPUT)) == _SENDINPUT_OK
    time.sleep(0.02)
    inp.ki.dwFlags = _KEYEVENTF_KEYUP
    ok2 = ctypes.windll.user32.SendInput(
        1, ctypes.byref(inp), ctypes.sizeof(_INPUT)) == _SENDINPUT_OK
    return ok1 and ok2


def cursor_nudge_check(dx: int = 3, dy: int = 0) -> tuple[tuple[int, int], tuple[int, int]] | None:
    """注入冒烟检查:光标移动 (dx,dy) 后立即复位。返回 (原位, 注入后位)。

    显式用户主动调用(aimlab_smoke.py --cursor-check),任何自动化测试
    都不得默认调用 —— 它会真的动鼠标。
    """
    before = cursor_position()
    if before is None:
        return None
    sink = SendInputSink()
    sink.send(dx, dy)
    time.sleep(0.05)
    after = cursor_position()
    sink.close()
    if after is None:
        return None
    back = SendInputSink()
    back.send(-dx, -dy)
    back.close()
    return before, after
