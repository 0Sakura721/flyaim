"""指针路径探针:测「注入计数 -> 光标像素」的真实增益与线性性。

===============================================================================
为什么用它取代 Raw Input(D27 的实机校验层)
===============================================================================
标定链条里,"注入 N 个计数"和"游戏收到 N 个计数"之间隔着一个 **系统指针路径**,
而它可能不是 1:1 —— 最常见的就是 Windows 的「提高指针精确度」(EPP/鼠标加速):

    注入 800 计数  ->  光标实际走了 959 px     (实测,本机,EPP 开启)
                     比值 1.199 ≠ 1.0

**Raw Input 读不到这个失真** —— 它在 EPP 之前就抓走了原始计数,永远读回 1.0,
等于把问题遮盖掉。而 `GetCursorPos` 读的是**经过 EPP 之后**的最终光标位置,
恰好是这个失真真正落脚的地方。所以本探针比 Raw Input 更有诊断价值。

三个能测出来的东西:
1. **增益 ratio** = 光标像素 / 注入计数。≠1 说明指针路径有缩放。
2. **线性性** = 换几个幅度(300 / 1200 / 300)比值是否恒定。
   EPP 的本质就是"随速度变化的非线性增益",所以**单点测量必然漏掉它**,
   必须多幅度。本探针就是为此设计。
3. **方向正确性** = dx>0 光标是否往 +x 走(y 同理,注意屏幕 y 向下)。

⚠️ 边界(必须诚实说明):
   - 本探针测的是 **Windows 桌面指针路径**。游戏若用 Raw Input(多数 FPS 都
     这样),**不吃 EPP**,所以本探针的 ratio≠1 不必然意味着游戏里也偏。
     它的价值是"抓 EPP 是否在起作用"这个事实,以及"注入通道是否畅通"。
   - 游戏内是否真吃 EPP,只能在游戏里量 —— 那是 `--verify-counts` 的活。
   - 探针会把光标移到指定锚点再测,测完**移回原位**。

⚠️ 合规:只用标准 `SetCursorPos` / `GetCursorPos` / `SendInput`,不做任何
   反作弊规避。探针本身不点击、不按键。
"""

from __future__ import annotations

import ctypes
import time
from dataclasses import dataclass
from ctypes import wintypes

_USER32 = ctypes.WinDLL("user32", use_last_error=True)


class POINT(ctypes.Structure):
    _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]


@dataclass
class PointerProbe:
    """一次"注入 -> 读光标"的结果。"""

    sent: tuple[int, int]
    moved: tuple[int, int]
    ax: int          # 锚点
    ay: int

    @property
    def ratio_x(self) -> float | None:
        return (self.moved[0] / self.sent[0]) if self.sent[0] else None

    @property
    def ratio_y(self) -> float | None:
        return (self.moved[1] / self.sent[1]) if self.sent[1] else None

    def describe(self) -> dict:
        return {
            "sent": list(self.sent),
            "moved": list(self.moved),
            "ratio_x": None if self.ratio_x is None else round(self.ratio_x, 4),
            "ratio_y": None if self.ratio_y is None else round(self.ratio_y, 4),
        }


def get_cursor() -> tuple[int, int]:
    p = POINT()
    _USER32.GetCursorPos(ctypes.byref(p))
    return int(p.x), int(p.y)


def set_cursor(x: int, y: int) -> None:
    _USER32.SetCursorPos(int(x), int(y))


def screen_size() -> tuple[int, int]:
    return (int(_USER32.GetSystemMetrics(0)),
            int(_USER32.GetSystemMetrics(1)))


def measure_once(
    sink, counts: tuple[int, int], anchor: tuple[int, int],
    settle_s: float = 0.04,
) -> PointerProbe:
    """把光标放到 anchor,注入 counts,读回实际位移。

    ⚠️ 必须先把光标放在**屏幕中央附近**再测:x 方向注入几百计数就会撞到屏幕
    边缘被钳位,量出来的 ratio 会假性偏小。调用方传 anchor = 屏幕中心。
    """
    set_cursor(*anchor)
    time.sleep(0.01)
    before = get_cursor()
    if counts != (0, 0):
        sink.send(int(counts[0]), int(counts[1]))
        time.sleep(settle_s)
    after = get_cursor()
    return PointerProbe(
        sent=counts,
        moved=(after[0] - before[0], after[1] - before[1]),
        ax=anchor[0], ay=anchor[1],
    )


def judge(rows: list[PointerProbe], tol: float = 0.02) -> tuple[bool, list[str]]:
    """判定探针结果。返回 (passed, 输出行)。"""
    lines: list[str] = []
    lines.append("  幅度      注入       光标位移    比值")
    ratios: list[float] = []
    for r in rows:
        rx = r.ratio_x if r.ratio_x is not None else r.ratio_y
        if rx is not None:
            ratios.append(rx)
        lines.append(
            f"  {abs(r.sent[0] or r.sent[1]):>5}  "
            f"({r.sent[0]:>5},{r.sent[1]:>4})  "
            f"({r.moved[0]:>6},{r.moved[1]:>4})  "
            f"{rx:.4f}" if rx is not None else
            f"  {abs(r.sent[0] or r.sent[1]):>5}  "
            f"({r.sent[0]:>5},{r.sent[1]:>4})  "
            f"({r.moved[0]:>6},{r.moved[1]:>4})  n/a"
        )
    if not ratios:
        return False, lines + ["  🔴 没有任何有效比值 —— 注入通道可能不通"]
    lo, hi = min(ratios), max(ratios)
    span = hi - lo
    passed = True

    lines.extend(clamp_warn(rows))

    # 判据 1:注入必须真的让光标动起来
    moved_any = any(r.moved != (0, 0) for r in rows)
    if not moved_any:
        passed = False
        lines.append("  🔴 光标完全没动 —— 注入未生效")
    else:
        lines.append("  ✅ 注入生效:光标确实移动了")

    # 判据 2:比值恒定 = 线性
    if span > tol:
        passed = False
        lines.append(
            f"  🔴 比值随幅度变化(区间 [{lo:.4f},{hi:.4f}],跨度 {span:.4f} > {tol})"
        )
        lines.append(
            "     这是**非线性**的直接证据:指针加速/角度捕捉还在起作用。"
        )
        lines.append(
            "     → 关掉「提高指针精确度」(控制面板→鼠标→指针选项)后重跑;"
        )
        lines.append(
            "     → 若你的游戏用 Raw Input,它不吃 EPP,这条不影响游戏内标定,"
            "但仍建议关掉以消除歧义。"
        )
    else:
        lines.append(
            f"  ✅ 比值在各幅度下恒定(跨度 {span:.4f} ≤ {tol}):指针路径线性"
        )

    # 判据 3:比值整体偏离 1 说明有固定缩放
    mean = sum(ratios) / len(ratios)
    if abs(mean - 1.0) > tol:
        lines.append(
            f"  ⚠️ 平均比值 {mean:.4f} ≠ 1:指针路径有固定缩放(EPP 的线性段,"
            f"或系统 DPI 缩放)。桌面指针会偏 {abs(mean-1)*100:.1f}%。"
        )
        lines.append(
            "     游戏若用 Raw Input 则不受影响;若吃 EPP,请在游戏里用 "
            "--verify-counts 实量。"
        )
    else:
        lines.append(f"  ✅ 平均比值 {mean:.4f} ≈ 1:指针路径无整体缩放")
    return passed, lines


def clamp_warn(rows: list[PointerProbe], margin: int = 8) -> list[str]:
    """检测"撞边缘钳位"造成的假性偏低比值。

    实测踩过:注入 800 与 1600 时位移都是 1899px —— 完全相同,因为光标已经
    顶到屏幕右边缘被钳住。此时比值 2.37 / 1.19 都是**假的**(真实增益更高)。
    判定:连续两个不同幅度给出几乎相同的位移(差 < margin),即视为钳位。
    """
    out: list[str] = []
    for i in range(1, len(rows)):
        a, b = rows[i - 1], rows[i]
        if abs(a.sent[0] or a.sent[1]) == abs(b.sent[0] or b.sent[1]):
            continue
        axis = 0 if a.sent[0] else 1
        if abs(b.moved[axis] - a.moved[axis]) < margin:
            out.append(
                f"  ⚠️ 幅度 {abs(a.sent[0] or a.sent[1])} 与 "
                f"{abs(b.sent[0] or b.sent[1])} 的位移几乎相同"
                f"({a.moved[axis]} vs {b.moved[axis]} px)= **撞屏幕边缘被钳位**,"
                "这两点的比值不可信。"
            )
    if out:
        out.append("     → 用更小的幅度,或把 anchor 移到屏幕左侧再测。")
    return out


def run_probe(
    sink, magnitudes: tuple[int, ...] = (300, 1200, 300),
    axis: str = "x", settle_s: float = 0.04, restore: bool = True,
) -> tuple[list[PointerProbe], int, int]:
    """在屏幕中心附近跑一组探针,测完把光标移回原处。

    返回 (rows, 原始光标 x, 原始光标 y)。

    ⚠️ anchor 的选取要保证"最大幅度也不会撞到边缘":否则大位移被钳位,
    比值假性偏低。见 `clamp_warn`。
    """
    sw, sh = screen_size()
    # anchor 尽量靠左,给正向注入留出最大空间
    ax = 20
    ay = sh // 2
    ox, oy = get_cursor()
    rows: list[PointerProbe] = []
    try:
        for m in magnitudes:
            counts = (m, 0) if axis == "x" else (0, m)
            # 每个幅度都回到同一个 anchor,保证量的是纯净位移
            rows.append(measure_once(sink, counts, (ax, ay), settle_s))
    finally:
        if restore:
            set_cursor(ox, oy)
    return rows, ox, oy
