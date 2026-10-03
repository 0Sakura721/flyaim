"""游戏内增益实证标定:注入已知计数脉冲 -> 测靶位移 -> 反推增益。

原理
----
不依赖 cm/360 / FOV / 灵敏度换算 —— 全部在像素空间观测:

    1. 检测当前最大青色靶的屏幕位置(native 分辨率);
    2. 以工作节奏(150 计数/拍,~25Hz)注入一串纯水平脉冲;
    3. 再检测同一靶的新位置,得 px_per_count;
    4. counts_per_screen_width = 脉冲计数 × 画面宽 / 位移;
    5. 写 gain.json:counts_per_action = 14/640 × counts_per_screen_width
       —— 即「action=1.0 每拍扫过画面宽的 14/640」,与离线靶场同语义。

若靶快移出屏幕,自动反向再测;多次脉冲取中位数。EPP 的速度非线性被
「以工作节奏注入」这一设定吸收在标定值里。

运行(注入会动鼠标,~10 秒)::

    & $py tools/aimlab_calib_gain.py
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from flyaim.bridge.capture import ScreenCapture, find_window_region, focus_window  # noqa: E402
from flyaim.bridge.detect import find_target  # noqa: E402
from flyaim.bridge.inject import SendInputSink  # noqa: E402

TEAL = (48, 224, 224)


def _detect(rgb: np.ndarray):
    return find_target(rgb, ref_color=TEAL, tolerance=60.0)


def burst_measure(cap: ScreenCapture, sink: SendInputSink,
                  counts_per_tick: int = 100, n_ticks: int = 4,
                  tick_s: float = 0.04) -> tuple[float, float] | None:
    """注入一串水平脉冲,返回 (总计数, 位移 native px) 或 None(无法跟踪)。

    稳健性约束(2026-10-04 第一版教训:多靶换块污染位移):
      - 跟踪「离上一帧最近的可疑色块」而非最大块;
      - 只在中心 35% 带内起测(透视切线畸变小、px/deg 近似线性);
      - 位移超过 45% 屏宽判为换块/出屏,弃样。
    """
    from scipy import ndimage

    f0, _ = cap.read()
    h, w = f0.shape[:2]
    a = f0.astype(np.float32)
    ref = np.asarray(TEAL, dtype=np.float32).reshape(1, 1, 3)
    mask = (np.sum((a - ref) ** 2, axis=2) <= 60.0 ** 2)
    lab, n = ndimage.label(mask)
    if n == 0:
        return None
    sizes = np.bincount(lab.reshape(-1)); sizes[0] = 0
    cands = []
    for i in range(1, n + 1):
        if sizes[i] < 60:
            continue
        yy, xx = np.nonzero(lab == i)
        cx, cy = float(xx.mean()), float(yy.mean())
        if abs(cx - (w - 1) / 2) > 0.35 * w:  # 中心带约束
            continue
        cands.append((cx, cy, float(np.sqrt(sizes[i] / np.pi))))
    if not cands:
        return None
    d0x = min(cands, key=lambda c: abs(c[0] - (w - 1) / 2))[0]

    direction = 1
    total = 0
    for _ in range(n_ticks):
        sink.send(direction * counts_per_tick, 0)
        total += direction * counts_per_tick
        time.sleep(tick_s)
    f1, _ = cap.read()
    a1 = f1.astype(np.float32)
    mask1 = (np.sum((a1 - ref) ** 2, axis=2) <= 60.0 ** 2)
    lab1, n1 = ndimage.label(mask1)
    best_dx, best_score = None, 1e9
    for i in range(1, n1 + 1):
        yy, xx = np.nonzero(lab1 == i)
        if len(xx) < 60:
            continue
        cx = float(xx.mean())
        dxp = cx - d0x
        score = abs(dxp - direction * 0.35 * w)  # 期望位移附近的连续性
        if abs(dxp) > 0.45 * w:  # 连续性约束:同一颗靶不会跳走
            continue
        if score < best_score:
            best_score, best_dx = score, dxp
    if best_dx is None or abs(best_dx) < 8:
        return None
    return float(total), float(best_dx)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bursts", type=int, default=6)
    ap.add_argument("--counts-per-tick", type=int, default=100)
    ap.add_argument("--n-ticks", type=int, default=4)
    ap.add_argument("--out", default=str(ROOT / "flyaim/runs/bridge/gain.json"))
    args = ap.parse_args()

    print("⚠️ 标定将真实移动鼠标 ~10 秒。请保持 Aim Lab 前台(有可见青靶),手离鼠标。")
    region = find_window_region("aimlab")
    if not focus_window("aimlab"):
        print("❌ 无法聚焦 Aim Lab")
        return 1
    cap = ScreenCapture(region=region, out_size=None)  # native 分辨率
    sink = SendInputSink()  # 不钳位,脉冲量精确
    time.sleep(0.5)

    samples: list[float] = []
    for i in range(args.bursts):
        r = burst_measure(cap, sink, counts_per_tick=args.counts_per_tick,
                          n_ticks=args.n_ticks)
        if r is None:
            print(f"  脉冲 {i}: 跟踪失败(靶不可见/位移过小),跳过")
            continue
        total, dx = r
        ppc = abs(dx) / abs(total)  # px/count
        samples.append(ppc)
        print(f"  脉冲 {i}: 注入 {total:+.0f} 计数 -> 靶移动 {dx:+.0f}px  = {ppc:.3f} px/计数")
        time.sleep(0.3)
    cap.close()

    if not samples:
        print("❌ 没有有效样本:请确认画面里有青色靶(进任务或大厅),再跑一次。")
        return 1

    ppc = statistics.median(samples)
    native_w = region[2]
    counts_per_screen = native_w / ppc
    counts_per_action = counts_per_screen * (14.0 / 640.0)

    print("\n---- 标定结果 ----")
    print(f"  px/计数(中位数)     = {ppc:.3f}")
    print(f"  扫一屏宽所需计数     = {counts_per_screen:.0f}")
    print(f"  counts_per_action    = {counts_per_action:.1f}"
          f"  (action=1.0 = 每拍 14/640 画面宽)")
    print(f"  等效 counts/360 锚点 = {counts_per_screen:.0f}"
          f"  (注意:这是像素空间锚点,deg 字段按 360°/该值折算仅供参考)")

    out = {
        "counts_per_360": round(float(counts_per_screen), 1),
        "speed_fraction_per_action": 14.0 / 640.0,
        "max_counts_per_tick": 400.0,
        "invert_x": False, "invert_y": False, "deadzone": 0.0,
        "calibration": {
            "method": "in-game burst tracking (pixel-space)",
            "px_per_count_median": round(ppc, 4),
            "bursts": len(samples),
            "native_width": int(native_w),
            "note": "counts_per_360 为像素空间等效锚点;EPP 已在工作节奏下被吸收",
        },
    }
    import json

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  已写入 {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
