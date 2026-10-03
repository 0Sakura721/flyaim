"""补救尝试:时间积分能否把被混沌淹没的确定性视觉成分提取出来?

Lead 持有。逻辑依据:
    方差分解显示 DN 输出 = 刺激成分(确定性、跨时间恒定) + 自身历史成分(混沌、零均值涨落)。
    实测比值 SS_stim/SS_state < 1,即**单帧**信噪比 < 1。

    但两者性质不同:刺激成分在时间上是**恒定的**,而混沌涨落是**零均值的**。
    因此对 DN 放电率做长时间积分(或对读出输出做强低通),
    理论上应能把信噪比按 sqrt(T) 提升 —— 这是标准信号处理结论。

    若积分 N 帧后信噪比能超过 ~3,则"用连接组做瞄准"仍然可行,
    代价是**响应变慢**(每 N 帧才更新一次方向),但闭环仍可成立。

本脚本测量:不同积分窗口下,刺激间差异 / 历史噪声 的比值。

运行::

    <捆绑 python> tools/test_temporal_integration.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from flyaim.config import ArenaConfig, BrainConfig, ReadoutConfig, RetinaConfig  # noqa: E402
from flyaim.pipeline import FlySystem  # noqa: E402

D = ROOT / "flyaim" / "data" / "build"
NORM = ROOT / "flyaim" / "runs" / "norm"


def make_frames(n: int = 4, seed: int = 0) -> list[np.ndarray]:
    from flyaim.arena.arena import Arena

    out = []
    for k in range(n):
        arena = Arena(ArenaConfig(), seed=seed + k)
        f = arena.reset()
        for _ in range(k * 7):
            f = arena.step(np.array([0.0, 0.0], np.float32)).frame
        out.append(f.copy())
    return out


def dn_trace(fs: FlySystem, frame: np.ndarray, rng, warmup_frames: int = 6,
             total_frames: int = 24) -> np.ndarray:
    """返回 DN 放电率的**时间序列** (total_frames, n_dn)。"""
    fs.reset()
    b = fs.brain
    dn = fs.roles.descending
    v = getattr(b, "_v", None)
    if v is not None:
        v += rng.normal(0.0, 0.02, size=v.shape).astype(v.dtype)

    drive = fs.retina.frame_to_spikes(frame)
    steps = b.cfg.steps_per_frame
    for _ in range(warmup_frames):
        for _s in range(steps):
            b.step(drive[:, 0], drive[:, 1], 1.0)

    trace = np.zeros((total_frames, dn.size), dtype=np.float64)
    for t in range(total_frames):
        for _s in range(steps):
            b.step(drive[:, 0], drive[:, 1], 1.0)
        trace[t] = b.rates[dn]
    return trace


def main() -> int:
    print("=" * 92)
    print("时间积分能否提取被混沌淹没的视觉成分?")
    print("=" * 92, flush=True)

    frames = make_frames(4, seed=0)
    n_stim, n_state, n_frames = len(frames), 4, 24
    windows = [1, 2, 4, 8, 16, 24]

    wp = NORM / "connectome_indeg.npz"
    w, g = 16.0, 1.5
    fs = FlySystem(D, RetinaConfig(),
                   BrainConfig(weight_scale_exc=w, weight_scale_inh=w, input_gain=g),
                   ReadoutConfig(), weights_override=wp)

    print(f"采集 {n_stim} 刺激 × {n_state} 初始状态 × {n_frames} 帧 DN 时间序列…", flush=True)
    t0 = time.perf_counter()
    T = np.zeros((n_stim, n_state, n_frames, fs.roles.descending.size), dtype=np.float64)
    for i, fr in enumerate(frames):
        for j in range(n_state):
            rng = np.random.default_rng(500 * i + j)
            T[i, j] = dn_trace(fs, fr, rng, total_frames=n_frames)
    print(f"  采集完成 {time.perf_counter()-t0:.0f}s\n", flush=True)

    print(f"{'窗口(帧)':>9} | {'刺激效应':>12} {'状态效应':>12} | {'比值':>8} | 判定")
    print("-" * 74, flush=True)
    results = []
    for win in windows:
        # 对每个 (i,j) 用最近 win 帧做时间平均
        A = T[:, :, -win:, :].mean(axis=2)          # (n_stim, n_state, n_dn)
        grand = A.mean(axis=(0, 1))
        stim_means = A.mean(axis=1)
        state_means = A.mean(axis=0)
        ss_stim = float(((stim_means - grand) ** 2).sum() / n_state) * n_stim / max(n_stim - 1, 1)
        ss_state = float(((state_means - grand) ** 2).sum() / n_stim) * n_state / max(n_state - 1, 1)
        ratio = ss_stim / max(ss_state, 1e-12)
        results.append((win, ss_stim, ss_state, ratio))
        if ratio >= 3:
            v = "✅ 可用"
        elif ratio >= 1.5:
            v = "⚠️ 弱"
        else:
            v = "❌ 不可用"
        print(f"{win:>9} | {ss_stim:>12.3f} {ss_state:>12.3f} | {ratio:>8.3f} | {v}", flush=True)

    print()
    best = max(results, key=lambda r: r[3])
    print(f"最佳窗口 = {best[0]} 帧, 比值 = {best[3]:.3f}")
    if best[3] >= 3:
        print("→ 时间积分有效。可用「每 N 帧更新一次方向」的慢闭环方案继续。")
    elif best[3] > results[0][3] * 1.3:
        print("→ 比值随窗口增长但未达可用阈值。")
        print("  说明刺激成分确实存在且是确定性的,但增长慢于 sqrt(T),")
        print("  很可能因为**刺激成分本身幅度太小**(一级靶中位数只有 1 个突触前伙伴)。")
    else:
        print("→ 时间积分**无效**:比值不随窗口改善。")
        print("  这意味着残差不是零均值噪声,而是**由初始状态决定的确定性轨迹**")
        print("  (混沌系统:同一刺激 + 不同初值 → 完全不同的吸引子)。")
        print("  这种情况下平均无用,因为差异不随 T 衰减。")
        print()
        print("  最终结论:**该 LIF 简化模型 + 静态连接组不承载可用视觉信号。**")
    print("=" * 92)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
