"""决定性检验 v2:DN 输出到底携带的是「刺激信息」还是「网络自身历史」?

Lead 持有。背景:两个测试给出相反结论,必须裁决:

  Lead 的 test_dn_information.py:
      噪声底 = 0(同画面重复 → 逐位相同),刺激差异 15~30%  → 判"承载信息"
      **缺陷**:噪声底取的是"完全相同的初始状态",而引擎是确定性的,
      所以噪声底必然为 0。这**无法区分**"输出由画面决定"与
      "输出由初始状态决定,画面只是无关扰动"。

  B 线的测试:
      同一批画面、**不同随机初始历史** → 噪声底 ≈ 刺激差异,比值 0.96~1.04
      → 判"不承载信息"。这个设计更严格。

本脚本用**方差分解**一次性裁决,而不是比较两个标量:
    对每个 (刺激 i, 初始状态 j) 组合,测 DN 稳态放电向量 r[i][j]
    然后计算:
        SS_stimulus = 刺激间方差(固定 j,跨 i 的方差)   <- 我们想要的"视觉信号"
        SS_state    = 初始状态间方差(固定 i,跨 j 的方差) <- 干扰项
        ratio = SS_stimulus / SS_state

    ratio > 3  → 画面是主导因素,DN 承载可用视觉信息
    ratio ~ 1  → 两者同量级,DN 输出无法与自发状态区分(**不可用**)

这个判据直接对应"读出层能否泛化到新的 episode" —— 这才是项目成败所在。

运行::

    <捆绑 python> tools/test_dn_variance.py
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
    """构造 n 个视觉上不同的静态画面。"""
    from flyaim.arena.arena import Arena

    out = []
    for k in range(n):
        arena = Arena(ArenaConfig(), seed=seed + k)
        f = arena.reset()
        for _ in range(k * 7):
            f = arena.step(np.array([0.0, 0.0], np.float32)).frame
        out.append(f.copy())
    return out


def steady_dn(fs: FlySystem, frame: np.ndarray, rng: np.random.Generator,
              warmup_steps: int = 200, measure_steps: int = 100) -> np.ndarray:
    """在**指定随机初始状态**下,给固定画面,测 DN 稳态平均放电率。

    关键:reset() 后**主动注入随机初始膜电位/状态扰动**,以模拟
    "同一个画面在不同历史下进入网络"。这是区分刺激效应与历史效应的核心。
    """
    fs.reset()
    b = fs.brain
    dn = fs.roles.descending

    # 注入随机初始状态扰动(若有内部状态向量)
    v = getattr(b, "_v", None)
    if v is not None and rng is not None:
        # 小幅度扰动:不足以单独引起发放,但足以让轨迹分岔
        v += rng.normal(0.0, 0.02, size=v.shape).astype(v.dtype)

    drive = fs.retina.frame_to_spikes(frame)
    steps = fs.brain.cfg.steps_per_frame
    for _ in range(warmup_steps // max(steps, 1) + 1):
        for _s in range(steps):
            b.step(drive[:, 0], drive[:, 1], 1.0)

    acc = np.zeros(dn.size, dtype=np.float64)
    for _ in range(measure_steps):
        b.step(drive[:, 0], drive[:, 1], 1.0)
        acc += b.rates[dn]
    return (acc / measure_steps).astype(np.float64)


def main() -> int:
    print("=" * 96)
    print("决定性检验 v2:方差分解 —— 刺激效应 vs 初始状态效应")
    print("=" * 96, flush=True)

    frame_sets = make_frames(4, seed=0)
    n_stim, n_state = len(frame_sets), 6
    print(f"刺激数 = {n_stim}, 每刺激的随机初始状态数 = {n_state}\n", flush=True)

    configs = [
        ("indeg", NORM / "connectome_indeg.npz", 12.0, 1.5),
        ("indeg", NORM / "connectome_indeg.npz", 16.0, 1.5),
        ("indeg", NORM / "connectome_indeg.npz", 20.0, 1.5),
    ]

    for norm, wp, w, g in configs:
        if not wp.exists():
            print(f"[skip] {wp} 不存在", flush=True)
            continue
        try:
            fs = FlySystem(D, RetinaConfig(),
                           BrainConfig(weight_scale_exc=w, weight_scale_inh=w,
                                       input_gain=g),
                           ReadoutConfig(), weights_override=wp)
        except Exception as e:
            print(f"[{norm} w={w}] 装配失败: {type(e).__name__}: {e}", flush=True)
            continue

        t0 = time.perf_counter()
        R = np.zeros((n_stim, n_state, fs.roles.descending.size), dtype=np.float64)
        for i, fr in enumerate(frame_sets):
            for j in range(n_state):
                rng = np.random.default_rng(1000 * i + j)
                R[i, j] = steady_dn(fs, fr, rng)
        dt = time.perf_counter() - t0

        mean_hz = float(R.mean())
        # ---- 方差分解 ----
        grand = R.mean(axis=(0, 1))
        # 刺激主效应(每个刺激在所有状态上的均值,再跨刺激求方差)
        stim_means = R.mean(axis=1)              # (n_stim, n_dn)
        state_means = R.mean(axis=0)             # (n_state, n_dn)

        ss_stim = float(((stim_means - grand) ** 2).sum() / n_state) * n_stim / max(n_stim - 1, 1)
        ss_state = float(((state_means - grand) ** 2).sum() / n_stim) * n_state / max(n_state - 1, 1)
        ss_resid = float(((R - grand) ** 2).sum()) - ss_stim * (n_stim - 1) * n_state \
            - ss_state * (n_state - 1) * n_stim
        ss_resid = max(ss_resid, 0.0)

        print(f"--- norm={norm} w_scale={w} in_gain={g} ---")
        print(f"    DN 平均放电率 = {mean_hz:.4f} Hz, 活跃 DN = {int((R.max(axis=(0,1))>0).sum())}")
        print(f"    SS_stimulus = {ss_stim:.4f}   SS_state = {ss_state:.4f}   "
              f"SS_resid = {ss_resid:.4f}")
        ratio = ss_stim / max(ss_state, 1e-12)
        print(f"    比值 SS_stim/SS_state = {ratio:.3f}   ({dt:.0f}s)")
        if ratio >= 3.0:
            print("    ✅ 画面是主导因素 → DN 承载可用视觉信息(读出应能泛化)")
        elif ratio >= 1.5:
            print("    ⚠️ 画面影响略强于历史噪声,泛化能力很可能不足")
        else:
            print("    ❌ 刺激效应与历史噪声同量级 → **DN 不承载可用视觉信息**")
            print("       机制:网络对初始状态混沌敏感,输出主要反映自身历史而非画面。")
            print("       后果:读出层会在训练集上 r2 很高(记住训练轨迹),但换 episode 即失效。")
        print(flush=True)

    print("=" * 96)
    print("解读:本测试同时控制刺激与初始状态,是裁决两个相反结论的正确判据。")
    print("      若比值 < 3,则最终结论应为「该 LIF 简化模型 + 静态连接组不承载可用视觉信号」,")
    print("      而非「果蝇不会瞄准」—— 这两者的区别是本项目可信度的核心。")
    print("=" * 96)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
