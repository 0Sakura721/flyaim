"""GPU vs CPU:数值等价性校验 + 性能基准。

先证明 GPU 引擎与 CPU 引擎**数值等价**,再证明它更快 —— 顺序不能反。
若两者结果不一致,"更快"就没有意义。

校验方法:
    用同一条外部驱动序列(来自真实视网膜编码)分别驱动 CPU / GPU 引擎,
    比较每步的发放数、final rates 的逐元素偏差、以及脉冲一致率。

运行::

    <捆绑 python> tools/bench_gpu.py            # 等价性 + 性能
    <捆绑 python> tools/bench_gpu.py --parity   # 只做等价性
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from flyaim.arena.arena import Arena  # noqa: E402
from flyaim.baselines.pid import PIDBaseline  # noqa: E402
from flyaim.config import ArenaConfig, BrainConfig, RetinaConfig  # noqa: E402
from flyaim.io import load_roles  # noqa: E402
from flyaim.retina.encoder import Retina  # noqa: E402

D = ROOT / "flyaim" / "data" / "build"
NORM = ROOT / "flyaim" / "runs" / "norm"
WEIGHTS = D / "connectome.npz"
W_SCALE, IN_GAIN = 16.0, 1.5
STEPS = 8


def cfg_for(dt_ms=4.0, steps=STEPS) -> BrainConfig:
    return BrainConfig(weight_scale_exc=W_SCALE, weight_scale_inh=W_SCALE,
                       input_gain=IN_GAIN, weight_norm="indeg",
                       dt_ms=dt_ms, steps_per_frame=steps)


def drive_sequence(n_frames: int = 8):
    """用真实视网膜编码产生驱动序列(与仿真同源,不造假数据)。"""
    for extra in ("roles.json", "neuron_index.parquet", "manifest.json"):
        dst = NORM / extra
        if not dst.exists():
            shutil.copy(D / extra, dst)
    roles = load_roles(D / "roles.json")
    ret = Retina(RetinaConfig(), input_neuron_ids=roles.visual_input)
    arena = Arena(ArenaConfig(), seed=0)
    pid = PIDBaseline()
    frame = arena.reset()
    seq = []
    for _ in range(n_frames):
        d = ret.frame_to_spikes(frame)
        seq.append((d[:, 0].copy(), d[:, 1].copy()))
        frame = arena.step(np.clip(pid.act(arena.get_state()), -1, 1)).frame
    return seq, roles


def parity_check(seq, roles) -> dict:
    from flyaim.brain.lif import Connectome
    from flyaim.brain.lif_gpu import ConnectomeGPU

    cfg = cfg_for()
    print("  构造 CPU 引擎…", flush=True)
    cpu = Connectome(str(WEIGHTS), cfg, input_neuron_ids=roles.visual_input)
    print("  构造 GPU 引擎…", flush=True)
    gpu = ConnectomeGPU(WEIGHTS, cfg, input_neuron_ids=roles.visual_input)

    n_steps = len(seq) * STEPS
    cpu_fired, gpu_fired = [], []
    for exc, inh in seq:
        for _ in range(STEPS):
            cpu.step(exc, inh, cfg.dt_ms)
            gpu.step(exc, inh, cfg.dt_ms)
            cpu_fired.append(int(cpu.spikes.sum()))
            gpu_fired.append(int(gpu.spikes.sum()))

    cpu_r = np.asarray(cpu.rates, np.float64)
    gpu_r = np.asarray(gpu.rates, np.float64)
    cpu_s = np.asarray(cpu.spikes, bool)
    gpu_s = np.asarray(gpu.spikes, bool)

    # ⚠️ 判据设计(重要)
    # 混沌循环网络里**逐位相等既不可能也不必要**:cuSPARSE 与 scipy 的求和时间
    # 顺序不同,阈值边缘的神经元会翻转,而混沌动力学把这种微差指数放大。
    # 实测特征:前 5 步发放数**逐位完全相同**,第 6 步起分岔 ——
    # 这正是"同一套数学 + 不同浮点求和顺序"的指纹;真 bug 会从第 1 步就分岔。
    #
    # 因此正确的判据是两条:
    #   A) **短时程精确一致**:起始若干步发放数完全相同 → 证明数学等价
    #   B) **长时程统计等价**:平均放电率无系统漂移 + 脉冲一致率高 → 证明无偏差
    # 不能用逐元素相对误差:rates≈0 的神经元会让分母爆掉(曾得到 9.99e7 的假告警)。
    n_exact = 0
    for a, b in zip(cpu_fired, gpu_fired):
        if a != b:
            break
        n_exact += 1

    denom = float(np.linalg.norm(cpu_r))
    rel_l2 = float(np.linalg.norm(gpu_r - cpu_r) / max(denom, 1e-12))
    abs_diff = np.abs(gpu_r - cpu_r)
    agree = float((cpu_s == gpu_s).mean())
    mean_cpu, mean_gpu = float(cpu_r.mean()), float(gpu_r.mean())
    rel_mean = abs(mean_gpu - mean_cpu) / max(abs(mean_cpu), 1e-12)

    return {
        "n_steps": n_steps,
        "cpu_fired": cpu_fired,
        "gpu_fired": gpu_fired,
        "n_exact_leading_steps": n_exact,
        "first_diff_step": (n_exact + 1) if n_exact < len(cpu_fired) else None,
        "spike_agreement": agree,
        "rate_rel_l2": rel_l2,
        "rate_max_abs_diff": float(abs_diff.max()),
        "rate_mean_abs_diff": float(abs_diff.mean()),
        "rate_mean_cpu": mean_cpu,
        "rate_mean_gpu": mean_gpu,
        "rate_rel_mean_diff": rel_mean,
        "rate_scale": float(np.abs(cpu_r).mean()),
        "p50_cpu": float(np.percentile(cpu_r, 50)),
        "p50_gpu": float(np.percentile(gpu_r, 50)),
        "p99_cpu": float(np.percentile(cpu_r, 99)),
        "p99_gpu": float(np.percentile(gpu_r, 99)),
    }


def bench(seq, roles, engine: str, steps: int = STEPS) -> dict:
    """测单帧墙钟(含 H2D 往返),口径与真实 pipeline 一致。"""
    cfg = cfg_for()
    if engine == "cpu":
        from flyaim.brain.lif import Connectome

        brain = Connectome(str(WEIGHTS), cfg, input_neuron_ids=roles.visual_input)
    else:
        from flyaim.brain.lif_gpu import ConnectomeGPU

        brain = ConnectomeGPU(WEIGHTS, cfg, input_neuron_ids=roles.visual_input)

    # 预热
    exc, inh = seq[0]
    for _ in range(steps):
        brain.step(exc, inh, cfg.dt_ms)
    if hasattr(brain, "sync"):
        brain.sync()
    brain.reset()

    frames = 6
    per_frame = []
    for f in range(frames):
        exc, inh = seq[f % len(seq)]
        t0 = time.perf_counter()
        for _ in range(steps):
            brain.step(exc, inh, cfg.dt_ms)
        _ = brain.rates          # 含一次 H2D(与真实读出口径一致)
        if hasattr(brain, "sync"):
            brain.sync()
        per_frame.append((time.perf_counter() - t0) * 1000.0)
    return {"engine": engine, "ms_per_frame": float(np.median(per_frame)),
            "ms_per_step": float(np.median(per_frame)) / steps,
            "n_fired": int(np.asarray(brain.spikes).sum())}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parity", action="store_true", help="只做等价性校验")
    ap.add_argument("--frames", type=int, default=8)
    ap.add_argument("--steps", type=int, default=STEPS)
    args = ap.parse_args()

    from flyaim.brain.lif_gpu import gpu_available, gpu_info

    print("=" * 96)
    print("GPU vs CPU:LIF 引擎")
    print("=" * 96)
    info = gpu_info()
    print(f"GPU: {info}")
    if not gpu_available():
        print("\n❌ GPU 不可用。先安装: pip install cupy-cuda12x")
        return 2

    seq, roles = drive_sequence(args.frames)
    print(f"驱动序列: {len(seq)} 帧 × {args.steps} 步;"
          f"感光细胞 {roles.visual_input.size} 个\n", flush=True)

    # ---------------- 1. 等价性 ----------------
    print("=" * 96)
    print("1. 数值等价性(先证明 GPU 没算错)")
    print("=" * 96, flush=True)
    pc = parity_check(seq, roles)
    print(f"  步数                      : {pc['n_steps']}")
    print(f"  CPU 每步发放数(前 8)      : {pc['cpu_fired'][:8]}")
    print(f"  GPU 每步发放数(前 8)      : {pc['gpu_fired'][:8]}")
    print(f"  A) 起始逐位一致步数       : {pc['n_exact_leading_steps']}"
          f"   (首次分岔于第 {pc['first_diff_step']} 步)")
    print("     → 前若干步完全一致 = 同一套数学;之后分岔 = 浮点求和顺序 + 混沌放大")
    print(f"  B) 脉冲逐元素一致率       : {pc['spike_agreement']*100:.4f}%")
    print(f"     平均放电率 (CPU/GPU)   : {pc['rate_mean_cpu']:.4f} / "
          f"{pc['rate_mean_gpu']:.4f} Hz  (相对差 {pc['rate_rel_mean_diff']*100:.3f}%)")
    print(f"     中位数 (CPU/GPU)       : {pc['p50_cpu']:.4f} / {pc['p50_gpu']:.4f} Hz")
    print(f"     p99    (CPU/GPU)       : {pc['p99_cpu']:.4f} / {pc['p99_gpu']:.4f} Hz")
    print(f"     rates 相对 L2 误差     : {pc['rate_rel_l2']:.3e}")
    print(f"     平均绝对偏差           : {pc['rate_mean_abs_diff']:.6f}"
          f"  (量级 {pc['rate_scale']:.3f} Hz)")
    print()
    print("  判据:A) 起始 ≥5 步逐位一致  且  B) 平均率相对差 <1% 且 脉冲一致率 >98%")
    ok = (pc["n_exact_leading_steps"] >= 5
          and pc["rate_rel_mean_diff"] < 0.01
          and pc["spike_agreement"] > 0.98)
    print(f"  → 判定: {'✅ 数学等价且无系统漂移' if ok else '❌ 不等价,先修正确性再谈性能'}")
    if not ok:
        return 1
    if args.parity:
        return 0

    # ---------------- 2. 性能 ----------------
    print()
    print("=" * 96)
    print("2. 性能(单帧 = 8 步 + 一次 rates 回读)")
    print("=" * 96, flush=True)
    rc = bench(seq, roles, "cpu", args.steps)
    rg = bench(seq, roles, "gpu", args.steps)
    print(f"  CPU : {rc['ms_per_frame']:9.2f} ms/帧   ({rc['ms_per_step']:7.2f} ms/步)")
    print(f"  GPU : {rg['ms_per_frame']:9.2f} ms/帧   ({rg['ms_per_step']:7.2f} ms/步)")
    sp = rc["ms_per_frame"] / max(rg["ms_per_frame"], 1e-9)
    print(f"  → 提速 {sp:.1f}x")
    print()
    print(f"  按当前帧预算(32 ms 模拟/帧)折算:")
    print(f"    CPU 约 {1000/rc['ms_per_frame']:.1f} 帧/秒   GPU 约 {1000/rg['ms_per_frame']:.1f} 帧/秒")
    print("=" * 96)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
