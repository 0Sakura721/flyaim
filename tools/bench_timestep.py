"""CPU 减负诊断:定位时间去哪了,并测试「零下载」的加速路径。

背景(用户诉求):CPU 单核满载,GPU 完全闲置。
但直接上 GPU 有两个问题:
    1. 本机无任何 GPGPU 绑定(torch/cupy/numba 全缺),CuPy wheel ~1GB,网络很慢;
    2. 需要先确认 CPU 时间到底花在哪 —— 若花在**过采样**上,改参数即可免费提速。

关键洞察:
    LIF 膜时间常数 tau_m = 20 ms,而 BrainConfig.dt_ms = 1 ms
    → 每步只推进 tau 的 1/20,**过采样 20 倍**。
    把 dt 提到 2~5 ms 时序动力学几乎不变,但每帧步数等比下降 → 线性省 CPU。

本脚本测量:
    A. 单步耗时分解(脉冲驱动稀疏传播 vs LIF 积分)
    B. 不同 (dt, steps_per_frame) 的**墙钟耗时** 与 **DN 活动保真度**

判定标准:DN 平均放电率 / 活跃数 / 发放稀疏度与基线(dt=1ms)接近,
且墙钟显著下降 → 该配置可用。

运行::

    <捆绑 python> tools/bench_gpu_vs_cpu.py
"""

from __future__ import annotations

import shutil
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from flyaim.arena.arena import Arena  # noqa: E402
from flyaim.config import ArenaConfig, BrainConfig, ReadoutConfig, RetinaConfig  # noqa: E402
from flyaim.pipeline import FlySystem  # noqa: E402

D = ROOT / "flyaim" / "data" / "build"
NORM = ROOT / "flyaim" / "runs" / "norm"
W_SCALE, IN_GAIN = 16.0, 1.5


def build(dt_ms: float, steps: int) -> FlySystem:
    """构造与 train_readout.py **完全一致**的配置。

    注意:这里用**原始** connectome.npz + `weight_norm="indeg"`(由 Connectome 内部归一化)。
    **绝不能**再传预归一化的 `connectome_indeg.npz` —— 那会把权重除以入度两次,
    网络直接静默(实测 DN=0 Hz),基准就失去意义。
    """
    for extra in ("roles.json", "neuron_index.parquet", "manifest.json"):
        dst = NORM / extra
        if not dst.exists():
            shutil.copy(D / extra, dst)
    return FlySystem(
        D, RetinaConfig(),
        BrainConfig(weight_scale_exc=W_SCALE, weight_scale_inh=W_SCALE,
                    input_gain=IN_GAIN, weight_norm="indeg",
                    dt_ms=dt_ms, steps_per_frame=steps),
        ReadoutConfig(),
    )


def measure(dt_ms: float, steps: int, frames: int = 30, use_pid: bool = True) -> dict:
    """测一个 (dt, steps) 组合。

    **必须用 PID 驱动靶场**:若用零动作,画面静止 → Retina 的 temporal_diff
    把驱动杀掉 → 网络静默 → 测到的是"静默成本"而非真实工作点成本。
    """
    from flyaim.baselines.pid import PIDBaseline

    fs = build(dt_ms, steps)
    arena = Arena(ArenaConfig(), seed=0)
    frame = arena.reset()
    pid = PIDBaseline()
    dn = fs.roles.descending

    # 预热(首次 matvec / 转置缓存)
    d0 = fs.retina.frame_to_spikes(frame)
    for _ in range(min(steps, 10)):
        fs.brain.step(d0[:, 0], d0[:, 1], dt_ms)

    fs.reset()
    step_ms = []
    t_retina = 0.0
    t_arena = 0.0
    t_wall = time.perf_counter()
    for _f in range(frames):
        t0 = time.perf_counter()
        drive = fs.retina.frame_to_spikes(frame)
        t_retina += time.perf_counter() - t0
        for _s in range(steps):
            t1 = time.perf_counter()
            fs.brain.step(drive[:, 0], drive[:, 1], dt_ms)
            step_ms.append((time.perf_counter() - t1) * 1000.0)
        t2 = time.perf_counter()
        if use_pid:
            act = np.clip(pid.act(arena.get_state()), -1, 1)
        else:
            act = np.zeros(2, np.float32)
        frame = arena.step(np.asarray(act, np.float32)).frame
        t_arena += time.perf_counter() - t2
    wall = time.perf_counter() - t_wall

    rates = np.asarray(fs.brain.rates, np.float64)
    sp_last = fs.brain.spikes
    return {
        "dt_ms": dt_ms, "steps": steps, "frames": frames,
        "wall_s": wall, "ms_per_frame": wall / frames * 1000.0,
        "step_p50_ms": float(np.median(step_ms)),
        "step_mean_ms": float(np.mean(step_ms)),
        "brain_ms_per_frame": float(np.sum(step_ms)) / frames,
        "retina_ms_per_frame": t_retina / frames * 1000.0,
        "arena_ms_per_frame": t_arena / frames * 1000.0,
        "dn_hz_mean": float(rates[dn].mean()),
        "dn_active": int((rates[dn] > 0).sum()),
        "active_total": int((rates > 0).sum()),
        "spike_rate": float(sp_last.mean()),
    }


def main() -> int:
    print("=" * 104)
    print("CPU 减负诊断:dt / steps_per_frame 对耗时与保真度的影响(真实工作点,PID 驱动)")
    print("=" * 104)
    print("tau_m = 20 ms  →  dt=1ms 相当于每步只推进 tau 的 1/20(过采样 20 倍)")
    print()
    hdr = (f"{'dt(ms)':>7} {'steps':>6} {'有效ms/帧':>10} | {'墙钟ms/帧':>10} "
           f"{'brain':>8} {'retina':>8} {'arena':>7} | {'DN Hz':>8} {'DN活跃':>7} "
           f"{'全网活跃':>8} | {'提速':>7}")
    print(hdr)
    print("-" * len(hdr), flush=True)

    grid = [(1.0, 33), (2.0, 16), (2.0, 8), (4.0, 8), (4.0, 4), (5.0, 7), (10.0, 3)]
    base = None
    rows = []
    for dt, steps in grid:
        try:
            r = measure(dt, steps)
        except Exception as e:
            print(f"{dt:>7.1f} {steps:>6} ERROR {type(e).__name__}: {e}", flush=True)
            continue
        if base is None:
            base = r
        speed = base["ms_per_frame"] / r["ms_per_frame"]
        rows.append(r)
        print(f"{dt:>7.1f} {steps:>6} {dt*steps:>10.1f} | {r['ms_per_frame']:>10.1f} "
              f"{r['brain_ms_per_frame']:>8.1f} {r['retina_ms_per_frame']:>8.1f} "
              f"{r['arena_ms_per_frame']:>7.1f} | "
              f"{r['dn_hz_mean']:>8.2f} {r['dn_active']:>7} {r['active_total']:>8} | "
              f"{speed:>6.2f}x", flush=True)

    print()
    print("=" * 104)
    if base:
        print("成本分解(基线 dt=1ms / 33 步):")
        tot = base["ms_per_frame"]
        for k, label in (("brain_ms_per_frame", "脑仿真(33 步)"),
                         ("retina_ms_per_frame", "视网膜编码"),
                         ("arena_ms_per_frame", "靶场渲染+判定")):
            print(f"  {label:18s} {base[k]:7.1f} ms  ({100*base[k]/tot:4.1f}%)")
        print()
        ok = [r for r in rows
              if r is not base
              and base["ms_per_frame"] / r["ms_per_frame"] >= 1.8
              and base["dn_hz_mean"] > 0.01
              and abs(r["dn_hz_mean"] - base["dn_hz_mean"]) / max(base["dn_hz_mean"], 1e-9) < 0.5]
        if ok:
            best = max(ok, key=lambda r: base["ms_per_frame"] / r["ms_per_frame"])
            speed = base["ms_per_frame"] / best["ms_per_frame"]
            print(f"★ 推荐(零下载):dt={best['dt_ms']}ms, steps_per_frame={best['steps']}")
            print(f"  {base['ms_per_frame']:.1f} -> {best['ms_per_frame']:.1f} ms/帧 "
                  f"({speed:.2f}x,CPU 等比下降)")
            print(f"  DN {base['dn_hz_mean']:.2f} -> {best['dn_hz_mean']:.2f} Hz,"
                  f"活跃 {base['dn_active']} -> {best['dn_active']}")
        else:
            print("未找到满足保真度的配置 —— 需 GPU 或子网方案。")
    print("=" * 104)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
