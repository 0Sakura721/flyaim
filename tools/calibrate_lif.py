"""标定 LIF 权重尺度与输入增益(Lead 独立复算)。

背景:拓扑已验证是通的(感光细胞 3 跳覆盖 1350/1360 个 DN,DN 入度中位数 321),
但 DN 零发放。原因在权重标定:
    _w_exc_eff = weight_scale_exc * (dt/tau) = 0.02 * 0.05 = 0.001
单个突触每步只加 0.001,而 v_thresh = 1.0 → 需要约 1000 个上游**同步**发放。

本脚本不靠猜:先实测真实运行时的每步发放比例,再据此推算出使目标层
达到合理放电率的权重尺度,并验证。

运行::

    <捆绑 python> tools/calibrate_lif.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from flyaim.arena.arena import Arena  # noqa: E402
from flyaim.baselines.pid import PIDBaseline  # noqa: E402
from flyaim.config import ArenaConfig, BrainConfig, ReadoutConfig, RetinaConfig  # noqa: E402
from flyaim.io import ConnectomeArtifacts, load_roles  # noqa: E402
from flyaim.pipeline import FlySystem  # noqa: E402

D = ROOT / "flyaim" / "data" / "build"


def measure(gain_w: float, input_gain: float, frames: int = 90, steps: int = 33) -> dict:
    """跑一段真实闭环(导师驱动靶场),统计各层放电与每步发放比例。"""
    fs = FlySystem(
        D,
        RetinaConfig(),
        BrainConfig(weight_scale_exc=gain_w, weight_scale_inh=gain_w, input_gain=input_gain),
        ReadoutConfig(),
    )
    b = fs.brain
    roles = fs.roles
    idx = pd.read_parquet(D / "neuron_index.parquet")
    sup = idx["superclass"].astype(str).to_numpy()

    arena = Arena(ArenaConfig(), seed=0)
    frame = arena.reset()
    pid = PIDBaseline()

    layers = {
        "ol_sensory": sup == "ol_sensory",
        "ol_intrinsic": sup == "ol_intrinsic",
        "visual_projection": sup == "visual_projection",
        "descending": sup == "descending_neuron",
        "vnc_motor": sup == "vnc_motor",
    }
    spike_counts = {k: 0 for k in layers}
    active_frames = {k: 0 for k in layers}
    n_steps = 0
    frac_fired = []  # 每步全网络发放比例

    for _ in range(frames):
        d = fs.retina.frame_to_spikes(frame)
        seen = {k: False for k in layers}
        for _s in range(steps):
            b.step(d[:, 0], d[:, 1], 1.0)
            sp = b.spikes
            n_steps += 1
            frac_fired.append(float(sp.mean()))
            for k, m in layers.items():
                c = int(sp[m].sum())
                spike_counts[k] += c
                if c:
                    seen[k] = True
        for k in layers:
            active_frames[k] += int(seen[k])
        res = arena.step(np.clip(pid.act(arena.get_state()), -1, 1))
        frame = res.frame

    # 层放电率(Hz) = 脉冲数 / (神经元数 * 步数 * dt_ms/1000)
    secs = n_steps * b.cfg.dt_ms / 1000.0
    out = {"weight_scale": gain_w, "input_gain": input_gain,
           "mean_frac_fired_per_step": float(np.mean(frac_fired)),
           "max_frac_fired_per_step": float(np.max(frac_fired))}
    for k, m in layers.items():
        n_neur = int(m.sum())
        out[f"hz_{k}"] = spike_counts[k] / max(n_neur * secs, 1e-9)
        out[f"active_frames_{k}"] = active_frames[k]
    return out


def main() -> int:
    print("=" * 100)
    print("LIF 标定:权重尺度 × 输入增益 扫描")
    print("=" * 100)
    print("先实测每步发放比例。目标:DN 有持续非零放电(合理区间 0.5~50 Hz)。")
    print()

    hdr = (f"{'w_scale':>8} {'in_gain':>8} | {'fired/step':>11} {'max':>8} | "
           f"{'R1R6 Hz':>9} {'oli Hz':>9} {'VPN Hz':>9} {'DN Hz':>9} {'MN Hz':>9} | "
           f"{'DN帧':>6}")
    print(hdr)
    print("-" * len(hdr))

    rows = []
    grid = [
        (0.02, 1.5),    # 当前默认
        (0.2, 1.5),
        (1.0, 1.5),
        (5.0, 1.5),
        (1.0, 0.5),
        (5.0, 0.5),
        (20.0, 0.5),
        (5.0, 0.1),
        (20.0, 0.1),
    ]
    for w, g in grid:
        t0 = time.perf_counter()
        try:
            r = measure(w, g)
        except Exception as e:
            print(f"{w:>8.2f} {g:>8.2f} | ERROR {type(e).__name__}: {e}")
            continue
        rows.append(r)
        print(f"{w:>8.2f} {g:>8.2f} | {r['mean_frac_fired_per_step']:>11.6f} "
              f"{r['max_frac_fired_per_step']:>8.4f} | "
              f"{r['hz_ol_sensory']:>9.3f} {r['hz_ol_intrinsic']:>9.3f} "
              f"{r['hz_visual_projection']:>9.3f} {r['hz_descending']:>9.3f} "
              f"{r['hz_vnc_motor']:>9.3f} | {r['active_frames_descending']:>6d} "
              f"[{time.perf_counter()-t0:.0f}s]", flush=True)

    print()
    ok = [r for r in rows if r["hz_descending"] > 0.1 and r["active_frames_descending"] > 5]
    if ok:
        print("可用配置(满足 DN 有持续非零放电):")
        for r in ok:
            print(f"  weight_scale={r['weight_scale']:.2f} input_gain={r['input_gain']:.2f} "
                  f"-> DN {r['hz_descending']:.3f} Hz, 活跃帧 {r['active_frames_descending']}")
    else:
        print("⚠️ 扫描范围内没有任何配置能让 DN 持续放电。")
        print("   → 问题可能不止于全局标量:需检查 DN 的上游是否存在**抑制主导**")
        print("     (W_inh 与 W_exc 比值)、或突触权重分布的长尾特性。")
    print("=" * 100)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
