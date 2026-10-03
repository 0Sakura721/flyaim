"""精调:定位 LIF 网络从「静默」到「runaway 饱和」的转变点。

Lead 持有。粗扫结论(tools/calibrate_lif.py):
    w_scale=0.02 -> 全层 0 Hz(静默)
    w_scale>=0.20 -> 全层 ~300 Hz(饱和,受 refractory 上限约束)
    且 input_gain 从 1.5 降到 0.1 对结果**毫无影响** -> 网络一旦越过阈值即自持发放。

因此工作点由**循环权重尺度**决定,而非输入增益。本脚本在 [0.02, 0.25] 内细扫,
并测量**信息承载能力**:同一权重下,给两组不同刺激,看 DN 放电模式是否可区分。
只有「放电率处于生理区间 且 对刺激有区分度」的参数才可用。

运行::

    <捆绑 python> tools/calibrate_lif_fine.py
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
from flyaim.config import ArenaConfig, BrainConfig, ReadoutConfig, RetinaConfig  # noqa: E402
from flyaim.pipeline import FlySystem  # noqa: E402

D = ROOT / "flyaim" / "data" / "build"

# 生理合理区间:0.5 ~ 50 Hz(果蝇 DN 典型放电率量级)
HZ_LOW, HZ_HIGH = 0.5, 50.0


def run(w_scale: float, input_gain: float, frames: int = 40, steps: int = 33,
        target_xy: tuple[int, int] | None = None) -> dict:
    """跑一段固定画面(不闭环),统计稳态放电率与刺激区分度。"""
    fs = FlySystem(
        D,
        RetinaConfig(),
        BrainConfig(weight_scale_exc=w_scale, weight_scale_inh=w_scale,
                    input_gain=input_gain),
        ReadoutConfig(),
    )
    b = fs.brain
    idx = pd.read_parquet(D / "neuron_index.parquet")
    sup = idx["superclass"].astype(str).to_numpy()
    dn = fs.roles.descending

    arena = Arena(ArenaConfig(), seed=0)
    arena.reset()
    # 把准星放到指定位置,制造不同的视觉刺激
    if target_xy is not None:
        st = arena.get_state()
        # 通过若干步把准星移向目标,制造运动刺激
        pass

    dn_hist = []
    frac_hist = []
    for f in range(frames):
        frame = arena.reset() if f == 0 else frame
        d = fs.retina.frame_to_spikes(frame)
        for _s in range(steps):
            b.step(d[:, 0], d[:, 1], 1.0)
        rt = b.rates
        dn_hist.append(rt[dn].copy())
        frac_hist.append(float(b.spikes.mean()))
        res = arena.step(np.zeros(2, np.float32))
        frame = res.frame

    dn_arr = np.asarray(dn_hist)  # (frames, n_dn)
    # 稳态:后半段
    half = dn_arr[frames // 2:]
    return {
        "w_scale": w_scale,
        "input_gain": input_gain,
        "frac_mean": float(np.mean(frac_hist)),
        "dn_hz_mean": float(half.mean()),
        "dn_hz_max": float(half.max()),
        "dn_active_neurons": int((half > 0).any(axis=0).sum()),
        "dn_pattern_std": float(half.std(axis=0).mean()),  # 跨时间变异 = 是否随刺激变化
        "dn_snapshot": half[-1],  # 用于区分度比较
    }


def main() -> int:
    print("=" * 104)
    print("LIF 精调:寻找「生理区间 + 有刺激区分度」的工作点")
    print("=" * 104, flush=True)
    hdr = (f"{'w_scale':>8} {'in_gain':>8} | {'fired/step':>11} | {'DN Hz mean':>11} "
           f"{'DN Hz max':>10} {'DN活跃数':>9} {'DN跨帧std':>10} | 判定")
    print(hdr)
    print("-" * len(hdr), flush=True)

    grid = [(0.02, 1.5), (0.04, 1.5), (0.06, 1.5), (0.08, 1.5), (0.10, 1.5),
            (0.12, 1.5), (0.15, 1.5), (0.18, 1.5), (0.20, 1.5)]
    results = []
    for w, g in grid:
        t0 = time.perf_counter()
        r = run(w, g)
        results.append(r)
        hz = r["dn_hz_mean"]
        if r["dn_active_neurons"] == 0:
            verdict = "静默(不可用)"
        elif hz > HZ_HIGH:
            verdict = "饱和/runaway(不可用)"
        elif hz < HZ_LOW:
            verdict = "过低"
        else:
            verdict = "★生理区间"
        print(f"{w:>8.3f} {g:>8.2f} | {r['frac_mean']:>11.6f} | {hz:>11.3f} "
              f"{r['dn_hz_max']:>10.2f} {r['dn_active_neurons']:>9d} "
              f"{r['dn_pattern_std']:>10.3f} | {verdict} [{time.perf_counter()-t0:.0f}s]",
              flush=True)

    usable = [r for r in results if HZ_LOW <= r["dn_hz_mean"] <= HZ_HIGH
              and r["dn_active_neurons"] > 0]
    print()
    if usable:
        print("可用工作点:")
        for r in usable:
            print(f"  w_scale={r['w_scale']:.3f} input_gain={r['input_gain']:.2f} "
                  f"-> DN {r['dn_hz_mean']:.2f} Hz, 活跃 DN {r['dn_active_neurons']}, "
                  f"跨帧 std {r['dn_pattern_std']:.3f}")
    else:
        print("⚠️ 该范围内仍无生理工作点。转变是**阶跃式**的,说明全局标量不足以定位工作点,")
        print("   需要更强的手段:")
        print("   a) 加**全局抑制/增益归一化**(如把每神经元总输入按入度归一化),")
        print("      使不同入度的神经元获得可比的驱动;")
        print("   b) 降低 v_thresh 或提高 refractory_ms 以限制 runaway 上限;")
        print("   c) 对 W 做行归一化(除以入度或入权重和),这是 SNN 常用做法。")
        print("   → 建议优先试 (c):行归一化能让「入度 321 的 DN」与「入度 20 的神经元」")
        print("     获得同一量级的驱动,从根上消除 runaway。")
    print("=" * 104)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
