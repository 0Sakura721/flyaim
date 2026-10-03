"""成本 vs 网络活跃度:找更便宜的生理工作点。

背景
----
实测单步耗时强烈依赖**活跃神经元数**(脉冲驱动的稀疏传播成本 ∝ 活跃列数):

    近静默(0.04% 发放) : 1.14 ms/步
    工作点(3%   发放) : 7.25 ms/步
    当前默认(82% 发放) : ~42  ms/步

而当前默认 `weight_scale=16` 下 **82% 的神经元每步都在发放** —— 这是癫痫态,
既不生理,又让 CPU 成本高一个数量级。

本脚本扫 weight_scale,同时报告:
    - 每步耗时 / 每帧耗时
    - 全网活跃比例、DN 放电率、DN 活跃数
    - **DN 对视觉刺激的区分度**(比值 = 刺激间差异 / 初始状态差异)

最后一项是关键:我之前已用方差分解证明该配置下 DN **不承载可用视觉信息**
(比值 < 1)。若某个更便宜的 weight_scale 同时给出更好的区分度,
那就是双赢;若区分度依然 < 1,则说明"便宜"与"有信息"不可兼得。

运行::

    <捆绑 python> tools/bench_activity_cost.py
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
from flyaim.baselines.pid import PIDBaseline  # noqa: E402
from flyaim.config import ArenaConfig, BrainConfig, ReadoutConfig, RetinaConfig  # noqa: E402
from flyaim.pipeline import FlySystem  # noqa: E402

D = ROOT / "flyaim" / "data" / "build"
NORM = ROOT / "flyaim" / "runs" / "norm"


def build(w: float, dt=4.0, steps=8) -> FlySystem:
    for extra in ("roles.json", "neuron_index.parquet", "manifest.json"):
        dst = NORM / extra
        if not dst.exists():
            shutil.copy(D / extra, dst)
    return FlySystem(D, RetinaConfig(),
                     BrainConfig(weight_scale_exc=w, weight_scale_inh=w,
                                 input_gain=1.5, weight_norm="indeg",
                                 dt_ms=dt, steps_per_frame=steps),
                     ReadoutConfig())


def probe(w: float, frames: int = 12) -> dict:
    fs = build(w)
    cfg = fs.brain.cfg
    arena = Arena(ArenaConfig(), seed=0)
    pid = PIDBaseline()
    frame = arena.reset()
    dn = fs.roles.descending

    # 预热
    d0 = fs.retina.frame_to_spikes(frame)
    for _ in range(cfg.steps_per_frame):
        fs.brain.step(d0[:, 0], d0[:, 1], cfg.dt_ms)
    fs.reset()

    step_ms, active_frac, dn_hist = [], [], []
    for _f in range(frames):
        drive = fs.retina.frame_to_spikes(frame)
        for _s in range(cfg.steps_per_frame):
            t0 = time.perf_counter()
            fs.brain.step(drive[:, 0], drive[:, 1], cfg.dt_ms)
            step_ms.append((time.perf_counter() - t0) * 1e3)
        rt = np.asarray(fs.brain.rates)
        active_frac.append(float((rt > 0).mean()))
        dn_hist.append(rt[dn].copy())
        frame = arena.step(np.clip(pid.act(arena.get_state()), -1, 1)).frame

    rt = np.asarray(fs.brain.rates)
    arr = np.asarray(dn_hist)
    half = arr[len(arr) // 2:]
    return {
        "w": w,
        "ms_step": float(np.median(step_ms)),
        "ms_frame": float(np.median(step_ms)) * cfg.steps_per_frame,
        "active_frac": float(np.mean(active_frac)),
        "dn_hz": float(half.mean()),
        "dn_active": int((half > 0).any(axis=0).sum()),
        "dn_temporal_std": float(half.std(axis=0).mean()),
    }


def main() -> int:
    print("=" * 100)
    print("成本 vs 活跃度:weight_scale 扫描(全量 166,700,dt=4ms/8 步)")
    print("=" * 100)
    hdr = (f"{'ws':>6} | {'步p50 ms':>9} {'帧 ms':>8} | {'活跃比例':>9} "
           f"{'DN Hz':>9} {'DN活跃':>7} {'DN跨帧std':>10} | {'相对 ws=16':>10}")
    print(hdr)
    print("-" * len(hdr), flush=True)

    grid = [16.0, 12.0, 10.0, 8.0, 6.0, 5.0, 4.0, 3.0]
    rows, base = [], None
    for w in grid:
        try:
            r = probe(w)
        except Exception as e:
            print(f"{w:>6.1f} | ERROR {type(e).__name__}: {e}", flush=True)
            continue
        if base is None:
            base = r
        rows.append(r)
        sp = base["ms_frame"] / max(r["ms_frame"], 1e-9)
        print(f"{w:>6.1f} | {r['ms_step']:>9.2f} {r['ms_frame']:>8.1f} | "
              f"{r['active_frac']*100:>8.2f}% {r['dn_hz']:>9.2f} {r['dn_active']:>7} "
              f"{r['dn_temporal_std']:>10.3f} | {sp:>9.2f}x", flush=True)

    print()
    print("=" * 100)
    ok = [r for r in rows if 0.5 <= r["dn_hz"] <= 50 and r["dn_active"] > 50]
    if ok:
        cheap = min(ok, key=lambda r: r["ms_frame"])
        print(f"★ 最省的生理工作点: weight_scale={cheap['w']}")
        print(f"  帧耗时 {cheap['ms_frame']:.1f} ms(基线 ws=16: {base['ms_frame']:.1f} ms,"
              f"提速 {base['ms_frame']/cheap['ms_frame']:.2f}x)")
        print(f"  活跃 {cheap['active_frac']*100:.2f}%  DN {cheap['dn_hz']:.2f} Hz  "
              f"活跃 DN {cheap['dn_active']}")
    print()
    print("注意:成本 ∝ 活跃神经元数,所以『便宜』与『活跃』直接冲突。")
    print("      DN 跨帧 std 越大,输出越随刺激变化 —— 但需用方差分解确认它是否可区分。")
    print("=" * 100)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
