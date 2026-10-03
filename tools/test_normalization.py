"""假设验证:按入度对 W 做行归一化能否消除 runaway / 死寂的二元现象。

Lead 持有。推理链:
    粗扫显示 w_scale 要么全静默(0.02)要么全饱和(>=0.2),且 input_gain 无效。
    → 工作点由循环权重决定,且转变是阶跃式的。

    原因假设:W[i,j] 的量纲是**突触计数**。一个入度 321 的 DN 与一个入度 20 的
    中间神经元,在**同一个全局权重尺度**下获得的驱动相差 16 倍:
        入度 20  的神经元: 20  × w_eff ≈ 0.004   → 永不发放
        入度 5000 的神经元: 5000 × w_eff ≈ 1.0    → 一触即发并自持
    因此不存在一个"对所有神经元都合适"的全局标量 —— 这就是二元现象的来源。

    修复:行归一化(Row-normalize)。把每行的输入权重除以该行的入度(或入权重和),
    使每个神经元的总驱动与"上游有多少个在发放"成比例,而与**入度大小无关**。
    这是 SNN 的标准做法(避免 hub 神经元主导)。

本脚本对比三种归一化策略下的工作点与刺激区分度:
    raw     : 原始突触计数(现状)
    indeg   : 除以行入度(非零元个数)
    wsum    : 除以行权重和

运行::

    <捆绑 python> tools/test_normalization.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from flyaim.arena.arena import Arena  # noqa: E402
from flyaim.config import ArenaConfig, BrainConfig, ReadoutConfig, RetinaConfig  # noqa: E402
from flyaim.io import ConnectomeArtifacts  # noqa: E402
from flyaim.pipeline import FlySystem  # noqa: E402

D = ROOT / "flyaim" / "data" / "build"
OUT = ROOT / "flyaim" / "runs" / "norm"


def normalize(mat: sp.csr_matrix, mode: str) -> sp.csr_matrix:
    """对 CSR 做行归一化。"""
    m = mat.tocsr().astype(np.float32)
    if mode == "raw":
        return m
    if mode == "indeg":
        deg = np.diff(m.indptr).astype(np.float32)
        scale = np.where(deg > 0, 1.0 / np.maximum(deg, 1.0), 0.0)
    elif mode == "wsum":
        s = np.asarray(m.sum(axis=1)).ravel().astype(np.float32)
        scale = np.where(s > 0, 1.0 / np.maximum(s, 1e-9), 0.0)
    else:
        raise ValueError(mode)
    # 每行乘以 scale[row]
    return sp.diags(scale).dot(m).tocsr().astype(np.float32)


def make_artifact(mode: str) -> Path:
    art = ConnectomeArtifacts.load(D / "connectome.npz")
    new = ConnectomeArtifacts(
        W_exc=normalize(art.W_exc, mode),
        W_inh=normalize(art.W_inh, mode),
        neuron_ids=art.neuron_ids,
        soma_pos=None,
    )
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / f"connectome_{mode}.npz"
    new.save(p)
    return p


def measure(weights_path: Path, w_scale: float, input_gain: float,
            frames: int = 40, steps: int = 33) -> dict:
    fs = FlySystem(D, RetinaConfig(),
                   BrainConfig(weight_scale_exc=w_scale, weight_scale_inh=w_scale,
                               input_gain=input_gain),
                   ReadoutConfig(), weights_override=weights_path)
    b = fs.brain
    dn = fs.roles.descending
    arena = Arena(ArenaConfig(), seed=0)
    frame = arena.reset()
    dn_hist, frac_hist = [], []
    for _f in range(frames):
        d = fs.retina.frame_to_spikes(frame)
        for _s in range(steps):
            b.step(d[:, 0], d[:, 1], 1.0)
        dn_hist.append(b.rates[dn].copy())
        frac_hist.append(float(b.spikes.mean()))
        res = arena.step(np.zeros(2, np.float32))
        frame = res.frame
    arr = np.asarray(dn_hist)
    half = arr[frames // 2:]
    return {
        "frac_mean": float(np.mean(frac_hist)),
        "dn_hz_mean": float(half.mean()),
        "dn_hz_max": float(half.max()),
        "dn_active": int((half > 0).any(axis=0).sum()),
        "dn_temporal_std": float(half.std(axis=0).mean()),
    }


def main() -> int:
    print("=" * 104)
    print("归一化策略对比:raw vs 行入度归一化 vs 行权重和归一化")
    print("=" * 104, flush=True)

    paths = {}
    for mode in ("raw", "indeg", "wsum"):
        t0 = time.perf_counter()
        paths[mode] = make_artifact(mode)
        print(f"[{mode}] 归一化产物 -> {paths[mode]}  ({time.perf_counter()-t0:.0f}s)", flush=True)

    print()
    hdr = (f"{'norm':>6} {'w_scale':>8} {'in_gain':>8} | {'fired/step':>11} | "
           f"{'DN Hz mean':>11} {'DN max':>9} {'DN活跃':>7} {'跨帧std':>9} | 判定")
    print(hdr)
    print("-" * len(hdr), flush=True)

    grid = [("raw", 0.02, 1.5), ("raw", 0.2, 1.5),
            ("indeg", 0.5, 1.5), ("indeg", 1.0, 1.5), ("indeg", 2.0, 1.5),
            ("indeg", 4.0, 1.5), ("indeg", 8.0, 1.5),
            ("wsum", 0.5, 1.5), ("wsum", 1.0, 1.5), ("wsum", 2.0, 1.5),
            ("wsum", 4.0, 1.5), ("wsum", 8.0, 1.5)]

    rows = []
    for mode, w, g in grid:
        t0 = time.perf_counter()
        try:
            r = measure(paths[mode], w, g)
        except Exception as e:
            print(f"{mode:>6} {w:>8.2f} {g:>8.2f} | ERROR {type(e).__name__}: {e}", flush=True)
            continue
        r.update(norm=mode, w_scale=w, input_gain=g)
        rows.append(r)
        hz = r["dn_hz_mean"]
        if r["dn_active"] == 0:
            v = "静默"
        elif hz > 50:
            v = "饱和"
        elif hz < 0.5:
            v = "过低"
        else:
            v = "★生理"
        print(f"{mode:>6} {w:>8.2f} {g:>8.2f} | {r['frac_mean']:>11.6f} | "
              f"{hz:>11.3f} {r['dn_hz_max']:>9.2f} {r['dn_active']:>7d} "
              f"{r['dn_temporal_std']:>9.3f} | {v} [{time.perf_counter()-t0:.0f}s]", flush=True)

    print()
    usable = [r for r in rows if 0.5 <= r["dn_hz_mean"] <= 50 and r["dn_active"] > 0]
    if usable:
        print(f"✅ 找到 {len(usable)} 个生理工作点:")
        for r in usable:
            print(f"   norm={r['norm']:>5} w_scale={r['w_scale']:<5} "
                  f"-> DN {r['dn_hz_mean']:.2f} Hz, 活跃 DN {r['dn_active']}, "
                  f"跨帧 std {r['dn_temporal_std']:.3f}")
        print("\n   → 行归一化若能把「静默/饱和」二元现象变成连续可调,即证实假设。")
    else:
        print("⚠️ 仍未找到生理工作点 —— 假设不成立或需要配合调整 v_thresh/refractory。")
    print("=" * 104)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
