"""诊断:感光细胞 → 下行神经元 的图可达性与哪些层真正被激活。

Lead 持有。背景:输入增益调好后,网络有 1133~2602 个神经元在放电,
但 1,360 个下行神经元**零脉冲**。本脚本用图论(不跑动力学)回答:
感光细胞是否在拓扑上能够影响到 DN;若不能,是哪里断了。

运行::

    <捆绑 python> tools/diag_reachability.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from flyaim.io import ConnectomeArtifacts, load_roles  # noqa: E402

D = ROOT / "flyaim" / "data" / "build"


def main() -> int:
    print("=" * 76)
    print("可达性诊断:感光细胞 → 下行神经元")
    print("=" * 76, flush=True)

    art = ConnectomeArtifacts.load(D / "connectome.npz")
    roles = load_roles(D / "roles.json")
    idx = pd.read_parquet(D / "neuron_index.parquet")
    n = art.n
    src, dst = roles.visual_input, roles.descending

    sup = idx["superclass"].astype(str).to_numpy()
    typ = idx["type"].astype(str).to_numpy()
    print(f"N = {n:,}  感光细胞 = {src.size}  DN = {dst.size}", flush=True)

    W = (art.W_exc + art.W_inh).tocsr()
    Wt = W.T.tocsr()
    print(f"W nnz = {W.nnz:,}", flush=True)

    # ---- 逐跳可达(反向传播:W[i,j]=j->i,故用 Wt 向前扩散)
    reach = np.zeros(n, dtype=np.float32)
    reach[src] = 1.0
    print("\n--- 逐跳可达(从感光细胞出发) ---", flush=True)
    t_hop = []
    for hop in range(1, 8):
        t0 = time.perf_counter()
        nxt = (reach > 0) | (Wt.dot(reach) > 0)
        dt = time.perf_counter() - t0
        t_hop.append(dt)
        reach = nxt.astype(np.float32)
        now = int((reach > 0).sum())
        hit = int((reach[dst] > 0).sum())
        print(f"  hop {hop}: 可达 {now:7,d} 神经元 | 覆盖 DN {hit:5d}/{dst.size} "
              f"| {dt:.1f}s", flush=True)

    reach_b = reach > 0
    print(f"\n最终:可达 {int(reach_b.sum()):,} / {n:,} ({100*reach_b.mean():.1f}%)")
    print(f"      DN 被覆盖 {int(reach_b[dst].sum()):,} / {dst.size} "
          f"({100*reach_b[dst].mean():.1f}%)")

    # ---- 可达集里各 superclass 的分布
    print("\n--- 可达集(hop<=7)按 superclass 分布 top 12 ---")
    vals, cnts = np.unique(sup[reach_b], return_counts=True)
    order = np.argsort(cnts)[::-1][:12]
    for o in order:
        print(f"  {vals[o]:26s} {cnts[o]:>8,}")

    # ---- 关键:中间层到底有没有被驱动
    print("\n--- 各功能层在可达集中的比例 ---")
    for layer in ("ol_sensory", "ol_intrinsic", "visual_projection", "visual_centrifugal",
                  "cb_intrinsic", "descending_neuron", "vnc_motor", "ascending_neuron"):
        m = sup == layer
        if m.sum():
            print(f"  {layer:22s} 总数 {int(m.sum()):>7,}  可达 {int((reach_b&m).sum()):>7,} "
                  f"({100*(reach_b&m).sum()/m.sum():5.1f}%)")

    # ---- DN 入度
    print("\n--- DN 入度(W[:, dst] 每列非零数) ---")
    sub = W.tocsc()[:, dst]
    indeg = np.diff(sub.indptr)
    print(f"  min={indeg.min()} median={int(np.median(indeg))} max={indeg.max()} "
          f"零入度={int((indeg==0).sum())}")

    # ---- 感光细胞出度
    print("\n--- 感光细胞出度 ---")
    sub2 = W.tocsr()[src, :]
    outdeg = np.diff(sub2.indptr)
    print(f"  min={outdeg.min()} median={int(np.median(outdeg))} max={outdeg.max()} "
          f"零出度={int((outdeg==0).sum())}")

    # ---- 感光细胞直接投向哪些 superclass
    print("\n--- 感光细胞(一级)直接投射目标的 superclass 分布 ---")
    cols = sub2.indices
    vals2, cnts2 = np.unique(sup[cols], return_counts=True)
    order2 = np.argsort(cnts2)[::-1][:10]
    for o in order2:
        print(f"  {vals2[o]:26s} {cnts2[o]:>8,}")

    print("\n" + "=" * 76, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
