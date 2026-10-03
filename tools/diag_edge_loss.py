"""诊断:原始边表 → CSR 的边丢失定位。

Lead 持有(放在 tools/ 以避免与 A 线 flyaim/data/ 写范围冲突)。
用途:确认 `connectome.npz` 是否发生了系统性边丢失,
并区分「有意的保边策略(如权重阈值)」与「映射 bug」。

运行::

    <捆绑 python> tools/diag_edge_loss.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.feather as feather

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ROOT = Path(r"D:\dsh\autoaim")
WEIGHTS = ROOT / ".cache" / "connectome-weights-male-cns-v1.0-minconf-0.5.feather"
BUILD = ROOT / "flyaim" / "data" / "build"


def main() -> int:
    print("=" * 74)
    print("边丢失诊断")
    print("=" * 74)

    idx = pd.read_parquet(BUILD / "neuron_index.parquet")
    print(f"neuron_index: N = {len(idx):,}, 列 = {list(idx.columns)[:8]}")
    body = idx["body_id"].to_numpy()
    print(f"  body_id 范围 = [{body.min():,}, {body.max():,}]")
    print(f"  唯一值       = {len(np.unique(body)):,}")
    print(f"  索引连续     = {bool(np.array_equal(np.asarray(idx.index), np.arange(len(idx))))}")

    lut = pd.Series(np.arange(len(body), dtype=np.int64), index=body)

    t = feather.read_table(WEIGHTS, memory_map=True)
    n_raw = t.num_rows
    w = t.column("weight").to_numpy()
    print(f"\n原始边表: {n_raw:,} 行")
    print(f"  weight 范围 = [{w.min():,}, {w.max():,}]")

    pre = t.column("body_pre").to_numpy()
    post = t.column("body_post").to_numpy()

    s_pre = lut.reindex(pre).to_numpy()
    s_post = lut.reindex(post).to_numpy()
    have_pre = ~np.isnan(s_pre)
    have_post = ~np.isnan(s_post)
    both = have_pre & have_post

    print("\n映射结果:")
    print(f"  body_pre  可定位 : {have_pre.sum():,} ({100*have_pre.mean():.1f}%)")
    print(f"  body_post 可定位 : {have_post.sum():,} ({100*have_post.mean():.1f}%)")
    print(f"  两端都在索引内   : {both.sum():,} ({100*both.mean():.1f}%)")
    print(f"  至少一端不在     : {(~both).sum():,}")

    self_loop = both & (pre == post)
    print(f"  其中自环         : {self_loop.sum():,}")

    n_uniq = 0
    if both.sum() > 0:
        a = s_pre[both].astype(np.int64)
        b = s_post[both].astype(np.int64)
        packed = a * np.int64(len(body)) + b
        n_uniq = int(np.unique(packed).size)
        print(f"  两端都在内的唯一边: {n_uniq:,}")
        del packed, a, b
    del s_pre, s_post, pre, post

    from flyaim.io import ConnectomeArtifacts

    art = ConnectomeArtifacts.load(BUILD / "connectome.npz")
    print("\nCSR 实际:")
    print(f"  N         = {art.n:,}")
    print(f"  W_exc.nnz = {art.W_exc.nnz:,}")
    print(f"  W_inh.nnz = {art.W_inh.nnz:,}")
    print(f"  合计      = {art.n_edges:,}")
    print(f"  相对原始   = {100*art.n_edges/n_raw:.1f}%")
    if n_uniq:
        print(f"  相对唯一边 = {100*art.n_edges/n_uniq:.1f}%")

    print("\n" + "=" * 74)
    if both.sum() == n_raw and n_uniq and art.n_edges < n_uniq * 0.99:
        print("结论: 边表全部可映射,但 CSR 边数远少于唯一边数 → 构建过程有丢失 BUG")
    elif both.sum() < n_raw * 0.99:
        print(f"结论: {n_raw - int(both.sum()):,} 条边的端点不在 neuron_index 中。")
        print("      若这不是有意策略,说明索引口径与边表不一致(疑似 BUG)。")
    else:
        print("结论: 边数一致,无丢失。")
    print("=" * 74)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
