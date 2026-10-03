"""诊断二:哪些 body 被排除在 neuron_index 之外,以及它们是什么。

Lead 持有。承接 diag_edge_loss.py 的结论:
    body_post 只有 20% 能映射到 neuron_index -> 丢失 1.26 亿条边的根因在此。

本脚本回答关键问题:被排除的是**非神经元(胶质细胞/伪影)**,
还是**真实神经元被误排除**(后者是 bug)。

运行::

    <捆绑 python> tools/diag_missing_bodies.py
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.feather as feather

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ROOT = Path(r"D:\dsh\autoaim")
CACHE = ROOT / ".cache"
WEIGHTS = CACHE / "connectome-weights-male-cns-v1.0-minconf-0.5.feather"
ANNOT = CACHE / "body-annotations-male-cns-v1.0-minconf-0.5.feather"
BUILD = ROOT / "flyaim" / "data" / "build"


def main() -> int:
    print("=" * 74)
    print("被排除 body 的身份诊断")
    print("=" * 74)

    idx = pd.read_parquet(BUILD / "neuron_index.parquet")
    in_index = set(idx["body_id"].to_numpy().tolist())
    print(f"neuron_index: N = {len(in_index):,}")

    # --- 边表里出现的所有 body ---
    t = feather.read_table(WEIGHTS, memory_map=True)
    pre = np.unique(t.column("body_pre").to_numpy())
    post = np.unique(t.column("body_post").to_numpy())
    all_bodies = np.union1d(pre, post)
    print(f"\n边表中出现的 distinct body: {len(all_bodies):,}")
    print(f"  body_pre  distinct : {len(pre):,}")
    print(f"  body_post distinct : {len(post):,}")

    miss_all = np.array([b for b in all_bodies if b not in in_index], dtype=np.int64)
    miss_post = np.array([b for b in post if b not in in_index], dtype=np.int64)
    print(f"\n不在 neuron_index 中的:")
    print(f"  任意端点 body : {len(miss_all):,} ({100*len(miss_all)/len(all_bodies):.1f}%)")
    print(f"  仅 body_post  : {len(miss_post):,} ({100*len(miss_post)/len(post):.1f}%)")

    # --- 这些缺失 body 在标注文件里是什么? ---
    ann = pd.read_feather(ANNOT)
    ann_cols = list(ann.columns)
    print(f"\n标注文件: {len(ann):,} 行, 列 = {ann_cols[:10]}")
    idcol = "bodyId" if "bodyId" in ann_cols else ann_cols[0]

    ann_idx = ann.set_index(idcol)
    present = ann_idx.index.isin(miss_all)
    print(f"  缺失 body 中能在标注文件里找到的: {int(present.sum()):,} / {len(miss_all):,}")

    sub = ann_idx.loc[ann_idx.index.isin(miss_all)]
    if len(sub):
        for col in ("statusLabel", "superclass", "class", "type"):
            if col in sub.columns:
                c = Counter(sub[col].fillna("<NA>").astype(str))
                print(f"\n  [{col}] 缺失体的分布 (top 12):")
                for k, v in c.most_common(12):
                    print(f"     {k:34s} {v:>10,}")

    # --- 对照:索引内的体 的 statusLabel 分布 ---
    if "statusLabel" in ann.columns:
        insub = ann_idx.loc[ann_idx.index.isin(in_index)]
        c_in = Counter(insub["statusLabel"].fillna("<NA>").astype(str))
        print("\n  [对照] 索引内体的 statusLabel 分布 (top 12):")
        for k, v in c_in.most_common(12):
            print(f"     {k:34s} {v:>10,}")

    # --- 关键判定 ---
    glia_like = 0
    if len(sub) and "statusLabel" in sub.columns:
        sl = sub["statusLabel"].fillna("<NA>").astype(str)
        glia_like = int(sl.isin(["Glia", "Orphan-artifact", "Unimportant", "Out of scope",
                                 "Orphan", "RT Orphan", "PRT Orphan"]).sum())
    print("\n" + "=" * 74)
    print(f"缺失体中被标注为 Glia/Orphan/Unimportant/Out-of-scope 类: {glia_like:,} / {len(miss_all):,}")
    if len(miss_all) and glia_like / len(miss_all) > 0.5:
        print("结论: 大部分缺失体是非神经元或低质量分割 -> 排除可能是**合理过滤**,")
        print("      但边被直接丢弃会让连接组缺失 83% 的边,必须显式说明并评估影响。")
    else:
        print("结论: 缺失体中**真实神经元占比高** -> 索引口径与边表不一致,疑似 BUG。")
    print("=" * 74)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
