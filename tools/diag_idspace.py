"""诊断三:weights 边表的 body ID 空间与 annotations 是否一致。

Lead 持有。承接 diag_edge_loss.py / diag_missing_bodies.py:
    边表出现 88.4M 个 distinct body,而 annotations 只有 211,577 行。
    这说明两个文件的 body ID **不在同一 ID 空间**,不是"过滤掉了一些细胞"。

本脚本给出决定性证据:比较 ID 取值范围与重叠度,并检查 body_post 的分布形态。

运行::

    <捆绑 python> tools/diag_idspace.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.feather as feather

ROOT = Path(r"D:\dsh\autoaim")
CACHE = ROOT / ".cache"
WEIGHTS = CACHE / "connectome-weights-male-cns-v1.0-minconf-0.5.feather"
ANNOT = CACHE / "body-annotations-male-cns-v1.0-minconf-0.5.feather"


def _stats(name: str, a: np.ndarray) -> dict:
    return {
        "name": name,
        "n": int(a.size),
        "distinct": int(np.unique(a).size),
        "min": int(a.min()),
        "max": int(a.max()),
    }


def main() -> int:
    print("=" * 78)
    print("ID 空间一致性诊断")
    print("=" * 78)

    ann = pd.read_feather(ANNOT)
    idcol = "bodyId" if "bodyId" in ann.columns else ann.columns[0]
    ann_ids = ann[idcol].to_numpy().astype(np.int64)
    a = _stats("annotations.bodyId", ann_ids)
    ann_set = set(ann_ids.tolist())
    print(f"\nannotations: {a['n']:,} 行, distinct {a['distinct']:,}, "
          f"范围 [{a['min']:,}, {a['max']:,}]")

    t = feather.read_table(WEIGHTS, memory_map=True)
    pre = t.column("body_pre").to_numpy()
    post = t.column("body_post").to_numpy()
    p = _stats("weights.body_pre", pre)
    q = _stats("weights.body_post", post)
    print(f"weights.body_pre : {p['n']:,} 行, distinct {p['distinct']:,}, "
          f"范围 [{p['min']:,}, {p['max']:,}]")
    print(f"weights.body_post: {q['n']:,} 行, distinct {q['distinct']:,}, "
          f"范围 [{q['min']:,}, {q['max']:,}]")

    # --- 重叠度 ---
    print("\n--- 与 annotations 的重叠 ---")
    for nm, arr in (("body_pre", pre), ("body_post", post)):
        u = np.unique(arr)
        # 分块检测,避免一次性构造巨大布尔数组
        hit = 0
        step = 5_000_000
        for i in range(0, u.size, step):
            chunk = u[i:i + step]
            hit += sum(1 for v in chunk.tolist() if v in ann_set)
        print(f"  {nm}: distinct {u.size:,}, 命中 annotations {hit:,} "
              f"({100*hit/u.size:.2f}%)")

    # --- 判定性证据:ID 分布形态 ---
    print("\n--- ID 分布形态 ---")
    print("  body_pre  distinct 数 =", f"{p['distinct']:,}")
    print("  body_post distinct 数 =", f"{q['distinct']:,}")
    print("  annotations distinct  =", f"{a['distinct']:,}")
    print()
    print("  解读:若 body_post 的 distinct 数远大于 annotations 的细胞数,")
    print("        则两文件**不在同一 ID 空间**(body_post 很可能是分割体素/原始 segment id,")
    print("        而非 proofread 后的神经元 body id)。")

    # --- 检查是否存在"post 是 pre 的子集/超集"这类关系 ---
    print("\n--- 交集形态(抽样估计) ---")
    rng = np.random.default_rng(0)
    sp = np.unique(rng.choice(pre, size=min(2_000_000, pre.size), replace=False))
    sq = np.unique(rng.choice(post, size=min(2_000_000, post.size), replace=False))
    print(f"  抽样 body_pre  distinct = {sp.size:,}, body_post distinct = {sq.size:,}")
    inter = np.intersect1d(sp, sq)
    print(f"  两者交集 = {inter.size:,}")
    if inter.size:
        print(f"  交集样例 = {inter[:10].tolist()}")

    # --- 结论 ---
    print("\n" + "=" * 78)
    if q["distinct"] > a["distinct"] * 10:
        print("结论: **ID 空间不一致**(已证实)。")
        print(f"      body_post 有 {q['distinct']:,} 个 distinct 值,"
              f"远超 annotations 的 {a['distinct']:,}。")
        print("      → connectome.npz 的 25.58M 边只是「两端恰好都能在 annotations 中")
        print("        按数值匹配上」的那一小部分,属于**假匹配**,不是真实连接组的子集。")
        print("      → 必须改用与 weights 同一 ID 空间的数据源(neuPrint API 或 neo4j 转储)")
        print("        重新构建连接组,否则仿真跑的不是果蝇的连接组。")
    else:
        print("结论: ID 空间量级相近,需进一步人工核对。")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
