"""A 线内部工具:用官方 traced-only 文件的 type_pre/type_post 列独立验证方向约定。

若 body_pre 的细胞类型 == body_pre 列指向的神经元 type,且 body_post 同理,
则字段语义 (pre=突触前, post=突触后) 得到官方文件的直接确认,
从而 W[i,j] = j->i 的落盘约定正确。
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa

ROOT = Path(r"D:\dsh\autoaim")
CACHE = ROOT / ".cache"
BUILD = ROOT / "flyaim" / "data" / "build"
F = CACHE / "connectome-weights-male-cns-v1.0-minconf-0.5-traced-only.feather"

t0 = time.time()
idx = pd.read_parquet(BUILD / "neuron_index.parquet")
bids = idx["body_id"].to_numpy(np.int64)
ty = idx["type"].astype(str).to_numpy()
sup = idx["superclass"].astype(str).to_numpy()
print(f"N={bids.size:,} 升序={bool((np.diff(bids)>0).all())}")

with pa.memory_map(str(F), "r") as src:
    r = pa.ipc.open_file(src)
    nb = r.num_record_batches
    b = r.get_batch(0)
    print("schema:", r.schema.names)
    tabs = []
    for i in range(0, nb, 40):
        bb = r.get_batch(i)
        tabs.append(bb.slice(0, min(bb.num_rows, 20000)).to_pandas())
df = pd.concat(tabs, ignore_index=True)
print(f"样本 {len(df):,} 行")
print(df.head(6).to_string(index=False))

for side in ("pre", "post"):
    body = df[f"body_{side}"].to_numpy(np.int64)
    tp = df[f"type_{side}"].astype("string").fillna("").to_numpy()
    p = np.searchsorted(bids, body)
    ok = p < bids.size
    pc = np.where(ok, p, 0)
    ok &= bids[pc] == body
    print(f"\n--- body_{side} ---")
    print(f"  在索引集内的比例 = {100.0*ok.mean():.2f}%")
    m = ok & (tp != "")
    same = (ty[pc][m] == tp[m])
    print(f"  有 type_{side} 的样本 {int(m.sum()):,} 行;"
          f"  与 neuron_index.type 完全一致 = {100.0*same.mean():.2f}%")
    if (~same).any():
        ex = np.flatnonzero(m & ~same)[:8]
        print("  不一致样例(官方 type vs 我们的 type):")
        for j in ex:
            print(f"    body={body[j]:>12,}  官方={tp[j]!r:24s} 我们={ty[pc[j]]!r}  "
                  f"superclass={sup[pc[j]]}")
    # 用 superclass 交叉:type_pre 应落在 ol_sensory 等真实类型里
    print(f"  type_{side} 取值样例 = {sorted(pd.unique(tp))[:8]}")

print(f"\n耗时 {time.time()-t0:.1f}s")
