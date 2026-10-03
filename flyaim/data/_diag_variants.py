"""A 线内部工具:对比三个 weights 变体。

- connectome-weights-...-minconf-0.5.feather             (全量,我们用的)
- connectome-weights-...-minconf-0.5-traced-only.feather
- connectome-weights-...-minconf-0.5-significant-only.feather

对每个文件:行数、schema、A-A 边数、突触合计。用来判断"全量文件 + A 集合过滤"
是否等价于官方提供的神经元级连接组。
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.feather as paf

ROOT = Path(r"D:\dsh\autoaim")
CACHE = ROOT / ".cache"
F_ANNOT = CACHE / "body-annotations-male-cns-v1.0-minconf-0.5.feather"
FILES = {
    "full": CACHE / "connectome-weights-male-cns-v1.0-minconf-0.5.feather",
    "traced-only": CACHE / "connectome-weights-male-cns-v1.0-minconf-0.5-traced-only.feather",
    "significant-only": CACHE / "connectome-weights-male-cns-v1.0-minconf-0.5-significant-only.feather",
}
t0 = time.time()


def log(m):
    print(f"[{time.time()-t0:6.1f}s] {m}", flush=True)


annot = paf.read_table(F_ANNOT, columns=["bodyId", "superclass"], memory_map=True).to_pandas()
neu = (annot["superclass"].notna() & (annot["superclass"].astype(str).str.len() > 0)).to_numpy()
aid = annot["bodyId"].to_numpy(np.int64)
o = np.argsort(aid, kind="stable")
aid, neu = aid[o], neu[o]
A = np.sort(aid[neu])
log(f"annotations {aid.size:,}(A={A.size:,})")


def match(arr, table):
    p = np.searchsorted(table, arr)
    ok = p < table.size
    pc = np.where(ok, p, 0)
    return ok & (table[pc] == arr)


for name, path in FILES.items():
    if not path.exists():
        print(f"\n### {name}: 文件不存在,跳过")
        continue
    print(f"\n{'='*90}\n### {name}   {path.stat().st_size:,} B\n{'='*90}")
    with pa.memory_map(str(path), "r") as src:
        r = pa.ipc.open_file(src)
        nb = r.num_record_batches
        print(f"schema = {r.schema.names}   batches={nb}")
        nrows = 0
        aa = 0
        aa_w = 0
        pre_uniq: set = set()
        post_uniq: set = set()
        for i in range(nb):
            b = r.get_batch(i)
            bp = b.column("body_pre").to_numpy(zero_copy_only=False)
            bq = b.column("body_post").to_numpy(zero_copy_only=False)
            w = b.column("weight").to_numpy(zero_copy_only=False)
            nrows += bp.size
            mp = match(bp, A)
            mq = match(bq, A)
            m = mp & mq
            aa += int(m.sum())
            aa_w += int(w[m].sum())
            if i % 200 == 0:                       # 采样唯一性
                if len(pre_uniq) < 3_000_000:
                    pre_uniq.update(bp[:5000].tolist())
                if len(post_uniq) < 3_000_000:
                    post_uniq.update(bq[:5000].tolist())
            if (i + 1) % 800 == 0:
                log(f"  {name}: {i+1}/{nb}")
        print(f"  行数(边)          = {nrows:,}")
        print(f"  其中两端都在 A 的边 = {aa:,}  ({100.0*aa/max(nrows,1):.2f}%)")
        print(f"  两端都在 A 的突触合计 = {aa_w:,}")
        print(f"  采样的 pre 唯一值 = {len(pre_uniq):,}   post 唯一值 = {len(post_uniq):,}")
log("完成")
