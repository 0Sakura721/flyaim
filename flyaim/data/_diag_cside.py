"""A 线内部工具:被丢弃端点(body_post 不在 annotations)到底是什么?

采样法,不整表载入。
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.feather as paf

ROOT = Path(r"D:\dsh\autoaim")
CACHE = ROOT / ".cache"
F_ANNOT = CACHE / "body-annotations-male-cns-v1.0-minconf-0.5.feather"
F_WEIGHTS = CACHE / "connectome-weights-male-cns-v1.0-minconf-0.5.feather"
F_NT = CACHE / "body-neurotransmitters-male-cns-v1.0.feather"

t0 = time.time()


def log(m):
    print(f"[{time.time()-t0:6.1f}s] {m}", flush=True)


annot = paf.read_table(F_ANNOT, columns=["bodyId", "superclass"], memory_map=True).to_pandas()
neu = annot["superclass"].notna() & (annot["superclass"].astype(str).str.len() > 0)
aid = annot["bodyId"].to_numpy(np.int64)
o = np.argsort(aid, kind="stable")
aid, neu = aid[o], neu.to_numpy()[o]
aid_set = aid  # sorted
A = np.sort(aid[neu])
nt_ids = np.sort(paf.read_table(F_NT, columns=["body"], memory_map=True)
                 .to_pandas()["body"].to_numpy(np.int64).copy())
log(f"annotations {aid.size:,} (A={A.size:,})  nt bodies {nt_ids.size:,}")


def member(arr, table):
    p = np.searchsorted(table, arr)
    ok = p < table.size
    pc = np.where(ok, p, 0)
    return ok & (table[pc] == arr)


SAMPLE_BATCHES = list(range(0, 2318, 40))          # ~58 批 ≈ 3.8M 行
s_pre, s_post = [], []
with pa.memory_map(str(F_WEIGHTS), "r") as src:
    r = pa.ipc.open_file(src)
    nb = r.num_record_batches
    for i in SAMPLE_BATCHES:
        b = r.get_batch(i)
        n = min(b.num_rows, 20_000)
        s_pre.append(b.column("body_pre").to_numpy(zero_copy_only=False)[:n])
        s_post.append(b.column("body_post").to_numpy(zero_copy_only=False)[:n])
    # 尾部(weight=1 区)单独看
    tail = r.get_batch(nb - 1)
    t_pre = tail.column("body_pre").to_numpy(zero_copy_only=False)
    t_post = tail.column("body_post").to_numpy(zero_copy_only=False)

s_pre = np.concatenate(s_pre)
s_post = np.concatenate(s_post)
log(f"样本行数 pre={s_pre.size:,} post={s_post.size:,};尾部单批 {t_post.size:,}")

for nm, arr in (("body_pre(全样本)", s_pre), ("body_post(全样本)", s_post),
                ("body_post(尾批,weight=1)", t_post)):
    in_a = int(member(arr, A).sum())
    in_ann = int(member(arr, aid_set).sum())
    in_nt = int(member(arr, nt_ids).sum())
    u = np.unique(arr)
    print(f"\n{nm}")
    print(f"  样本 {arr.size:,}  唯一 {u.size:,} ({100.0*u.size/arr.size:.1f}%)  "
          f"min={arr.min():,} max={arr.max():,}")
    print(f"  在 A(166,700 神经元) = {100.0*in_a/arr.size:6.2f}%")
    print(f"  在 annotations(211,577) = {100.0*in_ann/arr.size:6.2f}%")
    print(f"  在 nt bodies(1,835,518) = {100.0*in_nt/arr.size:6.2f}%")
    notann = ~member(arr, aid_set)
    if notann.any():
        v = arr[notann]
        print(f"  不在 annotations 的 {v.size:,} 个:唯一 {np.unique(v).size:,} "
              f"({100.0*np.unique(v).size/v.size:.1f}%)  min={v.min():,} max={v.max():,}")
        print(f"    其中在 nt bodies 里的比例 = {100.0*member(v, nt_ids).mean():.2f}%")
        # 是否只以 post 身份出现
        print(f"    样例 {v[:8].tolist()}")

# 这些"不在 annotations"的 body 是否也作为 body_pre 出现?
allpre = np.concatenate([s_pre, t_pre])
notann_post = s_post[~member(s_post, aid_set)]
uniq_notann = np.unique(notann_post)[:200_000]
u_allpre = np.unique(allpre)
hits = np.isin(uniq_notann, u_allpre, assume_unique=True)
print(f"\n交叉检验:不在 annotations 的 body_post 唯一值(取前 {uniq_notann.size:,} 个)中,"
      f"有 {100.0*hits.mean():.2f}% 也出现在 body_pre 里")
print(f"body_pre 唯一值数 = {u_allpre.size:,}")

# 用 body-stats 的 body 列表再查一次(如果它覆盖更大的 body 空间)
F_STATS = CACHE / "body-stats-male-cns-v1.0-minconf-0.5.feather"
try:
    with pa.memory_map(str(F_STATS), "r") as src:
        rs = pa.ipc.open_file(src)
        b0 = rs.get_batch(0)
        st_bodies = b0.column("body").to_numpy(zero_copy_only=False)
    print(f"\nbody-stats 第一批 body 取值样例 {st_bodies[:8].tolist()}  "
          f"min={st_bodies.min():,} max={st_bodies.max():,}")
    print(f"  这批 body 在 annotations 里的比例 = "
          f"{100.0*member(st_bodies, aid_set).mean():.2f}%")
    print(f"  这批 body 在 nt bodies 里的比例   = "
          f"{100.0*member(st_bodies, nt_ids).mean():.2f}%")
except Exception as e:  # noqa: BLE001
    print(f"body-stats 检查失败: {e}")
