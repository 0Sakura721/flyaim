"""A 线内部工具:边过滤诊断(决定性)。

要回答的问题:
  Q1. neuron_index 的 body_id 是否严格升序(否则 searchsorted 会误判)?
  Q2. 151.9M 条边的两个端点分别落在哪个集合?
      A = 166,700 神经元(superclass 非空) / B = 44,877 有标注非神经元 / C = 不在 annotations
  Q3. 为什么 pre 在集合内 112.6M 而 post 在集合内只有 4.8M?是数据性质还是映射错误?
  Q4. body_post 到底是不是 body ID(重复度/唯一值数)?
  Q5. 保留/丢弃是否与 weight 有关?
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.feather as paf

ROOT = Path(r"D:\dsh\autoaim")
sys.path.insert(0, str(ROOT))
CACHE = ROOT / ".cache"
BUILD = ROOT / "flyaim" / "data" / "build"
F_ANNOT = CACHE / "body-annotations-male-cns-v1.0-minconf-0.5.feather"
F_WEIGHTS = CACHE / "connectome-weights-male-cns-v1.0-minconf-0.5.feather"
F_NT = CACHE / "body-neurotransmitters-male-cns-v1.0.feather"

t0 = time.time()


def log(m: str) -> None:
    print(f"[{time.time()-t0:6.1f}s] {m}", flush=True)


# ---------------------------------------------------------------- Q1
idxdf = pd.read_parquet(BUILD / "neuron_index.parquet")
bids = idxdf["body_id"].to_numpy(dtype=np.int64)
log(f"Q1 neuron_index: N={bids.size:,}  严格升序={bool((np.diff(bids) > 0).all())}  "
    f"min={bids.min():,} max={bids.max():,}")

# ---------------------------------------------------------------- 准备查找表
annot = paf.read_table(F_ANNOT, columns=["bodyId", "superclass", "statusLabel", "class", "type"],
                       memory_map=True).to_pandas()
is_neuron = (annot["superclass"].notna()
             & (annot["superclass"].astype(str).str.len() > 0)).to_numpy()
all_ids = annot["bodyId"].to_numpy(dtype=np.int64)
ordr = np.argsort(all_ids, kind="stable")
all_ids = all_ids[ordr]
is_neuron = is_neuron[ordr]
code_tab = np.where(is_neuron, 1, 2).astype(np.int8)   # 1=A, 2=B
NB = all_ids.size
log(f"annotations: {NB:,}(A={int(is_neuron.sum()):,}, B={int((~is_neuron).sum()):,})")

nt_ids = np.sort(paf.read_table(F_NT, columns=["body"], memory_map=True)
                 .to_pandas()["body"].to_numpy(np.int64).copy())
log(f"nt file bodies: {nt_ids.size:,}")

LAB = {(1, 1): "A-A", (1, 2): "A-B", (1, 0): "A-C",
       (2, 1): "B-A", (2, 2): "B-B", (2, 0): "B-C",
       (0, 1): "C-A", (0, 2): "C-B", (0, 0): "C-C"}
counts = {v: 0 for v in LAB.values()}
wsum = {v: 0 for v in LAB.values()}
# weight 分桶 x 是否两端都在 A
wb_names = ["w=1", "w=2..9", "w=10..99", "w>=100"]
wb = {n: {"total": 0, "A-A": 0} for n in wb_names}

n_total = 0
samples: dict[str, list] = {"A-C": [], "C-A": [], "C-C": [], "A-B": [], "B-A": []}
with pa.memory_map(str(F_WEIGHTS), "r") as src:
    r = pa.ipc.open_file(src)
    nb = r.num_record_batches
    for i in range(nb):
        b = r.get_batch(i)
        bp = b.column("body_pre").to_numpy(zero_copy_only=False)
        bq = b.column("body_post").to_numpy(zero_copy_only=False)
        w = b.column("weight").to_numpy(zero_copy_only=False)
        n_total += bp.size
        ip = np.searchsorted(all_ids, bp)
        iq = np.searchsorted(all_ids, bq)
        okp = ip < NB
        ipc = np.where(okp, ip, 0)
        okp &= all_ids[ipc] == bp
        okq = iq < NB
        iqc = np.where(okq, iq, 0)
        okq &= all_ids[iqc] == bq
        cp = np.where(okp, code_tab[ipc], 0)
        cq = np.where(okq, code_tab[iqc], 0)
        key = cp.astype(np.int64) * 3 + cq
        for k in range(9):
            m = key == k
            nk = int(m.sum())
            if nk:
                name = LAB[(k // 3, k % 3)]
                counts[name] += nk
                wsum[name] += int(w[m].sum())
        aa = key == 4
        bucket = np.where(w == 1, 0, np.where(w < 10, 1, np.where(w < 100, 2, 3)))
        for bi, bn in enumerate(wb_names):
            mb = bucket == bi
            wb[bn]["total"] += int(mb.sum())
            wb[bn]["A-A"] += int((mb & aa).sum())
        # 采样被丢弃的端点
        if i < 60 or i % 500 == 0:
            for nm, mask, arr in (("A-C", key == 2, bq), ("C-A", key == 6, bp),
                                  ("C-C", key == 8, bq), ("A-B", key == 1, bq),
                                  ("B-A", key == 3, bp)):
                m = mask
                if m.any() and len(samples[nm]) < 300_000:
                    samples[nm].extend(arr[m][:1000].tolist())
        if (i + 1) % 800 == 0 or i == nb - 1:
            log(f"  batch {i+1}/{nb}")

log("扫描完成")
assert sum(counts.values()) == n_total
print(f"\n总边数 = {n_total:,}")
print(f"{'分类(前->后)':<14s} {'边数':>15s} {'占比':>8s} {'突触计数合计':>16s} {'占突触比':>9s}")
tsyn = sum(wsum.values())
for k, v in sorted(counts.items(), key=lambda kv: -kv[1]):
    print(f"{k:<14s} {v:>15,d} {100.0*v/n_total:>7.3f}% {wsum[k]:>16,d} {100.0*wsum[k]/tsyn:>8.3f}%")

print(f"\nQ5 按 weight 分桶:")
print(f"{'bucket':<12s} {'边数':>15s} {'其中 A-A':>15s} {'A-A 占比':>9s}")
for bn in wb_names:
    t, a = wb[bn]["total"], wb[bn]["A-A"]
    print(f"{bn:<12s} {t:>15,d} {a:>15,d} {100.0*a/max(t,1):>8.3f}%")

print("\nQ3/Q4 采样被丢弃端点的身份:")
for nm, vals in samples.items():
    if not vals:
        continue
    a = np.asarray(vals, dtype=np.int64)
    pos = np.searchsorted(all_ids, a)
    ok = pos < NB
    posc = np.where(ok, pos, 0)
    ok &= all_ids[posc] == a
    in_annot = int(ok.sum())
    in_b = int((ok & (code_tab[posc] == 2)).sum())
    p2 = np.searchsorted(nt_ids, a)
    ok2 = p2 < nt_ids.size
    p2c = np.where(ok2, p2, 0)
    ok2 &= nt_ids[p2c] == a
    uniq = len(set(a.tolist()))
    print(f"  {nm:<5s} n_sample={a.size:>7,} uniq={uniq:>7,} ({100.0*uniq/a.size:5.1f}% 唯一)  "
          f"在annotations={100.0*in_annot/a.size:6.2f}% 其中B={100.0*in_b/a.size:6.2f}%  "
          f"在nt文件={100.0*int(ok2.sum())/a.size:6.2f}%  min={a.min():,} max={a.max():,}")

# 前 20 行的成员资格
with pa.memory_map(str(F_WEIGHTS), "r") as src:
    r = pa.ipc.open_file(src)
    b0 = r.get_batch(0).slice(0, 20).to_pandas()
b0["pre_in_A"] = b0["body_pre"].isin(set(bids.tolist()))
b0["post_in_A"] = b0["body_post"].isin(set(bids.tolist()))
print("\n前 20 行成员资格:")
print(b0.to_string(index=False))

res = {"n_total": int(n_total), "counts": counts, "weight_sums": wsum,
       "by_weight": wb,
       "sample": {k: {"n": len(v),
                      "uniq": len(set(v)),
                      "min": int(min(v)) if v else None,
                      "max": int(max(v)) if v else None} for k, v in samples.items() if v}}
(CACHE / "edge_partition.json").write_text(json.dumps(res, ensure_ascii=False, indent=2),
                                           encoding="utf-8")
log(f"wrote {CACHE/'edge_partition.json'}")
