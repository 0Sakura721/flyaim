"""A 线内部工具:生物学合理性验证 —— 保留的 A-A 子图是否符合已知视叶解剖。

如果 R1-R6 的输出主要落在 L1-L5 / Mi / Tm / C2 / C3(lamina 与 medulla 靶细胞),
而 T4/T5 的输入主要来自 L1-L5,就说明:
  (1) 方向约定 W[i,j] = j->i 正确;
  (2) 保留的 25.58M 条 A-A 边是真实且有意义的连接组。
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

ROOT = Path(r"D:\dsh\autoaim")
sys.path.insert(0, str(ROOT))
from flyaim.io import ConnectomeArtifacts  # noqa: E402

BUILD = ROOT / "flyaim" / "data" / "build"
t0 = time.time()
idx = pd.read_parquet(BUILD / "neuron_index.parquet")
art = ConnectomeArtifacts.load(BUILD / "connectome.npz")
W = (art.W_exc + art.W_inh).tocsr()
print(f"载入 {time.time()-t0:.1f}s;  W.nnz={W.nnz:,}  N={W.shape[0]:,}")

ty = idx["type"].astype("string").fillna("?").to_numpy()
sup = idx["superclass"].astype("string").fillna("?").to_numpy()
cla = idx["class"].astype("string").fillna("?").to_numpy()


def show(title: str, sel: np.ndarray, k: int = 14, outgoing: bool = True) -> None:
    if sel.size == 0:
        print(f"\n{title}: 无细胞")
        return
    sub = W[sel, :] if outgoing else W[:, sel]
    tot = np.asarray(sub.sum(axis=1 if outgoing else 0)).ravel()
    print(f"\n{title}  n={sel.size:,}  出/入权重合计={tot.sum():,.0f}  "
          f"非零={int((tot>0).sum()):,}")
    agg: dict[str, float] = {}
    partner_of = (lambda i: np.asarray(sub[i].todense()).ravel() if outgoing
                  else np.asarray(sub[:, i].todense()).ravel())
    for i in range(min(sel.size, 60)):
        v = partner_of(i)
        for j in np.flatnonzero(v):
            key = f"{sup[j]}/{ty[j]}"
            agg[key] = agg.get(key, 0.0) + float(v[j])
    top = sorted(agg.items(), key=lambda kv: -kv[1])[:k]
    print(f"  取 {min(sel.size,60)} 个代表细胞的 top-{k} 靶(按 type 聚合权重):")
    for k2, v in top:
        print(f"    {k2:<40s} {v:>12,.0f}")


def pick(mask: np.ndarray) -> np.ndarray:
    return np.flatnonzero(mask)


# 1. 感光细胞 -> 下游
r16 = pick(ty == "R1-R6")
show("R1-R6 感光细胞(外)", r16[:60], outgoing=True)
show("R1-R6 感光细胞(外)[反向,看谁输入给它们]", r16[:60], outgoing=False)

# 2. T4 / T5 运动检测:应接收 L1-L5 / Mi / Tm
t4 = pick(np.char.startswith(ty.astype(str), "T4"))
t5 = pick(np.char.startswith(ty.astype(str), "T5"))
show("T4(ON 运动通路)", t4[:60], outgoing=False)
show("T5(OFF 运动通路)", t5[:60], outgoing=False)

# 3. L1-L5 应接收 R1-R6
l15 = pick(np.isin(ty, ["L1", "L2", "L3", "L4", "L5"]))
show("L1-L5(lamina 单极细胞)", l15[:60], outgoing=False)

# 4. 下行神经元:应接收大量上行/中央复合体输入
dn = pick(sup == "descending_neuron")
show("下行神经元 DN", dn[:60], outgoing=False)
show("下行神经元 DN(输出)", dn[:60], outgoing=True)

# 5. 运动神经元
mn = pick(sup == "vnc_motor")
show("运动神经元 MN", mn[:60], outgoing=False)

# 6. 全局:出/入度分布
deg_out = np.asarray(W.sum(axis=0)).ravel()
deg_in = np.asarray(W.sum(axis=1)).ravel()
print(f"\n全局权重出度: 非零 {int((deg_out>0).sum()):,}/{W.shape[0]:,}  "
      f"中位(非零) {np.median(deg_out[deg_out>0]):,.0f}  最大 {deg_out.max():,.0f}")
print(f"全局权重入度: 非零 {int((deg_in>0).sum()):,}/{W.shape[0]:,}  "
      f"中位(非零) {np.median(deg_in[deg_in>0]):,.0f}  最大 {deg_in.max():,.0f}")
print(f"孤立神经元(无任何出入边)= {int(((deg_out==0)&(deg_in==0)).sum()):,}")
print(f"总突触计数 = {W.sum():,.0f}")
print(f"\n耗时 {time.time()-t0:.1f}s")
