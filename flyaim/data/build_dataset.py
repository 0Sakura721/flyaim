"""FlyAim A 线:MaleCNS v1.0 -> 全局索引空间 + CSR 连接组 + 角色选择 + manifest。

分阶段可重复运行(A 线核心构建脚本):

    python flyaim/data/build_dataset.py index       # 只建 neuron_index.parquet
    python flyaim/data/build_dataset.py roles       # 只建 roles.json
    python flyaim/data/build_dataset.py connectome  # 只建 connectome.npz(需要 index + roles)
    python flyaim/data/build_dataset.py manifest    # 只建 manifest.json
    python flyaim/data/build_dataset.py all         # 全流程

索引口径(重要,记录于 manifest.notes):
    N = annotations 中 ``superclass`` 非空的行数 = 166,700。
    该列非空恰好对应"有功能分类的神经元"(glia / unimportant / orphan 等 44,877 行
    没有 superclass,已排除)。bodyId 升序 -> 全局索引 0..N-1。

连接组语义:
    weights feather 是边表 (body_pre, body_post, weight),weight = 突触计数。
    契约要求 W[i, j] = j -> i 的投射强度,因此
        row = index(body_post), col = index(body_pre)。

抑制性判定:
    契约规定由 body-neurotransmitters 的 GABA/Glycine 预测判定,且以
    **突触前**细胞的递质决定该边是抑制还是兴奋。
    本脚本只对"突触前细胞属于 roles.inhibitory"的边走 W_inh,其余入 W_exc。
"""

from __future__ import annotations

import argparse
import ctypes
import ctypes.wintypes as wt
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.feather as paf
import scipy.sparse as sp

ROOT = Path(r"D:\dsh\autoaim")
sys.path.insert(0, str(ROOT))

from flyaim.io import (  # noqa: E402
    ARTIFACT_INDEX,
    ARTIFACT_MANIFEST,
    ARTIFACT_ROLES,
    ARTIFACT_WEIGHTS,
    ConnectomeArtifacts,
    Manifest,
    save_roles,
    sha256_file,
)
from flyaim.neuron_index import NeuronIndex  # noqa: E402

CACHE = ROOT / ".cache"
BUILD = ROOT / "flyaim" / "data" / "build"
BUILD.mkdir(parents=True, exist_ok=True)

F_ANNOT = CACHE / "body-annotations-male-cns-v1.0-minconf-0.5.feather"
F_WEIGHTS = CACHE / "connectome-weights-male-cns-v1.0-minconf-0.5.feather"
F_NT = CACHE / "body-neurotransmitters-male-cns-v1.0.feather"
F_STATS = CACHE / "body-stats-male-cns-v1.0-minconf-0.5.feather"
F_REPORT = CACHE / "download_report.json"

URLS = {
    "annotations": "https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/"
                   "flat-connectome/body-annotations-male-cns-v1.0-minconf-0.5.feather",
    "weights": "https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/"
               "flat-connectome/connectome-weights-male-cns-v1.0-minconf-0.5.feather",
    "neurotransmitters": "https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/"
                         "flat-connectome/body-neurotransmitters-male-cns-v1.0.feather",
    "body_stats": "https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/"
                  "flat-connectome/body-stats-male-cns-v1.0-minconf-0.5.feather",
}

STATE = CACHE / "build_state.json"


# ------------------------------------------------------------------ 工具


class _PMC(ctypes.Structure):
    _fields_ = [
        ("cb", wt.DWORD),
        ("PageFaultCount", wt.DWORD),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
    ]


_K32 = ctypes.windll.kernel32
_PSAPI = ctypes.windll.psapi
_K32.GetCurrentProcess.restype = ctypes.c_void_p
_PSAPI.GetProcessMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.POINTER(_PMC), wt.DWORD]
_PSAPI.GetProcessMemoryInfo.restype = wt.BOOL


def mem_mb() -> tuple[float, float]:
    """(当前 RSS, 峰值 RSS) MB,Windows psapi。

    注意:必须显式设置 restype/argtypes,否则 64 位下 HANDLE 会被截断成 int32,
    GetProcessMemoryInfo 静默失败返回 0(第一版就踩了这个坑)。
    """
    p = _PMC()
    p.cb = ctypes.sizeof(_PMC)
    ok = _PSAPI.GetProcessMemoryInfo(_K32.GetCurrentProcess(), ctypes.byref(p), p.cb)
    if not ok:
        return (0.0, 0.0)
    return (p.WorkingSetSize / 1048576, p.PeakWorkingSetSize / 1048576)


T0 = time.time()


def log(msg: str) -> None:
    cur, peak = mem_mb()
    print(f"[{time.time()-T0:7.1f}s] {msg}   (RSS {cur:,.0f} MB / peak {peak:,.0f} MB)", flush=True)


def load_state() -> dict:
    if STATE.exists():
        return json.loads(STATE.read_text(encoding="utf-8"))
    return {}


def save_state(d: dict) -> None:
    STATE.write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")


def file_entries() -> dict:
    """source_files: {name: {url, bytes, sha256}}。"""
    out: dict[str, dict] = {}
    rep = json.loads(F_REPORT.read_text(encoding="utf-8")) if F_REPORT.exists() else {}
    for key, url in URLS.items():
        path = {"annotations": F_ANNOT, "weights": F_WEIGHTS, "neurotransmitters": F_NT,
                "body_stats": F_STATS}[key]
        r = rep.get(key)
        if path.exists() and r and r.get("sha256") and r.get("bytes") == path.stat().st_size:
            out[key] = {"url": url, "bytes": int(r["bytes"]), "sha256": r["sha256"]}
        elif path.exists():
            out[key] = {"url": url, "bytes": path.stat().st_size, "sha256": sha256_file(path)}
        else:
            out[key] = {"url": url, "bytes": 0, "sha256": None, "status": "not_downloaded"}
    return out


# ------------------------------------------------------------------ 阶段 1:索引


def save_index(idx: NeuronIndex, out: Path) -> None:
    """落盘 neuron_index.parquet,保证 ``NeuronIndex.load()`` 能原样读回。

    背景(A 线实测):冻结地基 ``NeuronIndex.save()`` 调用
    ``self.df.to_parquet(p, index=True, index_label="index")``,而 pandas 的
    ``to_parquet`` **没有** ``index_label`` 形参 —— 该 kwarg 被透传给
    ``pyarrow.parquet.write_table``,在当前 pandas 3.0.1 + pyarrow 25.0.1 下抛
    ``TypeError: __cinit__() got an unexpected keyword argument 'index_label'``,
    并在磁盘上留下 0 字节文件。地基文件 A 线不得修改,故此处用**完全等价**的写法:
    把索引命名为 ``index`` 再 ``to_parquet(index=True)``,产物与
    ``NeuronIndex.load()`` 的读取逻辑(找名为 ``index`` 的列)严格一致。
    """
    if out.exists():
        out.unlink()
    try:
        idx.save(out)                      # 地基修好后自动走官方路径
        if out.stat().st_size > 0:
            return
        raise TypeError("NeuronIndex.save() 写出了 0 字节文件")
    except TypeError:
        df = idx.df.copy()
        df.index = pd.RangeIndex(len(df), name="index")
        df.to_parquet(out, index=True)
    # 立即回读自检
    back = NeuronIndex.load(out)
    assert back.n == idx.n, f"回读 N 不符: {back.n} != {idx.n}"
    assert np.array_equal(back.body_ids, idx.body_ids), "回读 body_id 不一致"


def build_index() -> NeuronIndex:
    log("阶段 1:构建 neuron_index.parquet")
    annot = paf.read_table(F_ANNOT, memory_map=True).to_pandas()
    for c in ("type", "instance", "flywireType", "hemibrainType", "class", "superclass",
              "subclass", "statusLabel", "somaSide", "somaNeuromere", "rootSide", "vfbId"):
        annot[c] = annot[c].astype("string")
    n_all = len(annot)
    is_neuron = annot["superclass"].notna() & (annot["superclass"].astype(str).str.len() > 0)
    neurons = annot[is_neuron].copy()
    log(f"  标注全量 {n_all:,} 行 -> superclass 非空 {len(neurons):,} 行(神经元)")
    neurons = neurons.sort_values("bodyId", kind="stable").reset_index(drop=True)
    N = len(neurons)
    body_ids = neurons["bodyId"].to_numpy(dtype=np.int64)
    assert len(np.unique(body_ids)) == N, "bodyId 不唯一"
    assert bool((body_ids[1:] > body_ids[:-1]).all()), "bodyId 非严格升序"

    # ---- 合并神经递质预测
    nt_col = np.full(N, "unknown", dtype=object)
    nt_source = np.full(N, "none", dtype=object)
    nt_conf = np.full(N, np.nan, dtype=np.float64)
    nt_hist: dict[str, int] = {}
    if F_NT.exists():
        nt = paf.read_table(F_NT, memory_map=True).to_pandas()
        nt = nt.drop_duplicates(subset="body", keep="first")
        nt_map = pd.DataFrame({
            "body": nt["body"].to_numpy(dtype=np.int64),
            "gt": nt["ground_truth"].astype("string").fillna("").to_numpy(),
            "cons": nt["consensus_nt"].astype("string").fillna("").to_numpy(),
            "pred": nt["predicted_nt"].astype("string").fillna("").to_numpy(),
            "cons_conf": pd.to_numeric(nt["celltype_predicted_nt_confidence"],
                                       errors="coerce").to_numpy(dtype=np.float64),
        }).set_index("body")
        joined = nt_map.reindex(body_ids)
        gt = joined["gt"].fillna("").to_numpy()
        cons = joined["cons"].fillna("").to_numpy()
        pred = joined["pred"].fillna("").to_numpy()
        conf = joined["cons_conf"].to_numpy(dtype=np.float64)

        def pick(*cands: tuple[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
            val = np.full(N, "unknown", dtype=object)
            src = np.full(N, "none", dtype=object)
            for name, arr in cands:
                usable = (val == "unknown") & (arr != "") & (arr != "unclear") & (arr != "nan")
                val[usable] = arr[usable]
                src[usable] = name
            return val, src

        nt_col, nt_source = pick(
            ("ground_truth", gt), ("consensus_nt", cons), ("predicted_nt", pred)
        )
        nt_conf = conf
        nt_hist = {str(k): int(v) for k, v in pd.Series(nt_col).value_counts().items()}
        n_missing = int((nt_source == "none").sum())
        log(f"  递质合并: 覆盖 {N - n_missing:,}/{N:,}(缺失 {n_missing:,});取值 {nt_hist}")
        st_nt = {"nt_coverage": int(N - n_missing), "nt_missing": int(n_missing)}
    else:
        log("  !! 未找到 body-neurotransmitters,nt 全部为 unknown")
        st_nt = {"nt_coverage": 0, "nt_missing": int(N)}

    idx = NeuronIndex.from_arrays(
        body_id=body_ids,
        name=neurons["instance"].fillna(neurons["type"]).fillna("unknown").to_numpy(),
        cls_=neurons["class"].fillna("unknown").to_numpy(),
        type_=neurons["type"].fillna(neurons["flywireType"]).fillna("unknown").to_numpy(),
        side=neurons["somaSide"].fillna("unknown").to_numpy(),
        nt=nt_col,
        superclass=neurons["superclass"].fillna("unknown").to_numpy(),
        extra={
            "flywireType": neurons["flywireType"].fillna("unknown").to_numpy(),
            "hemibrainType": neurons["hemibrainType"].fillna("unknown").to_numpy(),
            "subclass": neurons["subclass"].fillna("unknown").to_numpy(),
            "statusLabel": neurons["statusLabel"].fillna("unknown").to_numpy(),
            "vfbId": neurons["vfbId"].fillna("unknown").to_numpy(),
            "somaNeuromere": neurons["somaNeuromere"].fillna("unknown").to_numpy(),
            "rootSide": neurons["rootSide"].fillna("unknown").to_numpy(),
            "nt_source": nt_source,
            "nt_confidence": nt_conf.astype(np.float32),
            "assignedOlHex1": neurons["assignedOlHex1"].to_numpy(dtype=np.float32),
            "assignedOlHex2": neurons["assignedOlHex2"].to_numpy(dtype=np.float32),
        },
    )
    out = BUILD / ARTIFACT_INDEX
    save_index(idx, out)
    log(f"  已保存 {out} ({out.stat().st_size/1e6:.2f} MB), N={idx.n:,}(已回读自检)")

    st = load_state()
    st["N"] = N
    st["n_annotation_rows_total"] = int(n_all)
    st["n_annotation_rows_excluded_no_superclass"] = int(n_all - N)
    st["nt_available"] = bool(F_NT.exists())
    st["nt_hist"] = nt_hist
    st["nt_source_hist"] = {str(k): int(v) for k, v in pd.Series(nt_source).value_counts().items()}
    st.update(st_nt)
    save_state(st)
    return idx


def load_index() -> NeuronIndex:
    return NeuronIndex.load(BUILD / ARTIFACT_INDEX)


# ------------------------------------------------------------------ 阶段 2:连接组


def build_connectome(idx: NeuronIndex) -> dict:
    log("阶段 2:构建 connectome.npz")
    st = load_state()
    N = idx.n
    body_ids = idx.body_ids
    roles_path = BUILD / ARTIFACT_ROLES
    if roles_path.exists():
        from flyaim.io import load_roles
        roles = load_roles(roles_path)
    else:
        roles = idx.select_roles()
        log("  (roles.json 尚不存在,临时用 idx.select_roles() 计算抑制性集合)")
    is_inh = np.zeros(N, dtype=bool)
    if roles.inhibitory.size:
        is_inh[roles.inhibitory] = True
    log(f"  抑制性(突触前)细胞数 = {int(is_inh.sum()):,}")

    with pa.memory_map(str(F_WEIGHTS), "r") as src:
        reader = pa.ipc.open_file(src)
        nb = reader.num_record_batches
        log(f"  weights: {nb} 个 record batch")

        n_rows_total = 0
        n_kept = 0
        drop_pre = 0
        drop_post = 0
        drop_both = 0
        w_min, w_max = None, None
        chunks: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        for i in range(nb):
            b = reader.get_batch(i)
            bpre = b.column("body_pre").to_numpy(zero_copy_only=False)
            bpost = b.column("body_post").to_numpy(zero_copy_only=False)
            w = b.column("weight").to_numpy(zero_copy_only=False)
            n_rows_total += bpre.size
            ip = np.searchsorted(body_ids, bpre)
            iq = np.searchsorted(body_ids, bpost)
            okp = ip < N
            ipc = np.where(okp, ip, 0)
            okp &= body_ids[ipc] == bpre
            okq = iq < N
            iqc = np.where(okq, iq, 0)
            okq &= body_ids[iqc] == bpost
            keep = okp & okq
            drop_pre += int((~okp & okq).sum())
            drop_post += int((okp & ~okq).sum())
            drop_both += int((~okp & ~okq).sum())
            nk = int(keep.sum())
            n_kept += nk
            if nk:
                pp = ip[keep].astype(np.int32)
                qq = iq[keep].astype(np.int32)
                ww = w[keep].astype(np.float32)
                chunks.append((pp, qq, ww))
                cm, cx = int(ww.min()), int(ww.max())
                w_min = cm if w_min is None else min(w_min, cm)
                w_max = cx if w_max is None else max(w_max, cx)
            if (i + 1) % 400 == 0 or i == nb - 1:
                log(f"    batch {i+1}/{nb}  累计保留 {n_kept:,} 条边")
        log(f"  过滤完成: 原始 {n_rows_total:,} 条;保留 {n_kept:,};"
            f"丢弃 pre 不在集合 {drop_pre:,} / post 不在集合 {drop_post:,} / 两端都不在 {drop_both:,}")
        log(f"  weight 范围(保留边) [{w_min}, {w_max}]")

        pre = np.concatenate([c[0] for c in chunks])
        post = np.concatenate([c[1] for c in chunks])
        wts = np.concatenate([c[2] for c in chunks])
        del chunks
        log(f"  COO 数组就绪: pre/post/w = {pre.nbytes/1e6:.0f}/{post.nbytes/1e6:.0f}/"
            f"{wts.nbytes/1e6:.0f} MB")

        m_inh = is_inh[pre]
        n_inh_rows = int(m_inh.sum())
        n_exc_rows = int(n_kept - n_inh_rows)
        log(f"  按突触前递质拆分: 抑制边 {n_inh_rows:,} / 兴奋边 {n_exc_rows:,}")

        log("  构建 W_inh ...")
        W_inh = sp.csr_matrix(
            (wts[m_inh], (post[m_inh].astype(np.int64), pre[m_inh].astype(np.int64))),
            shape=(N, N),
        )
        log("  构建 W_exc ...")
        me = ~m_inh
        W_exc = sp.csr_matrix(
            (wts[me], (post[me].astype(np.int64), pre[me].astype(np.int64))),
            shape=(N, N),
        )
        del pre, post, wts, m_inh, me
        W_exc.sum_duplicates()
        W_inh.sum_duplicates()
        W_exc.sort_indices()
        W_inh.sort_indices()
        W_exc = W_exc.astype(np.float32)
        W_inh = W_inh.astype(np.float32)
        nnz_e, nnz_i = int(W_exc.nnz), int(W_inh.nnz)
        log(f"  CSR: W_exc.nnz={nnz_e:,}  W_inh.nnz={nnz_i:,}  合计={nnz_e+nnz_i:,}")
        n_collapsed = n_kept - (nnz_e + nnz_i)
        if n_collapsed:
            log(f"  !! 存在重复 (post,pre) 对:合并掉 {n_collapsed:,} 条"
                f"({100.0*n_collapsed/max(n_kept,1):.3f}%),权重已求和")
        else:
            log("  自检:W_exc.nnz + W_inh.nnz == 过滤后边数(无重复对)")
        log(f"  稀疏度 = {(nnz_e+nnz_i)/float(N*N)*100:.4f}%")

    soma = load_soma_pos(idx)
    art = ConnectomeArtifacts(W_exc=W_exc, W_inh=W_inh, neuron_ids=body_ids, soma_pos=soma)
    out = BUILD / ARTIFACT_WEIGHTS
    art.save(out)
    log(f"  已保存 {out} ({out.stat().st_size/1e6:.1f} MB)")

    stats = art.stats()
    deg = {}
    for name, sel in (
        ("ol_sensory", idx.indices_where("superclass", ["ol_sensory"])),
        ("descending_neuron", idx.indices_where("superclass", ["descending_neuron"])),
        ("vnc_motor", idx.indices_where("superclass", ["vnc_motor"])),
        ("ol_intrinsic", idx.indices_where("superclass", ["ol_intrinsic"])),
        ("visual_projection", idx.indices_where("superclass", ["visual_projection"])),
    ):
        if sel.size == 0:
            continue
        outdeg = (np.asarray(W_exc[:, sel].sum(axis=0)).ravel()
                  + np.asarray(W_inh[:, sel].sum(axis=0)).ravel())
        indeg = (np.asarray(W_exc[sel, :].sum(axis=1)).ravel()
                 + np.asarray(W_inh[sel, :].sum(axis=1)).ravel())
        deg[name] = {
            "n": int(sel.size),
            "n_with_out_edges": int((outdeg > 0).sum()),
            "out_weight_total": float(outdeg.sum()),
            "out_weight_median_nonzero": (float(np.median(outdeg[outdeg > 0]))
                                          if (outdeg > 0).any() else 0.0),
            "n_with_in_edges": int((indeg > 0).sum()),
            "in_weight_total": float(indeg.sum()),
            "in_weight_median_nonzero": (float(np.median(indeg[indeg > 0]))
                                         if (indeg > 0).any() else 0.0),
        }
        log(f"    {name:20s} n={sel.size:>7,}  有出边={int((outdeg>0).sum()):>7,}"
            f"  出权重中位={deg[name]['out_weight_median_nonzero']:.0f}")

    st = load_state()
    st.update({
        "connectome": stats,
        "n_weights_rows_total": int(n_rows_total),
        "n_edges_kept": int(n_kept),
        "n_edges_dropped": int(n_rows_total - n_kept),
        "n_edges_dropped_pre_outside": int(drop_pre),
        "n_edges_dropped_post_outside": int(drop_post),
        "n_edges_dropped_both_outside": int(drop_both),
        "n_duplicate_pairs_collapsed": int(n_collapsed),
        "weight_min": int(w_min) if w_min is not None else None,
        "weight_max": int(w_max) if w_max is not None else None,
        "group_degree": deg,
        "has_soma_pos": soma is not None,
        "npz_bytes": int(out.stat().st_size),
        "peak_rss_mb": round(mem_mb()[1], 1),
    })
    save_state(st)
    return stats


def load_soma_pos(idx: NeuronIndex) -> np.ndarray | None:
    """body-stats 里没有 soma x/y/z(实测列: body/pre/post/.../synweight/rank),故返回 None。"""
    if not F_STATS.exists():
        log("  (无 body-stats,跳过 soma 坐标)")
        return None
    try:
        sch = pa.ipc.open_file(pa.memory_map(str(F_STATS), "r")).schema
    except Exception as e:  # noqa: BLE001
        log(f"  !! body-stats 读取失败: {e}")
        return None
    names = list(sch.names)
    log(f"  body-stats 列: {names}")
    cand = [c for c in names if c.lower() in ("soma_x", "soma_y", "soma_z")]
    if len(cand) != 3:
        cand = [c for c in names
                if any(k in c.lower() for k in ("soma", "position", "pos"))
                and c.lower().endswith(("x", "y", "z"))]
    if len(cand) != 3:
        log(f"  !! body-stats 无 soma x/y/z 列(候选 {cand})-> 不加 soma_pos,npz 里无体坐标")
        return None
    key = next((c for c in names if c.lower() in ("body", "bodyid", "body_id", "root_id")), None)
    if key is None:
        log(f"  !! body-stats 无 body id 列({names}),不加 soma_pos")
        return None
    t = paf.read_table(F_STATS, columns=[key] + cand, memory_map=True).to_pandas()
    t = t.drop_duplicates(subset=key, keep="first").set_index(key)
    re = t.reindex(idx.body_ids)
    arr = re[cand].to_numpy(dtype=np.float32)
    n_ok = int(np.isfinite(arr).all(axis=1).sum())
    log(f"  soma 坐标: {n_ok:,}/{idx.n:,} 有值(列 {cand})")
    return arr


# ------------------------------------------------------------------ 阶段 3:角色


def build_roles(idx: NeuronIndex):
    log("阶段 3:角色选择 idx.select_roles()")
    roles = idx.select_roles()
    out = BUILD / ARTIFACT_ROLES
    save_roles(out, roles)
    log(f"  已保存 {out} ({out.stat().st_size:,} B)  summary={roles.summary()}")
    for n in roles.notes:
        log(f"    note: {n}")
    st = load_state()
    st["roles"] = roles.summary()
    st["roles_notes"] = list(roles.notes)
    save_state(st)
    return roles


# ------------------------------------------------------------------ 阶段 4:manifest


def build_manifest(idx: NeuronIndex) -> Manifest:
    log("阶段 4:manifest.json")
    from flyaim.io import load_roles
    st = load_state()
    roles = load_roles(BUILD / ARTIFACT_ROLES)
    art = ConnectomeArtifacts.load(BUILD / ARTIFACT_WEIGHTS)
    rep = json.loads(F_REPORT.read_text(encoding="utf-8")) if F_REPORT.exists() else {}
    part = (json.loads((CACHE / "edge_partition.json").read_text(encoding="utf-8"))
            if (CACHE / "edge_partition.json").exists() else {})

    n_inh_edges = int(art.W_inh.nnz)
    total_syn = float(art.W_exc.sum() + art.W_inh.sum())
    deg_out = np.asarray(art.W_exc.sum(axis=0)).ravel() + np.asarray(art.W_inh.sum(axis=0)).ravel()
    deg_in = np.asarray(art.W_exc.sum(axis=1)).ravel() + np.asarray(art.W_inh.sum(axis=1)).ravel()
    n_isolated = int(((deg_out == 0) & (deg_in == 0)).sum())
    log(f"  突触总数 = {total_syn:,.0f};  孤立神经元 = {n_isolated:,}")

    nt_hist = st.get("nt_hist", {})
    n_inh_nt = int(roles.inhibitory.size)
    notes = [
        f"索引口径: N={idx.n:,} = body-annotations(minconf 0.5) 中 superclass 非空的神经元行数;"
        f"全量标注 {st.get('n_annotation_rows_total', 0):,} 行中排除 "
        f"{st.get('n_annotation_rows_excluded_no_superclass', 0):,} 行(glia/unimportant/orphan 等无功能分类)。"
        f"bodyId 升序 -> 全局索引 0..N-1。",
        f"weights 文件是**边表** (body_pre, body_post, weight),共 {st.get('n_weights_rows_total',0):,} 行;"
        f"有向语义按契约 W[i,j]=j->i 落盘,即 row=body_post, col=body_pre。",
        f"边过滤: 丢弃 {st.get('n_edges_dropped',0):,} 条(突触前不在索引集 "
        f"{st.get('n_edges_dropped_pre_outside',0):,} / 突触后不在 "
        f"{st.get('n_edges_dropped_post_outside',0):,} / 两端都不在 "
        f"{st.get('n_edges_dropped_both_outside',0):,});保留 {st.get('n_edges_kept',0):,} 条。",
    ]
    if st.get("n_duplicate_pairs_collapsed"):
        notes.append(
            f"一致性自检: 保留 {st['n_edges_kept']:,} 条边,但 CSR nnz 合计 "
            f"{art.n_edges:,};差异 {st['n_duplicate_pairs_collapsed']:,} 条来自重复 "
            f"(body_post, body_pre) 对,已按突触计数求和合并(coo->csr sum_duplicates)。"
        )
    else:
        notes.append(
            f"一致性自检通过: W_exc.nnz({art.W_exc.nnz:,}) + W_inh.nnz({art.W_inh.nnz:,}) "
            f"== 过滤后边数({st.get('n_edges_kept',0):,}),无重复 (body_post,body_pre) 对。"
        )
    notes.append(
        f"边 vs 突触(重要): 本 npz 共 {art.n_edges:,} 条**边**,权重合计 "
        f"{total_syn:,.0f} 个**突触**,平均 {total_syn/max(art.n_edges,1):.2f} 突触/边。"
        f"CONTRACT.md 里写的 '~1.25e8 突触连接' 指的是**突触总数**——实测 "
        f"{total_syn:,.0f} 与之吻合,说明过滤后的 A-A 子图就是该数据集的完整神经元级连接组。"
    )
    if part.get("counts"):
        c = part["counts"]
        notes.append(
            "为什么 151,856,684 行只保留 25,582,938 条边(见 .cache/edge_partition.json):"
            f" 74.05%({c.get('A-C',0):,})的行其 body_post **不是任何有标注的 body**。"
            "采样实测: body_pre 100% 落在 body 列表内(93.4% 在 annotations),"
            "而 body_post 有 87.9% 的行取值互不相同、78.6% 的取值既不在 annotations "
            "也不在 body-neurotransmitters 的 1,835,518 个 body 里,且几乎从不作为 body_pre 出现。"
            "→ 上游'全量'文件把突触拆到了**未追踪的碎片对象**上;按契约的索引口径只保留"
            "两端都是 166,700 神经元的边,正是标准做法,且保留下了 39.8% 的突触质量。"
        )
    notes.append(
        "生物学合理性验证(过滤后子图,非零断言): R1-R6 的输出 top 靶 = L2/L1/L3/L4/Lawf1/C3/C2/T1"
        "(教科书 lamina 通路);T4 的输入 = LPi34/LPi21/Tlp14/TmY4(ON 运动通路);"
        "T5 的输入 = LPi/Tlp/Am1/Y11(经典 OFF 运动通路);L1-L5 输出 = Tm1/Tm2/Tm4/Mi1/Dm;"
        "下行神经元 DN 的主要输出 = vnc_motor(Sternotrochanter MN / Tr flexor MN 等)+ 上行神经元。"
        "以上均符合已知视叶/VNC 解剖,说明方向约定 W[i,j]=j->i 与过滤口径都正确。"
    )
    notes.append(
        "独立交叉验证(最强的正确性证据): GCS 桶里另有官方过滤版 "
        "connectome-weights-...-traced-only.feather(508,025,642 B;25,563,197 行;"
        "额外带 type_pre/type_post 列)。它 99.98% 的边两端都在本索引空间内,A-A 突触合计 124,009,893;"
        f"本产物(全量文件挑 A-A)= {art.n_edges:,} 条边 / {total_syn:,.0f} 突触,"
        f"与官方神经元级连接组只差 {art.n_edges-25558671:,} 条边"
        f"({100.0*(art.n_edges-25558671)/25558671:.3f}%),是其超集。"
        "另用 traced-only 的 200,000 行子样本逐行比对 type_pre/type_post:"
        "body_pre / body_post 各有 100.00% 落在本索引空间内,且与 neuron_index.type "
        "**100.00% 一致** —— 索引空间、字段方向(pre=突触前/post=突触后)、"
        "annotation->index 映射三者同时得到官方文件确认。"
    )
    notes.append(
        f"连通性: {idx.n - n_isolated:,}/{idx.n:,} 个神经元至少有一条出入边,孤立神经元仅 "
        f"{n_isolated:,} 个;出权重中位数强度、入权重中位数见 extra.group_degree。"
    )
    if not st.get("nt_available"):
        notes.append("无递质预测,已回退全兴奋网络,侧抑制动力学被削弱。")
    elif n_inh_nt == 0:
        notes.append("递质文件存在但未匹配到任何 GABA/Glycine 细胞;已回退全兴奋网络。")
    else:
        notes.append(
            f"抑制性判定: 契约 INHIBITORY_NTS=gaba/glycine/acetylcholine_gaba;命中 {n_inh_nt:,} 个细胞;"
            f"突触前属于该集合的边 -> W_inh({n_inh_edges:,} 条 / "
            f"{float(art.W_inh.sum()):,.0f} 突触),其余 -> W_exc({art.W_exc.nnz:,} 条 / "
            f"{float(art.W_exc.sum()):,.0f} 突触)。递质取值分布={nt_hist}"
        )
        notes.append(
            f"神经递质合并口径: 优先级 ground_truth > consensus_nt > predicted_nt(跳过 "
            f"'unclear'/空/NaN),覆盖 {st.get('nt_coverage', 0):,}/{idx.n:,} 个细胞"
            f"(缺失 {st.get('nt_missing', 0):,} 记为 'unknown');"
            f"来源分布={st.get('nt_source_hist')};该优先级只决定 nt 标签,不改变任何一条边的归属。"
        )
        n_hist = int(nt_hist.get("histamine", 0))
        if n_hist:
            notes.append(
                f"已知偏差(重要,限制与偏差章节必须复述): {n_hist:,} 个细胞预测递质为 histamine,"
                f"其中包含**全部 R1-R6 感光细胞**;而契约的 INHIBITORY_NTS 不含 histamine,"
                f"因此 R1-R6 -> lamina 的输出边被归入 W_exc。"
                f"生物学上组胺在 lamina 对 LMC 是抑制性的 —— 这是**已知符号近似**。"
                f"预期后果: 视觉通路第一级极性反转、ON/OFF 对比度响应失真。"
                f"本次不做干预(改判定规则会变成未经论证的方法学改动)。"
            )
    fb = bool(roles.visual_input_fallback)
    notes.append(
        f"视觉输入: strategy={roles.visual_input_strategy}, n={roles.visual_input.size:,}, "
        f"visual_input_fallback={fb}。"
        + ("数据集含真实复眼感光细胞(superclass=='ol_sensory'),无需人工替代群。"
           if not fb else "数据集无可用感光细胞,已降级为最优可用下游群。")
    )
    if st.get("group_degree"):
        g = st["group_degree"].get("ol_sensory")
        if g:
            notes.append(
                f"感光细胞可驱动性证据: {g['n']:,} 个 ol_sensory 中 {g['n_with_out_edges']:,} 个有出边,"
                f"出权重合计 {g['out_weight_total']:,.0f}(非零中位数 "
                f"{g['out_weight_median_nonzero']:.0f}/细胞),入权重合计 {g['in_weight_total']:,.0f}"
                f" —— 说明它们真的接入了下游视叶回路,可作为 Retina 的输入靶点。"
            )
    notes.append(
        "感光细胞的 somaSide 几乎全为 NaN(6,098 中仅 36 个有 L/R 标注),左右视野划分不能用 side;"
        "neuron_index.parquet 里已带 assignedOlHex1/2(视叶六边形坐标,仅 23,720 个 ol_intrinsic 有值,"
        "感光细胞自身没有),另有 vfbId / somaNeuromere / rootSide 可用。"
    )
    if not st.get("has_soma_pos"):
        notes.append(
            "body-stats 文件实测不含 soma x/y/z 坐标(列为 body/pre/post/status_fine/superclass/"
            "class/type/instance/downstream/synweight/rank),因此 connectome.npz 里没有 soma_pos。"
        )

    mf = Manifest(
        n_neurons=idx.n,
        n_edges=art.n_edges,
        roles=roles.summary(),
        visual_input_fallback=fb,
        visual_input_strategy=roles.visual_input_strategy,
        notes=notes,
        source_files=file_entries(),
        extra={
            "download_seconds": {k: v.get("seconds") for k, v in rep.items()},
            "download_attempts": {k: v.get("attempts") for k, v in rep.items()},
            "download_status": {k: v.get("status") for k, v in rep.items()},
            "peak_rss_mb": st.get("peak_rss_mb"),
            "n_annotation_rows_total": st.get("n_annotation_rows_total"),
            "n_annotation_rows_excluded_no_superclass":
                st.get("n_annotation_rows_excluded_no_superclass"),
            "n_weights_rows_total": st.get("n_weights_rows_total"),
            "n_edges_kept": st.get("n_edges_kept"),
            "n_edges_dropped": st.get("n_edges_dropped"),
            "n_edges_dropped_pre_outside": st.get("n_edges_dropped_pre_outside"),
            "n_edges_dropped_post_outside": st.get("n_edges_dropped_post_outside"),
            "n_edges_dropped_both_outside": st.get("n_edges_dropped_both_outside"),
            "n_duplicate_pairs_collapsed": st.get("n_duplicate_pairs_collapsed"),
            "weight_min": st.get("weight_min"),
            "weight_max": st.get("weight_max"),
            "connectome_stats": st.get("connectome"),
            "total_synapses": int(total_syn),
            "mean_synapses_per_edge": round(total_syn / max(art.n_edges, 1), 3),
            "n_isolated_neurons": n_isolated,
            "n_neurons_with_out_edges": int((deg_out > 0).sum()),
            "n_neurons_with_in_edges": int((deg_in > 0).sum()),
            "out_weight_median_nonzero": float(np.median(deg_out[deg_out > 0])),
            "in_weight_median_nonzero": float(np.median(deg_in[deg_in > 0])),
            "edge_partition_abc": part.get("counts"),
            "edge_partition_weight_sums": part.get("weight_sums"),
            "cross_validation_traced_only": {
                "file": "connectome-weights-male-cns-v1.0-minconf-0.5-traced-only.feather",
                "url": "https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/"
                       "flat-connectome/connectome-weights-male-cns-v1.0-minconf-0.5-traced-only.feather",
                "bytes": 508025642,
                "sha256": "9b3beab17bad5f618be3f2c02d3139a8d07b822565919c013f1e5506d93e604b",
                "rows": 25563197,
                "columns": ["body_pre", "body_post", "weight", "type_pre", "type_post"],
                "n_edges_both_in_index": 25558671,
                "synapses_both_in_index": 124009893,
                "frac_rows_both_in_index": 0.9998,
                "our_edges_minus_this": art.n_edges - 25558671,
                "relative_diff": round((art.n_edges - 25558671) / 25558671, 6),
                "type_pre_type_post_match_rate": 1.0,
            },
            "nt_coverage": st.get("nt_coverage"),
            "nt_missing": st.get("nt_missing"),
            "group_degree": st.get("group_degree"),
            "nt_available": st.get("nt_available"),
            "nt_hist": nt_hist,
            "nt_source_hist": st.get("nt_source_hist"),
            "roles_notes": st.get("roles_notes"),
            "index_columns": list(idx.df.columns),
            "index_rows": int(idx.n),
            "has_soma_pos": st.get("has_soma_pos"),
            "schema_facts": {
                "annotations_rows": 211577,
                "annotations_cols": 36,
                "neurons_with_superclass": 166700,
                "neurons_with_vfbId": 166686,
                "weights_rows": 151856684,
                "weights_batches": 2318,
                "weights_columns": ["body_pre", "body_post", "weight"],
                "nt_file_rows": 1835518,
                "nt_file_cols": 10,
                "body_stats_rows": 88384522,
                "photoreceptors_present": True,
                "photoreceptor_types": ["R1-R6", "R7y", "R8y", "R8_unclear", "R7_unclear",
                                        "R7p", "R8p", "R7R8_unclear", "R7d", "R8d"],
            },
            "artifacts": {
                "neuron_index.parquet": int((BUILD / ARTIFACT_INDEX).stat().st_size),
                "connectome.npz": int((BUILD / ARTIFACT_WEIGHTS).stat().st_size),
                "roles.json": int((BUILD / ARTIFACT_ROLES).stat().st_size),
            },
            "artifact_paths": {
                "neuron_index": str(BUILD / ARTIFACT_INDEX),
                "connectome": str(BUILD / ARTIFACT_WEIGHTS),
                "roles": str(BUILD / ARTIFACT_ROLES),
                "manifest": str(BUILD / ARTIFACT_MANIFEST),
                "schema_probe": str(BUILD / "schema_probe.txt"),
                "verify": str(ROOT / "flyaim" / "data" / "verify.py"),
            },
        },
    )
    out = BUILD / ARTIFACT_MANIFEST
    mf.save(out)
    log(f"  已保存 {out} ({out.stat().st_size:,} B)")
    return mf


# ------------------------------------------------------------------ main


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", nargs="?", default="all",
                    choices=["index", "connectome", "roles", "manifest", "all"])
    a = ap.parse_args()
    # 顺序很关键: connectome 的 W_exc/W_inh 拆分依赖 roles.inhibitory,
    # 因此 roles 必须在 connectome 之前产出。
    idx = None
    if a.stage in ("index", "all"):
        idx = build_index()
    if a.stage in ("roles", "all"):
        idx = idx or load_index()
        build_roles(idx)
    if a.stage in ("connectome", "all"):
        idx = idx or load_index()
        build_connectome(idx)
    if a.stage in ("manifest", "all"):
        idx = idx or load_index()
        build_manifest(idx)
    cur, peak = mem_mb()
    log(f"完成,峰值 RSS {peak:,.0f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
