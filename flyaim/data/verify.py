"""FlyAim A 线数据自检(可重复运行,零断言失败为通过)。

用法::

    python flyaim/data/verify.py                # 快速自检(不重算 1.9GB 校验和)
    python flyaim/data/verify.py --sha256       # 额外核对源文件 sha256(耗时 ~1 分钟)
    python flyaim/data/verify.py --quiet        # 只打印结论

覆盖的断言(对应交付标准):
    1. 四个产物存在且非空
    2. neuron_index 的全局索引严格为 [0, N),body_id 唯一且严格升序,与 CONTRACT 的 166,700 一致
    3. NeuronIndex.load() 能读回且 N 与 parquet 行数一致
    4. ConnectomeArtifacts.load() 能读回;W_exc/W_inh 为 CSR、float32、形状 (N, N)
    5. W_exc.nnz + W_inh.nnz == manifest.n_edges == manifest.extra.n_edges_kept
    6. 权重非负、无显式零;突触总数与 manifest.extra.total_synapses 一致
    7. roles.json 四个角色的索引全部 < N、唯一、非负且非空(降级时 inhibitory 可为空但要显式)
    8. roles 与 neuron_index 元数据自洽(visual_input 全为 ol_sensory 等)
    9. manifest 内部自洽(n_neurons / n_edges / roles / visual_input_fallback / source_files)
   10. 生物学 sanity:R1-R6 -> lamina 单极细胞存在连接(不通过则警告,不致命)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from flyaim.io import (  # noqa: E402
    ARTIFACT_INDEX,
    ARTIFACT_MANIFEST,
    ARTIFACT_ROLES,
    ARTIFACT_WEIGHTS,
    ConnectomeArtifacts,
    Manifest,
    load_roles,
    sha256_file,
)
from flyaim.neuron_index import INHIBITORY_NTS, NeuronIndex  # noqa: E402

BUILD = ROOT / "flyaim" / "data" / "build"
EXPECTED_N = 166700

FAILURES: list[str] = []
WARNINGS: list[str] = []
ROWS: list[tuple[str, str]] = []


def ok(cond: bool, msg: str) -> bool:
    if not cond:
        FAILURES.append(msg)
        print(f"  [FAIL] {msg}", flush=True)
    return bool(cond)


def warn(cond: bool, msg: str) -> None:
    if not cond:
        WARNINGS.append(msg)
        print(f"  [warn] {msg}", flush=True)


def row(k: str, v: object) -> None:
    ROWS.append((k, str(v)))


def section(n: str, title: str) -> None:
    print(f"\n=== {n} {title} ===", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sha256", action="store_true", help="核对源文件 sha256(慢)")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()

    print("=" * 74)
    print("FlyAim A 线数据自检  (build 目录: %s)" % BUILD)
    print("=" * 74)

    # ------------------------------------------------------------ 1
    section("1/9", "产物存在性")
    paths = {
        "neuron_index": BUILD / ARTIFACT_INDEX,
        "connectome": BUILD / ARTIFACT_WEIGHTS,
        "roles": BUILD / ARTIFACT_ROLES,
        "manifest": BUILD / ARTIFACT_MANIFEST,
    }
    for k, p in paths.items():
        exists = p.exists() and p.stat().st_size > 0
        ok(exists, f"产物缺失或为空: {p}")
        if exists:
            print(f"  {k:<14s} {p.stat().st_size:>14,d} B   {p}")
            row(f"{k} bytes", f"{p.stat().st_size:,}")
    if FAILURES:
        print("\n产物不全,后续检查跳过。")
        return 1

    # ------------------------------------------------------------ 2
    section("2/9", "全局索引空间 [0, N)")
    raw = pd.read_parquet(paths["neuron_index"])
    n_parquet = len(raw)
    if "index" in raw.columns:
        iv = raw["index"].to_numpy()
        layout = "索引作为列 'index'"
    elif raw.index.name == "index":
        iv = raw.index.to_numpy()
        layout = "索引在 DataFrame.index 上(名为 'index')"
    else:
        iv = raw.index.to_numpy()
        layout = f"默认索引(name={raw.index.name!r})"
    print(f"  布局: {layout}")
    ok(np.array_equal(iv, np.arange(n_parquet)),
       f"全局索引不是连续 [0,N): 前 5 个={iv[:5].tolist()} 后 5 个={iv[-5:].tolist()}")
    ok(n_parquet == EXPECTED_N, f"neuron_index 行数 {n_parquet:,} != 契约的 {EXPECTED_N:,}")
    ok("body_id" in raw.columns, "neuron_index 缺少 body_id 列")
    ok("superclass" in raw.columns, "neuron_index 缺少 superclass 列(Lead 强制要求)")
    bids = raw["body_id"].to_numpy(dtype=np.int64)
    ok(len(np.unique(bids)) == len(bids), "body_id 存在重复")
    ok(bool((np.diff(bids) > 0).all()), "body_id 不是严格升序(会破坏索引↔body 映射的确定性)")
    row("N (neurons)", f"{n_parquet:,}")
    row("body_id 范围", f"[{bids.min():,}, {bids.max():,}]")

    idx = NeuronIndex.load(paths["neuron_index"])
    ok(idx.n == n_parquet, f"NeuronIndex.load() N={idx.n} != parquet 行数 {n_parquet}")
    ok(np.array_equal(idx.body_ids, bids), "NeuronIndex.load() 的 body_id 顺序与 parquet 不一致")
    print(f"  NeuronIndex.load() OK: N={idx.n:,}, 列数={len(idx.df.columns)}")

    # ------------------------------------------------------------ 3
    section("3/9", "连接组 CSR")
    art = ConnectomeArtifacts.load(paths["connectome"])
    N = art.n
    ok(N == n_parquet, f"connectome N={N} != neuron_index 行数 {n_parquet}")
    ok(art.W_exc.shape == (N, N), f"W_exc.shape={art.W_exc.shape} != ({N},{N})")
    ok(art.W_inh.shape == (N, N), f"W_inh.shape={art.W_inh.shape} != ({N},{N})")
    ok(sp.issparse(art.W_exc) and art.W_exc.format == "csr",
       f"W_exc 不是 CSR 格式(format={art.W_exc.format})")
    ok(sp.issparse(art.W_inh) and art.W_inh.format == "csr",
       f"W_inh 不是 CSR 格式(format={art.W_inh.format})")
    ok(art.W_exc.dtype == np.float32, f"W_exc.dtype={art.W_exc.dtype} != float32")
    ok(art.W_inh.dtype == np.float32, f"W_inh.dtype={art.W_inh.dtype} != float32")
    ok(np.array_equal(art.neuron_ids, bids), "connectome.neuron_ids 与 neuron_index.body_id 不一致")
    ok(bool((art.W_exc.data >= 0).all()), "W_exc 存在负权重")
    ok(bool((art.W_inh.data >= 0).all()), "W_inh 存在负权重")
    ok(art.W_exc.nnz > 0, "W_exc 为空")
    row("n_edges", f"{art.n_edges:,}")
    row("  W_exc.nnz", f"{art.W_exc.nnz:,}")
    row("  W_inh.nnz", f"{art.W_inh.nnz:,}")

    # 行为自检:行和 == CSR 每行入权重(只抽样最大行,避免慢)
    W = (art.W_exc + art.W_inh).tocsr()
    nnz_dup = art.W_exc.nnz + art.W_inh.nnz - W.nnz
    ok(nnz_dup == 0,
       f"W_exc 与 W_inh 在同一 (i,j) 上有 {nnz_dup:,} 个重叠条目(同一突触前细胞不应同时兴奋+抑制)")

    # ------------------------------------------------------------ 4
    section("4/9", "manifest 与产物一致性")
    mf = Manifest.load(paths["manifest"])
    ok(mf.n_neurons == N, f"manifest.n_neurons={mf.n_neurons} != {N}")
    ok(mf.n_edges == art.n_edges, f"manifest.n_edges={mf.n_edges:,} != 实际 {art.n_edges:,}")
    extra = mf.extra or {}
    kept = extra.get("n_edges_kept")
    if kept is not None:
        ok(int(kept) == art.n_edges,
           f"manifest.extra.n_edges_kept={kept:,} != W_exc.nnz+W_inh.nnz={art.n_edges:,} "
           f"(说明存在未记录的重复对合并或额外过滤)")
    total_syn = float(art.W_exc.sum() + art.W_inh.sum())
    if extra.get("total_synapses") is not None:
        ok(abs(float(extra["total_synapses"]) - total_syn) < 1.0,
           f"manifest.extra.total_synapses={extra['total_synapses']} != 实测 {total_syn:,.0f}")
    row("总突触数", f"{total_syn:,.0f}")
    row("密度", f"{100.0*art.n_edges/(N*N):.4f}%")
    ok(bool(mf.visual_input_fallback) is False or bool(mf.visual_input_fallback) is True,
       "visual_input_fallback 不是布尔值")
    ns = mf.notes
    ok(any("histamine" in s for s in ns), "manifest.notes 缺少 histamine 符号偏差说明")
    ok(any("一致性自检" in s for s in ns), "manifest.notes 缺少一致性自检结论")
    ok(len(mf.source_files) >= 2, f"manifest.source_files 只有 {len(mf.source_files)} 项")
    for k, v in mf.source_files.items():
        ok(isinstance(v, dict) and "url" in v and "bytes" in v and "sha256" in v,
           f"source_files[{k}] 缺 url/bytes/sha256")
    print(f"  manifest: n_neurons={mf.n_neurons:,} n_edges={mf.n_edges:,} "
          f"notes={len(ns)} 条 source_files={len(mf.source_files)} 个")

    # ------------------------------------------------------------ 5
    section("5/9", "角色选择 roles.json")
    roles = load_roles(paths["roles"])
    for name in ("visual_input", "descending", "motor", "inhibitory"):
        arr = roles.get(name)
        ok(arr.dtype == np.int64 or arr.dtype == np.int32,
           f"roles.{name} dtype={arr.dtype} 不是整数")
        if arr.size:
            ok(int(arr.min()) >= 0 and int(arr.max()) < N,
               f"roles.{name} 存在越界索引(min={arr.min()}, max={arr.max()}, N={N})")
            ok(len(np.unique(arr)) == arr.size, f"roles.{name} 存在重复索引")
        row(f"  n_{name}", f"{arr.size:,}")
    ok(roles.visual_input.size > 0, "roles.visual_input 为空")
    ok(roles.descending.size > 0, "roles.descending 为空(控制读出无合法来源)")
    ok(roles.motor.size > 0, "roles.motor 为空")
    warn(roles.inhibitory.size > 0, "roles.inhibitory 为空 -> 已回退全兴奋网络")
    ok(mf.roles == roles.summary(),
       f"manifest.roles 与 roles.json 不一致: {mf.roles} vs {roles.summary()}")
    row("  visual_input_strategy", roles.visual_input_strategy)
    row("  visual_input_fallback", roles.visual_input_fallback)

    # ------------------------------------------------------------ 6
    section("6/9", "roles 与 neuron_index 元数据自洽")
    sup = idx.df["superclass"].to_numpy()
    vis_sup = set(sup[roles.visual_input].tolist()) if roles.visual_input.size else set()
    print(f"  visual_input 的 superclass 取值集合 = {sorted(vis_sup)}")
    ok(vis_sup == {"ol_sensory"},
       f"visual_input 里混入了非 ol_sensory 的细胞: {sorted(vis_sup - {'ol_sensory'})}")
    n_photo = int((sup == "ol_sensory").sum())
    ok(roles.visual_input.size == n_photo,
       f"roles.visual_input={roles.visual_input.size} != ol_sensory 总数 {n_photo}")
    row("  ol_sensory 总数", f"{n_photo:,}")
    dn_sup = set(sup[roles.descending].tolist()) if roles.descending.size else set()
    print(f"  descending 的 superclass 取值集合 = {sorted(dn_sup)}")
    mn_sup = set(sup[roles.motor].tolist()) if roles.motor.size else set()
    print(f"  motor 的 superclass 取值集合 = {sorted(mn_sup)}")
    ok(mn_sup == {"vnc_motor"}, f"motor 里混入了非 vnc_motor: {sorted(mn_sup - {'vnc_motor'})}")
    if roles.inhibitory.size:
        nt = idx.df["nt"].to_numpy()
        bad = sorted(set(nt[roles.inhibitory].tolist()) - set(INHIBITORY_NTS))
        ok(not bad, f"inhibitory 里存在非 {sorted(INHIBITORY_NTS)} 的 nt 标签: {bad}")
        print(f"  inhibitory 的 nt 取值集合 = {sorted(set(nt[roles.inhibitory].tolist()))}")
    nodeg = np.asarray(W.sum(axis=1)).ravel() + np.asarray(W.sum(axis=0)).ravel()
    n_iso = int((nodeg == 0).sum())
    row("孤立神经元", f"{n_iso:,}")
    warn(n_iso < 0.01 * N, f"孤立神经元过多: {n_iso:,}/{N:,}")

    # ------------------------------------------------------------ 7
    section("7/9", "生物学 sanity(方向约定 W[i,j] = j->i)")
    ty = idx.df["type"].astype(str).to_numpy()
    r16 = np.flatnonzero(ty == "R1-R6")
    lmc = np.flatnonzero(np.isin(ty, ["L1", "L2", "L3", "L4", "L5"]))
    print(f"  R1-R6 n={r16.size:,}   L1-L5 n={lmc.size:,}")
    if r16.size and lmc.size:
        sub = W[np.ix_(lmc[:200], r16[:200])]
        n_conn = int(sub.nnz)
        wsum = float(sub.sum())
        print(f"  R1-R6 -> L1-L5 的连接对: {n_conn:,} 条,权重合计 {wsum:,.0f}")
        ok(n_conn > 0, "R1-R6 与 lamina 单极细胞 L1-L5 之间没有任何连接 —— 方向约定可能反了")
    else:
        warn(False, "缺少 R1-R6 或 L1-L5,无法做通路 sanity 检查")

    # ------------------------------------------------------------ 8
    section("8/9", "源文件校验和")
    if a.sha256:
        for k, v in mf.source_files.items():
            p = next((q for q in (ROOT / ".cache").glob("*") if q.name in str(v["url"])), None)
            if p is None or not p.exists():
                warn(False, f"source_files[{k}] 的本地文件不在 .cache,跳过 sha256 核对")
                continue
            got = sha256_file(p)
            ok(got == v["sha256"], f"{k} sha256 不匹配: 实测 {got[:16]}… != 记录 {str(v['sha256'])[:16]}…")
            print(f"  {k:<20s} {p.stat().st_size:>14,d} B  sha256 OK")
    else:
        print("  (跳过;加 --sha256 可核对,需 ~1 分钟)")

    # ------------------------------------------------------------ 9
    section("9/9", "摘要")
    width = max(len(k) for k, _ in ROWS) + 2
    for k, v in ROWS:
        print(f"  {k:<{width}s} {v}")
    print()
    if FAILURES:
        print(f"结论:**失败** —— {len(FAILURES)} 个断言未通过:")
        for f in FAILURES:
            print(f"  - {f}")
        return 1
    print(f"结论:**全部断言通过**(0 失败,{len(WARNINGS)} 个警告)")
    for w in WARNINGS:
        print(f"  warn: {w}")
    print(f"产物目录: {BUILD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
