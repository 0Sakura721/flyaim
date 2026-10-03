"""A 线内部工具:weights / neurotransmitters schema 探查(不整表载入内存)。"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.feather as feather

CACHE = Path(r"D:\dsh\autoaim\.cache")
W = CACHE / "connectome-weights-male-cns-v1.0-minconf-0.5.feather"
NT = CACHE / "body-neurotransmitters-male-cns-v1.0.feather"


def probe_weights() -> None:
    print("=" * 100)
    print(f"WEIGHTS: {W.name}  {W.stat().st_size/1e6:.2f} MB")
    print("=" * 100)
    md = feather.read_table(W, memory_map=True, columns=None).schema.metadata or {}
    fp = md.get(b"ARROW:extension:name")
    # 只读 schema:用 ipc open + 不读 body
    with pa.memory_map(W, "r") as src:
        reader = pa.ipc.open_file(src)
        print(f"num_record_batches = {reader.num_record_batches}")
        sch = reader.schema
        print(f"--- schema ---\n{sch}")
        nrows = 0
        for i in range(reader.num_record_batches):
            nrows += reader.get_batch(i).num_rows
        print(f"总行数 = {nrows:,}")
        b0 = reader.get_batch(0)
        print(f"batch0 行数 = {b0.num_rows}")
        print("--- head(5) ---")
        print(b0.slice(0, 5).to_pandas().to_string())
        blast = reader.get_batch(reader.num_record_batches - 1)
        print("--- tail(5) ---")
        print(blast.slice(max(0, blast.num_rows - 5), 5).to_pandas().to_string())

        # 列统计:遍历所有 batch 累加,只取数值列 min/max/sum
        print("\n--- 逐列统计(全表扫描)---")
        for j, name in enumerate(sch.names):
            col = sch.field(j)
            mn = None
            mx = None
            tot = 0
            nn = 0
            for i in range(reader.num_record_batches):
                arr = reader.get_batch(i).column(j)
                try:
                    a = arr.to_numpy(zero_copy_only=False)
                except Exception:  # noqa: BLE001
                    a = np.asarray(arr.to_pylist())
                if a.dtype.kind in "iuf":
                    mn = a.min() if mn is None else min(mn, a.min())
                    mx = a.max() if mx is None else max(mx, a.max())
                    tot += a.sum()
                    nn += int(np.count_nonzero(a))
            print(f"  {name!r:20s} type={col.type}  min={mn}  max={mx}  sum={tot}  非零数={nn}")


def probe_nt() -> None:
    print("\n" + "=" * 100)
    print(f"NEUROTRANSMITTERS: {NT.name}  {NT.stat().st_size/1e6:.2f} MB")
    print("=" * 100)
    t = feather.read_table(NT, memory_map=True)
    print(t.schema)
    df = t.to_pandas()
    print(f"shape = {df.shape}")
    print("--- head(8) ---")
    with pd.option_context("display.width", 250, "display.max_columns", 60):
        print(df.head(8).to_string())
    print("\n--- 逐列 ---")
    for c in df.columns:
        s = df[c]
        print(f"  {c!r:26s} dtype={str(s.dtype):10s} nunique={s.nunique(dropna=True):>8} "
              f"nan={int(s.isna().sum()):>7} sample={list(s.dropna().unique()[:4])}")
    # 找字符串列做值统计
    for c in df.columns:
        if df[c].dtype == object or str(df[c].dtype).startswith("str") or str(df[c].dtype) == "category":
            print(f"\n--- {c} 取值分布 ---")
            print(df[c].astype("string").fillna("<NA>").value_counts().head(40).to_string())


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "both"
    if which in ("both", "w"):
        probe_weights()
    if which in ("both", "nt"):
        probe_nt()
