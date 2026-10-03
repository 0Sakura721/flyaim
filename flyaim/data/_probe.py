"""A 线内部工具:schema 探查。只读,不写产物。"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pyarrow.feather as feather

sys.path.insert(0, r"D:\dsh\autoaim")

CACHE = Path(r"D:\dsh\autoaim\.cache")


def probe(path: Path) -> pd.DataFrame:
    print("=" * 100)
    print(f"FILE: {path.name}  ({path.stat().st_size/1e6:.2f} MB)")
    print("=" * 100)
    sch = feather.read_table(path, memory_map=True).schema
    print("--- arrow schema ---")
    print(sch)
    df = feather.read_table(path, memory_map=True).to_pandas()
    print(f"--- shape: {df.shape} ---")
    print("--- dtypes / 唯一值数 / 缺失数 / 样例 ---")
    for c in df.columns:
        s = df[c]
        nun = s.nunique(dropna=True)
        nna = int(s.isna().sum())
        vals = s.dropna().unique()[:5]
        print(f"  {c!r:28s} dtype={str(s.dtype):16s} nunique={nun:>9} nan={nna:>8}  sample={list(vals)[:5]}")
    print("--- head(5) ---")
    with pd.option_context("display.width", 220, "display.max_columns", 50):
        print(df.head(5).to_string())
    return df


if __name__ == "__main__":
    for name in sys.argv[1:]:
        probe(CACHE / name)
