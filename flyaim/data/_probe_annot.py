"""A 线内部工具:annotations 深度探查(找感光细胞 / class 取值)。"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd
import pyarrow.feather as feather

CACHE = Path(r"D:\dsh\autoaim\.cache")
ANNOT = CACHE / "body-annotations-male-cns-v1.0-minconf-0.5.feather"


def vc(df: pd.DataFrame, col: str, top: int | None = None) -> None:
    if col not in df.columns:
        print(f"  [缺列] {col}")
        return
    s = df[col].astype("string").fillna("<NA>")
    counts = s.value_counts()
    nna = int(df[col].isna().sum())
    print(f"\n--- {col}: {counts.size} 个取值 (NaN={nna}) ---")
    show = counts if top is None else counts.head(top)
    for k, v in show.items():
        print(f"    {str(k):42s} {v:>8}")


def main() -> None:
    tbl = feather.read_table(ANNOT, memory_map=True)
    print(f"rows={tbl.num_rows}")
    df = tbl.to_pandas()

    print("\n" + "=" * 90)
    print("A. 关键分类列全量取值")
    print("=" * 90)
    for c in ("statusLabel", "superclass", "class", "subclass", "somaSide", "somaNeuromere"):
        vc(df, c)

    print("\n" + "=" * 90)
    print("B. 神经元 vs 非神经元:按 vfbId / superclass 是否存在交叉统计")
    print("=" * 90)
    has_vfb = df["vfbId"].notna()
    has_sc = df["superclass"].notna()
    print(pd.crosstab(has_vfb.rename("has_vfbId"), has_sc.rename("has_superclass")))
    print(f"\nvfbId 非空行数 = {int(has_vfb.sum())}  (去重 = {df.loc[has_vfb,'bodyId'].nunique()})")
    print(f"superclass 非空行数 = {int(has_sc.sum())}")
    print("\n--- statusLabel × has_vfbId ---")
    print(pd.crosstab(df["statusLabel"].astype("string").fillna("<NA>"), has_vfb.rename("has_vfbId")))

    print("\n" + "=" * 90)
    print("C. 感光细胞搜索(photoreceptor)")
    print("=" * 90)
    ty = df["type"].astype("string").fillna("")
    inst = df["instance"].astype("string").fillna("")
    sc = df["superclass"].astype("string").fillna("")
    cls = df["class"].astype("string").fillna("")
    st = df["statusLabel"].astype("string").fillna("")

    patterns = {
        "superclass~photoreceptor": sc.str.contains("photorecept", case=False),
        "superclass~sensory": sc.str.contains("sensory", case=False),
        "type~^R[1-8]": ty.str.match(r"^R[1-8][^0-9a-zA-Z]?$") | ty.str.match(r"^R[1-8][a-z]?"),
        "instance~^R[1-8]": inst.str.match(r"^R[1-8]"),
        "type/inst~photo": ty.str.contains("photo", case=False) | inst.str.contains("photo", case=False),
        "type/inst~lamina": ty.str.contains("lamina", case=False) | inst.str.contains("lamina", case=False),
        "type~^L[1-5]$": ty.str.match(r"^L[1-5]$"),
        "type~^T[45]": ty.str.match(r"^T[45]"),
        "type~^Mi[0-9]": ty.str.match(r"^Mi[0-9]"),
        "type~^Tm[0-9]": ty.str.match(r"^Tm[0-9]"),
        "type~^Lawf": ty.str.match(r"^Lawf"),
        "type~^R[78]": ty.str.match(r"^R[78]"),
        "statusLabel~Glia": st.str.contains("glia", case=False),
        "statusLabel~Trachea": st.str.contains("trachea", case=False),
    }
    for name, m in patterns.items():
        m = m.fillna(False).to_numpy(dtype=bool)
        print(f"  {name:32s} n={int(m.sum()):>7}", end="")
        if m.sum():
            ex = sorted(set(ty[m].tolist()))[:12]
            print(f"   examples={ex}")
        else:
            print()

    print("\n--- 含 'R1'..'R8' 前缀的 type 名(前 60) ---")
    hits = sorted({t for t in ty.unique().tolist() if re.match(r"^R[1-8](\b|[a-zA-Z]?\d*$)", str(t))})
    print(hits[:60], f"... total {len(hits)}")

    print("\n--- type 中含 'R' + 数字 且 短名 的(疑似感光) ---")
    short = sorted({t for t in ty.unique().tolist() if re.fullmatch(r"R\d+[a-zA-Z]?", str(t))})
    print(short)

    print("\n" + "=" * 90)
    print("D. 视叶 / 视觉投射相关统计(type 前缀)")
    print("=" * 90)
    for pref in ("VPN", "LPC", "LC", "ME", "LO", "T4", "T5", "R1", "R2", "R7", "R8",
                 "L1", "L2", "L3", "L4", "L5", "Mi", "Tm", "T1", "T2", "T3", "Lawf", "C2", "C3"):
        m = ty.str.upper().str.startswith(pref).fillna(False).to_numpy(dtype=bool)
        print(f"  type 前缀 {pref:5s}: {int(m.sum()):>7}")

    print("\n--- 所有含 'visual' 的 superclass/class/type 取值 ---")
    for col in ("superclass", "class", "subclass", "type"):
        s = df[col].astype("string").fillna("")
        vals = sorted({v for v in s.unique().tolist() if "visual" in str(v).lower() or "optic" in str(v).lower() or "ol_" in str(v).lower()})
        print(f"  {col}: {vals[:40]}")

    if "assignedOlHex1" in df.columns:
        print(f"\nassignedOlHex1 非空 = {int(df['assignedOlHex1'].notna().sum())}")
        ol = df["assignedOlHex1"].notna()
        print("OL 非空行的 superclass 取值:")
        print(df.loc[ol, "superclass"].astype("string").fillna("<NA>").value_counts().to_string())

    print("\n--- bodyId 范围 ---")
    b = df["bodyId"].to_numpy()
    print(f"  min={b.min()} max={b.max()} n={b.size} unique={len(set(b.tolist()))}")
    print(f"  dtype={b.dtype}  是否单调={bool((b[1:]>b[:-1]).all())}")


if __name__ == "__main__":
    main()
