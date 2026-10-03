"""A 线内部工具:感光细胞证据链(交叉验证 type <-> superclass <-> statusLabel)。"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pyarrow.feather as feather

CACHE = Path(r"D:\dsh\autoaim\.cache")
ANNOT = CACHE / "body-annotations-male-cns-v1.0-minconf-0.5.feather"
pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 60)
pd.set_option("display.max_rows", 300)


def main() -> None:
    df = feather.read_table(ANNOT, memory_map=True).to_pandas()
    df = df[df["superclass"].notna()].copy()          # 166,700 神经元
    print(f"神经元行数(superclass 非空) = {len(df)}")
    print(f"bodyId 唯一数 = {df['bodyId'].nunique()}  单调 = {bool((df['bodyId'].to_numpy()[1:] > df['bodyId'].to_numpy()[:-1]).all())}")

    ty = df["type"].astype("string").fillna("<NA>")
    sc = df["superclass"].astype("string").fillna("<NA>")
    st = df["statusLabel"].astype("string").fillna("<NA>")

    print("\n=== 1. superclass 全量取值(仅神经元) ===")
    print(sc.value_counts().to_string())

    print("\n=== 2. 'ol_sensory' 到底是不是感光细胞?type 全量取值 ===")
    ols = sc == "ol_sensory"
    print(f"ol_sensory n = {int(ols.sum())}")
    print(ty[ols].value_counts().to_string())

    print("\n=== 3. type 以 R 开头(感光细胞命名法)的完整列表 ===")
    rmask = ty.str.match(r"^R\d").fillna(False).to_numpy(dtype=bool)
    print(f"n = {int(rmask.sum())}")
    print(ty[rmask].value_counts().to_string())
    print("\n这些 R 细胞的 superclass:")
    print(sc[rmask].value_counts().to_string())
    print("\n这些 R 细胞的 statusLabel:")
    print(st[rmask].value_counts().to_string())
    print("\n这些 R 细胞与 ol_sensory 集合是否一致?")
    same = ((rmask) == ols.to_numpy(dtype=bool))
    print(f"  完全一致 = {bool(same.all())};  仅R非ol_sensory = {int((rmask & ~ols.to_numpy(dtype=bool)).sum())};"
          f"  仅ol_sensory非R = {int((~rmask & ols.to_numpy(dtype=bool)).sum())}")

    print("\n=== 4. 视叶细胞分类体系(superclass x type 前缀) ===")
    for pre in ("R1", "R7", "R8", "T4", "T5", "L1", "Mi", "Tm", "ME", "LO", "LC", "LPC", "Lawf"):
        m = ty.str.upper().str.startswith(pre).fillna(False).to_numpy(dtype=bool)
        if m.sum() == 0:
            print(f"  {pre:5s}    0")
            continue
        print(f"  {pre:5s} {int(m.sum()):>6}  superclass={dict(sc[m].value_counts().head(3))}")

    print("\n=== 5. 视觉通路各层级细胞数(用于选择 visual_input) ===")
    print(f"  感光细胞 R1-R6/R7/R8           = {int(rmask.sum())}")
    print(f"  视叶中间神经元 ol_intrinsic     = {int((sc=='ol_intrinsic').sum())}")
    print(f"  视觉投射 visual_projection      = {int((sc=='visual_projection').sum())}")
    print(f"  视觉离心 visual_centrifugal     = {int((sc=='visual_centrifugal').sum())}")
    print(f"  class=='visual'                 = {int((df['class'].astype('string')=='visual').sum())}")
    print(f"  class 含 'ol_'                   = {int(df['class'].astype('string').fillna('').str.startswith('ol_').sum())}")

    print("\n=== 6. 下行神经元(DN)候选 ===")
    dn_sc = sc == "descending_neuron"
    dn_ty = ty.str.upper().str.startswith("DN").fillna(False).to_numpy(dtype=bool)
    print(f"  superclass=='descending_neuron'      n={int(dn_sc.sum())}")
    print(f"  type 前缀 'DN'                        n={int(dn_ty.sum())}")
    print(f"  并集                                  n={int((dn_sc.to_numpy(dtype=bool)|dn_ty).sum())}")
    print(f"  交集                                  n={int((dn_sc.to_numpy(dtype=bool)&dn_ty).sum())}")
    print("  class=='descending'?  ->",
          int((df['class'].astype('string').fillna('') == 'descending').sum()), "(class 列里没有 'descending')")
    print("\n  descending_neuron 的 class 取值:")
    print(df.loc[dn_sc, "class"].astype("string").fillna("<NA>").value_counts().to_string())
    print("\n  descending_neuron 的 type 样例(前 25):", sorted(ty[dn_sc].unique().tolist())[:25])

    print("\n=== 7. 运动神经元(MN)候选 ===")
    print("  superclass=='vnc_motor'  n =", int((sc == "vnc_motor").sum()))
    print("  superclass=='cb_motor'   n =", int((sc == "cb_motor").sum()))
    print("  class=='motor'?          n =", int((df['class'].astype('string').fillna('') == 'motor').sum()))
    print("  class 取值(全 21 个,仅神经元):")
    print(df["class"].astype("string").fillna("<NA>").value_counts().to_string())

    print("\n=== 8. 嗅觉/其他感觉(供对照) ===")
    print(sc.value_counts().to_string())


if __name__ == "__main__":
    sys.exit(main())
