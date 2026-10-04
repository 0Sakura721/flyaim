"""从 GPU 战役结果计算 C-G1/G2/G3 预注册门判定(与 CPU 版同判据)。

    & $py tools/verdict_ann2_gpu.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
ADIR = ROOT / "flyaim" / "runs" / "ann2_gpu"


def main() -> int:
    from scipy import stats
    from flyaim.io import save_json

    d = json.loads((ADIR / "train_gpu.json").read_text(encoding="utf-8"))
    r = d["arm_results"]
    ra, rb, sa, rd = (r["real-ann2"], r["real-bc"], r["shuffle-ann2"], r["random"])
    g = {}
    t1 = stats.ttest_rel(ra, rd)
    g["C-G1"] = {"passed": bool(t1.pvalue < 0.05 and np.mean(ra) < np.mean(rd)
                                and (np.mean(rd) - np.mean(ra)) > 50),
                 "real_ann2": float(np.mean(ra)), "random": float(np.mean(rd)),
                 "p": float(t1.pvalue)}
    t2 = stats.ttest_rel(ra, rb)
    g["C-G2"] = {"passed": bool(t2.pvalue < 0.05 and np.mean(ra) < np.mean(rb)),
                 "real_bc": float(np.mean(rb)), "p": float(t2.pvalue)}
    t3 = stats.ttest_rel(ra, sa)
    g["C-G3"] = {"passed": bool(t3.pvalue < 0.05 and np.mean(ra) < np.mean(sa)),
                 "shuffle_ann2": float(np.mean(sa)), "p": float(t3.pvalue),
                 "effect_px": float(abs(np.mean(ra) - np.mean(sa)))}
    verdict = {"results": r, "gates": g, "wall_s": d["wall_s"]}
    save_json(str(ADIR / "verdict.json"), verdict)
    print("---- C 门判定(GPU 战役) ----")
    for k, v in g.items():
        print(f"  {k}: {'PASS' if v['passed'] else 'FAIL'}  {v}")
    print("\n  real-ann2   %.1f px" % np.mean(ra))
    print("  shuffle-ann2 %.1f px" % np.mean(sa))
    print("  real-bc     %.1f px" % np.mean(rb))
    print("  random      %.1f px" % np.mean(rd))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
