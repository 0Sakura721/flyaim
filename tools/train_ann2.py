"""CONTRACT_ANN.md 附录 A2 执行入口:门控连接组网络 + 预算补足 + 三门判定。

    & $py tools/train_ann2.py                # 两臂完整 A2 + 四臂评估 + 判门
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from flyaim.ann_gated import GatedConnectomeRNN, eval_arm, train_arm  # noqa: E402
from flyaim.io import save_json  # noqa: E402

ADIR = ROOT / "flyaim" / "runs" / "ann2"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=12000)
    ap.add_argument("--seeds", type=int, default=10)
    args = ap.parse_args()

    sys.path.insert(0, str(ROOT / "tools"))
    from ann_dashboard import StatePublisher
    import json as _json

    layout_path = ROOT / ".cache/ann_layout.json"
    sample_idx = None
    if layout_path.exists():
        sample_idx = np.array(_json.loads(layout_path.read_text(encoding="utf-8"))["sample_idx"])
    pub = StatePublisher(ROOT / ".cache/ann2_state.json", sample_idx=sample_idx,
                         total_frames=args.frames)

    t0 = time.perf_counter()
    all_results = {}
    for arm in ("real", "shuffle"):
        pub.set_frame(0)
        out = train_arm(arm, frames=args.frames, out_dir=ADIR / arm, publish=pub)
        all_results[f"{arm}-bc"] = out["eval_bc"]
        net = GatedConnectomeRNN(arm)
        z = np.load(ADIR / arm / f"{arm}_ann2_best.npz")
        net.load_state({k: z[k] for k in z.files})
        all_results[f"{arm}-ann2"] = eval_arm(net, seeds=args.seeds,
                                              tag=f"{arm}-ann2")
        print(f"[campaign] {arm} 完成 {time.perf_counter()-t0:.0f}s", flush=True)

    import cupy as cp

    class RandomNet:
        cp = cp
        n = 166700

        @staticmethod
        def act_closed_loop(u, h):
            return np.random.default_rng().uniform(
                -1, 1, 2).astype(np.float32), h

    all_results["random"] = eval_arm(RandomNet, seeds=args.seeds, tag="random")

    def md(name):
        return [r["mean_dist_px"] for r in all_results[name]]

    from scipy import stats

    verdict = {"results": {k: {"mean_dist": md(k), "hits": [r["hits"] for r in v]}
                           for k, v in all_results.items()}, "gates": {}}
    ra, rd = md("real-ann2"), md("random")
    t1 = stats.ttest_rel(ra, rd)
    verdict["gates"]["C-G1"] = {
        "passed": bool(t1.pvalue < 0.05 and np.mean(ra) < np.mean(rd)
                       and (np.mean(rd) - np.mean(ra)) > 50),
        "real_ann2": float(np.mean(ra)), "random": float(np.mean(rd)),
        "p": float(t1.pvalue)}
    rb = md("real-bc")
    t2 = stats.ttest_rel(ra, rb)
    verdict["gates"]["C-G2"] = {
        "passed": bool(t2.pvalue < 0.05 and np.mean(ra) < np.mean(rb)),
        "real_bc": float(np.mean(rb)), "p": float(t2.pvalue)}
    sa = md("shuffle-ann2")
    t3 = stats.ttest_rel(ra, sa)
    verdict["gates"]["C-G3"] = {
        "passed": bool(t3.pvalue < 0.05 and np.mean(ra) < np.mean(sa)),
        "shuffle_ann2": float(np.mean(sa)), "p": float(t3.pvalue)}
    save_json(str(ADIR / "verdict.json"), verdict)

    print("\n---- 预注册门判定(C2) ----")
    for g, v in verdict["gates"].items():
        print(f"  {g}: {'✅ 通过' if v['passed'] else '❌ 未通过'}  {v}")
    for name, res in all_results.items():
        mdv = [r["mean_dist_px"] for r in res]
        print(f"  {name:16s} mean_dist = {np.mean(mdv):7.1f} ± {np.std(mdv):5.1f} px"
              f"  hits = {sum(r['hits'] for r in res)}")
    print(f"\n  判定与数据 -> {ADIR / 'verdict.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
