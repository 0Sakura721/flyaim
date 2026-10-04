"""CONTRACT_ANN.md 附录 A 的执行入口:连接组约束网络训练 + 四臂评估 + 门判定。

运行::

    & $py tools/train_ann.py                  # 训练 real + shuffle,评估四臂,判门
    & $py tools/train_ann.py --frames 12000 --seeds 10
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

from flyaim.io import save_json  # noqa: E402
from flyaim.ann_train import ConnectomeRNN, eval_arm, train_arm  # noqa: E402

ADIR = ROOT / "flyaim" / "runs" / "ann"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=12000)
    ap.add_argument("--seeds", type=int, default=10)
    args = ap.parse_args()

    t0 = time.perf_counter()
    nets = {}
    results = {}
    for arm in ("real", "shuffle"):
        out = train_arm(arm, frames=args.frames, out_dir=ADIR / arm)
        net = ConnectomeRNN(arm)
        # 载入期间的谱半径维护(训练中已每 16 chunk 重投影,载入后再一次)
        import json as _json
        _t = json.loads((ADIR / arm / f"{arm}_train.json").read_text(encoding="utf-8"))
        # 载回训练后的 W.data/g/R(从 npz)
        z = np.load(ADIR / arm / f"{arm}_ann_best.npz")  # A1 v5:评估最优检查点
        net.W.data[...] = net.cp.asarray(z["W_data"])
        net.g[...] = net.cp.asarray(z["g"])
        net.R[...] = net.cp.asarray(z["R"])
        nets[arm] = net
        results[f"{arm}-ann"] = eval_arm(net, seeds=args.seeds, tag=f"{arm}-ann")
        print(f"[campaign] {arm} 完成 {time.perf_counter()-t0:.0f}s", flush=True)

    # real-untrained:零初始化读出 → 恒零动作(准星静止对照)
    results["real-untrained"] = eval_arm(nets["real"], seeds=args.seeds,
                                         tag="real-untrained")
    # random:均匀随机
    class _R:
        cp = nets["real"].cp

        @staticmethod
        def act_closed_loop(u, h):
            import numpy as np

            rng = np.random.default_rng()
            return rng.uniform(-1, 1, 2).astype(np.float32), h
    results["random"] = eval_arm(_R, seeds=args.seeds, tag="random")

    # ---- 预注册门判定(A3) ----
    def md(name):
        return [r["mean_dist_px"] for r in results[name]]

    from scipy import stats

    verdict = {"results": {k: {"mean_dist": md(k), "hits": [r["hits"] for r in v]}
                           for k, v in results.items()}, "gates": {}}
    if all(k in results for k in ("real-ann", "random", "shuffle-ann")):
        ra, rd = md("real-ann"), md("random")
        t1 = stats.ttest_rel(ra, rd)
        verdict["gates"]["A-G1"] = {
            "passed": bool(t1.pvalue < 0.05 and np.mean(ra) < np.mean(rd)
                           and (np.mean(rd) - np.mean(ra)) > 50),
            "real_ann_mean": float(np.mean(ra)), "random_mean": float(np.mean(rd)),
            "p": float(t1.pvalue)}
        sa = md("shuffle-ann")
        t3 = stats.ttest_rel(ra, sa)
        verdict["gates"]["A-G3"] = {
            "passed": bool(t3.pvalue < 0.05 and np.mean(ra) < np.mean(sa)),
            "shuffle_ann_mean": float(np.mean(sa)), "p": float(t3.pvalue)}
    tr_p = ADIR / "real" / "real_train.json"
    if tr_p.exists():
        tr = json.loads(tr_p.read_text(encoding="utf-8"))
        losses = tr["losses"]
        k = max(1, len(losses) // 3)
        tail = losses[-k:]
        x = np.arange(len(tail))
        slope, _, _, _, se = stats.linregress(x, tail)
        ci = 1.96 * se
        verdict["gates"]["A-G2"] = {
            "passed": bool(slope < 0 and slope + ci < 0),
            "tail_slope": float(slope), "ci95": [float(slope - ci), float(slope + ci)]}
    save_json(str(ADIR / "verdict.json"), verdict)

    print("\n---- 预注册门判定(A3) ----")
    for g, v in verdict["gates"].items():
        print(f"  {g}: {'✅ 通过' if v['passed'] else '❌ 未通过'}  {v}")
    for name, res in results.items():
        mdv = [r["mean_dist_px"] for r in res]
        print(f"  {name:16s} mean_dist = {np.mean(mdv):7.1f} ± {np.std(mdv):5.1f} px"
              f"  hits = {sum(r['hits'] for r in res)}")
    print(f"\n  判定与数据 -> {ADIR / 'verdict.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
