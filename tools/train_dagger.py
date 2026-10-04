"""CONTRACT_ANN.md 附录 B(DAgger)执行入口。

    & $py tools/train_dagger.py            # 两臂完整 DAgger + 评估 + 判门
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

from flyaim.ann_train import ConnectomeRNN, eval_arm  # noqa: E402
from flyaim.ann_train import dagger, train_on_buffer  # noqa: E402
from flyaim.io import save_json  # noqa: E402

ADIR = ROOT / "flyaim" / "runs" / "ann-dagger"


def snapshot(net: ConnectomeRNN) -> dict:
    cp = net.cp
    return {"W_data": cp.asnumpy(net.W.data).copy(),
            "g": cp.asnumpy(net.g).copy(),
            "R": cp.asnumpy(net.R).copy(),
            "b": cp.asnumpy(net.b).copy()}


def load_state(net: ConnectomeRNN, st: dict) -> None:
    net.W.data[...] = net.cp.asarray(st["W_data"])
    net.g[...] = net.cp.asarray(st["g"])
    net.R[...] = net.cp.asarray(st["R"])
    net.b[...] = net.cp.asarray(st["b"])


def run_arm(arm: str, round_frames: int, seeds: int) -> dict:
    net = ConnectomeRNN(arm)
    buffer: list = []
    out: dict = {"arm": arm, "rounds": []}

    # ---- 轮 0:BC(教师驱动)
    dagger(net, round_frames, policy_driven=False, buffer=buffer,
           episode_seeds=range(0, 4))
    loss0 = train_on_buffer(net, buffer, passes=1)
    bc_state = snapshot(net)
    print(f"[{arm}] 轮0(BC): loss={loss0:.4f} buffer={sum(len(e) for e in buffer)}",
          flush=True)
    res_bc = eval_arm(net, seeds=seeds, tag=f"{arm}-bc")
    out["rounds"].append({"round": 0, "loss": loss0, "eval": res_bc})

    # ---- 轮 1/2:DAgger(策略驱动 + 教师标注)
    for r, seed_off in ((1, 4), (2, 8)):
        dagger(net, round_frames, policy_driven=True, buffer=buffer,
               episode_seeds=range(seed_off, seed_off + 4))
        loss = train_on_buffer(net, buffer, passes=1)
        print(f"[{arm}] 轮{r}(DAgger): loss={loss:.4f} "
              f"buffer={sum(len(e) for e in buffer)}", flush=True)
        out["rounds"].append({"round": r, "loss": loss})

    # ---- 最终评估(同时给出 BC 检查点对照,B-G2)
    res_final = eval_arm(net, seeds=seeds, tag=f"{arm}-dagger")
    load_state(net, bc_state)
    res_bc_full = eval_arm(net, seeds=seeds, tag=f"{arm}-bc-ckpt")
    out.update({"eval_final": res_final, "eval_bc": res_bc_full,
                "eval_bc_head": res_bc})
    return out, net, bc_state


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--round-frames", type=int, default=3000)
    ap.add_argument("--seeds", type=int, default=10)
    args = ap.parse_args()

    t0 = time.perf_counter()
    all_results = {}
    finals = {}
    for arm in ("real", "shuffle"):
        out, net, bc_state = run_arm(arm, args.round_frames, args.seeds)
        all_results[f"{arm}-dagger"] = out["eval_final"]
        all_results[f"{arm}-bc"] = out["eval_bc"]
        finals[arm] = {"dagger": snapshot(net), "bc": bc_state}
        net.save(ADIR / arm / f"{arm}_dagger_final.npz")
        (ADIR / arm / f"{arm}_dagger.json").write_text(
            json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[campaign] {arm} 完成 {time.perf_counter()-t0:.0f}s", flush=True)

    # random 对照
    import cupy as _cp

    class RandomNet:
        cp = _cp
        n = 166700

        @staticmethod
        def act_closed_loop(u, h):
            return np.random.default_rng().uniform(
                -1, 1, 2).astype(np.float32), h

    all_results["random"] = eval_arm(RandomNet, seeds=args.seeds, tag="random")

    # ---- 门判定(B2) ----
    def md(name):
        return [r["mean_dist_px"] for r in all_results[name]]

    from scipy import stats

    verdict = {"results": {k: {"mean_dist": md(k), "hits": [r["hits"] for r in v]}
                           for k, v in all_results.items()}, "gates": {}}
    rd_, rn = md("real-dagger"), md("random")
    t1 = stats.ttest_rel(rd_, rn)
    verdict["gates"]["B-G1"] = {
        "passed": bool(t1.pvalue < 0.05 and np.mean(rd_) < np.mean(rn)
                       and (np.mean(rn) - np.mean(rd_)) > 50),
        "real_dagger": float(np.mean(rd_)), "random": float(np.mean(rn)),
        "p": float(t1.pvalue)}
    rb = md("real-bc")
    t2 = stats.ttest_rel(rd_, rb)
    verdict["gates"]["B-G2"] = {
        "passed": bool(t2.pvalue < 0.05 and np.mean(rd_) < np.mean(rb)),
        "real_bc": float(np.mean(rb)), "p": float(t2.pvalue)}
    sd_ = md("shuffle-dagger")
    t3 = stats.ttest_rel(rd_, sd_)
    verdict["gates"]["B-G3"] = {
        "passed": bool(t3.pvalue < 0.05 and np.mean(rd_) < np.mean(sd_)),
        "shuffle_dagger": float(np.mean(sd_)), "p": float(t3.pvalue)}
    save_json(str(ADIR / "verdict.json"), verdict)

    print("\n---- 预注册门判定(B2) ----")
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
