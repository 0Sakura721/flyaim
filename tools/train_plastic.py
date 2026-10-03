"""CONTRACT_PLASTIC.md 附录 P 的执行入口:训练 + 四臂评估 + 预注册门判定。

运行::

    # 1) 训练(真实 + shuffle,各 ~1h GPU;断点快照每 9000 帧)
    & $py tools/train_plastic.py --phase train --arm real
    & $py tools/train_plastic.py --phase train --arm shuffle

    # 2) 评估四臂 + 门判定
    & $py tools/train_plastic.py --phase eval

两阶段分离:训练崩溃不影响已落盘的快照;评估阶段读快照。
判定规则冻结在 CONTRACT_PLASTIC.md P5,本工具只是执行者。
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

from flyaim.config import PlasticityConfig  # noqa: E402
from flyaim.io import save_json  # noqa: E402
from flyaim.plastic_train import arm_weights, eval_arm, train_arm  # noqa: E402

PDIR = ROOT / "flyaim" / "runs" / "plastic"


def do_train(args: argparse.Namespace) -> int:
    arm = args.arm
    if (PDIR / arm / f"{arm}_wfinal.npz").exists() and not args.force:
        print(f"[{arm}] 已有成品权重({arm}_wfinal.npz),跳过(--force 重跑)")
        return 0
    pcfg = PlasticityConfig(
        probe_frames=args.probe_frames,
        m_sigma_px=args.sigma,
        plastic_every=args.plastic_every,
    )
    t0 = time.perf_counter()
    out = train_arm(arm=arm, frames=args.frames, device=args.device,
                    out_dir=PDIR / arm, pcfg=pcfg, log_every=args.log_every)
    print(f"[{arm}] 完成,用时 {time.perf_counter()-t0:.0f}s,η={out['eta']:.4g}")
    return 0


def do_eval(args: argparse.Namespace) -> int:
    pdir = PDIR
    arms = {
        "real-plastic": pdir / "real" / "real_wfinal.npz",
        "shuffle-plastic": pdir / "shuffle" / "shuffle_wfinal.npz",
        "real-frozen": arm_weights("real"),
        "shuffle-frozen": arm_weights("shuffle"),
    }
    # g 取各训练 json 里冻结的值(real 与 shuffle 各自标定;frozen 臂用 real 的)
    g_map = {}
    for name in ("real", "shuffle"):
        p = pdir / name / f"{name}_train.json"
        if p.exists():
            g_map[name] = json.loads(p.read_text(encoding="utf-8"))["readout_g"]
    results = {}
    for name, wp in arms.items():
        if not wp.exists():
            print(f"[skip] {name}: {wp} 不存在")
            continue
        g = g_map.get("real", 1.0)
        if name.startswith("shuffle") and "shuffle" in g_map:
            g = g_map["shuffle"]
        if "plastic" in name:
            t0 = time.perf_counter()
            res = eval_arm(wp, readout_g=g, seeds=args.seeds, frames=args.frames,
                           tag=name)
            print(f"  ({name} 评估用时 {time.perf_counter()-t0:.0f}s)")
        else:
            res = eval_arm(wp, readout_g=g, seeds=args.seeds, frames=args.frames,
                           tag=name)
        results[name] = res

    # ---- 预注册门判定(P5) ----
    def mean_dist(name: str) -> list[float]:
        return [r["mean_dist_px"] for r in results.get(name, [])]

    verdict = {"gates": {}, "results": {k: {"mean_dist": mean_dist(k),
                                            "hits": [r["hits"] for r in v]}
                                        for k, v in results.items()}}
    if all(name in results for name in
           ("real-plastic", "real-frozen", "shuffle-plastic")):
        from scipy import stats

        rp, rf = mean_dist("real-plastic"), mean_dist("real-frozen")
        sp_ = mean_dist("shuffle-plastic")
        t1 = stats.ttest_rel(rp, rf)
        g1 = bool(t1.pvalue < 0.05 and np.mean(rp) < np.mean(rf)
                  and (np.mean(rf) - np.mean(rp)) > 20)
        verdict["gates"]["G1"] = {
            "passed": g1,
            "real_plastic_mean": float(np.mean(rp)),
            "real_frozen_mean": float(np.mean(rf)),
            "p": float(t1.pvalue),
        }
        t3 = stats.ttest_rel(rp, sp_)
        g3 = bool(t3.pvalue < 0.05 and np.mean(rp) < np.mean(sp_))
        verdict["gates"]["G3"] = {
            "passed": g3,
            "shuffle_plastic_mean": float(np.mean(sp_)),
            "p": float(t3.pvalue),
        }
        # G2:训练曲线末端斜率(real)
        tr_p = pdir / "real" / "real_train.json"
        if tr_p.exists():
            tr = json.loads(tr_p.read_text(encoding="utf-8"))
            errs = [m["mean_err_px"] for m in tr["metrics"]]
            k = max(1, len(errs) // 3)
            tail = errs[-k:]
            x = np.arange(len(tail))
            slope, inter, r, p, se = stats.linregress(x, tail)
            ci = 1.96 * se
            g2 = bool(slope < 0 and (slope + ci) < 0)  # 斜率<0 且 95%CI 上界<0
            verdict["gates"]["G2"] = {
                "passed": g2, "tail_slope_px_per_200f": float(slope),
                "ci95": [float(slope - ci), float(slope + ci)],
            }
    save_json(str(PDIR / "verdict.json"), verdict)
    print("\n---- 预注册门判定(P5) ----")
    for g, v in verdict["gates"].items():
        print(f"  {g}: {'✅ 通过' if v['passed'] else '❌ 未通过'}  {v}")
    for name, res in results.items():
        md = [r["mean_dist_px"] for r in res]
        print(f"  {name:16s} mean_dist = {np.mean(md):7.1f} ± {np.std(md):.1f} px"
              f"  hits = {sum(r['hits'] for r in res)}")
    print(f"\n  判定与数据 -> {PDIR / 'verdict.json'}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True, choices=["train", "eval"])
    ap.add_argument("--arm", default="real", choices=["real", "shuffle"])
    ap.add_argument("--frames", type=int, default=36000)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--frames-eval", dest="frames", type=int, default=900)
    ap.add_argument("--probe-frames", type=int, default=200)
    ap.add_argument("--sigma", type=float, default=60.0)
    ap.add_argument("--plastic-every", type=int, default=1)
    ap.add_argument("--device", default="cuda", choices=["cuda"])
    ap.add_argument("--log-every", type=int, default=200)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    if args.phase == "train":
        return do_train(args)
    return do_eval(args)


if __name__ == "__main__":
    raise SystemExit(main())
