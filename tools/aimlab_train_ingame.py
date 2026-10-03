"""游戏域读出层重训:seek 导师采集的 (DN rates, action) 数据对 -> 岭回归。

运行::

    & $py tools/aimlab_train_ingame.py --npz flyaim/runs/bridge/collect_1.npz `
           --npz flyaim/runs/bridge/collect_2.npz `
           --out flyaim/runs/bridge/readout_ingame.npz

方法与离线 fit_readout 完全一致(冻结连接组,学习只发生在 Readout 线性层):
特征 = DN 群(1360)放电率,标签 = seek 导师的归一化 action,岭回归,
偏置不正则化。数据来自真实游戏帧(seek 驱动 + 开火推进目标轮换),
天然带游戏域的输入统计(全域光流)。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from flyaim.config import ReadoutConfig  # noqa: E402
from flyaim.io import load_roles  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", nargs="+", required=True, help="collect 落盘的 npz(可多个)")
    ap.add_argument("--out", default=str(ROOT / "flyaim/runs/bridge/readout_ingame.npz"))
    ap.add_argument("--ridge", type=float, default=10.0)
    ap.add_argument("--eval-frac", type=float, default=0.2,
                    help="尾部留出比例(时间块留出,不做随机打乱 —— 防同轨迹泄漏)")
    args = ap.parse_args()

    Xs, Ys, fid = [], [], None
    for p in args.npz:
        z = np.load(p)
        Xs.append(z["X"])
        Ys.append(z["Y"])
        f = z["feature_ids"]
        if fid is None:
            fid = f
        elif not np.array_equal(fid, f):
            print(f"❌ {p} 的 feature_ids 与其它文件不一致")
            return 1
        print(f"  {p}: {z['X'].shape[0]} 样本")
    X = np.concatenate(Xs, 0).astype(np.float64)
    Y = np.concatenate(Ys, 0).astype(np.float64)

    roles = load_roles(ROOT / "flyaim" / "data" / "build" / "roles.json")
    cfg = ReadoutConfig(mode="trained", ridge_lambda=args.ridge)
    from flyaim.brain.readout import Readout

    ro = Readout(cfg, roles=roles)
    if not np.array_equal(np.asarray(ro.feature_ids), fid):
        print("❌ 数据里的 feature_ids 与 roles.json 的 DN 群不一致(数据过期?)")
        return 1

    # 时间块留出:前 (1-frac) 训练,尾 frac 评估(相邻帧强相关,随机打分会泄漏)
    n = X.shape[0]
    n_tr = int(n * (1.0 - args.eval_frac))
    Xtr, Ytr, Xev, Yev = X[:n_tr], Y[:n_tr], X[n_tr:], Y[n_tr:]
    print(f"样本 {n}(训练 {n_tr} / 留出 {n - n_tr}),特征 {X.shape[1]},ridge={args.ridge}")

    ro.fit(Xtr, Ytr)
    ev = ro.evaluate(Xev, Yev, prefix="holdout_")
    print("\n---- 拟合诊断 ----")
    print(f"  train r2(dx/dy/all) = {ro.train_info['train_r2_dx']:.3f} / "
          f"{ro.train_info['train_r2_dy']:.3f} / {ro.train_info['train_r2_all']:.3f}")
    print(f"  holdout r2(dx/dy/all) = {ev['holdout_r2_dx']:.3f} / "
          f"{ev['holdout_r2_dy']:.3f} / {ev['holdout_r2_all']:.3f}")
    print(f"  holdout rmse = {ev['holdout_rmse']:.3f}")

    # 常数预测器基线(预测均值):holdout r2 必须显著超过它才有增量信息
    ym = Ytr.mean(0)
    ss_res = ((Yev - ym) ** 2).sum(0)
    ss_tot = ((Yev - Ytr.mean(0)) ** 2).sum(0)
    r2_const = 1 - ss_res.sum() / max(ss_tot.sum(), 1e-12)
    print(f"  holdout 常数基线 r2_all = {r2_const:.3f}"
          f"  (读出超出 {ev['holdout_r2_all'] - r2_const:+.3f})")

    p = ro.save(args.out)
    print(f"\n  权重 -> {p}")
    print(f"  试跑: tools/aimlab_play.py --mode fly --readout \"{p}\"")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
