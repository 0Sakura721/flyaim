"""机制诊断:果蝇臂的动作是「恒定偏置」还是「在变化」?

Lead 持有。从录屏目视发现:fly 臂准星被顶死在右上角,而靶在左上角。
提出假设:
    果蝇臂输出的 action 近似**恒定**(被读出层的 bias 主导),
    于是准星被一路推到墙角并卡住;训练集的 r2=0.75 可能只是
    在拟合「期望动作的均值」这个近似常数,而不是真的在瞄准。

判据:
    1. 评估时 action 的时间标准差 / action 均值范数 —— 接近 0 说明动作恒定
    2. 评估时 DN 放电率 vs 训练时 DN 放电率 —— 分布是否漂移(协变量偏移)
    3. 用「常数预测器」基线的 r2 对照:
       若读出的 r2 不显著高于「永远预测训练集均值」的 r2,
       则读出层学到的就是常数。

运行::

    <捆绑 python> tools/diag_readout_mechanism.py
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from flyaim.arena.arena import Arena  # noqa: E402
from flyaim.baselines.pid import PIDBaseline  # noqa: E402
from flyaim.config import ArenaConfig, BrainConfig, ReadoutConfig, RetinaConfig  # noqa: E402
from flyaim.fit_readout import collect_training_set, fit_readout  # noqa: E402
from flyaim.pipeline import FlySystem  # noqa: E402

D = ROOT / "flyaim" / "data" / "build"
NORM = ROOT / "flyaim" / "runs" / "norm"
W, G = 16.0, 1.5


def main() -> int:
    print("=" * 88)
    print("机制诊断:果蝇臂的动作是恒定偏置还是有效控制?")
    print("=" * 88, flush=True)

    wp = NORM / "connectome_indeg.npz"
    # roles.json 等上下文
    for extra in ("roles.json", "neuron_index.parquet", "manifest.json"):
        dst = wp.parent / extra
        if not dst.exists():
            shutil.copy(D / extra, dst)

    rc = RetinaConfig()
    bc = BrainConfig(weight_scale_exc=W, weight_scale_inh=W, input_gain=G)
    ro = ReadoutConfig(mode="trained", ridge_lambda=10.0)
    fs = FlySystem(D, rc, bc, ro, weights_override=wp)

    # ---- 1. 训练(短) ----
    print("\n[1] 训练读出层(200 帧)", flush=True)
    pid = PIDBaseline()
    ts = collect_training_set(fs, Arena(ArenaConfig(), seed=9000), 9000, 200,
                              lambda st: pid.act(st))
    diag = fit_readout(fs.readout, [ts], ridge_lambda=10.0)
    print(f"    r2 = {diag['r2']:.4f}  死特征 = {diag['n_dead_features']}/{diag['n_features']}")

    # ---- 常数预测器基线 ----
    Ymean = ts.Y.mean(axis=0, keepdims=True)
    ss_tot = float(((ts.Y - ts.Y.mean(axis=0)) ** 2).sum())
    ss_const = float(((ts.Y - Ymean) ** 2).sum())
    r2_const = 1.0 - ss_const / max(ss_tot, 1e-12)
    print(f"    [对照] 「永远预测训练集均值」的 r2 = {r2_const:.4f}")
    print(f"    [对照] 训练集 Y 的标准差 = {ts.Y.std(axis=0)}  (接近 0 说明目标近似常数)")
    if diag["r2"] <= r2_const + 0.05:
        print("    ⚠️ 读出的 r2 未显著超过常数预测器 → **读出层学到的基本是常数**")
    else:
        print("    ✅ 读出的 r2 显著超过常数预测器 → 确实学到了随输入变化的映射")

    # ---- 2. 评估时动作的时间结构 ----
    print("\n[2] 评估阶段(被测果蝇)的动作统计", flush=True)
    arena = Arena(ArenaConfig(), seed=0)
    frame = arena.reset()
    fs.reset()
    acts, dnr = [], []
    for i in range(200):
        d = fs.retina.frame_to_spikes(frame)
        for _s in range(fs.brain.cfg.steps_per_frame):
            fs.brain.step(d[:, 0], d[:, 1], 1.0)
        a = fs.readout.act(fs.brain)
        acts.append(np.asarray(a, dtype=np.float64).copy())
        dnr.append(float(fs.brain.rates[fs.roles.descending].mean()))
        frame = arena.step(a).frame

    A = np.asarray(acts)
    print(f"    action 均值   = [{A[:,0].mean():+.4f}, {A[:,1].mean():+.4f}]")
    print(f"    action 标准差 = [{A[:,0].std():.4f}, {A[:,1].std():.4f}]")
    print(f"    动作的时间变异系数 = {(A.std(axis=0) / (np.abs(A.mean(axis=0)) + 1e-9))}")
    if A.std(axis=0).max() < 0.05:
        print("    ❌ 动作**近似恒定**(std < 0.05) → 准星被恒定推向墙角并卡住")
        print("       这正是「比随机还差」的机制:随机至少会游走,恒定偏置会顶死在边界")
    else:
        print("    → 动作确实在变化,不是恒定偏置")

    # ---- 3. DN 分布的协变量偏移 ----
    print("\n[3] DN 放电率:训练 vs 评估", flush=True)
    tr_dn = ts.X.mean(axis=0)
    ev_dn = np.asarray(dnr)
    print(f"    训练 DN 率均值 = {tr_dn.mean():.4f} Hz  (非零 {int((tr_dn>0).sum())}/{tr_dn.size})")
    print(f"    评估 DN 率均值 = {ev_dn.mean():.4f} Hz")
    nz_tr = tr_dn[tr_dn > 0]
    if nz_tr.size:
        print(f"    训练非零 DN 率的分布: p50={np.median(nz_tr):.4f}  max={nz_tr.max():.4f} Hz")
    print(f"    评估阶段 DN 率范围: [{ev_dn.min():.4f}, {ev_dn.max():.4f}] Hz")
    print()
    print("=" * 88)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
