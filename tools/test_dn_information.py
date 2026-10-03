"""决定性检验:DN 放电是否真的承载视觉信息?

Lead 持有。本项目成败的关键问题,必须在标定之后立即回答:

    在这些参数下,DN 的放电模式是否**随视觉刺激变化**?

若 DN 输出与刺激无关(无论画面怎么变,输出都一样),那么:
    - 读出层无论如何训练都无法得到 r2 > 0;
    - 任何"命中率"都只是网络自发活动的副产品;
    - 结论必须是「该配置下连接组不承载可用的视觉信号」,而不是"果蝇不会瞄准"。

检验方法(判别式,不依赖任何学习):
    1. 固定一个静态画面,跑 N 步,记录 DN 稳态放电向量 r_static
    2. 换成另一个静态画面(靶在别处),同样记录 r_other
    3. 计算两者差异 ||r_static - r_other|| / ||r_static||
    4. 再与**同一画面的两次重复**(应有微小差异,作为噪声底)对比

    若 刺激差异 >> 噪声底 → DN 承载视觉信息(可训练)
    若 刺激差异 ≈ 噪声底 → DN 输出与画面无关(不可训练)

运行::

    <捆绑 python> tools/test_dn_information.py
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from flyaim.config import ArenaConfig, BrainConfig, ReadoutConfig, RetinaConfig  # noqa: E402
from flyaim.io import save_roles, load_roles  # noqa: E402
from flyaim.pipeline import FlySystem  # noqa: E402

D = ROOT / "flyaim" / "data" / "build"
NORM = ROOT / "flyaim" / "runs" / "norm"


def make_frames(n: int = 4, seed: int = 0) -> list[np.ndarray]:
    """用靶场渲染几个**视觉上不同**的静态画面。

    做法:让准星走到不同位置(改变画面内容),但都取静止靶的靶场。
    注意这里只用像素——不把坐标喂给网络。
    """
    from flyaim.arena.arena import Arena

    frames = []
    for k in range(n):
        arena = Arena(ArenaConfig(), seed=seed + k)
        f = arena.reset()
        # 推进若干步产生不同的画面状态
        for _ in range(k * 7):
            res = arena.step(np.array([0.0, 0.0], np.float32))
            f = res.frame
        frames.append(f.copy())
    return frames


def steady_dn(fs: FlySystem, frame: np.ndarray, warmup: int = 33, steps: int = 66) -> np.ndarray:
    """给一个固定画面,跑够步数取 DN 稳态平均放电率。"""
    fs.reset()
    dn = fs.roles.descending
    drive = fs.retina.frame_to_spikes(frame)
    for _ in range(warmup // fs.brain.cfg.steps_per_frame + 1):
        for _s in range(fs.brain.cfg.steps_per_frame):
            fs.brain.step(drive[:, 0], drive[:, 1], 1.0)
    acc = np.zeros(dn.size, dtype=np.float64)
    n = 0
    for _ in range(steps):
        fs.brain.step(drive[:, 0], drive[:, 1], 1.0)
        acc += fs.brain.rates[dn]
        n += 1
    return (acc / n).astype(np.float32)


def main() -> int:
    print("=" * 84)
    print("决定性检验:DN 放电是否承载视觉信息")
    print("=" * 84, flush=True)

    # 为归一化产物补 roles.json(Connectome 会在权重同目录查找)
    src_roles = D / "roles.json"
    for sub in ("norm",):
        tgt = ROOT / "flyaim" / "runs" / sub / "roles.json"
        if not tgt.exists() and src_roles.exists():
            shutil.copy(src_roles, tgt)
            print(f"[setup] 复制 roles.json -> {tgt}", flush=True)

    configs = [
        ("raw", D / "connectome.npz", 0.20, 1.5),
        ("indeg", NORM / "connectome_indeg.npz", 8.0, 1.5),
        ("indeg", NORM / "connectome_indeg.npz", 12.0, 1.5),
        ("indeg", NORM / "connectome_indeg.npz", 16.0, 1.5),
    ]

    frames = make_frames(4, seed=0)
    print(f"构造了 {len(frames)} 个视觉不同的静态画面\n", flush=True)

    for norm, wp, w, g in configs:
        if not wp.exists():
            print(f"[skip] {wp} 不存在", flush=True)
            continue
        try:
            fs = FlySystem(D, RetinaConfig(),
                           BrainConfig(weight_scale_exc=w, weight_scale_inh=w, input_gain=g),
                           ReadoutConfig(), weights_override=wp)
        except Exception as e:
            print(f"[{norm} w={w}] 装配失败: {type(e).__name__}: {e}", flush=True)
            continue

        reps = [steady_dn(fs, f) for f in frames]
        # 噪声底:同一画面重复两次
        r_a = steady_dn(fs, frames[0])
        r_a2 = steady_dn(fs, frames[0])

        noise = float(np.linalg.norm(r_a - r_a2))
        baseline = float(np.linalg.norm(r_a)) + 1e-12
        print(f"--- norm={norm} w_scale={w} in_gain={g} ---")
        print(f"    DN 平均放电率 = {np.mean([r.mean() for r in reps]):.4f} Hz, "
              f"活跃 DN 数 = {int((np.asarray(reps).max(axis=0) > 0).sum())}")
        print(f"    重复噪声底 ||Δ|| = {noise:.4f}  (相对 {noise/baseline*100:.2f}%)")

        pairs = []
        for i in range(len(reps)):
            for j in range(i + 1, len(reps)):
                d = float(np.linalg.norm(reps[i] - reps[j]))
                pairs.append(d)
        mean_pair = float(np.mean(pairs))
        print(f"    刺激间差异 ||Δ|| 均值 = {mean_pair:.4f}  (相对 {mean_pair/baseline*100:.2f}%)")
        ratio = mean_pair / max(noise, 1e-9)
        if mean_pair <= noise:
            print(f"    比值 = {ratio:.2f}  ❌ 刺激差异 <= 噪声底 → **DN 不承载视觉信息**")
        elif ratio < 3:
            print(f"    比值 = {ratio:.2f}  ⚠️ 弱可辨(勉强)")
        else:
            print(f"    比值 = {ratio:.2f}  ✅ 刺激差异显著大于噪声 → DN 承载视觉信息")
        print(flush=True)

    print("=" * 84)
    print("解读:比值 < ~3 时,读出层训练出的 r2 会接近 0,任何命中率都不可归因于视觉。")
    print("=" * 84)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
