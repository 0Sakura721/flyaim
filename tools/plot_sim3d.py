"""实战彩排可视化:模拟 FPS 里复眼看到的画面、它的两张空间图、以及角误差曲线。

三格:
  A) 模拟游戏帧(准星钉在中心、青色靶、世界点阵、红枪模型 —— 与 Aim Lab 同构图)
     + 复眼估计的靶位(observer 侧真值只用于标注,**不回流**)
  B) R1-R6 亮度图 / R7-R8 色觉图并排(青靶只出现在 R8,红准星只出现在 R7)
  C) 角误差时间曲线 + 靶角半径判定带 + 命中/换位时刻

运行: <捆绑 python> tools/plot_sim3d.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import rcParams  # noqa: E402

from flyaim.baselines.eye_servo import make_eye_servo  # noqa: E402
from flyaim.bridge import GainConfig, GainModel  # noqa: E402
from flyaim.bridge.sim3d import SimFPS3DConfig, SimFPS3DSource  # noqa: E402

rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
rcParams["axes.unicode_minus"] = False

DATA_DIR = ROOT / "flyaim" / "data" / "build"
OUT = ROOT / "flyaim" / "runs" / "sim3d-figure.png"
SECONDS_TICKS = 260


def main() -> int:
    gain = GainModel(GainConfig.from_cm360(800, 40, fov_h_deg=103.0))
    sim = SimFPS3DConfig(target_ang_deg=11.0, respawn_on_hit=False)
    src = SimFPS3DSource(sim, gain=gain)
    arm = make_eye_servo(str(DATA_DIR), chroma="r8", aim_mode="center",
                         search="scan")
    arm.reset(0)

    frame, meta = src.read()
    snap_frame, snap_maps, snap_aim = None, None, None
    errs, hits = [], []
    for i in range(SECONDS_TICKS):
        a = arm.act(frame)
        if i == 6:                       # 锁定前后各取一帧
            snap_frame = frame.copy()
            snap_maps = {k: v.copy() for k, v in (arm.last_maps or {}).items()}
            snap_aim = arm._aim
        src.push(a)
        frame, meta = src.read()
        errs.append(meta["err_deg"])
        hits.append(bool(meta["hit"]))
    errs = np.asarray(errs, dtype=float)
    hits = np.asarray(hits, dtype=bool)

    fig = plt.figure(figsize=(15.0, 9.4))
    gs = fig.add_gridspec(2, 3, hspace=0.26, wspace=0.20,
                          left=0.045, right=0.985, top=0.90, bottom=0.07)

    # A) 帧
    ax = fig.add_subplot(gs[0, :2])
    ax.imshow(snap_frame)
    ax.plot([sim.width / 2], [sim.height / 2], marker="+", ms=16, mew=2.4,
            color="#ffd166")
    ax.set_title("A 模拟游戏帧(第 7 拍):准星钉在画面中心(黄 +),动的是相机。"
                 "红枪模型在下方 —— 这就是「红色找不准星」的真实原因", fontsize=10.5)
    ax.set_xticks([]); ax.set_yticks([])

    # B) 两张复眼空间图
    for j, (key, cmap, ttl) in enumerate((
            ("lum", "inferno", "B1 R1-R6 亮度图\n(准星/枪走这条)"),
            ("r7", "cividis", "B2 R7 长波(偏红)\n(红准星、红枪在这)"),
            ("r8", "viridis", "B3 R8 短波(偏青)\n(只有青靶在这 ← 用它瞄准)"))):
        ax = fig.add_subplot(gs[1, j])
        ax.imshow(snap_maps[key], cmap=cmap, interpolation="nearest")
        if key == "r8" and snap_aim is not None:
            ax.plot([(sim.width / 2) / (sim.width / arm.eye_cols)],
                    [(sim.height / 2) / (sim.height / arm.eye_rows)],
                    marker="s", ms=8, mfc="none", mec="#00E5FF", mew=1.8)
        ax.set_title(ttl, fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])

    # C) 角误差
    ax = fig.add_subplot(gs[0, 2])
    ax.axhspan(0, sim.target_ang_deg / 2, color="#2ecc71", alpha=0.18,
               label=f"命中判定(<= {sim.target_ang_deg/2:.1f}°)")
    ax.plot(errs, color="#4dd2ff", lw=1.5, label="角误差")
    ax.set_xlabel("拍(tick)")
    ax.set_ylabel("与靶的角度差 (°)")
    ax.set_title(f"C 角误差(锁定后 {hits[3:].mean()*100:.1f}% 在判定圈内,\n"
                 f"首达第 {int(np.argmax(hits))} 拍)", fontsize=10.5)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=0.25)
    ax.set_ylim(-0.5, max(12.0, float(errs.max()) * 1.05))

    fig.suptitle("投入实战前的彩排:复眼在模拟 FPS 里瞄准(针孔投影 / 准星居中 / "
                 "全域光流 / 注入走 GainModel)", fontsize=13.5)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=130)
    print(f"figure -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
