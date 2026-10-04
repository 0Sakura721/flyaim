"""为 tools/aim_perfect.py 的结果画图:复眼怎么看靶,以及它怎么贴住。

四格:
  A) 靶场帧 + 复眼估计的靶心/准星 + 指令箭头(true 位置只用于标注,不回流)
  B) R1-R6 亮度图 (24x32) + 估计准星      —— 准星的专属探测器
  C) R7-R8 色觉图 (24x32) + 估计靶心      —— 靶的专属探测器
  D) 靶距时间曲线:eye-servo / oracle / random,22px 判定带阴影

只读比赛:图里的 "true" 数据来自 observer 侧(arena 状态),**绝不回流给臂**。

运行::

    <捆绑 python> tools/plot_aim_perfect.py
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

from flyaim.arena.arena import Arena  # noqa: E402
from flyaim.baselines.eye_servo import make_eye_servo  # noqa: E402
from flyaim.config import ArenaConfig  # noqa: E402

# 中文字形(D13:DejaVu Sans 无 CJK 字形,必须显式指定)
rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
rcParams["axes.unicode_minus"] = False

DATA_DIR = ROOT / "flyaim" / "data" / "build"
OUT = ROOT / "flyaim" / "runs" / "aim-perfect-figure.png"


def main() -> int:
    cfg = ArenaConfig()
    cfg.max_frames = 900
    cfg.respawn_on_hit = False

    arm = make_eye_servo(str(DATA_DIR))   # 用类默认参数(标定见 DECISIONS D25)
    arena = Arena(cfg, seed=0)
    frame = arena.reset()
    arm.reset(0)

    dists, snaps = [], {}
    st = arena.get_state()
    for i in range(900):
        act = arm.act(frame)
        st = arena.get_state()
        if i in (0, 3, 7):
            snaps[i] = {
                "frame": frame.copy(),
                "maps": {k: v.copy() for k, v in (arm.last_maps or {}).items()},
                "aim": arm._aim,
                "err": arm.last_err_px,
                "act": np.asarray(act, dtype=float),
                "true_cross": np.asarray(st["crosshair"], dtype=float),
                "true_target": np.asarray(st["targets"], dtype=float).reshape(-1, 2)[0],
            }
        res = arena.step(act)
        frame = res.frame
        dists.append(float(res.target_dist))

    d = np.asarray(dists)
    snap = snaps[3]

    fig = plt.figure(figsize=(15.0, 9.6))
    gs = fig.add_gridspec(2, 3, hspace=0.28, wspace=0.22,
                          left=0.05, right=0.98, top=0.90, bottom=0.07)

    # ---------------- A) 帧 + 标注 ----------------
    ax = fig.add_subplot(gs[0, :2])
    ax.imshow(snap["frame"])
    ax.scatter([snap["true_target"][0]], [snap["true_target"][1]], s=90,
               facecolors="none", edgecolors="#4dd2ff", linewidths=1.6,
               label="true 靶心 (observer 侧)")
    ax.scatter([snap["true_cross"][0]], [snap["true_cross"][1]], s=90,
               facecolors="none", edgecolors="#ffffff", linewidths=1.4,
               label="true 准星 (observer 侧)")
    ax.annotate("", xy=tuple(snap["true_target"]), xytext=tuple(snap["true_cross"]),
                arrowprops=dict(arrowstyle="->", color="#ffd166", lw=2.0))
    ax.set_title("A 靶场帧(第 4 帧):复眼只看到像素;箭头 = 真值误差 "
                 f"|e|={np.linalg.norm(snap['true_target']-snap['true_cross']):.0f}px,"
                 f"复眼估计 {snap['err']:.0f}px", fontsize=11)
    ax.legend(loc="lower right", fontsize=9, framealpha=0.85)
    ax.set_xticks([]); ax.set_yticks([])

    # ---------------- B) 亮度图 ----------------
    ax = fig.add_subplot(gs[0, 2])
    im = ax.imshow(snap["maps"]["lum"], cmap="inferno", interpolation="nearest")
    ax.plot([snap["aim"][0]], [snap["aim"][1]], marker="+", ms=14, mew=2.2,
            color="#7CFC00")
    ax.set_title("B R1-R6 亮度图 24x32\n(绿 + = 复眼估计的准星)", fontsize=11)
    ax.set_xticks([]); ax.set_yticks([])
    plt.colorbar(im, ax=ax, fraction=0.046, label="Hz")

    # ---------------- C) 色觉图 ----------------
    chroma = np.clip(snap["maps"]["r7"] - snap["maps"]["r8"], 0.0, None)
    ax = fig.add_subplot(gs[1, 0])
    im = ax.imshow(chroma, cmap="magma", interpolation="nearest")
    cy, cx = np.unravel_index(np.argmax(chroma), chroma.shape)
    ax.plot([cx], [cy], marker="x", ms=12, mew=2.4, color="#00E5FF")
    ax.set_title("C R7-R8 色觉图 24x32\n(青 x = 复眼估计的靶心)", fontsize=11)
    ax.set_xticks([]); ax.set_yticks([])
    plt.colorbar(im, ax=ax, fraction=0.046, label="Hz")

    # ---------------- D) 靶距曲线 ----------------
    ax = fig.add_subplot(gs[1, 1:])
    ax.axhspan(0, float(cfg.target_radius), color="#2ecc71", alpha=0.18,
               label=f"命中判定圈 (<= {cfg.target_radius}px)")
    ax.plot(d, color="#4dd2ff", lw=1.6, label="eye-servo(6,098 感光细胞)")
    ax.set_xlabel("帧")
    ax.set_ylabel("准星中心到靶心的距离 (px)")
    ax.set_title(f"D 靶距时间曲线:首达第 {int(np.argmax(d <= cfg.target_radius))} 帧,"
                 f"此后稳靶帧占比 {float((d[7:] <= cfg.target_radius).mean())*100:.1f}%"
                 f"(respawn=False, seed=0)", fontsize=11)
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(alpha=0.25)
    ax.set_xlim(0, len(d))
    ax.set_ylim(-3, max(60.0, float(d.max()) * 1.05))

    fig.suptitle("让果蝇百发百中:复眼(6,098 感光细胞)+ 2 维刹停读出 —— 连接组被绕过",
                 fontsize=14)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=130)
    print(f"figure -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
