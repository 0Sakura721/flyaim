"""实时训练曲线查看器:盯战役日志,自动解析 loss/评估点,弹 matplotlib 窗口。

运行(另开一个终端,不影响训练)::

    & $py tools/live_loss.py --file .cache/dagger_campaign2.log

解析的行模式(flyaim 各战役日志通用):
    f=NNNN bc_loss=X.XXXX        (ANN/BC 逐 500 帧)
    轮N(...): loss=X.XXXX        (DAgger 轮)
    eval[臂] seed=N mean_dist=NNNpx hits=K
窗口关闭(Ctrl-C 或点 X)不影响训练。
"""

from __future__ import annotations

import argparse
import re
import time
from pathlib import Path

import matplotlib.pyplot as plt

RE_FRAME = re.compile(r"f=(\d+)\s+bc_loss=([\d.]+)")
RE_ROUND = re.compile(r"轮(\d+)\([^)]*\):\s*loss=([\d.]+)")
RE_EVAL = re.compile(r"eval\[([^\]]+)\]\s+seed=(\d+)\s+mean_dist=(\d+)px")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default=".cache/dagger_campaign2.log")
    ap.add_argument("--interval", type=float, default=2.0)
    args = ap.parse_args()

    path = Path(args.file)
    print(f"盯日志: {path}(每 {args.interval}s 刷新;关窗即退出,不影响训练)")

    plt.ion()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("FlyAim 实时训练监控")

    last_size = 0
    while True:
        if path.exists():
            text = path.read_text(encoding="utf-8", errors="replace")
            if len(text) != last_size:
                last_size = len(text)
                xs, ys = [], []
                rx, ry, rlabel = [], [], []
                evals: dict[str, list[tuple[int, float]]] = {}
                for line in text.splitlines():
                    m = RE_FRAME.search(line)
                    if m:
                        xs.append(int(m.group(1)))
                        ys.append(float(m.group(2)))
                    m = RE_ROUND.search(line)
                    if m:
                        rx.append(int(m.group(1)))
                        ry.append(float(m.group(2)))
                        rlabel.append(f"轮{m.group(1)}")
                    m = RE_EVAL.search(line)
                    if m:
                        evals.setdefault(m.group(1), []).append(
                            (int(m.group(2)), float(m.group(3))))

                ax1.clear()
                if xs:
                    ax1.plot(xs, ys, ".-", color="#2060c0")
                ax1.set_xlabel("帧")
                ax1.set_ylabel("行为克隆 loss")
                ax1.set_title("BC loss")
                if rx:
                    for x, y, lab in zip(rx, ry, rlabel):
                        ax1.axvline(x, color="gray", ls="--", lw=0.8, alpha=0.5)
                        ax1.annotate(lab, (x, ax1.get_ylim()[1]), fontsize=8,
                                     rotation=90, va="top")

                ax2.clear()
                for name, pts in evals.items():
                    pts.sort()
                    ax2.plot([p[0] for p in pts], [p[1] for p in pts], "o",
                             label=name, alpha=0.7)
                ax2.set_xlabel("seed")
                ax2.set_ylabel("闭环平均靶距 px")
                ax2.set_title("评估(越低越好)")
                if evals:
                    ax2.legend(fontsize=8)
                fig.canvas.draw()
                fig.canvas.flush_events()
        plt.pause(args.interval)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        pass
