"""曲线绘制:学习曲线(acq_curve)与命中率对比,供 Phase 3 报告使用。

只依赖 matplotlib(Agg 后端,无显示环境也能跑)。所有图内文字用英文,
避免缺中文字体导致方块(报告正文里再写中文说明)。

用法::

    from flyaim.arena.plot import plot_acq_curves, plot_hit_rate_bars
    plot_acq_curves({"pid": [...10 个数...], "random": [...]}, "runs/x/acq.png")
    plot_hit_rate_bars({"pid": summary_pid, "random": summary_random}, "runs/x/bars.png")

    # 或者直接跑一个最小 demo(真跑 pid / random,不造假数据):
    python -m flyaim.arena.plot
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

__all__ = [
    "plot_acq_curves",
    "plot_hit_rate_bars",
    "make_report_figure",
    "demo",
]

# 各臂固定配色(报告里颜色一致,便于人眼比对)
_ARM_COLORS = {
    "fly": "#1f77b4",
    "shuffle": "#ff7f0e",
    "pid": "#2ca02c",
    "random": "#d62728",
}


def _color(name: str, i: int) -> str:
    return _ARM_COLORS.get(str(name), plt.get_cmap("tab10")(i % 10))


def plot_acq_curves(
    curves: Mapping[str, Sequence[float]] | Sequence[float],
    out_path: str | Path,
    title: str = "Acquisition curve: hit rate per bin (does it get better over time?)",
    xlabel: str = "Episode progress (bin, equal-length)",
    ylabel: str = "Hit rate within bin",
    dpi: int = 150,
    show_mean_line: bool = True,
) -> Path:
    """画分箱学习曲线。

    Parameters
    ----------
    curves:
        ``{arm: [bin 命中率, ...]}``;也可以直接给一条曲线(``list[float]``)。
    out_path:
        输出 PNG 路径(父目录自动创建)。
    """
    if isinstance(curves, Mapping):
        items = list(curves.items())
    else:
        items = [("curve", list(curves))]

    fig, ax = plt.subplots(figsize=(8.0, 4.5), dpi=dpi)
    for i, (name, curve) in enumerate(items):
        y = np.asarray(list(curve), dtype=np.float64)
        x = np.arange(y.size)
        c = _color(str(name), i)
        ax.plot(x, y, marker="o", ms=4, lw=1.8, color=c, label=f"{name} (n={y.size} bins)")
        if show_mean_line and y.size:
            ax.axhline(float(y.mean()), color=c, ls=":", lw=1.0, alpha=0.7)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p)
    plt.close(fig)
    return p


def plot_hit_rate_bars(
    summaries: Mapping[str, dict],
    out_path: str | Path,
    title: str = "Hit rate by arm (same arena seeds / same frame budget)",
    dpi: int = 150,
    errors: Mapping[str, float] | None = None,
    order: Sequence[str] | None = None,
    annotate_counts: bool = True,
) -> Path:
    """画各臂命中率柱状图(可选误差棒)。

    Parameters
    ----------
    summaries:
        ``{arm: Metrics.summary()}``。误差棒优先取 summary 里的
        ``hit_rate_se``,其次 ``hit_rate_std``;也可用 ``errors`` 显式覆盖。
    errors:
        ``{arm: 误差}``,显式指定误差棒(优先于 summary 里的键)。
    order:
        柱子顺序(默认按 mapping 顺序)。
    """
    names = list(order) if order is not None else list(summaries.keys())
    rates = [float(summaries[n].get("hit_rate", 0.0)) for n in names]
    errs = []
    for n in names:
        if errors is not None and n in errors:
            errs.append(float(errors[n]))
        else:
            s = summaries[n]
            if "hit_rate_se" in s:
                errs.append(float(s["hit_rate_se"]))
            elif "hit_rate_std" in s:
                errs.append(float(s["hit_rate_std"]))
            else:
                errs.append(0.0)

    fig, ax = plt.subplots(figsize=(7.0, 4.5), dpi=dpi)
    x = np.arange(len(names))
    colors = [_color(n, i) for i, n in enumerate(names)]
    bars = ax.bar(x, rates, yerr=errs if any(errs) else None, capsize=4, color=colors, alpha=0.85)
    for i, (b, n) in enumerate(zip(bars, names)):
        s = summaries[n]
        txt = f"{rates[i]:.4f}"
        if annotate_counts and "hits" in s and "frames" in s:
            txt += f"\n{int(s['hits'])}/{int(s['frames'])}"
        ax.annotate(
            txt,
            (b.get_x() + b.get_width() / 2, b.get_height()),
            ha="center",
            va="bottom",
            fontsize=8,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=9)
    ax.set_ylabel("hit_rate (hits / frames)", fontsize=9)
    ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.25, axis="y")
    top = max(rates + [0.0]) * 1.35 + 1e-6
    ax.set_ylim(0, top)
    fig.tight_layout()
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p)
    plt.close(fig)
    return p


def make_report_figure(
    curves: Mapping[str, Sequence[float]],
    summaries: Mapping[str, dict],
    out_path: str | Path,
    dpi: int = 150,
) -> Path:
    """把学习曲线 + 命中率柱状图合成一张报告用图。"""
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.6), dpi=dpi)

    for i, (name, curve) in enumerate(curves.items()):
        y = np.asarray(list(curve), dtype=np.float64)
        axes[0].plot(np.arange(y.size), y, marker="o", ms=4, lw=1.8,
                     color=_color(str(name), i), label=name)
    axes[0].set_title("Acquisition curve (hit rate per bin)", fontsize=10)
    axes[0].set_xlabel("bin", fontsize=9)
    axes[0].set_ylabel("hit rate within bin", fontsize=9)
    axes[0].grid(alpha=0.25)
    axes[0].legend(fontsize=8)

    names = list(summaries.keys())
    rates = [float(summaries[n].get("hit_rate", 0.0)) for n in names]
    errs = [float(summaries[n].get("hit_rate_se", summaries[n].get("hit_rate_std", 0.0)))
            for n in names]
    xs = np.arange(len(names))
    axes[1].bar(xs, rates, yerr=errs if any(errs) else None, capsize=4,
                color=[_color(n, i) for i, n in enumerate(names)], alpha=0.85)
    for xi, r in zip(xs, rates):
        axes[1].annotate(f"{r:.4f}", (xi, r), ha="center", va="bottom", fontsize=8)
    axes[1].set_xticks(xs)
    axes[1].set_xticklabels(names, fontsize=9)
    axes[1].set_ylabel("hit_rate", fontsize=9)
    axes[1].set_title("Hit rate by arm", fontsize=10)
    axes[1].grid(alpha=0.25, axis="y")
    axes[1].set_ylim(0, max(rates + [0.0]) * 1.35 + 1e-6)

    fig.tight_layout()
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p)
    plt.close(fig)
    return p


def demo(
    out_dir: str | Path | None = None,
    seeds: Sequence[int] = (0, 1, 2),
    frames: int = 900,
    n_bins: int = 10,
    fps: float = 30.0,
) -> dict:
    """真跑 pid / random 两个臂(不造假数据),产出两张 PNG。

    Returns
    -------
    dict
        ``{"acq_png": Path, "bars_png": Path, "report_png": Path,
           "summaries": {...}, "curves": {...}}``
    """
    from flyaim.arena.rollout import as_act_fn, run_episode
    from flyaim.arena.arena import Arena
    from flyaim.baselines.pid import PIDBaseline
    from flyaim.baselines.random import RandomBaseline
    from flyaim.config import ArenaConfig

    root = Path(out_dir) if out_dir is not None else Path(__file__).resolve().parent
    cfg = ArenaConfig()
    summaries: dict[str, dict] = {}
    curves: dict[str, list[float]] = {}

    for arm in ("pid", "random"):
        rates, curve_stack, hits, dists = [], [], [], []
        for s in seeds:
            controller = PIDBaseline(cfg) if arm == "pid" else RandomBaseline(seed=int(s))
            arena = Arena(cfg, int(s))
            m, _ = run_episode(
                arena, as_act_fn(controller, "state"),
                max_frames=frames, n_bins=n_bins, fps=fps,
            )
            sm = m.summary()
            rates.append(sm["hit_rate"])
            curve_stack.append(sm["acq_curve"])
            hits.append(sm["hits"])
            dists.append(sm["mean_target_dist_px"])
        arr = np.asarray(curve_stack, dtype=np.float64)
        summaries[arm] = {
            "hit_rate": float(np.mean(rates)),
            "hit_rate_std": float(np.std(rates, ddof=1)) if len(rates) > 1 else 0.0,
            "hit_rate_se": (
                float(np.std(rates, ddof=1) / np.sqrt(len(rates))) if len(rates) > 1 else 0.0
            ),
            "hits": int(np.sum(hits)),
            "frames": int(frames * len(seeds)),
            "mean_target_dist_px": float(np.mean(dists)),
        }
        curves[arm] = [float(v) for v in arr.mean(axis=0)]

    acq = plot_acq_curves(curves, root / "acq_curve.png")
    bars = plot_hit_rate_bars(summaries, root / "hit_rate_comparison.png")
    rep = make_report_figure(curves, summaries, root / "report_panel.png")
    return {"acq_png": acq, "bars_png": bars, "report_png": rep,
            "summaries": summaries, "curves": curves}


def main() -> None:
    out = demo()
    print("[plot] summaries:")
    for arm, s in out["summaries"].items():
        print(f"  {arm:7s} hit_rate={s['hit_rate']:.4f} +/- {s['hit_rate_se']:.4f} (se)")
    for k in ("acq_png", "bars_png", "report_png"):
        print(f"[plot] {k}: {Path(out[k]).resolve()}")


if __name__ == "__main__":
    main()
