"""实时查看训练/评估过程(独立进程,读 telemetry.jsonl 刷新)。

用法::

    <捆绑 python> tools/live_view.py --dir <run 目录> [--interval 1.5]

显示 4 格:
    [靶场画面]   [全网 166,700 神经元活动热图(按解剖层次分块)]
    [DN 时序]     [平均靶距 / 训练 R²]

进程只读文件,训练进程不受影响;关掉窗口训练照常进行。
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
from pathlib import Path

import numpy as np


def _unpack(s):
    return np.frombuffer(base64.b64decode(s), np.float16).astype(np.float32)


def _num(s):
    """JSON 里可能是 base64 数组(打包)或标量。返回 np.ndarray 或 float。"""
    if isinstance(s, str):
        return _unpack(s)
    return s

HEADER = None
REC = []
FRAMES = []
DIAGS = []


def read_jsonl(path: Path) -> int:
    """增量读 JSONL,返回新条数。容忍末行被截断。"""
    global HEADER
    n_new = 0
    if not path.exists():
        return 0
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue  # 末行可能正在写
            k = d.get("kind")
            if k == "header":
                HEADER = d
            elif k == "frame":
                REC.append(d)
                n_new += 1
            elif k == "diag":
                DIAGS.append(d)
                n_new += 1
    return n_new


def build(plt):
    """构造图窗,返回句柄 dict。"""
    fig = plt.figure(figsize=(13.5, 8.2))
    try:
        fig.canvas.manager.set_window_title("FlyAim — 实时训练监视")
    except Exception:
        pass
    gs = fig.add_gridspec(2, 2, hspace=0.28, wspace=0.22,
                          left=0.06, right=0.985, top=0.925, bottom=0.08)

    ax_prev = fig.add_subplot(gs[0, 0])
    ax_rast = fig.add_subplot(gs[0, 1])
    ax_dn = fig.add_subplot(gs[1, 0])
    ax_met = fig.add_subplot(gs[1, 1])

    ax_prev.set_title("靶场画面", fontsize=10)
    ax_prev.set_xticks([])
    ax_prev.set_yticks([])
    prev_img = ax_prev.imshow(np.zeros((48, 64, 3), np.uint8), animated=True)

    ax_rast.set_title("全网活动热图(按解剖层次分块)", fontsize=10)
    shape = HEADER.get("sample_shape") if HEADER else [160, 160]
    h, w = int(shape[0]), int(shape[1])
    rast_img = ax_rast.imshow(np.zeros((h, w), np.float32),
                              aspect="auto", cmap="inferno", animated=True,
                              interpolation="nearest")
    ax_rast.set_xticks([])
    ax_rast.set_ylabel("神经元(层序)", fontsize=9)

    ax_dn.set_title("下行神经元放电率(Hz)时序", fontsize=10)
    ax_dn.set_xlabel("帧", fontsize=9)
    ax_dn.set_ylabel("DN 索引", fontsize=9)
    dn_img = ax_dn.imshow(np.zeros((1, 1), np.float32), aspect="auto",
                          cmap="magma", animated=True, interpolation="nearest")

    ax_met.set_title("平均靶距 / 训练 R²", fontsize=10)
    ax_met.set_xlabel("帧", fontsize=9)
    ax_met.set_ylabel("靶距 (px)", fontsize=9)
    (line_dist,) = ax_met.plot([], [], lw=1.8, color="#1f77b4", label="mean_target_dist_px")
    ax_r2 = ax_met.twinx()
    ax_r2.set_ylabel("R²", fontsize=9)
    (line_r2,) = ax_r2.plot([], [], lw=1.6, color="#d62728", ls="--", label="readout R²")
    ax_met.legend(handles=[line_dist, line_r2], loc="upper right", fontsize=8,
                  framealpha=0.7)

    fig.suptitle("等待数据…", fontsize=11)
    return dict(fig=fig, ax_prev=ax_prev, prev_img=prev_img,
                ax_rast=ax_rast, rast_img=rast_img,
                ax_dn=ax_dn, dn_img=dn_img,
                ax_met=ax_met, line_dist=line_dist,
                ax_r2=ax_r2, line_r2=line_r2)


def refresh(h):
    if not REC:
        return
    last = REC[-1]
    n = len(REC)

    # ---- 靶场画面 ----
    png = last.get("frame_png")
    if png:
        from PIL import Image
        with Image.open(Path(TELE_DIR) / png) as im:
            h["prev_img"].set_data(np.asarray(im.convert("RGB")))
            h["prev_img"].set_extent((0, im.width, im.height, 0))

    # ---- 全网热图 ----
    act_raw = last.get("act", "")
    act = _unpack(act_raw) if isinstance(act_raw, str) else np.asarray(act_raw, np.float32)
    if act.size and HEADER:
        k = int(np.prod(HEADER["sample_shape"]))
        if act.size > k:
            full = act[:k].reshape(HEADER["sample_shape"])
            h["rast_img"].set_data(full)
            vmax = max(float(full.max()), 1e-3)
            if vmax > 0:
                h["rast_img"].set_clim(0.0, vmax)
            # 层边界线
            bounds = HEADER.get("layer_bounds") or []
            if bounds and not getattr(h["ax_rast"], "_bounds_drawn", False):
                names = HEADER.get("layer_names") or []
                for bi, b in enumerate(bounds):
                    h["ax_rast"].axhline(b, color="w", lw=0.7, alpha=0.55)
                    if bi < len(names) and bounds[bi] > 0:
                        y = (0 if bi == 0 else bounds[bi - 1] + bounds[bi]) / 2.0
                        h["ax_rast"].text(2, y, names[bi], color="w", fontsize=6.5,
                                          va="center", ha="left", alpha=0.85)
                h["ax_rast"].set_ylim(bounds[-1], 0)
                h["ax_rast"].set_xlim(0, HEADER["sample_shape"][1])
                h["ax_rast"].set_yticks([])
                h["ax_rast"]._bounds_drawn = True

    # ---- DN 时序(保留最近 400 帧) ----
    if act.size and HEADER and HEADER.get("n_dn", 0) > 0:
        dn_all = []
        for r in REC[-400:]:
            a = r.get("act", "")
            arr = _unpack(a) if isinstance(a, str) else np.asarray(a, np.float32)
            if arr.size >= HEADER["n_dn"]:
                dn_all.append(arr[-HEADER["n_dn"]:])
        if dn_all:
            m = np.stack(dn_all)
            h["dn_img"].set_data(m)
            h["dn_img"].set_extent((max(0, n - 400), n, m.shape[0], 0))
            h["dn_img"].set_clim(0.0, max(float(m.max()), 1e-3))

    # ---- 指标曲线 ----
    xs = np.arange(n)
    dists = np.asarray([r.get("meta", {}).get("target_dist",
                                              r.get("mean_target_dist_px", np.nan))
                        for r in REC], np.float64)
    h["line_dist"].set_data(xs, dists)
    h["ax_met"].set_xlim(0, max(40, n))
    ok = np.isfinite(dists)
    if ok.any():
        h["ax_met"].set_ylim(0.0, max(float(dists[ok].max()) * 1.15, 40.0))

    if DIAGS:
        rx = np.asarray([d.get("i", 0) for d in DIAGS], np.float64)
        ry = np.asarray([d.get("r2", np.nan) for d in DIAGS], np.float64)
        h["line_r2"].set_data(rx, ry)
        h["ax_r2"].set_xlim(0, max(40, n))
        ok2 = np.isfinite(ry)
        if ok2.any():
            lo = min(0.0, float(ry[ok2].min()) - 0.05)
            hi = max(1.0, float(ry[ok2].max()) + 0.05)
            h["ax_r2"].set_ylim(lo, hi)

    # ---- 标题 ----
    lm = last.get("layer_mean_hz", {})
    parts = [f"帧 {last.get('i', n - 1)}",
             f"spike_rate={last.get('spike_rate', 0):.4f}",
             f"活跃={last.get('active', 0)}"]
    meta = last.get("meta", {})
    for key in ("target_dist", "hits", "hit_rate"):
        if key in meta:
            v = meta[key]
            parts.append(f"{key}={v:.3f}" if isinstance(v, float) else f"{key}={v}")
    for k in ("ol_sensory", "ol_intrinsic", "visual_projection", "descending_neuron"):
        if k in lm:
            parts.append(f"{k}={lm[k]:.4f}")
    if DIAGS:
        parts.append(f"R²={DIAGS[-1].get('r2', float('nan')):.4f}"
                     f" (n={DIAGS[-1].get('n_samples', '?')})")
    h["fig"].suptitle("  |  ".join(parts), fontsize=10)


def main() -> int:
    global TELE_DIR
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="含 telemetry.jsonl 的 run 目录")
    ap.add_argument("--interval", type=float, default=1.5, help="刷新间隔(秒)")
    ap.add_argument("--no-show", action="store_true", help="只打印状态,不开窗口")
    args = ap.parse_args()

    TELE_DIR = Path(args.dir)
    jsonl = TELE_DIR / "telemetry.jsonl"
    print(f"[live_view] 监视 {jsonl}")
    print(f"[live_view] 等待数据(训练进程启动后自动出现)…", flush=True)

    while not jsonl.exists():
        try:
            import time
            time.sleep(1.0)
        except KeyboardInterrupt:
            return 0
    read_jsonl(jsonl)

    if args.no_show:
        for _ in range(10**9):
            import time
            n = read_jsonl(jsonl)
            if n:
                last = REC[-1] if REC else {}
                print(f"帧 {last.get('i', '?')}  spike={last.get('spike_rate', 0):.4f}  "
                      f"活跃={last.get('active', 0)}  wall={last.get('wall_s', 0):.1f}s",
                      flush=True)
            time.sleep(args.interval)
        return 0

    import matplotlib
    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt

    h = build(plt)
    plt.show(block=False)

    import time
    while True:
        try:
            n = read_jsonl(jsonl)
            if n or not plt.fignum_exists(h["fig"].number):
                if not plt.fignum_exists(h["fig"].number):
                    print("[live_view] 窗口已关闭,退出监视(训练照常进行)")
                    return 0
                refresh(h)
                h["fig"].canvas.draw_idle()
                h["fig"].canvas.flush_events()
            plt.pause(args.interval)
        except KeyboardInterrupt:
            print("\n[live_view] 退出监视(训练照常进行)")
            return 0


TELE_DIR = Path(".")

if __name__ == "__main__":
    raise SystemExit(main())
