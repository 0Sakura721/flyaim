"""实时训练曲线(Web 版):盯战役日志 → 生成自刷新 HTML,浏览器打开即可。

    & $py tools/live_loss_web.py --file .cache/dagger_campaign2.log
    # 然后浏览器打开 .cache/live_loss.html(一次即可,页面每 5s 自刷新)

为什么是 Web 而不是 matplotlib 窗口:D13 的教训——GUI 窗口周期性抢焦点、
CJK 字体缺字;浏览器标签页两者皆无。零依赖:纯 SVG,无 JS 库。
"""

from __future__ import annotations

import argparse
import html
import re
import time
from pathlib import Path

RE_FRAME = re.compile(r"f=(\d+)\s+bc_loss=([\d.]+)")
RE_ROUND = re.compile(r"轮(\d+)\([^)]*\):\s*loss=([\d.]+)")
RE_EVAL = re.compile(r"eval\[([^\]]+)\]\s+seed=(\d+)\s+mean_dist=(\d+)px")
RE_BEST = re.compile(r"最优检查点 @帧(\d+) loss=([\d.]+)")

ARM_COLORS = {"real-dagger": "#2060c0", "shuffle-dagger": "#c05020",
              "real-bc-ckpt": "#8080c0", "shuffle-bc": "#c09060",
              "random": "#20a040", "real-untrained": "#a0a0a0"}


def parse(text: str) -> dict:
    xs, ys = [], []
    rounds: list[tuple[int, float, str]] = []
    evals: dict[str, list[tuple[int, float]]] = {}
    best: list[tuple[str, int, float]] = []
    for line in text.splitlines():
        m = RE_FRAME.search(line)
        if m:
            xs.append((int(m.group(1)), float(m.group(2))))
        m = RE_ROUND.search(line)
        if m:
            rounds.append((int(m.group(1)), float(m.group(2)), m.group(0)))
        m = RE_EVAL.search(line)
        if m:
            evals.setdefault(m.group(1), []).append((int(m.group(2)), float(m.group(3))))
        m = RE_BEST.search(line)
        if m:
            best.append((line.split("]")[0].strip("["), int(m.group(1)), float(m.group(2))))
    return {"xs": xs, "rounds": rounds, "evals": evals, "best": best}


def svg_polyline(points: list[tuple[float, float]], w, h, color) -> str:
    if len(points) < 2:
        return ""
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    x0, x1 = min(xs), max(xs) if max(xs) > min(xs) else min(xs) + 1
    y0, y1 = min(ys), max(ys) if max(ys) > min(ys) else min(ys) + 1
    pad = (y1 - y0) * 0.1 or 0.1
    y0, y1 = y0 - pad, y1 + pad
    pts = " ".join(
        f"{(x - x0) / (x1 - x0) * w:.1f},{h - (y - y0) / (y1 - y0) * h:.1f}"
        for x, y in points)
    return (f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="2"/>')


def render(d: dict) -> str:
    W, H = 560, 260
    loss_pts = [(x, y) for x, y in d["xs"]]
    round_marks = "".join(
        f'<line x1="{x}" y1="0" x2="{x}" y2="{H}" stroke="#bbb" stroke-dasharray="3,3"/>'
        f'<text x="{x + 3}" y="14" font-size="11" fill="#666">{html.escape(lab[:14])}</text>'
        for x, y, lab in
        [(_rel_round(d["xs"], r), y, lab) for r, y, lab in d["rounds"]])
    best_marks = "".join(
        f'<text x="8" y="{20 + 16 * i}" font-size="12" fill="#2060c0">'
        f'✦ {html.escape(arm)}: 帧{f} loss={y:.4f}</text>'
        for i, (arm, f, y) in enumerate(d["best"]))

    import statistics

    eval_scatter = []
    eval_legend = ""
    random_line = ""
    if d["evals"]:
        max_seed = max(max((s for s, _ in pts), default=0) for pts in d["evals"].values()) + 1
        max_d = max(max((v for _, v in pts), default=1) for pts in d["evals"].values()) or 1
        if "random" in d["evals"]:
            r_mean = statistics.mean(v for _, v in d["evals"]["random"])
            ry = H - 20 - r_mean / max_d * (H - 50)
            random_line = (f'<line x1="40" y1="{ry:.0f}" x2="{W - 40}" y2="{ry:.0f}" '
                           f'stroke="#20a040" stroke-dasharray="4,4"/>'
                           f'<text x="{W - 150}" y="{ry - 6:.0f}" font-size="11" '
                           f'fill="#20a040">random 基准 {r_mean:.0f}px</text>')
        for i, (arm, pts) in enumerate(sorted(d["evals"].items())):
            color = ARM_COLORS.get(arm, f"hsl({i * 60 % 360},70%,45%)")
            m = statistics.mean(v for _, v in pts) if pts else 0
            eval_legend += (f'<span style="color:{color};margin-right:14px">'
                            f"■ {html.escape(arm)} ({m:.0f}px)</span>")
            for s, v in sorted(pts):
                cx = 40 + s / max(1, max_seed) * (W - 80)
                cy = H - 20 - v / max_d * (H - 50)
                eval_scatter.append(
                    f'<circle cx="{cx:.0f}" cy="{cy:.0f}" r="4" fill="{color}" '
                    f'fill-opacity="0.75"/>')

    ts = time.strftime("%H:%M:%S")
    return f"""<!DOCTYPE html><html lang="zh"><head><meta charset="utf-8">
<meta http-equiv="refresh" content="5">
<title>FlyAim 实时训练</title></head>
<body style="font-family:'Microsoft YaHei',sans-serif;background:#14181d;color:#dde">
<h2 style="margin:6px">FlyAim 实时训练监控 <span style="color:#678;font-size:14px">刷新于 {ts}</span></h2>
<h3 style="margin:4px 12px">行为克隆 loss</h3>
<svg width="{W + 20}" height="{H}" style="background:#1b2128;border-radius:6px">
{round_marks}{svg_polyline(loss_pts, W, H, "#40a0ff")}
</svg>
<div style="margin:4px 12px">{best_marks}</div>
<h3 style="margin:8px 12px">闭环评估(平均靶距 px,越低越好)</h3>
<div style="margin:2px 12px">{eval_legend}</div>
<svg width="{W + 20}" height="{H}" style="background:#1b2128;border-radius:6px">
{random_line}
{''.join(eval_scatter)}
</svg>
<p style="color:#678;margin:6px 12px">页面每 5s 自刷新;关掉不影响训练。绿虚线 = random 基准均值。</p>
</body></html>"""


def _rel_round(xs, r):
    return xs[0][0] if xs else 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default=".cache/dagger_campaign2.log")
    ap.add_argument("--out", default=".cache/live_loss.html")
    ap.add_argument("--watch", action="store_true", help="持续生成(默认生成一次)")
    args = ap.parse_args()
    path = Path(args.file)
    out = Path(args.out)

    def once():
        if path.exists():
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(render(parse(path.read_text(encoding="utf-8",
                                                          errors="replace"))),
                           encoding="utf-8")
            print(f"已更新 {out}({time.strftime('%H:%M:%S')})", flush=True)

    once()
    if args.watch:
        last = path.stat().st_size if path.exists() else 0
        while True:
            time.sleep(2.0)
            try:
                cur = path.stat().st_size
                if cur != last:
                    last = cur
                    once()
            except FileNotFoundError:
                pass
    print(f"浏览器打开 {out}(若用 --watch 则自动持续更新)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
