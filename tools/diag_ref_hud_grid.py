"""把理想基准片(手机录屏)的 HUD 按秒裁出来拼成一张网格图 + 落盘人工读数。

为什么这样做:手机片的 HUD 布局与 PC 的 1920x1080 不同,复用手写 OCR 需要重新
标定;而这一轮真正需要的只是**分数随时间的形状**(终局值、增长率、是否有平台)。
每秒一个读数、拼成一张图一次看完,比写一个不可靠的 OCR 便宜得多,也**不会
产生"看着精确的错数字"** —— 这是 D42.4 的教训。

流程:ffmpeg 每秒抽 1 帧 → 裁三框区域 → 2x 放大 → 拼 3 列网格 → **人工目检读数**
→ 读数写进 hud_ground_truth.csv(带 `read_by` 字段标明来源是目检而非 OCR)。

布局(前一轮逐像素测量,见 .cache/video_ref/layout_measured.json):
    三框 x = [557,723] [728,872] [877,1043]  数字行 y = [44,58]

用法::

    & $py tools/diag_ref_hud_grid.py            # 只出网格图 + 白像素旁证
    & $py tools/diag_ref_hud_grid.py --write    # 另外落盘目检读数
"""
import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / ".cache" / "video_ref"
# 地面真值 CSV 落到 **tools/fixtures/**(会被 git 跟踪)而不是 .cache/:
# 参考视频本身不在仓库里,这份 63 点轨迹是**唯一**能长期留存的基准记录。
# 网格 PNG 仍留在 .cache/(1.4MB,可再生产物,不进仓库)。
TRUTH_CSV = ROOT / "tools" / "fixtures" / "ref_hud_ground_truth.csv"
VIDEO = Path(r"C:\Users\Admin\Downloads\Screenrecording_20261005_230205.mp4")
FFMPEG = Path(
    r"C:\Users\Admin\.dsh\dsh-runtimes\dsh-primary-runtime\dependencies"
    r"\python\Lib\site-packages\imageio_ffmpeg\binaries\ffmpeg-win-x86_64-v7.1.exe"
)

CROP = (553, 38, 1047, 64)      # 三框整体(含分隔线)
SCALE = 2
COLS = 3

# 2026-10-06 由 Lead 目检 hud_grid_per_sec.png 读出(3 列 × 21 行,行优先 = 第 0..62 秒)。
# (points, time_s, acc_pct);None = 该格不可读 / 已过结算。
# 自检:点数必须单调不减、TIME 必须逐秒递减、末格应为 130,335 / 00:00。
READINGS: list[tuple[int | None, str | None, int | None]] = [
    (0, "01:00", 100), (0, "01:00", 100), (1490, "00:59", 100),
    (4092, "00:58", 100), (6331, "00:57", 100), (8952, "00:56", 100),
    (11578, "00:55", 100), (14192, "00:54", 100), (16809, "00:53", 100),
    (18472, "00:52", 98), (20506, "00:51", 97), (22738, "00:50", 97),
    (24792, "00:49", 96), (27009, "00:48", 96), (29617, "00:47", 96),
    (31436, "00:46", 97), (33660, "00:45", 97), (35881, "00:44", 97),
    (38498, "00:43", 97), (40724, "00:42", 97), (None, "00:41", 98),   # 该格数字被压糊
    (45957, "00:40", 98), (47808, "00:39", 97), (49679, "00:38", 96),
    (52276, "00:37", 97), (54497, "00:36", 97), (56717, "00:35", 97),
    (58928, "00:34", 97), (60198, "00:33", 96), (62401, "00:32", 97),
    (64617, "00:31", 97), (66845, "00:30", 97), (68878, "00:29", 96),
    (70711, "00:28", 97), (73305, "00:27", 97), (75556, "00:26", 97),
    (77743, "00:25", 97), (79979, "00:24", 97), (82583, "00:23", 97),
    (84815, "00:22", 97), (87037, "00:21", 97), (88280, "00:20", 97),
    (90886, "00:19", 97), (93503, "00:18", 97), (95728, "00:17", 97),
    (97008, "00:16", 97), (99039, "00:15", 96), (101640, "00:14", 97),
    (103885, "00:13", 97), (105919, "00:12", 96), (108146, "00:11", 96),
    (110768, "00:10", 97), (113381, "00:09", 97), (114630, "00:08", 96),
    (116847, "00:07", 96), (119069, "00:06", 96), (121107, "00:05", 96),
    (122191, "00:04", 96), (124806, "00:03", 96), (126456, "00:02", 96),
    (128683, "00:01", 96), (130335, "00:00", 95), (None, None, None),
]


def extract_frames() -> list[Path]:
    OUT.mkdir(parents=True, exist_ok=True)
    tmp = OUT / "per_sec"
    tmp.mkdir(exist_ok=True)
    for p in tmp.glob("s_*.png"):
        p.unlink()
    # fps=1:每秒一帧;ffmpeg 默认应用 display matrix 旋转
    subprocess.run(
        [str(FFMPEG), "-hide_banner", "-loglevel", "error", "-y",
         "-i", str(VIDEO), "-vf", "fps=1", "-pix_fmt", "rgb24",
         str(tmp / "s_%03d.png")],
        check=True,
    )
    return sorted(tmp.glob("s_*.png"))


# --- 网格法本身的已知偏差(2026-10-06 踩过,必须写下来)----------------------
# `ffmpeg -vf fps=1` 有自己的相位:实测 cell k 落在 **t ≈ k + 0.45s**,不是 t = k。
# 后果:cell 61 落在 t≈61.45(分数正好经过 130,335),而**真正的终局值 130,709
# 出现在 t≈61.60,并且 HUD 在 t≈62.2 就消失了 —— 恰好落在 cell 61 与 cell 62
# 之间被采样漏掉**。Lead 因此把 130,335 当成了终局,还据此"纠正"了用户原始
# 记录里的 130,709 —— 那个纠正是错的(DECISIONS D42.11)。
# 教训:**用固定相位抽样去读"末值",必须自己验相位,不能假设它对齐。**
GRID_PHASE_S = 0.45

# --- 密集尾部读数(0.15s 步长,-ss 精确抽帧)-------------------------------
# **终局值的唯一权威来源。** 它给出了网格法漏掉的那几帧,并暴露了
# 「TIME 已经是 00:00,但分数仍在上涨」这个此前没人注意的事实。
TAIL: list[tuple[float, int | None, str | None, int | None]] = [
    (61.00, 129221, "00:00", 95),
    (61.15, 129596, "00:00", 95),
    (61.30, 129971, "00:00", 95),
    (61.45, 130335, "00:00", 95),
    (61.60, 130709, "00:00", 96),
    (61.75, 130709, "00:00", 96),
    (61.90, 130709, "00:00", 96),
    (62.05, 130709, "00:00", 96),
    (62.20, None, None, None),        # HUD 已消失(白像素 807 -> 0)
]
FINAL_POINTS = 130709                 # 权威终局值(见上)
FINAL_ACC = 96


def _mmss(s: str) -> int:
    m, sec = s.split(":")
    return int(m) * 60 + int(sec)


def write_ground_truth(frames: list[Path]) -> None:
    """落盘目检读数(网格 + 密集尾部)+ 自洽性自检。

    两段来源都写进同一个 CSV,用 `source` 列区分:
        grid_fps1  逐秒网格(t ≈ cell + GRID_PHASE_S,±0.5s)
        tail_ss    密集尾部(-ss 精确,权威)
    """
    if len(frames) != len(READINGS):
        print(f"⚠️ 帧数 {len(frames)} != 读数条数 {len(READINGS)},跳过落盘")
        return
    rows = ["source,t_s,points,time_disp,time_s,acc_pct,read_by"]
    for i, (pts, tdisp, acc) in enumerate(READINGS):
        ts = "" if tdisp is None else _mmss(tdisp)
        rows.append(f"grid_fps1,{i + GRID_PHASE_S:.2f},"
                    f"{'' if pts is None else pts},{tdisp or ''},{ts},"
                    f"{'' if acc is None else acc},lead-visual")
    for t, pts, tdisp, acc in TAIL:
        ts = "" if tdisp is None else _mmss(tdisp)
        rows.append(f"tail_ss,{t:.2f},{'' if pts is None else pts},"
                    f"{tdisp or ''},{ts},{'' if acc is None else acc},lead-visual")
    TRUTH_CSV.parent.mkdir(parents=True, exist_ok=True)
    body = "\n".join(rows) + "\n"
    TRUTH_CSV.write_text(body, encoding="utf-8")
    print(f"落盘 {TRUTH_CSV}({len(READINGS)} 网格 + {len(TAIL)} 尾部 = "
          f"{len(rows) - 1} 行数据,来源=目检)")
    (OUT / "hud_ground_truth.csv").write_text(body, encoding="utf-8")

    # ---- 自洽性自检:这些是"读数可信"的必要条件 ----
    pts = [p for p, _, _ in READINGS if p is not None]
    tis = [_mmss(t) for _, t, _ in READINGS if t is not None]
    ok_mono = all(b >= a for a, b in zip(pts, pts[1:]))
    drops = [i for i in range(1, len(tis)) if tis[i] < tis[i - 1]]
    first_drop = drops[0] if drops else len(tis)
    ok_clk = (all(a >= b for a, b in zip(tis, tis[1:]))
              and all(a - b == 1 for a, b in zip(tis, tis[1:]) if a > b))
    print(f"自检 网格点数单调不减: {'✅' if ok_mono else '❌'}  "
          f"({pts[0]} → {pts[-1]})")
    print(f"自检 倒计时钟: {'✅' if ok_clk else '❌'}  "
          f"({tis[0]}s → {tis[-1]}s,共 {len(tis)} 个读数)")
    print(f"自检 **开局前置期**: 前 {first_drop} 秒 HUD 显示 {tis[0]}s 但时钟未起跳")
    # 终局:必须用密集尾部,不能用网格(网格漏掉了它 —— 见 GRID_PHASE_S 说明)
    tail_pts = [p for _, p, _, _ in TAIL if p is not None]
    ok_final = tail_pts and tail_pts[-1] == FINAL_POINTS
    print(f"自检 **终局值来自密集尾部**: {tail_pts[-1] if tail_pts else '?'} "
          f"(期望 {FINAL_POINTS}) {'✅' if ok_final else '❌'}")
    print(f"     ⚠️ 网格法只到 {pts[-1]} —— 它漏掉了 t≈61.6 之后的那一截,"
          f"因为 fps=1 的相位是 +{GRID_PHASE_S}s 而 HUD 在 t≈62.2 消失")
    print(f"     ⚠️ 且 TIME 早已是 00:00 —— **分数在计时结束后仍继续上涨**,"
          f"读「末值」必须多抽几帧,不能只抽到 00:00 就停")
    by = {t: p for p, t, _ in
          [(p, _mmss(tt), a) for p, tt, a in READINGS if p is not None and tt]}
    print("自检 10 秒窗口得分(应大致平稳):")
    for t in (50, 40, 30, 20, 10, 0):
        if t in by and (t + 10) in by:
            print(f"    {t+10:>2}s → {t:>2}s : {by[t] - by[t+10]:>6d} 分")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true", help="落盘目检读数 CSV")
    args = ap.parse_args()

    frames = extract_frames()
    print(f"抽出 {len(frames)} 帧(每秒 1 帧)")
    cells = []
    for i, p in enumerate(frames):
        im = Image.open(p).convert("RGB").crop(CROP)
        cells.append((i, im.resize((im.width * SCALE, im.height * SCALE),
                                   Image.LANCZOS)))
    cw, ch = cells[0][1].width, cells[0][1].height
    pad, label_w = 4, 60
    rows_n = (len(cells) + COLS - 1) // COLS
    canvas = Image.new("RGB", (COLS * (cw + label_w + pad) + pad,
                               rows_n * (ch + pad) + pad), (20, 20, 20))
    for idx, (_sec, im) in enumerate(cells):
        r, c = divmod(idx, COLS)
        canvas.paste(im, (pad + c * (cw + label_w + pad) + label_w,
                          pad + r * (ch + pad)))
    out = OUT / "hud_grid_per_sec.png"
    canvas.save(out)
    print(f"网格图 {canvas.width}x{canvas.height} -> {out}")
    print(f"单元格 {cw}x{ch},共 {len(cells)} 格,{COLS} 列 × {rows_n} 行,"
          f"**行优先 = 第 0,1,2 / 3,4,5 / … 秒**")

    print("\n每秒 HUD 白像素数(>190) —— 不依赖 OCR 的客观旁证,确认数字在变:")
    vals = []
    for p in frames:
        a = np.asarray(Image.open(p).convert("RGB").crop(CROP))
        vals.append(int(((a[:, :, 0] > 190) & (a[:, :, 1] > 190)
                         & (a[:, :, 2] > 190)).sum()))
    for i in range(0, len(vals), 10):
        print("  t=%2d-%2ds: %s" % (i, min(i + 9, len(vals) - 1),
                                    " ".join(f"{v:4d}" for v in vals[i:i + 10])))

    if args.write:
        write_ground_truth(frames)
    return 0


if __name__ == "__main__":
    sys.exit(main())
