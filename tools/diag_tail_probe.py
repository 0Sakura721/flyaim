"""裁决 130,335 vs 130,709:密集抽取回合末尾的 HUD,看分数到底停在哪。

背景:Lead 用 fps=1 抽帧,在 t≈61s 读到 130,335(00:00/95%),把 t≈62s 那格
记成 None(以为是结算过渡);子代理用全帧率读到 t=62.12s 有 **130,709/96%**,
即**分数在 00:00 之后还在涨**。若子代理对,则用户原始记录 130,709 是对的,
Lead 的"修正"是错的 —— 必须自己验。

做法:0.15s 步长抽 t∈[61.0, 63.2],裁三框 → 2x → 竖排成一张图,一次目检。
"""
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / ".cache" / "video_ref"
VIDEO = Path(r"C:\Users\Admin\Downloads\Screenrecording_20261005_230205.mp4")
FFMPEG = Path(
    r"C:\Users\Admin\.dsh\dsh-runtimes\dsh-primary-runtime\dependencies"
    r"\python\Lib\site-packages\imageio_ffmpeg\binaries\ffmpeg-win-x86_64-v7.1.exe"
)

T0, T1, STEP = 61.00, 63.15, 0.15
CROP = (553, 38, 1047, 64)      # 三框整体(含分隔线)
SCALE = 2


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    tmp = OUT / "tail"
    tmp.mkdir(exist_ok=True)
    for p in tmp.glob("t_*.png"):
        p.unlink()
    times = []
    t = T0
    while t <= T1 + 1e-9:
        times.append(round(t, 3))
        t += STEP
    # 逐个 -ss 精确抽帧(避免 fps 滤镜的相位不确定性)
    for i, ts in enumerate(times):
        subprocess.run(
            [str(FFMPEG), "-hide_banner", "-loglevel", "error", "-y",
             "-ss", f"{ts:.3f}", "-i", str(VIDEO), "-frames:v", "1",
             "-pix_fmt", "rgb24", str(tmp / f"t_{i:03d}.png")],
            check=True)
    cells = []
    for i, ts in enumerate(times):
        p = tmp / f"t_{i:03d}.png"
        if not p.exists():
            continue
        im = Image.open(p).convert("RGB").crop(CROP)
        im = im.resize((im.width * SCALE, im.height * SCALE), Image.LANCZOS)
        cells.append((ts, im))
    cw, ch = cells[0][1].width, cells[0][1].height
    gap = 3
    canvas = Image.new("RGB", (cw + 2 * gap, len(cells) * (ch + gap) + gap),
                       (12, 12, 12))
    for k, (_ts, im) in enumerate(cells):
        canvas.paste(im, (gap, gap + k * (ch + gap)))
    out = OUT / "tail_hud_stack.png"
    canvas.save(out)
    print(f"{len(cells)} 帧,步长 {STEP}s,t∈[{times[0]},{times[-1]}]")
    print(f"竖排图 {canvas.width}x{canvas.height} -> {out}")
    print(f"**第 k 行 = t={T0:.2f}+{STEP}*k 秒**")
    for k, (ts, _im) in enumerate(cells):
        print(f"  行{k:2d}  t={ts:.2f}s")

    # 不依赖 OCR 的客观旁证:白色像素数逐帧
    print("\n每秒白像素数(>190) —— 判断 HUD 何时消失:")
    vals = []
    for i, ts in enumerate(times):
        p = tmp / f"t_{i:03d}.png"
        if not p.exists():
            continue
        a = np.asarray(Image.open(p).convert("RGB").crop(CROP))
        vals.append((ts, int(((a[:, :, 0] > 190) & (a[:, :, 1] > 190)
                              & (a[:, :, 2] > 190)).sum())))
    for ts, v in vals:
        print(f"  t={ts:.2f}  {v:5d}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
