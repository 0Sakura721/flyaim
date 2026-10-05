"""从真实 PC HUD 帧里提取数字字形,产出可直接嵌进 score_ocr.py 的 ASCII 种子。

**为什么必须用真实帧**:T24 已证 —— 手画字形(如 `ring`)并不匹配真实 HUD 的
`0`(MSE 超阈)。OCR 的匹配阈值是 0.075,只有**同字体同尺度的真实字形**才靠得住。

输入是本次实盘落下的两张 HUD 裁剪(beat10 与局末),它们是同一次渲染:
    hud_00_b10_g1.png : POINTS=0      TIME=∞  ACC=100%
    hud_01_last.png   : POINTS=10820  TIME=∞  ACC=48%
已知真值 → 可提取的数字字形集合 = {0,1,2,4,8,%}

用法::

    & $py tools/extract_hud_seeds.py
"""
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from flyaim.bridge.score_ocr import _BOX_X, _DIGIT_Y, _GH, _GW, _WHITE  # noqa: E402

RUN = Path(r"E:\flyaim_runs\20261006-024357-d42-hudfix")
# 文件名 -> {框: 该框的真实字符串}
SAMPLES = {
    "hud_00_b10_g1.png": {"points": "0", "acc": "100%"},
    "hud_01_last.png": {"points": "10820", "acc": "48%"},
}


def split_glyphs(img: np.ndarray, box: str) -> list[np.ndarray]:
    x0, x1 = _BOX_X[box]
    reg = img[_DIGIT_Y[0]:_DIGIT_Y[1], x0:x1]
    white = ((reg[:, :, 0] > _WHITE) & (reg[:, :, 1] > _WHITE)
             & (reg[:, :, 2] > _WHITE))
    cols = white.sum(axis=0)
    xs = np.where(cols > 0)[0]
    if xs.size == 0:
        return []
    segs, s = [], int(xs[0])
    for i in range(1, xs.size):
        if int(xs[i]) - int(xs[i - 1]) >= 3:
            segs.append((s, int(xs[i - 1]) + 1))
            s = int(xs[i])
    segs.append((s, int(xs[-1]) + 1))
    out = []
    for a, b in segs:
        g = white[:, a:b].astype(np.uint8) * 255
        g = np.asarray(Image.fromarray(g).resize((_GW, _GH), Image.BILINEAR),
                       dtype=np.float32) / 255.0
        out.append(g)
    return out


def main() -> int:
    # 注意:落盘的是 3x 最近邻放大图,需先缩回原尺寸
    found: dict[str, list[np.ndarray]] = {}
    for fname, truth in SAMPLES.items():
        p = RUN / fname
        if not p.exists():
            print(f"缺 {p}")
            continue
        im = Image.open(p).convert("RGB")
        im = im.resize((im.width // 3, im.height // 3), Image.NEAREST)
        img = np.asarray(im, dtype=np.int32)
        print(f"--- {fname}  {img.shape} ---")
        for box, text in truth.items():
            gs = split_glyphs(img, box)
            print(f"  {box:6s} 期望 {text!r}  实得 {len(gs)} 段"
                  f"  {'✅' if len(gs) == len(text) else '❌ 段数不符,跳过'}")
            if len(gs) == len(text):
                for g, ch in zip(gs, text):
                    found.setdefault(ch, []).append(g)

    print("\n可用字形:", {k: len(v) for k, v in sorted(found.items())})
    np.savez_compressed(ROOT / ".cache" / "hud_seed_glyphs.npz",
                        **{k: np.stack(v) for k, v in found.items()})
    print("已存 .cache/hud_seed_glyphs.npz")

    # 交叉验证:新提取的 0/1 应当能匹配已内嵌的种子(否则说明提取链路错了)
    from flyaim.bridge.score_ocr import _parse_seed, _SEED_0, _SEED_1, _SEED_PCT
    for ch, seed in (("0", _SEED_0), ("1", _SEED_1), ("%", _SEED_PCT)):
        if ch not in found:
            continue
        t = _parse_seed(seed)
        ds = [float(np.mean((g - t) ** 2)) for g in found[ch]]
        print(f"  交叉验证 {ch!r}: 与内嵌种子的 MSE = "
              f"{['%.4f' % d for d in ds]}  (阈值 0.075)")

    # 打印可直接嵌入的 ASCII(仅打印新字形)
    for ch in sorted(c for c in found if c.isdigit()):
        print(f"\n_SEED_{ch} = \"\"\"\\")
        g = found[ch][0]
        for row in g:
            print("".join("#" if v >= 0.5 else "." for v in row))
        print('"""')
    return 0


if __name__ == "__main__":
    sys.exit(main())
