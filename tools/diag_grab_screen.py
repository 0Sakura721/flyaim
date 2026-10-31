"""抓一张当前屏幕(只读),用于人眼看游戏 UI —— 定位"开始"按钮在哪。

不需要游戏在特定状态;给的是"此刻屏幕上是什么"。用于回答
「R 键按不动一局 Gridshot,那到底该怎么开始」这类问题:
比起猜热键,直接看画面更快也更可靠。
"""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from flyaim.bridge.capture import ScreenCapture, find_window_region  # noqa: E402

OUT = ROOT / ".cache" / "screen_shot"


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    try:
        region = find_window_region("aimlab")
    except Exception as exc:
        print(f"找不到 aimlab 窗口: {exc}")
        return 1
    print(f"窗口客户区 = {region}")
    cap = ScreenCapture(region=region, out_size=None, backend="dxcam",
                        grab_retry=4, grab_retry_ms=4.0)
    frame, meta = cap.read()
    cap.close()
    print(f"抓到帧 {frame.shape}  stale={meta.get('grab_stale')}")
    im = Image.fromarray(np.asarray(frame, dtype=np.uint8))
    full = OUT / "now_full.png"
    im.save(full)
    print(f"整屏 -> {full}")
    # 再存两张下半屏放大图,便于看按钮上的字
    h, w = frame.shape[:2]
    for name, box in (("now_bottom", (0, int(h * 0.55), w, h)),
                      ("now_center", (int(w * 0.20), int(h * 0.15),
                                      int(w * 0.80), int(h * 0.75)))):
        crop = im.crop(box)
        crop = crop.resize((crop.width * 2, crop.height * 2), Image.LANCZOS)
        p = OUT / f"{name}.png"
        crop.save(p)
        print(f"{name} -> {p}  {crop.size}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
