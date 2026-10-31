"""实时监视 Aim Lab 的 HUD 状态 —— 用来找"怎么把一局开起来"。

背景:靶场里 `TIME = ∞`(TIME 框 1 段),而脚本需要 `MM:SS`(5 段)才算计分局。
实测 **R 键按 7 次无效**,且画面上**没有任何开始按钮/提示文字**,所以自动开局
这条路走不通。与其继续猜热键,不如一边看状态一边试。

**必须用 `python -u` 跑**(stdout 不缓冲),否则长期任务的中间输出看不到 ——
这在 job pwsh-264 上已经吃过一次。

用法::

    & $py -u tools/diag_hud_watch.py --seconds 120
"""
import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from flyaim.bridge.capture import find_window_region  # noqa: E402
from flyaim.bridge.score_ocr import ScoreHUD, hud_region_for, time_glyph_count  # noqa: E402

# TIME 分段数的含义(不依赖任何 OCR 模板)
MEANING = {0: "空白/无 HUD", 1: "练习态 ∞  ← 没开局", 5: "MM:SS 倒计时  ← 计分局!"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=120.0)
    ap.add_argument("--every", type=float, default=0.5)
    args = ap.parse_args()

    region = find_window_region("aimlab")
    bbox = hud_region_for(region)
    print(f"窗口 region={region}  HUD 框={bbox}")
    print("=" * 78)
    print("现在请**在 Aim Lab 里尝试各种开局方式**(空格 / 回车 / 鼠标左键 /")
    print("退回任务菜单重进 …)。哪个动作让下面从 `分段=1` 变成 `分段=5`,")
    print("那个动作就是答案。整行会一直刷新,直到超时。")
    print("=" * 78)

    from flyaim.bridge.capture import ScreenCapture
    left, top, w, h = bbox
    cap = ScreenCapture(region=(left, top, w, h), out_size=None, backend="dxcam",
                        grab_retry=4, grab_retry_ms=4.0)
    hud = ScoreHUD(grab_fn=None)   # 只用于解析;帧由下面注入
    t0 = time.perf_counter()
    last_ng: int | None = None
    n_seen5 = 0
    while time.perf_counter() - t0 < args.seconds:
        el = time.perf_counter() - t0
        try:
            frame, _meta = cap.read()
        except Exception as exc:
            print(f"[{el:6.1f}s] 抓帧失败: {type(exc).__name__}: {exc}", flush=True)
            time.sleep(args.every)
            continue
        img = np.asarray(frame, dtype=np.int32)
        ng = time_glyph_count(img)
        rd = hud.read(img)
        mark = ""
        if last_ng is not None and ng != last_ng:
            mark = f"   <<<< 分段 {last_ng} -> {ng} 变化! "
        if ng == 5:
            n_seen5 += 1
            mark += "  ★★★ 计分局! "
        print(f"[{el:6.1f}s] 分段={ng} ({MEANING.get(ng, '其它/可疑')})  "
              f"POINTS={rd['points']}  TIME={rd['time_s']}  ACC={rd['acc_pct']}"
              f"{mark}", flush=True)
        last_ng = ng
        time.sleep(args.every)
    cap.close()
    print("=" * 78)
    if n_seen5:
        print(f"✅ 期间见到 {n_seen5} 次计分局状态 —— 开局方式有效,可据此接进自动开局")
    else:
        print("❌ 全程未出现计分局状态。请确认是从**任务菜单**进 Gridshot 并点开始,")
        print("   而不是停在某个 ∞ 自由练习态。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
