"""Phase B 探测:定位 Aim Lab 窗口 → 截真实画面 → 分析 → 存标注预览图。

运行::

    & $py tools/aimlab_probe.py                 # 自动找 "aimlab" 窗口
    & $py tools/aimlab_probe.py --find "aim lab"
    & $py tools/aimlab_probe.py --list          # 只列窗口不截屏
    & $py tools/aimlab_probe.py --frames 10     # 多抓几帧算稳定帧率

输出:
    - 窗口客户区 (left, top, w, h) —— 直接可作 aimlab_bridge.py --region;
    - 每帧:亮度/饱和度统计、高饱和主色 top5(候选靶色)、
      蓝靶/白准星检测结果、画面年龄;
    - 标注预览图存 .cache/aimlab_probe/*.png(检测框/十字 + 色块直方),
      **请人工目检** —— 检测阈值是否成立由你的眼睛最终裁决。

本工具只读屏幕,不注入任何输入。
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from flyaim.bridge.capture import ScreenCapture, find_window_region, list_windows  # noqa: E402
from flyaim.bridge.detect import DEFAULT_BLUE, annotate, find_target  # noqa: E402

OUT = ROOT / ".cache" / "aimlab_probe"


def dominant_saturated_colors(rgb: np.ndarray, top: int = 5) -> list[tuple[tuple[int, int, int], int]]:
    """高饱和高亮度像素的主色聚类(步长 32 量化),供选靶色参考。"""
    a = rgb.astype(np.float32) / 255.0
    mx, mn = a.max(2), a.min(2)
    sat = np.where(mx > 0, (mx - mn) / np.maximum(mx, 1e-6), 0.0)
    mask = (sat > 0.45) & (mx > 0.45)
    if not mask.any():
        return []
    q = (rgb[mask] // 32 * 32 + 16).astype(np.int64)
    cnt = Counter(map(tuple, q.tolist()))
    return cnt.most_common(top)


def find_white(rgb: np.ndarray) -> tuple[int, int] | None:
    """找近似纯白的最大连通块质心(候选准星)。"""
    a = rgb.astype(np.int16)
    mask = (a.min(2) > 200) & (a.max(2) - a.min(2) < 30)
    if mask.sum() < 4:
        return None
    from scipy import ndimage

    labels, n = ndimage.label(mask)
    sizes = np.bincount(labels.reshape(-1))
    sizes[0] = 0
    best = int(np.argmax(sizes))
    if sizes[best] < 4:
        return None
    yy, xx = np.nonzero(labels == best)
    return float(xx.mean()), float(yy.mean())


def topmost_window_at(x: int, y: int) -> str:
    """返回屏幕点 (x,y) 处最顶层窗口的标题(判断截到的是谁)。"""
    import ctypes
    from ctypes import wintypes

    user32 = ctypes.windll.user32
    pt = wintypes.POINT(int(x), int(y))
    hwnd = user32.WindowFromPoint(pt)
    if not hwnd:
        return "(无)"
    root = user32.GetAncestor(hwnd, 2)  # GA_ROOT
    buf = ctypes.create_unicode_buffer(256)
    user32.GetWindowTextW(root or hwnd, buf, 256)
    return buf.value or "(无标题)"


def probe_window(args: argparse.Namespace) -> int:
    region = find_window_region(args.find)
    l, t, w, h = region
    print(f"窗口客户区: left={l} top={t} w={w} h={h}")
    top = topmost_window_at(l + w // 2, t + h // 2)
    print(f"区域中心最顶层窗口: \"{top}\""
          + ("  ✅ 是目标窗口" if args.find.lower() in top.lower()
             else f"  ⚠️ 不是目标窗口!截到的是别的窗口(如聊天/浏览器),"
                  "请点击 Aim Lab 窗口把它带到最前面"))
    print(f"  -> aimlab_bridge.py 可直接用: --region {l},{t},{w},{h}")

    cap = ScreenCapture(region=region, out_size=None, backend=args.backend)
    OUT.mkdir(parents=True, exist_ok=True)
    times = []
    for i in range(max(1, args.frames)):
        frame, meta = cap.read()
        times.append(meta.get("grab_ms", -1))
        if i % max(1, args.frames // 3) != 0 and i != args.frames - 1:
            continue

        rgb = np.asarray(frame, dtype=np.uint8)
        mean = float(rgb.mean())
        dom = dominant_saturated_colors(rgb)
        blue = find_target(rgb, ref_color=DEFAULT_BLUE, tolerance=60.0)
        white = find_white(rgb)
        white_det = (
            find_target(rgb, ref_color=(240, 240, 240), tolerance=50.0, min_area_px=4)
        )
        ann = annotate(rgb, blue)
        if white_det.ok:
            ann = annotate(ann, white_det, color=(255, 200, 0))
        p = OUT / f"probe_{i:03d}.png"
        from PIL import Image

        Image.fromarray(ann).save(p)

        print(f"\n[帧 {i}] grab={meta.get('grab_ms')}ms mean={mean:.1f} "
              f"shape={rgb.shape[1]}x{rgb.shape[0]}")
        if dom:
            print("  高饱和主色 top5(候选靶色,RGB): "
                  + ", ".join(f"{c}×{n}" for c, n in dom))
        else:
            print("  高饱和主色: 无(可能是菜单页/暗场景/黑帧)")
        print(f"  蓝靶检测: {'ok' if blue.ok else 'miss'}"
              + (f" cx={blue.cx:.0f} cy={blue.cy:.0f} r={blue.radius_px:.0f}" if blue.ok else ""))
        print(f"  白块检测: {'ok' if white_det.ok else 'miss'}"
              + (f" cx={white_det.cx:.0f} cy={white_det.cy:.0f}" if white_det.ok else "")
              + (f"  (质心 {white[0]:.0f},{white[1]:.0f})" if white else ""))
        print(f"  标注图 -> {p}")
    cap.close()
    if times:
        print(f"\ngrab_ms 列表: {times}")
    print("请人工目检标注图:蓝框是否套住靶?黄框是否套住准星?"
          "若靶色不是蓝色,把 top5 主色里最像的一个配进 SeekController(ref_color=...)。")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Aim Lab 窗口/画面探测(只读)")
    ap.add_argument("--find", default="aimlab", help="窗口标题子串(默认 aimlab)")
    ap.add_argument("--list", action="store_true", help="只列出可见窗口")
    ap.add_argument("--frames", type=int, default=6)
    ap.add_argument("--backend", default="auto", choices=["auto", "pil", "mss", "dxcam"])
    args = ap.parse_args()

    if args.list:
        for w in list_windows():
            print(f'  "{w["title"]}"  region=({w["left"]},{w["top"]},{w["width"]},{w["height"]})')
        return 0
    try:
        return probe_window(args)
    except RuntimeError as e:
        print(f"❌ {e}")
        print("提示: --list 查看全部窗口标题,再用 --find \"标题子串\" 指定。")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
