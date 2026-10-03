"""增益标定:生成 GainConfig JSON(tools/aimlab_bridge.py --gain-json 消费)。

===============================================================================
两条路线
===============================================================================
A. 由 cm/360 推导(推荐,量一次就够):
       1. 游戏里准星对着墙上参照物,鼠标从桌面起点匀速滑动回到几乎同一起点,
          使游戏内视角恰好转一整圈;量这段位移的厘米数 = cm/360;
       2. 查/量你的鼠标 DPI(驱动里看,通常 400/800/1600);
       3. 运行:
              & $py tools/aimlab_gain.py --from-cm360 800 40
          得到 counts_per_360 = 800 * 40 / 2.54 ≈ 12598。

B. 直接给计数(显示器上量):有的转换网站直接给 "counts/360",
   用 --counts-per-360 填进去。

换算与打印(本工具会输出,方便核对):
    deg_per_count        = 360 / counts_per_360
    counts_per_action    = speed_fraction_per_action * counts_per_360
    deg_per_action       = counts_per_action * deg_per_count
    画面比例语义:action=1.0 每拍扫过画面宽度 speed_fraction(默认 14/640,
    与离线靶场一致)。deg/s = deg_per_action * tick_hz —— tick 越快转得越快,
    **必须连同拍频一起报告/设置**。

FOV 参考(--fov 可选打印):水平 FOV 度数除以捕获宽度像素 = 屏幕像素↔角度
的线性近似换算,用于把「检测到的像素误差」翻译成「需要转的角度」。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from flyaim.bridge.gain import GainConfig, GainModel  # noqa: E402

DEFAULT_OUT = ROOT / "flyaim" / "runs" / "bridge" / "gain.json"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="FlyAim 增益标定")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--from-cm360", nargs=2, type=float, metavar=("DPI", "CM360"),
                     help="由鼠标 DPI 与 cm/360 推导")
    src.add_argument("--counts-per-360", type=float, help="直接指定 counts/360")
    ap.add_argument("--speed-fraction", type=float,
                    default=14.0 / 640.0, help="action=1.0 每拍扫过画面宽度比例(默认靶场语义)")
    ap.add_argument("--max-counts", type=float, default=600.0, help="每 tick 计数钳位")
    ap.add_argument("--deadzone", type=float, default=0.0)
    ap.add_argument("--invert-x", action="store_true")
    ap.add_argument("--invert-y", action="store_true")
    ap.add_argument("--fov", type=float, default=None,
                    help="游戏水平 FOV(度),可选,只用于打印像素↔角度表")
    ap.add_argument("--capture-width", type=int, default=640, help="捕获宽度像素")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    kwargs = dict(
        speed_fraction_per_action=args.speed_fraction,
        max_counts_per_tick=args.max_counts,
        invert_x=args.invert_x,
        invert_y=args.invert_y,
        deadzone=args.deadzone,
    )
    if args.from_cm360:
        dpi, cm360 = args.from_cm360
        cfg = GainConfig.from_cm360(dpi, cm360, **kwargs)
        print(f"  输入: {dpi} DPI × {cm360} cm/360")
    else:
        cfg = GainConfig(counts_per_360=args.counts_per_360, **kwargs)  # type: ignore[arg-type]
        print(f"  输入: counts_per_360={args.counts_per_360}")

    g = GainModel(cfg)
    p = g.save(args.out)

    print("\n---- 换算表(核对用) ----")
    print(f"  counts_per_360      = {cfg.counts_per_360:.0f} 计数/整圈")
    print(f"  deg_per_count       = {g.deg_per_count():.5f} °/计数")
    print(f"  speed_fraction      = {cfg.speed_fraction_per_action:.5f} 画面宽/拍"
          f"  (靶场默认 {14.0/640.0:.5f})")
    print(f"  counts_per_action   = {g.counts_per_action:.1f} 计数/满量程 action")
    print(f"  deg_per_action      = {g.deg_per_action():.3f} °/满量程 action")
    print("  不同拍频下的满量程角速度:")
    for hz in (4, 8, 10, 15, 30):
        print(f"    tick={hz:>2} Hz -> {g.deg_per_action() * hz:8.1f} °/s")
    if args.fov:
        dpp = args.fov / max(1, args.capture_width)
        print(f"\n  FOV={args.fov}° / 捕获宽 {args.capture_width}px -> "
              f"{dpp:.4f} °/像素(线性近似)")
        print(f"    半屏误差({args.capture_width // 2}px)≈ {dpp * args.capture_width / 2:.1f}°")
        print("    (透视投影下边缘会偏,中心区域较准 —— 增益层不使用该值,仅供人工核对)")
    print(f"\n  已写入 {p}")
    print("  下一步: tools/aimlab_bridge.py --sink sendinput --gain-json "
          f"\"{p}\"")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
