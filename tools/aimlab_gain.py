"""增益标定:生成 GainConfig JSON(tools/aimlab_bridge.py --gain-json 消费)。

===============================================================================
三条路线(都不必真的拿尺子量桌面滑块距离)
===============================================================================
A. 由 cm/360 推导(如果你已经量过,或者从换算网站抄来):
       1. 游戏里准星对着墙上参照物,鼠标从桌面起点匀速滑动回到几乎同一起点,
          使游戏内视角恰好转一整圈;量这段位移的厘米数 = cm/360;
       2. 查/量你的鼠标 DPI(驱动里看,通常 400/800/1600);
       3. 运行:
              & $py tools/aimlab_gain.py --from-cm360 800 40
          得到 counts_per_360 = 800 * 40 / 2.54 ≈ 12598。

B. 由游戏内灵敏度推导(**推荐起点:不用测量,只用两个游戏里读得到的数**):
              & $py tools/aimlab_gain.py --from-sens 1.5 --engine source
      公式   counts_per_360 = 360 / (sens × yaw_coef)
      ⚠️ **DPI 不进这个式子**。DPI 只决定"手滑 1 厘米 = 多少计数";
      (400DPI, 2 sens) 与 (800DPI, 1 sens) 完全等价 —— 这就是 eDPI 的含义。
      ⚠️ **但"游戏内灵敏度"的语义因引擎而异**,给错系数是系统性偏差,
      而且看起来"能跑"(闭环把误差吸收成速度差)。见 --engine 列表。

C. 直接给计数:转换网站常直接给 "counts/360",用 --counts-per-360 填进去。

   🔁 **两条路线要互为校验**:--from-cm360 与 --from-sens 同时给,本工具会
   把两边对拍并报差异。diff > 5% 说明至少一边错了(常见:引擎系数错、
   标称 DPI 与真实 DPI 不符、游戏内 DPI 缩放、鼠标加速没关)。

===============================================================================
🔴 Aim Lab 特有的坑:DPI 缩放
===============================================================================
Aim Lab 在高 DPI 模式(1600/3200)下会**把计数归一化回 800 基准**:
    3200 -> 每次原始计数 × 0.25,1600 -> × 0.5
此时 `360/(0.022·sens)` **不成立**,要把缩放乘回去:
              & $py tools/aimlab_gain.py --from-sens 1.0 --engine aimlab --dpi-scale 0.25
       或用 --aimlab-dpi 3200 让工具自动补。
不处理这一项,注入的转动会**过转 2~4 倍**,靶一拍就飞过中心 ——
这正是 D19/D20「视角一开跑就转离靶区、全程 staring at wall」的成因形态。

===============================================================================
换算与打印(本工具会输出,方便核对)
===============================================================================
    deg_per_count     = 360 / counts_per_360
    deg_per_action    = speed_fraction_per_action * linFOV      (给了 --fov)
                      = speed_fraction_per_action * 360        (没给 --fov,旧语义)
    linFOV            = 2·tan(FOV/2)   —— 把"扫过画面宽度的比例"换算成角度
    counts_per_action = deg_per_action * counts_per_360 / 360

    deg/s = deg_per_action * tick_hz —— tick 越快转得越快,
    **必须连同拍频一起报告/设置**。

🔴 `--fov` **不是可选打印项,是语义的一部分**(2026-10-04 由模拟 FPS 复查发现):
    action=1.0 的靶场语义是「靶在画面里移动 14px = 画面宽度的 2.19%」。
    不给 FOV 时本工具按旧实现把它当成「整圈的 2.19%」= 7.875°/拍;
    而 FOV=103° 的正确值是 2.19% × linFOV(144.06°) = 3.151°/拍。
    **差 2.50 倍** —— 一拍就把靶推过中心 35px(而不是 14px)。
    这正是 D19/D20 记录的「视角一开跑就转离靶区、全程 staring at wall」的
    直接成因之一。**上线务必显式给 --fov。**
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from flyaim.bridge.gain import (  # noqa: E402
    ENGINE_COEFS,
    GainConfig,
    GainModel,
    SensModel,
)

DEFAULT_OUT = ROOT / "flyaim" / "runs" / "bridge" / "gain.json"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="FlyAim 增益标定")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--from-cm360", nargs=2, type=float, metavar=("DPI", "CM360"),
                     help="路线 A:由鼠标 DPI 与 cm/360 推导")
    src.add_argument("--from-sens", type=float, metavar="SENS",
                     help="路线 B:由游戏内灵敏度推导(**不用桌面上量**)")
    src.add_argument("--counts-per-360", type=float,
                     help="路线 C:直接指定 counts/360(换算网站常直接给)")
    ap.add_argument("--engine", default="source", choices=sorted(ENGINE_COEFS),
                    help=f"路线 B 的引擎系数来源(默认 source);可选 {sorted(ENGINE_COEFS)}")
    ap.add_argument("--yaw-coef", type=float, default=None,
                    help="路线 B:直接给「每计数转过多少度」,给了则忽略 --engine")
    ap.add_argument("--dpi-scale", type=float, default=None,
                    help="路线 B:计数进引擎前的缩放(见文件头 Aim Lab 的坑)")
    ap.add_argument("--aimlab-dpi", type=float, default=None,
                    help="路线 B:填 Aim Lab 界面上的 DPI,工具自动补 --dpi-scale"
                         "(800→1.0, 1600→0.5, 3200→0.25)")
    ap.add_argument("--cross-check-dpi", type=float, default=800.0,
                    help="交叉校验用的 DPI(仅影响 cm/360 显示口径,默认 800)")
    ap.add_argument("--verify-counts", type=float, default=None,
                    help="实机校验:注入该计数后量到的等效 counts/360,与生成值对拍")
    ap.add_argument("--speed-fraction", type=float,
                    default=14.0 / 640.0, help="action=1.0 每拍扫过画面宽度比例(默认靶场语义)")
    ap.add_argument("--max-counts", type=float, default=600.0, help="每 tick 计数钳位")
    ap.add_argument("--deadzone", type=float, default=0.0)
    ap.add_argument("--invert-x", action="store_true")
    ap.add_argument("--invert-y", action="store_true")
    ap.add_argument("--fov", type=float, default=None,
                    help="游戏水平 FOV(度)。**强烈建议给**;不给则退回旧语义"
                         "(deg_per_action = speed_fraction×360),FOV≈103° 时会过转 2.5 倍")
    ap.add_argument("--capture-width", type=int, default=640, help="捕获宽度像素")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    return ap.parse_args()


def _resolve_dpi_scale(args: argparse.Namespace) -> tuple[float, str]:
    """把 --aimlab-dpi / --dpi-scale 归一到 (scale, 说明)。"""
    if args.dpi_scale is not None:
        return float(args.dpi_scale), f"显式 --dpi-scale {args.dpi_scale}"
    if args.aimlab_dpi is not None:
        a = float(args.aimlab_dpi)
        if a <= 800.0:
            return 1.0, f"Aim Lab DPI={a:g} ≤800 → 不缩放"
        s = 800.0 / a
        return s, (f"Aim Lab DPI={a:g} >800 → 计数被归一化 ×{s:g} "
                   f"(不补会过转 {1/s:g} 倍)")
    return 1.0, "未指定 Aim Lab DPI,按不缩放"


def _route_b(args: argparse.Namespace) -> tuple[float, SensModel, str]:
    scale, scale_note = _resolve_dpi_scale(args)
    sm = SensModel(sens=args.from_sens, yaw_coef=(args.yaw_coef
                  if args.yaw_coef is not None
                  else ENGINE_COEFS[args.engine]["yaw_coef"]),
                   dpi_scale=scale)
    src = (f"yaw_coef={sm.yaw_coef} (--yaw-coef)"
           if args.yaw_coef is not None else f"engine={args.engine}")
    return sm.counts_per_360(), sm, f"{src}; {scale_note}"


def main() -> int:
    args = parse_args()
    kwargs = dict(
        speed_fraction_per_action=args.speed_fraction,
        fov_h_deg=args.fov,
        max_counts_per_tick=args.max_counts,
        invert_x=args.invert_x,
        invert_y=args.invert_y,
        deadzone=args.deadzone,
    )
    sm: SensModel | None = None
    if args.from_cm360:
        dpi, cm360 = args.from_cm360
        cfg = GainConfig.from_cm360(dpi, cm360, **kwargs)
        print(f"  路线 A: {dpi} DPI × {cm360} cm/360")
    elif args.from_sens is not None:
        c, sm, note = _route_b(args)
        cfg = GainConfig(counts_per_360=c, **kwargs)
        print(f"  路线 B: sens={args.from_sens:g}  ({note})")
        print(f"           counts_per_360 = 360/({sm.yaw_deg_per_count:.6f}°/计数"
              f" × {sm.dpi_scale:g}) = {c:.1f}")
        print("           注意 DPI 不进这个式子 —— 它只决定手滑 1cm 产生多少计数")
    else:
        cfg = GainConfig(counts_per_360=args.counts_per_360, **kwargs)  # type: ignore[arg-type]
        print(f"  路线 C: counts_per_360={args.counts_per_360}")

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
        import math as _m
        f_px = (args.capture_width / 2.0) / _m.tan(_m.radians(args.fov) / 2.0)
        ppp = f_px * _m.pi / 180.0
        print(f"\n  针孔核对:FOV={args.fov}° / 捕获宽 {args.capture_width}px")
        print(f"    f = {f_px:.1f} px/rad;中心处 {ppp:.3f} px/°")
        print(f"    action=1 想移动 14px -> 应转 {14.0 / ppp:.3f}°"
              f"(本工具给出的 {g.deg_per_action():.3f}° —— 应对得上)")
        print(f"    linFOV = 2·tan(FOV/2) = {cfg.lin_fov_deg:.2f}°"
              f"(= 画面宽度换算成角度的线性化值)")
        print("    (透视投影下边缘 px/度 更大,中心区域最准;实测见 tools/aimlab_sim3d.py §5)")
    else:
        old = 0.021875 * 360.0
        print("\n  🔴 未指定 --fov:deg_per_action 用的是旧语义"
              "(speed_fraction×360)。")
        print(f"     若 FOV=103°,正确值应为 {0.021875 * 144.06:.3f}°/拍,"
              f"现在是 {old:.3f}°/拍 = 过转 {old / (0.021875 * 144.06):.2f} 倍。")
        print("     请加 --fov <你的游戏 FOV> 后重新生成 gain.json。")

    # -- 交叉校验:路线 A/B 互拍 -------------------------------------------
    if args.from_sens is not None and sm is not None:
        cdpi = float(args.cross_check_dpi)
        cm_eq = cfg.counts_per_360 * 2.54 / cdpi
        print(f"\n---- 互换核对(按 {cdpi:g} DPI 口径) ----")
        print(f"  路线 B 给出的等效 cm/360 = {cm_eq:.2f} cm"
              f"(社区数据通常按 800 DPI 报,换算成其他 DPI 时 cm 同比变化)")
        print("  ⚠️ 这类数字**依赖 DPI 口径**,不同来源对比前先统一 DPI。")
        print("  想要硬校验,用 --verify-counts 报实机量的值:")
        print("    注入 N 个计数 -> 量准星移过画面宽度百分比 p")
        print("    等效 counts/360 = N * 360 / (p * FOV角宽)")
    if args.verify_counts is not None and args.verify_counts > 0:
        v = float(args.verify_counts)
        ratio = cfg.counts_per_360 / v
        print(f"\n---- 实机校验 ----")
        print(f"  生成值 {cfg.counts_per_360:.0f} vs 实机量到 {v:.0f}"
              f" -> 比值 {ratio:.3f}")
        if abs(ratio - 1.0) <= 0.05:
            print("  ✅ 差异 ≤5%,标定可信,可直接上线。")
        else:
            pc = (ratio - 1.0) * 100.0
            print(f"  🔴 差异 {pc:+.1f}%,**不要上线**。按下面顺序查:")
            print("     1) 鼠标加速/角度捕捉是否真的关了(最常见)")
            print("     2) 引擎系数是否配对(--engine / --yaw-coef)")
            print("     3) Aim Lab DPI 缩放(--aimlab-dpi)")
            print("     4) 鼠标标称 DPI 与真实 DPI 的偏差")
            print(f"     → 以实机值为准:--counts-per-360 {v:.0f}")

    print(f"\n  已写入 {p}")
    print("  下一步: tools/aimlab_bridge.py --sink sendinput --gain-json "
          f"\"{p}\"")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
