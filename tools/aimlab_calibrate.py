"""端到端定标向导:把「三个数 + 鼠标」变成一份可信的 gain.json。

===============================================================================
它在整条链里的位置
===============================================================================
    [你给 3 个数]  --(sens / DPI / FOV)-->  counts_per_360(公式)
                                              |
    [系统指针路径] --(注入计数->读光标)-->  指针增益 / 线性性(实测)
                                              |
    [游戏内]       --(可选 --verify-counts)--> 游戏内计数(只手测一次)
                                              v
                                        gain.json(带置信度报告)

前两层本工具全自动,第三层可选。**跑完会打印一份"能不能上线"的判定**。

第 2 层用的是 **光标位移探针**(不是 Raw Input):因为鼠标加速(EPP)的失真
发生在"注入 -> 光标"这段路径上,而 Raw Input 在 EPP 之前取值,**永远读回 1.0,
等于把这个失真遮盖掉**。探针会在屏幕中心注入几组不同幅度的计数,看比值是否
恒定 —— EPP 的非线性只能靠"换速度再测"抓出来,单点测量必然漏掉。

===============================================================================
用法(最小)
===============================================================================
    & $py tools/aimlab_calibrate.py --sens 1.5 --aimlab-dpi 800 --fov 103
    & $py tools/aimlab_calibrate.py --edpi 1200 --dpi 800 --fov 103
    & $py tools/aimlab_calibrate.py --cm360 34.64 --dpi 800 --fov 103   # 你已量过

    # 探针默认就开;不想动鼠标就用 --no-probe
    & $py tools/aimlab_calibrate.py --sens 1.5 --aimlab-dpi 800 --fov 103 --no-probe

⚠️ 探针会移动你的鼠标(几百毫秒),测完自动移回原位。它不点击、不按键。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from flyaim.bridge.gain import ENGINE_COEFS, GainConfig, GainModel  # noqa: E402
from flyaim.bridge.inject import SendInputSink, pointer_accel_enabled  # noqa: E402
from flyaim.bridge.pointer_probe import judge as probe_judge  # noqa: E402
from flyaim.bridge.pointer_probe import run_probe  # noqa: E402

DEFAULT_OUT = ROOT / "flyaim" / "runs" / "bridge" / "gain.json"

#: 探针注入幅度(计数)。**必须多幅度**:EPP 的非线性只能靠"换速度再测"抓出来,
#: 单点测量会给出一个看似正常的比值。
PROBE_MAGNITUDES = (300, 1200, 300)


def _pass(msg: str) -> None:
    print(f"  PASS  {msg}")


def _fail(msg: str) -> None:
    print(f"  FAIL  {msg}")


def _warn(msg: str) -> None:
    print(f"  WARN  {msg}")


def _probe_hardware(
    magnitudes: tuple[int, ...], settle_s: float
) -> list[dict]:
    """在屏幕中心附近注入多组计数,读回光标实际位移。

    ⚠️ 这里量的是 **Windows 桌面指针路径**(EPP 之后),不是 Raw Input。
    选它的理由:EPP 的失真恰恰发生在这个路径上;而 Raw Input 在 EPP 之前
    取值,**永远读回 1.0,等于把这个失真遮盖掉**。
    """
    sink = SendInputSink()
    rows, _ox, _oy = run_probe(sink, magnitudes, axis="x",
                               settle_s=settle_s, restore=True)
    return [r.describe() for r in rows]


def _judge_probe(rows: list[dict], tol: float) -> tuple[bool, list[str]]:
    """判定探针结果。返回 (ok, 结论文本行)。"""
    lines: list[str] = []
    if not rows:
        return False, ["  没拿到任何探针数据"]
    print_rows = []
    ratios: list[float] = []
    for r in rows:
        m = r["moved"]
        s = r["sent"]
        rx = r["ratio_x"] if r.get("ratio_x") is not None else r.get("ratio_y")
        if rx is not None:
            ratios.append(rx)
        print_rows.append(
            f"    幅度 {abs(s[0] or s[1]):>5}: 注入 ({s[0]:>5},{s[1]:>4}) -> "
            f"光标 ({m[0]:>6},{m[1]:>4})  比值 "
            + (f"{rx:.4f}" if rx is not None else "n/a")
        )
    if not ratios:
        return False, print_rows + ["  🔴 没有有效比值 —— 注入通道可能不通"]

    lines.extend(print_rows)
    lo, hi = min(ratios), max(ratios)
    span = hi - lo
    mean = sum(ratios) / len(ratios)
    ok = True

    if all(r["moved"] == [0, 0] for r in rows):
        ok = False
        lines.append("  🔴 光标完全没动 —— 注入未生效")
        return ok, lines
    lines.append("  ✅ 注入生效:光标确实移动")

    if span > tol:
        ok = False
        lines.append(
            f"  🔴 比值随幅度变化(区间 [{lo:.4f},{hi:.4f}],跨度 {span:.4f} > {tol})"
            " = **非线性**:鼠标加速/角度捕捉确实在改写注入量。"
        )
        lines.append(
            "     → 关掉「提高指针精确度」后重跑(控制面板→鼠标→指针选项)。"
        )
        lines.append(
            "     → 若游戏用 Raw Input(多数 FPS),它不吃 EPP,游戏内不一定受影响;"
            "但桌面指针会,且歧义会污染归因,建议关掉。"
        )
    else:
        lines.append(f"  ✅ 比值恒定(跨度 {span:.4f} ≤ {tol}):指针路径线性")

    if abs(mean - 1.0) > tol:
        lines.append(
            f"  ⚠️ 平均比值 {mean:.4f} ≠ 1(偏 {(mean-1)*100:+.1f}%):"
            "指针路径有整体缩放(EPP 线性段或系统 DPI 缩放)。"
        )
        lines.append(
            "     → 游戏若吃 EPP,请在游戏内用 --verify-counts 实量,并以实机值为准。"
        )
    else:
        lines.append(f"  ✅ 平均比值 {mean:.4f} ≈ 1:无整体缩放")
    return ok, lines


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="FlyAim 端到端定标向导(sens/cm → gain.json + 硬件校验)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--sens", type=float, help="游戏内灵敏度(推荐起点)")
    src.add_argument("--edpi", type=float, help="eDPI(=DPI×sens);需配合 --dpi")
    src.add_argument("--cm360", type=float, help="已知的 cm/360(需配合 --dpi)")
    src.add_argument("--counts-per-360", type=float, help="直接给计数")

    ap.add_argument("--dpi", type=float, default=800.0,
                    help="鼠标标称 DPI(--edpi/--cm360 必需;默认 800)")
    ap.add_argument("--engine", default="aimlab", choices=sorted(ENGINE_COEFS),
                    help=f"灵敏度语义来源(默认 aimlab);可选 {sorted(ENGINE_COEFS)}")
    ap.add_argument("--yaw-coef", type=float, default=None,
                    help="直接给每计数转过的度,给了则忽略 --engine")
    ap.add_argument("--aimlab-dpi", type=float, default=None,
                    help="Aim Lab 界面上那个 DPI;>800 时自动补归一化缩放")
    ap.add_argument("--dpi-scale", type=float, default=None, help="直接给缩放")
    ap.add_argument("--fov", type=float, required=True,
                    help="游戏水平 FOV(度)。**必需** —— 不给会过转 2.5 倍")
    ap.add_argument("--speed-fraction", type=float, default=14.0 / 640.0)
    ap.add_argument("--max-counts", type=float, default=600.0)
    ap.add_argument("--deadzone", type=float, default=0.0)
    ap.add_argument("--invert-x", action="store_true")
    ap.add_argument("--invert-y", action="store_true")

    ap.add_argument("--probe", dest="probe", action="store_true", default=None,
                    help="做硬件校验(默认开;会动鼠标)")
    ap.add_argument("--no-probe", dest="probe", action="store_false",
                    help="跳过硬件校验")
    ap.add_argument("--probe-tol", type=float, default=0.02,
                    help="硬件校验容差(默认 2%%)")
    ap.add_argument("--settle", type=float, default=0.12, help="每组后泵消息时长")
    ap.add_argument("--verify-counts", type=float, default=None,
                    help="游戏内手测的等效 counts/360(可选,最高优先级证据)")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    ok_all = True

    print("=" * 84)
    print("FlyAim 端到端定标向导")
    print("=" * 84)

    # ---- 0. 指针加速状态(最廉价的先验) ---------------------------------
    print("\n[0] 系统指针加速状态")
    epp = pointer_accel_enabled()
    if epp is None:
        _warn("读不到 EPP 设置(非 Windows?注册表不可读?)")
    elif epp:
        _warn("**指针加速=开**。若你做硬件校验,比值可能随幅度变化 —— 那一项会替你判死")
    else:
        _pass("指针加速=关(注入的计数不会被系统再加工)")

    # ---- 1. 解析输入 -> counts_per_360 ---------------------------------
    print("\n[1] 由输入推导 counts_per_360")
    kwargs = dict(
        speed_fraction_per_action=args.speed_fraction,
        fov_h_deg=args.fov,
        max_counts_per_tick=args.max_counts,
        invert_x=args.invert_x,
        invert_y=args.invert_y,
        deadzone=args.deadzone,
    )
    if args.dpi_scale is not None:
        scale, note = float(args.dpi_scale), f"显式 --dpi-scale {args.dpi_scale}"
    elif args.aimlab_dpi is not None:
        a = float(args.aimlab_dpi)
        if a <= 800.0:
            scale, note = 1.0, f"Aim Lab DPI={a:g} ≤800 → 不缩放"
        else:
            scale, note = 800.0 / a, (f"Aim Lab DPI={a:g} >800 → 计数归一化 "
                                      f"×{800.0/a:g}(不补会过转 {a/800.0:g} 倍)")
    else:
        scale, note = 1.0, "未给 Aim Lab DPI,按不缩放(高 DPI 模式会错)"

    if args.sens is not None:
        cfg = GainConfig.from_sens(args.sens, args.engine, scale,
                                   args.yaw_coef, **kwargs)
        print(f"  输入: sens={args.sens:g}  engine={args.engine}  {note}")
        print(f"        counts_per_360 = 360/({args.sens:g}×"
              f"{cfg.counts_per_360 and (360.0/cfg.counts_per_360/args.sens):.6f}"
              f" × {scale:g}) = {cfg.counts_per_360:.1f}")
    elif args.edpi is not None:
        cfg = GainConfig.from_edpi(args.edpi, args.dpi, args.engine, scale,
                                   args.yaw_coef, **kwargs)
        sens = args.edpi / args.dpi
        print(f"  输入: eDPI={args.edpi:g}  鼠标 DPI={args.dpi:g}  {note}")
        print(f"        → 反解 sens = {args.edpi:g}/{args.dpi:g} = {sens:g}")
        print(f"        counts_per_360 = {cfg.counts_per_360:.1f}")
    elif args.cm360 is not None:
        cfg = GainConfig.from_cm360(args.dpi, args.cm360, **kwargs)
        print(f"  输入: {args.dpi:g} DPI × {args.cm360:g} cm/360  {note}")
        print(f"        counts_per_360 = {args.dpi:g}×{args.cm360:g}/2.54"
              f" = {cfg.counts_per_360:.1f}")
    else:
        cfg = GainConfig(counts_per_360=args.counts_per_360, **kwargs)
        print(f"  输入: counts_per_360={args.counts_per_360:g}")

    g = GainModel(cfg)
    print(f"  deg_per_count  = {g.deg_per_count():.6f} °/计数")
    print(f"  deg_per_action = {g.deg_per_action():.4f} °/满量程 action"
          f"  (linFOV={cfg.lin_fov_deg:.3f}°)")
    print(f"  等效 cm/360(按 {args.dpi:g} DPI 口径) = {g.cm360(args.dpi):.2f} cm")

    # ---- 2. 硬件校验 ---------------------------------------------------
    do_probe = True if args.probe is None else bool(args.probe)
    if do_probe:
        print("\n[2] 指针路径探针:注入已知计数 -> 读光标实际位移")
        print(f"  幅度序列 {PROBE_MAGNITUDES}(会动鼠标,结束后移回原位)")
        try:
            rows = _probe_hardware(PROBE_MAGNITUDES, args.settle)
        except Exception as e:
            rows, ok_all = [], False
            _fail(f"探针失败:{e!r}")
            print("     → 用 --no-probe 跳过,或改用手测 --verify-counts")
        if rows:
            ok, lines = _judge_probe(rows, args.probe_tol)
            for ln in lines:
                print(ln)
            ok_all = ok_all and ok
    else:
        rows = []
        print("\n[2] 指针路径探针:已跳过(--no-probe)")

    # ---- 3. 游戏内手测(可选,最高优先级) -------------------------------
    if args.verify_counts and args.verify_counts > 0:
        print("\n[3] 游戏内校验(手测)")
        v = float(args.verify_counts)
        ratio = cfg.counts_per_360 / v
        print(f"  公式 {cfg.counts_per_360:.0f} vs 实机 {v:.0f} -> 比值 {ratio:.3f}")
        if abs(ratio - 1.0) <= 0.05:
            _pass("差异 ≤5%,公式与游戏一致")
        else:
            ok_all = False
            _fail(f"差异 {(ratio-1)*100:+.1f}% —— 不要上线")
            print("     排查顺序:①鼠标加速 ②引擎系数 ③Aim Lab DPI 缩放 "
                  "④标称 DPI 与真实 DPI")
            print(f"     → 若要以实机为准:--counts-per-360 {v:.0f}")
    else:
        print("\n[3] 游戏内校验:未做(可选,给 --verify-counts)")

    # ---- 4. 落盘 + 结论 -------------------------------------------------
    p = g.save(args.out)
    report = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "inputs": {
            "sens": args.sens, "edpi": args.edpi, "dpi": args.dpi,
            "cm360": args.cm360, "engine": args.engine,
            "yaw_coef": args.yaw_coef, "aimlab_dpi": args.aimlab_dpi,
            "dpi_scale": scale, "fov": args.fov,
        },
        "model": g.describe(),
        "epp_enabled": epp,
        "probe": rows,
        "verify_counts": args.verify_counts,
        "verdict": "PASS" if ok_all else "FAIL",
    }
    rp = Path(args.out).with_suffix(".report.json")
    rp.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                  encoding="utf-8")

    print("\n" + "=" * 84)
    if ok_all:
        print("✅ 判定:PASS —— 可以进下一步(--sink null 目检,再 sendinput)")
    else:
        print("🔴 判定:FAIL —— 先按上面 FAIL 行的提示修,别进注入阶段")
    print("=" * 84)
    print(f"  增益已写入 {p}")
    print(f"  报告已写入 {rp}")
    print("\n  下一步(只读,不动鼠标):")
    print(f"    & $py tools/aimlab_probe.py --find aimlab")
    print(f"    & $py tools/aimlab_bridge.py --controller eye --source screen "
          f"--window aimlab --sink null --fov {args.fov:g} --target-hue auto "
          f"--eye 24x32 --eye-aim center --eye-search scan --frames 300 --live")
    return 0 if ok_all else 1


if __name__ == "__main__":
    raise SystemExit(main())
