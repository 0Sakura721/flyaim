"""桥接层冒烟测试(不依赖 GPU / 游戏 / 真实鼠标注入)。

运行::

    & $py tools/aimlab_smoke.py                 # 全部离线检查
    & $py tools/aimlab_smoke.py --screen        # + 真实屏幕捕获 5 帧(只读)
    & $py tools/aimlab_smoke.py --cursor-check  # + 光标微移注入并复位(显式要求)

检查项:
    T1  GainModel 数学与 JSON 往返
    T2  Win32 INPUT 结构体尺寸/打包(SendInput 不实际调用)
    T3  指针加速度检测可用(返回 bool/None,不抛错)
    T4  ArraySource + SeekController + NullSink 端到端闭环(含遥测落盘)
    T5  ArenaSource 闭环彩排:靶距应显著下降(seek 收敛),hit 至少 1 次
    T6  detect.find_target 在合成帧上检出/漏检行为正确
    T7  (可选 --screen)屏幕捕获真实出图,尺寸正确,≥3 FPS
    T8  (可选 --cursor-check)注入微小位移后光标实际移动且已复位

退出码 0 = 全部通过。任何一项失败打印 FAIL 并返回 1。
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from flyaim.arena.arena import Arena  # noqa: E402
from flyaim.bridge.capture import ArraySource, ArenaSource, ScreenCapture, centered_region  # noqa: E402
from flyaim.bridge.controllers import RandomController, SeekController, ZeroController  # noqa: E402
from flyaim.bridge.detect import DEFAULT_BLUE, annotate, find_target  # noqa: E402
from flyaim.bridge.gain import ENGINE_COEFS, GainConfig, GainModel  # noqa: E402
from flyaim.bridge.inject import NullSink, pointer_accel_enabled  # noqa: E402
from flyaim.bridge.loop import BridgeLoop  # noqa: E402
from flyaim.config import ArenaConfig  # noqa: E402

PASS: list[str] = []
FAIL: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    if cond:
        PASS.append(name)
        print(f"  PASS  {name}" + (f"  [{detail}]" if detail else ""))
    else:
        FAIL.append(name)
        print(f"  FAIL  {name}  {detail}")


def synth_frame(cx: float, cy: float, w=640, h=480, r=20) -> np.ndarray:
    f = np.full((h, w, 3), 24, dtype=np.uint8)
    yy, xx = np.mgrid[0:h, 0:w]
    m = (xx - cx) ** 2 + (yy - cy) ** 2 <= r * r
    f[m] = DEFAULT_BLUE
    return f


# ---------------------------------------------------------------- T1 增益


def t1_gain() -> None:
    print("[T1] GainModel")
    g = GainModel(GainConfig.from_cm360(800, 40))
    check("T1.from_cm360", abs(g.cfg.counts_per_360 - 800 * 40 / 2.54) < 1e-6,
          f"counts_per_360={g.cfg.counts_per_360:.0f}")
    dx, dy = g.to_counts(np.array([1.0, -1.0]))
    expect = g.counts_per_action
    check("T1.to_counts", dx == round(expect) and dy == -round(expect),
          f"(dx,dy)=({dx},{dy}) expect≈{expect:.1f}")
    dx2, _ = g.to_counts(np.array([5.0, 0.0]))
    check("T1.clip", dx2 == round(expect), f"超界 action 被裁剪: {dx2}")
    m = abs(g.cfg.max_counts_per_tick)
    g2 = GainModel(GainConfig(counts_per_360=12000, max_counts_per_tick=50))
    dx3, _ = g2.to_counts(np.array([1.0, 0.0]))
    check("T1.max_counts", dx3 == 50, f"钳位生效: {dx3}")
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "gain.json"
        g.save(p)
        g3 = GainModel.load(p)
        check("T1.json_roundtrip", abs(g3.cfg.counts_per_360 - g.cfg.counts_per_360) < 1e-6
              and g3.cfg.invert_x == g.cfg.invert_x)
    ginv = GainModel(GainConfig(invert_y=True))
    _, dy4 = ginv.to_counts(np.array([0.0, 0.3]))
    _, dy5 = GainModel(GainConfig()).to_counts(np.array([0.0, 0.3]))
    check("T1.invert_y", np.sign(dy4) == -np.sign(dy5), f"{dy4} vs {dy5}")


# ---------------------------------------------------------------- T2/T3 注入结构


def t2_inject_struct() -> None:
    print("[T2] Win32 注入结构(不实际调用 SendInput)")
    from flyaim.bridge.inject import _INPUT, _INPUT_MOUSE, SendInputSink

    check("T2.input_size", ctypes_size(_INPUT) >= 40,
          f"sizeof(INPUT)={ctypes_size(_INPUT)}(x64 应为 40)")
    check("T2.input_type", _INPUT_MOUSE == 0)
    s = NullSink()
    ok = s.send(3, -4)
    check("T2.nullsink", ok and s.total_counts.tolist() == [3, -4])
    check("T2.sendinput_ctor", SendInputSink(max_counts_per_tick=10) is not None)


def ctypes_size(cls) -> int:
    import ctypes

    return ctypes.sizeof(cls)


def t3_pointer_accel() -> None:
    print("[T3] 指针加速度检测")
    v = pointer_accel_enabled()
    check("T3.detect_no_throw", v is None or isinstance(v, bool), f"结果={v}")
    if v is True:
        print("        ⚠️ 「提高指针精确度」已开启 —— 受控实验前建议在系统设置关闭")


# ---------------------------------------------------------------- T4 array 闭环


def t4_array_loop() -> None:
    print("[T4] ArraySource + Seek + NullSink 端到端(含遥测)")
    frames = [synth_frame(80 + (640 - 160) * i / 29, 240) for i in range(30)]
    src = ArraySource(frames)
    sink = NullSink()
    ctl = SeekController(ref_color=DEFAULT_BLUE)
    g = GainModel(GainConfig())

    import sys as _s

    _s.path.insert(0, str(ROOT / "tools"))
    from telemetry import TelemetryWriter  # type: ignore

    with tempfile.TemporaryDirectory() as td:
        w = TelemetryWriter(td, preview_every=10, run_tag="smoke")
        loop = BridgeLoop(source=src, sink=sink, controller=ctl, gain=g,
                          telemetry=w, max_frames=25, min_tick_interval_s=0.002)
        t0 = time.perf_counter()
        summary = loop.run()
        wall = time.perf_counter() - t0
        check("T4.ran", summary["frames"] == 25, f"frames={summary['frames']}")
        check("T4.tick_hz", summary["tick_hz"] > 5, f"{summary['tick_hz']} Hz (wall={wall:.2f}s)")
        check("T4.telemetry", Path(td, "telemetry.jsonl").exists()
              and Path(td, "telemetry.jsonl").stat().st_size > 0)
        recs = [json.loads(x) for x in
                Path(td, "telemetry.jsonl").read_text(encoding="utf-8").splitlines()]
        check("T4.telemetry_header", recs[0].get("kind") == "header")
        check("T4.telemetry_no_brain", any(r.get("meta", {}).get("no_brain")
                                           for r in recs if r.get("kind") == "frame"))
        previews = list((Path(td) / "preview").glob("*.png")) if (Path(td) / "preview").exists() else []
        check("T4.preview", len(previews) >= 1, f"{len(previews)} 张预览图")
        check("T4.counts_flow", sink.n_calls == 25 and np.abs(sink.total_counts).sum() > 0,
              f"calls={sink.n_calls} total={sink.total_counts.tolist()}")


# ---------------------------------------------------------------- T5 arena 彩排收敛


def t5_arena_loop() -> None:
    print("[T5] ArenaSource 闭环彩排(seek 应把准星收敛到靶上并反复命中)")
    cfg = ArenaConfig()
    arena = Arena(cfg, seed=7)
    src = ArenaSource(arena, closed_loop=True)
    ctl = SeekController(ref_color=(235, 70, 70), tolerance=80.0)  # 本项目红靶
    g = GainModel(GainConfig())
    loop = BridgeLoop(source=src, sink=NullSink(), controller=ctl, gain=g,
                      max_frames=300, min_tick_interval_s=0.001)
    summary = loop.run()
    check("T5.ran", summary["frames"] == 300, f"frames={summary['frames']}")
    check("T5.detection_worked", ctl.n_not_found < 300 * 0.5,
          f"漏检率 {ctl.n_not_found}/{300}")
    # 命中 = 准星进入靶半径 22px 内(命中后靶 respawn 到远处),
    # 是「瞄准点收敛」最干净的行为学证据。
    check("T5.converge", summary["n_hits_env"] >= 3,
          f"300 帧内命中 {summary['n_hits_env']} 次(命中判定半径 22px)")


# ---------------------------------------------------------------- T6 检测


def t6_detect() -> None:
    print("[T6] find_target")
    d = find_target(synth_frame(320, 240), ref_color=DEFAULT_BLUE)
    check("T6.hit", d.ok and abs(d.cx - 320) < 1.5 and abs(d.cy - 240) < 1.5,
          f"cx={d.cx:.1f} cy={d.cy:.1f} r={d.radius_px:.1f}")
    check("T6.radius", abs(d.radius_px - 20) < 1.5, f"r={d.radius_px:.2f} (期望 20)")
    d2 = find_target(np.full((480, 640, 3), 24, dtype=np.uint8), ref_color=DEFAULT_BLUE)
    check("T6.miss", not d2.ok)
    ann = annotate(synth_frame(320, 240), d)
    check("T6.annotate", ann.shape == (480, 640, 3) and ann.dtype == np.uint8)


# ---------------------------------------------------------------- T7 屏幕捕获(可选)


def t7_screen() -> None:
    print("[T7] 屏幕捕获(真实截屏,只读)")
    try:
        cap = ScreenCapture()  # 主屏中央 640x480
    except Exception as e:
        check("T7.capture_init", False, f"{type(e).__name__}: {e}")
        return
    times = []
    last = None
    for _ in range(5):
        t0 = time.perf_counter()
        f, meta = cap.read()
        times.append(time.perf_counter() - t0)
        last = (f, meta)
        time.sleep(0.05)
    f, meta = last
    check("T7.shape", f.shape == (480, 640, 3) and f.dtype == np.uint8, f"{f.shape} {f.dtype}")
    fps = 1.0 / float(np.mean(times))
    check("T7.fps", fps >= 3.0, f"PIL 捕获 {fps:.1f} FPS (grab={meta.get('grab_ms')}ms)")
    out = ROOT / ".cache" / "aimlab_capture_probe.png"
    from PIL import Image

    Image.fromarray(f).save(out)
    print(f"        探针帧已存 {out}(自行打开核对区域是否正确,本测试不展示内容)")
    check("T7.region_hint", True, f"region={cap.region} backend={cap.backend_name}")
    cap.close()


# ---------------------------------------------------------------- T8 光标注入(可选)


def t8_cursor_check() -> None:
    print("[T8] 光标微移注入(会真实动鼠标,移动 3px 后立即复位)")
    from flyaim.bridge.inject import cursor_nudge_check

    r = cursor_nudge_check(3, 0)
    if r is None:
        check("T8.cursor", False, "GetCursorPos 不可用")
        return
    (x0, y0), (x1, y1) = r
    moved = (x1 - x0, y1 - y0)
    check("T8.moved", abs(moved[0]) >= 1, f"位移 {moved}(SendInput 生效)")
    (xb, yb), _ = r
    from flyaim.bridge.inject import cursor_position

    now = cursor_position()
    check("T8.restored", now is not None and abs(now[0] - xb) <= 1 and abs(now[1] - yb) <= 1,
          f"复位到 {now}(原 {xb},{yb})")


# ---------------------------------------------------------------- 附:零控制器 3 帧


def t9_zero_random() -> None:
    print("[T9] zero/random 控制器经 BridgeLoop")
    for name, ctl in (("zero", ZeroController()), ("random", RandomController(seed=1))):
        sink = NullSink()
        loop = BridgeLoop(source=ArraySource([synth_frame(320, 240)]), sink=sink,
                          controller=ctl, gain=GainModel(GainConfig()),
                          max_frames=3, min_tick_interval_s=0.001)
        s = loop.run()
        check(f"T9.{name}", s["frames"] == 3 and sink.n_calls == 3)


# ---------------------------------------------------------------- 附:增益语义(D26)

def t10_gain_fov_semantics() -> None:
    """D26:增益的「比例语义」必须随 FOV 走,否则过转 2.50 倍。

    这是被真实查出过的一处标定缺陷(它让恒定量程控制器从"锁定 100%"掉到
    "锁定 0.3%"),所以把它固化成冒烟项 —— 以后谁再动 gain.py,这里会立刻响。
    """
    print("[T10] 增益语义:FOV 一致性与过转倍数")
    legacy = GainModel(GainConfig(counts_per_360=12598.0))
    fixed = GainModel(GainConfig(counts_per_360=12598.0, fov_h_deg=103.0))
    # 旧语义 = speed_fraction × 360
    check("T10.legacy", abs(legacy.deg_per_action() - 0.021875 * 360.0) < 1e-6,
          f"旧语义 {legacy.deg_per_action():.3f}°/拍")
    # 新语义 = speed_fraction × linFOV,linFOV = 2·tan(FOV/2)
    check("T10.linfov", abs(fixed.cfg.lin_fov_deg - 144.061) < 0.01,
          f"linFOV={fixed.cfg.lin_fov_deg:.3f}°")
    check("T10.deg_per_action", abs(fixed.deg_per_action() - 3.1513) < 1e-3,
          f"FOV=103° -> {fixed.deg_per_action():.4f}°/拍")
    check("T10.ratio", abs(legacy.deg_per_action() / fixed.deg_per_action() - 2.50) < 0.01,
          f"过转 {legacy.deg_per_action()/fixed.deg_per_action():.2f} 倍")
    # counts_per_action 必须与 deg_per_action 自洽(否则注入量与名义角度不符)
    for g in (legacy, fixed):
        want = g.deg_per_action() * g.cfg.counts_per_360 / 360.0
        check("T10.counts_consistent", abs(g.counts_per_action - want) < 1e-6,
              f"counts_per_action={g.counts_per_action:.2f} vs {want:.2f}")
    check("T10.fov_validated", _t10_raises(lambda: GainConfig(fov_h_deg=200.0)),
          "非法 FOV 应被拒绝")


def _t10_raises(fn) -> bool:
    try:
        fn()
    except ValueError:
        return True
    return False


def t11_sens_model() -> None:
    """路线 B:由「游戏内灵敏度」推 counts_per_360,必须与路线 A 自洽。

    这是 D27 固化的东西:cm/360 **不必在桌面上量**。但三条不变式必须成立,
    否则标定会以「能跑但系统地偏」的形式悄悄出错(与 D26 的 2.5 倍同族)。
    """
    print("[T11] 灵敏度模型:路线 B ↔ 路线 A 互换自洽")
    # 不变式 1:360 / (sens × yaw_coef) 的定义
    c = GainConfig.counts_per_360_from_sens(1.5, yaw_coef=0.022)
    check("T11.def", abs(c - 360.0 / (1.5 * 0.022)) < 1e-6,
          f"360/(1.5×0.022) = {c:.1f}")

    # 不变式 2:与路线 A 对拍(D=800 时 34.64cm/360 应给出同一个数)
    a = GainConfig.from_cm360(800.0, 34.64)
    check("T11.routeA_agree", abs(a.counts_per_360 - 10909.1) / 10909.1 < 3e-3,
          f"路线A {a.counts_per_360:.0f} vs 路线B {c:.0f}")

    # 不变式 3:「DPI 不进 counts_per_360」—— eDPI 等价
    #   公式里没有 DPI 项,只有 sens。所以 sens 与 counts/360 严格成反比。
    #   ⚠️ 注意别把这条写成 "两式相等"(首版就写错了):是**比例**关系。
    c2 = GainConfig.counts_per_360_from_sens(2.0, yaw_coef=0.022)
    c1 = GainConfig.counts_per_360_from_sens(1.0, yaw_coef=0.022)
    check("T11.dpi_free", abs(c1 / c2 - 2.0) < 1e-9,
          f"sens×2 -> counts/360 ÷2({c2:.0f} vs {c1:.0f});公式无 DPI 项")
    #   而唯一让 DPI 重新进来的是 dpi_scale(游戏内归一化),它必须线性可乘。
    c1s = GainConfig.counts_per_360_from_sens(1.0, yaw_coef=0.022, dpi_scale=0.5)
    check("T11.scale_linear", abs(c1s / c1 - 2.0) < 1e-9,
          f"dpi_scale 与 counts/360 成反比,系数恰为 1/scale({c1s:.0f} vs {c1:.0f});"
          "这是 DPI 影响结果的唯一通道")

    # 不变式 4:幂等互逆 —— A->B->A 必须回到原点
    r = GainConfig.sens_for_cm360(800.0, 40.0, yaw_coef=0.022)
    back = GainConfig.cm360_from_sens(r, 800.0, yaw_coef=0.022)
    check("T11.roundtrip", abs(back - 40.0) < 1e-9,
          f"40cm/360 -> sens {r:.4f} -> {back:.6f}cm/360")

    # 不变式 5:Aim Lab 高 DPI 缩放:3200 模式计数被归一化,过转 4 倍
    hi = GainConfig.counts_per_360_from_sens(1.5, dpi_scale=800.0 / 3200.0,
                                             yaw_coef=0.022)
    check("T11.dpi_scale", abs(hi / c - 4.0) < 1e-9,
          f"3200DPI 归一化 -> counts×4(过转 {hi/c:.1f} 倍)")

    # 不变式 6:未知 engine 必须报错,不能静默退默认
    check("T11.engine_validated",
          _t10_raises(lambda: GainConfig.from_sens(1.0, engine="csgo_but_typo")),
          "未知引擎应被拒绝")
    check("T11.sens_pos", _t10_raises(lambda: GainConfig.from_sens(0.0)),
          "sens=0 应被拒绝")

    # 不变式 7:所有内置引擎系数的 counts_per_360 都落在合理量级
    for k, v in ENGINE_COEFS.items():
        cc = GainConfig.counts_per_360_from_sens(1.0, engine=k)
        check(f"T11.{k}", 100.0 < cc < 1e6, f"{k}: sens=1 -> {cc:.0f} counts/360")


def t12_pointer_probe() -> None:
    """D27.5:指针路径探针的判定逻辑(不注入,只测纯函数)。

    探针本身会动鼠标,所以这里只测"给定输入 -> 判定输出"的**纯逻辑**:
    这是本项目反复吃过亏的地方(判定函数写错 = 假阳性/假阴性),
    必须能在零副作用下回归。
    """
    print("[T12] 指针探针判定逻辑")
    from flyaim.bridge.pointer_probe import PointerProbe, judge

    def rows(*specs) -> list:
        return [PointerProbe(sent=(s, 0), moved=(m, 0), ax=0, ay=0)
                for s, m in specs]

    # 理想:比值恒为 1
    ok, _ = judge(rows((300, 300), (1200, 1200), (300, 300)))
    check("T12.linear_ideal", ok, "比值恒 1 应通过")

    # 线性但有固定缩放:比值恒为 2 —— 应通过线性判据但给出缩放告警
    ok2, lines2 = judge(rows((300, 600), (1200, 2400), (300, 600)))
    check("T12.scaled_linear", ok2, "恒定比值 2 仍算线性(有缩放告警)")

    # 非线性:本机实测形态(EPP 开)必须被判死
    ok3, lines3 = judge(rows((300, 981), (1200, 1859), (300, 981)), tol=0.02)
    check("T12.epp_nonlinear", not ok3, "比值 3.27/1.55 必须判不通过")
    check("T12.epp_diagnosed",
          any("非线性" in ln or "变化" in ln for ln in lines3),
          "应指出非线性")

    # 完全不动:注入未生效
    ok4, _ = judge(rows((300, 0), (1200, 0)))
    check("T12.no_move", not ok4, "光标没动应判不通过")

    # 空输入
    ok5, _ = judge([])
    check("T12.empty", not ok5, "空输入应判不通过")

    # 比值属性自洽
    p = PointerProbe(sent=(400, 0), moved=(1000, 0), ax=0, ay=0)
    check("T12.ratio_prop", abs(p.ratio_x - 2.5) < 1e-9, "ratio_x=2.5")
    check("T12.ratio_y_none", p.ratio_y is None, "sent_y=0 时 ratio_y=None")

    # describe 可序列化(报告要 json.dump)
    d = p.describe()
    check("T12.describe", d["ratio_x"] == 2.5 and d["sent"] == [400, 0],
          "describe 字段正确")


def t13_fov_convention() -> None:
    """D28:FOV 口径守卫 —— 防止把「游戏设置里的 fov 值」当成渲染 FOV。

    这个坑的特点:数值看着都合理(90 也在 [5,179) 内,不会被校验拦),
    但会让 deg_per_action 差 33%。所以**必须靠固定数值回归**守住。
    """
    print("[T13] FOV 口径守卫 (D28)")
    import math
    from flyaim.bridge.gain import GainConfig, GainModel

    C360 = 360.0 / (2.0 * 0.022)      # 8181.818...

    def dpa(fov: float) -> float:
        cfg = GainConfig(counts_per_360=C360, fov_h_deg=fov)
        return GainModel(cfg).deg_per_action()

    # CS2@16:9 的渲染 FOV 必须给 106.26,而不是设置里的 90
    # linFOV(90)=114.592°, linFOV(106.26)=152.788°(角度制)
    lin90 = 2.0 * math.tan(math.radians(90.0) / 2.0) * 180.0 / math.pi
    lin106 = 2.0 * math.tan(math.radians(106.26) / 2.0) * 180.0 / math.pi
    check("T13.lin90", abs(lin90 - 114.592) < 0.01, f"linFOV(90)={lin90:.3f}°")
    check("T13.lin106", abs(lin106 - 152.788) < 0.01, f"linFOV(106.26)={lin106:.3f}°")

    # deg_per_action 与 FOV 单调正相关,且两条已知值必须命中
    a90, a106 = dpa(90.0), dpa(106.26)
    check("T13.dpa90", abs(a90 - 2.5067) < 1e-3, f"FOV=90 -> {a90:.4f}°/action")
    check("T13.dpa106", abs(a106 - 3.3422) < 1e-3, f"FOV=106.26 -> {a106:.4f}°/action")

    # 误填 90 的相对误差应是 -25%(=1 - 114.592/152.788)
    rel = a90 / a106 - 1.0
    check("T13.misconfig_25pct", abs(rel + 0.25) < 2e-3,
          f"照界面填 90 会少转 {abs(rel):.1%}")

    # 106.26 必须严格大于 90 —— 守住"Valve 按纵横比放大"这个事实
    check("T13.valve_expand", a106 > a90 * 1.3,
          "CS2 渲染 FOV 应比 fov 值大 30% 以上")

    # 两个语义必须分开:counts_per_360 与 FOV 完全无关
    c1 = GainModel(GainConfig(counts_per_360=C360, fov_h_deg=90.0))
    c2 = GainModel(GainConfig(counts_per_360=C360, fov_h_deg=106.26))
    check("T13.fov_free_c360",
          c1.cfg.counts_per_360 == c2.cfg.counts_per_360,
          "counts_per_360 不应随 FOV 变化(核心等式里没有 FOV)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--screen", action="store_true", help="加测真实屏幕捕获(只读)")
    ap.add_argument("--cursor-check", action="store_true",
                    help="加测光标微移注入(会真实移动鼠标 3px 并复位)")
    args = ap.parse_args()

    print("=" * 84)
    print("FlyAim 桥接层冒烟测试")
    print("=" * 84)
    t1_gain()
    t2_inject_struct()
    t3_pointer_accel()
    t4_array_loop()
    t5_arena_loop()
    t6_detect()
    t9_zero_random()
    t10_gain_fov_semantics()
    t11_sens_model()
    t12_pointer_probe()
    t13_fov_convention()
    if args.screen:
        t7_screen()
    if args.cursor_check:
        t8_cursor_check()

    print("-" * 84)
    print(f"通过 {len(PASS)} / 失败 {len(FAIL)}")
    if FAIL:
        print("失败项: " + ", ".join(FAIL))
        return 1
    print("全部通过 ✅")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
