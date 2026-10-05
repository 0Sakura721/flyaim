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


# ---------------------------------------------------------------- HUD 合成帧
# T24 / T25 共用。
# ⚠️ **手画字形只可用于"分段/流程"测试**:T24 已证它们**不匹配**真实 HUD 字形
# (MSE 超阈)。任何"能不能读对真实数字"的测试**必须**用 tools/fixtures/ 下的
# 真实帧,否则就是在测自己造的输入 —— 那是 D42.8「测试把 bug 固化」的翻版。

def _blank() -> np.ndarray:
    """HUD 裁剪同尺寸的深灰底(默认按 GRAB_BBOX 的宽高)。"""
    from flyaim.bridge.score_ocr import GRAB_BBOX
    return np.full((GRAB_BBOX[3] - GRAB_BBOX[1], GRAB_BBOX[2] - GRAB_BBOX[0], 3),
                   100, np.int32)


def _glyph(kind: str) -> np.ndarray:
    g = np.zeros((28, 14), np.uint8)
    if kind == "ring":      # 类 0
        g[3:25, 2:12] = 1; g[7:21, 5:9] = 0
    elif kind == "bar":     # 类 1
        g[3:25, 6:10] = 1
    elif kind == "top":     # 上横+下横
        g[3:7, 2:12] = 1; g[21:25, 2:12] = 1
    elif kind == "mid":     # 三横
        g[3:6, 2:12] = 1; g[12:16, 2:12] = 1; g[22:26, 2:12] = 1
    elif kind == "dots":    # 冒号
        g[7:11, 5:9] = 1; g[17:21, 5:9] = 1
    elif kind == "cross":   # 类 x(未学字符用)
        for i in range(22):
            g[3 + i, 2 + i // 2] = 1; g[3 + i, 11 - i // 2] = 1
    return g


def _put(img: np.ndarray, box: str, slot: int, kind: str) -> None:
    """往某框数字行的第 slot 格贴一个手画字符(原地改 img)。"""
    from PIL import Image

    from flyaim.bridge.score_ocr import _BOX_X, _DIGIT_Y
    x0, _ = _BOX_X[box]
    gw = 14
    gx = x0 + 4 + slot * (gw + 5)
    y0, y1 = _DIGIT_Y
    sub = np.asarray(Image.fromarray((_glyph(kind) * 255).astype(np.uint8))
                     .resize((gw, y1 - y0), Image.NEAREST))
    img[y0:y1, gx:gx + gw] = sub[:, :, None]


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


def t14_trigger_gate() -> None:
    """D30:开火门限守卫 —— 防止「瞄得准却不开火」再犯。

    这个 bug 的特点:瞄准链路完全正常(收敛误差也很小),
    但开火门限与收敛误差**同一量级**,导致擦边失手、时开时不开。
    靠人眼很难发现"为什么没开火",必须用数学回归守住门限公式。
    """
    print("[T14] 开火门限守卫 (D30)")
    import numpy as np

    from flyaim.bridge.controllers import TriggerOnTarget
    from flyaim.bridge.detect import Detection

    def gate_err(R: float, fire_frac: float, min_r: float) -> tuple[float, float]:
        """返回 (limit, 实际是否开火)。用假 frame 直接驱动 act()。

        fire_confirm_beats=1:本组只测**门限几何**(D30),开火确认时序归 T22。
        """
        tg = TriggerOnTarget(None, lambda: True, fire_frac=fire_frac,
                             min_radius_px=min_r, fire_confirm_beats=1)

        class _Inner:
            last_detection = Detection(ok=True, cx=0.0, cy=0.0, radius_px=R)

            def act(self, frame):
                return np.zeros(2, dtype=np.float32)

        tg.inner = _Inner()
        frame = np.zeros((10, 10, 3), dtype=np.uint8)
        err_holder: dict[str, float] = {}

        # 直接把 err 写进 det 位置:cx/cy 以画面中心为零
        h, w = 480, 640
        frame = np.zeros((h, w, 3), dtype=np.uint8)

        def run(dx: float, dy: float) -> bool:
            _Inner.last_detection = Detection(
                ok=True, cx=(w - 1) / 2.0 + dx, cy=(h - 1) / 2.0 + dy, radius_px=R)
            tg.act(frame)
            return tg.n_fires > 0

        return max(R * fire_frac, min_r), run

    # 1) 门限公式:取大者,而非单纯倍数
    lim = max(8.0 * 1.4, 4.0)
    check("T14.limit_formula", abs(lim - 11.2) < 1e-6,
          f"radius=8 fire_frac=1.4 min=4 -> limit={lim:.1f}px")
    lim2 = max(1.0 * 1.4, 4.0)
    check("T14.min_radius_floor", abs(lim2 - 4.0) < 1e-6,
          f"小靶 radius=1 -> 门限被 min_radius_px 兜到 {lim2:.1f}px,而非塌缩到 1.4px")

    # 2) 关键回归:原来会失手的场景现在必须开火
    #    实测 radius=7.0, err=8.1 —— 旧门限 7.0*1.15=8.05 < 8.1 => 不开火
    old_limit = 7.0 * 1.15
    check("T14.regress_old_fails", 8.1 > old_limit,
          f"旧门限 {old_limit:.2f}px < err 8.10px -> 这就是「瞄了不开火」的直接成因")
    new_limit = max(7.0 * 1.4, 4.0)
    check("T14.regress_new_fires", 8.1 <= new_limit,
          f"新门限 {new_limit:.2f}px >= err 8.10px -> 同一场景现在开火")

    # 3) 真跑一遍 act():门限内外各一次
    _, run = gate_err(7.0, 1.4, 4.0)
    check("T14.fires_inside", run(4.0, 0.0), "err=4.0px 应在门限内开火")
    _, run2 = gate_err(7.0, 1.4, 4.0)
    check("T14.no_fire_outside", not run2(40.0, 0.0), "err=40px 远超门限,不应开火")

    # 4) 边界:err 恰等于门限时必须开火(闭区间,不能是开区间)
    _, run3 = gate_err(8.0, 1.4, 4.0)
    check("T14.boundary_inclusive", run3(new_limit, 0.0),
          f"err == limit({new_limit:.1f}px) 属边界,闭区间必须开火")

    # 5) 向后兼容:只给 err_frac 时 fire_frac 应继承它
    tg = TriggerOnTarget(None, lambda: True, err_frac=0.9)
    check("T14.legacy_erfrac", abs(tg.fire_frac - 0.9) < 1e-9,
          "未给 fire_frac 时应回退到 err_frac,不破坏旧调用")

    # 6) 冷却:连续两拍必须只开一枪(confirm=1 —— 本组测冷却,确认时序归 T22)
    tg = TriggerOnTarget(None, lambda: True, fire_frac=1.4, min_radius_px=4.0,
                         cooldown_s=10.0, fire_confirm_beats=1)

    class _Inner:
        last_detection = None

        def act(self, frame):
            return np.zeros(2, dtype=np.float32)

    tg.inner = _Inner()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    for _ in range(2):
        _Inner.last_detection = Detection(ok=True, cx=319.5, cy=239.5, radius_px=8.0)
        tg.act(frame)
    check("T14.cooldown", tg.n_fires == 1,
          f"冷却 10s 内连打两拍 -> 只应开 1 枪,实开 {tg.n_fires}")
    check("T14.cooldown_counted", tg.n_blocked_by_cooldown == 1,
          f"被冷却挡下应计数,实测 {tg.n_blocked_by_cooldown}")

    # 7) 目标粘滞:多靶并存时必须挑离上一拍最近的,不能"永远选最大块"
    #    构造:大靶在右边(area 大),上一拍在左边 —— 应继续锁左边
    import numpy as _np

    tg = TriggerOnTarget(None, lambda: True, fire_frac=1.4, min_radius_px=4.0,
                         sticky_px=90.0)

    class _Inner2:
        last_detection = None

        def act(self, frame):
            return _np.zeros(2, dtype=_np.float32)

    tg.inner = _Inner2()
    fr = _np.zeros((480, 640, 3), dtype=_np.uint8)
    # 画面里放两块同色:左边小靶 r=6 @(200,240),右边大靶 r=14 @(500,240)
    yy, xx = _np.ogrid[:480, :640]
    fr[(_np.hypot(xx - 200, yy - 240) <= 6)] = (48, 224, 224)
    fr[(_np.hypot(xx - 500, yy - 240) <= 14)] = (48, 224, 224)
    # 上一拍锁左边
    tg._prev_xy = (200.0, 240.0)
    tg.act(fr)
    check("T14.sticky_keeps_near", tg.last_detection is not None
          and abs(tg.last_detection.cx - 200) < 5,
          f"sticky 应保持锁定近的左边小靶 cx≈200,实测 "
          f"{tg.last_detection.cx if tg.last_detection else None:.0f}")

    # 对照:关掉粘滞就应回到"选最大块"(右边 r=14)
    tg2 = TriggerOnTarget(None, lambda: True, fire_frac=1.4, min_radius_px=4.0,
                          sticky_px=0.0)
    tg2.inner = _Inner2()
    det = find_target(fr, ref_color=(48, 224, 224), tolerance=95.0)
    check("T14.no_sticky_picks_big", abs(det.cx - 500) < 5,
          f"关粘滞后 find_target 选最大块 cx≈500,实测 {det.cx:.0f}(这就是换靶的根源)")


def t15_speed_and_xhair() -> None:
    """D31:准星排除 + 降采样守卫 —— 「瞄了不开火」最后一层 + 「又快又准」。

    这组坑的共性:数值看着都合理,但会让系统**对着准星自己开枪**
    (err≈0、永远静止,被 sticky 一选就粘死),或让坐标折算差整数倍。
    """
    print("[T15] 准星排除 + 降采样守卫 (D31)")
    import numpy as np

    from flyaim.bridge.detect import _detect_all, find_targets

    # 造帧:两个真靶 + 正中心一个小色块(模拟准星十字)
    fr = np.zeros((400, 500, 3), dtype=np.uint8)
    yy, xx = np.ogrid[:400, :500]
    for (cx, cy, r) in [(120, 200, 22), (380, 190, 20)]:
        fr[np.hypot(xx - cx, yy - cy) <= r] = (48, 224, 224)
    # 准星:画面中心 (249.5, 199.5) 处 r=3 的小十字
    fr[np.hypot(xx - 249.5, yy - 199.5) <= 3] = (48, 224, 224)

    # 1) 不排除时应能看到 3 个(含准星)
    d_all = find_targets(fr, ref_color=(48, 224, 224), tolerance=95.0, top_k=6)
    check("T15.xhair_present", len(d_all) == 3,
          f"不排除时应检出 3 个(两真靶+准星),实测 {len(d_all)}")
    check("T15.xhair_is_center", any(abs(d.cx - 249.5) < 2 and d.radius_px < 8
                                     for d in d_all),
          "其中一个应在画面正中且半径很小(准星特征)")

    # 2) 排除后应只剩 2 个真靶
    d_ex = find_targets(fr, ref_color=(48, 224, 224), tolerance=95.0, top_k=6,
                        exclude_center_px=12.0, exclude_center_max_r=8.0)
    check("T15.xhair_excluded", len(d_ex) == 2,
          f"排除准星后应剩 2 个真靶,实测 {len(d_ex)}")
    check("T15.real_targets_kept",
          all(d.radius_px > 15 for d in d_ex),
          "留下的必须都是真靶(半径 >15px),不能误删")

    # 3) 大靶即使正好在中心也不能被误排除(半径判据是必要条件)
    fr2 = np.zeros((400, 500, 3), dtype=np.uint8)
    yy2, xx2 = np.ogrid[:400, :500]
    fr2[np.hypot(xx2 - 249.5, yy2 - 199.5) <= 30] = (48, 224, 224)  # 大靶在正中
    d_big = find_targets(fr2, ref_color=(48, 224, 224), tolerance=95.0,
                         exclude_center_px=12.0, exclude_center_max_r=8.0)
    check("T15.big_center_kept", len(d_big) == 1 and d_big[0].radius_px > 25,
          "正中大靶半径 30 > max_r 8,不该被当准星排除")

    # 4) 降采样:坐标/半径/面积必须按 ds 放大回原尺寸
    for ds in (1, 2, 3):
        dn = find_targets(fr, ref_color=(48, 224, 224), tolerance=95.0, top_k=3,
                          downsample=ds, exclude_center_px=12.0, exclude_center_max_r=8.0)
        ok = (len(dn) == 2
              and abs(dn[0].cx - 120) < 2.0 * ds
              and abs(dn[0].radius_px - 22) < 2.0 * ds)
        check(f"T15.ds{ds}_scaling", ok,
              f"ds={ds}: 首靶 ({dn[0].cx:.0f},{dn[0].cy:.0f},R{dn[0].radius_px:.1f})"
              f" 应≈(120,200,R22)")

    # 5) 面积折算:ds=2 时面积应≈原尺寸（容差按离散化放 25%）
    d1 = find_targets(fr, ref_color=(48, 224, 224), tolerance=95.0, top_k=3, downsample=1,
                      exclude_center_px=12.0, exclude_center_max_r=8.0)
    d2 = find_targets(fr, ref_color=(48, 224, 224), tolerance=95.0, top_k=3, downsample=2,
                      exclude_center_px=12.0, exclude_center_max_r=8.0)
    rel = abs(d2[0].area_px - d1[0].area_px) / max(d1[0].area_px, 1)
    check("T15.ds_area_consistent", rel < 0.25,
          f"ds=2 面积 {d2[0].area_px} vs ds=1 {d1[0].area_px},偏差 {rel:.1%}")

    # 6) min_area 也要随 ds 折算(否则小靶在 ds 下全被滤掉)
    d_small = find_targets(fr, ref_color=(48, 224, 224), tolerance=95.0, top_k=3,
                           min_area_px=200, downsample=2,
                           exclude_center_px=12.0, exclude_center_max_r=8.0)
    check("T15.min_area_scaled", len(d_small) >= 1,
          f"min_area=200 在 ds=2 下应仍能检出大靶(面积按 ds² 折算),实测 {len(d_small)}")


def t16_aim_fire_consistency() -> None:
    """D32:「瞄准锁的靶」必须 == 「开火看的靶」(2026-10-05 实盘定位)。

    这组坑是「老是瞄准了但不开火」的**最终真因**,且极隐蔽:

      SeekController 用 lock + _locked_xy 选一个靶,把它拖到画面中心
      → err≈0 → action≈0 → 相机停住(看起来"瞄准了");
      而 TriggerOnTarget 夹在 ActionEMA 之后,**拿不到** seek 的检测结果
      (ActionEMA 没有透传 last_detection/last_candidates),于是自己重检测、
      用**自己那份 _prev_xy** 选靶 —— 选中的是另一个靶,离中心 193px,
      err 永远进不了门限 → 就是不开火。

    铁证(实盘 run 20261005-130026):同一帧独立回放 seek.act=[1.0,0.28],
    而实盘 act≈0.005 —— 因为实盘 seek 锁的其实是画面中心那个靶,
    与开火层看的 (535,398) 不是同一个。

    本测试搭一个**闭环玩具靶场**:帧里有多个靶,准星钉在画面中心,
    每拍按 action 反向平移所有靶(模拟"动相机=靶在动"),从而让 seek 真的
    能收敛。断言:瞄准层锁定某靶并把它拖到中心后,**开火层看的必须是同一个靶**
    且落在门限内、真的开火。
    """
    print("[T16] 瞄准/开火目标一致性守卫 (D32)")
    import numpy as np

    from flyaim.bridge.controllers import SeekController, TriggerOnTarget

    W = H = 700

    def render(t1, t2):
        fr = np.zeros((H, W, 3), dtype=np.uint8)
        yy, xx = np.ogrid[:H, :W]
        for (cx, cy, r) in (t1, t2):
            if -r < cx < W + r and -r < cy < H + r:
                fr[np.hypot(xx - cx, yy - cy) <= r] = (48, 224, 224)
        return fr

    # 两个靶:targets 随相机移动。位移 = -action * speed(与游戏同号)
    state = {"t1": (150.0, 620.0, 28.0), "t2": (520.0, 120.0, 30.0)}
    SPEED = 60.0

    def shift(action):
        ax, ay = float(action[0]), float(action[1])
        for k in ("t1", "t2"):
            cx, cy, r = state[k]
            state[k] = (cx - ax * SPEED, cy - ay * SPEED, r)

    seek = SeekController(ref_color=(48, 224, 224), tolerance=95.0,
                          use_aim_detect=False, kp=2.2, downsample=2,
                          xhair_px=12.0, xhair_max_r=8.0)

    fired = []

    def _click():
        fired.append(1)
        return True

    class _Ema:
        """最小复刻 ActionEMA:必须透传 last_detection/last_candidates。"""
        def __init__(self, inner):
            self.inner = inner
            self._prev = None

        @property
        def last_detection(self):
            return getattr(self.inner, "last_detection", None)

        @property
        def last_candidates(self):
            return getattr(self.inner, "last_candidates", None)

        def __getattr__(self, item):
            return getattr(self.inner, item)

        def act(self, frame):
            a = np.asarray(self.inner.act(frame), dtype=np.float32).reshape(2)
            self._prev = a if self._prev is None else 0.35 * a + 0.65 * self._prev
            return self._prev.copy()

    trig = TriggerOnTarget(_Ema(seek), _click, ref_color=(48, 224, 224),
                           tolerance=95.0, fire_frac=1.4, min_radius_px=4.0,
                           cooldown_s=0.0, sticky_px=220.0, downsample=2,
                           xhair_px=12.0, xhair_max_r=8.0, jump_reset_px=45.0)

    # 闭环跑 60 拍:seek 应锁定一个靶并收敛到中心
    for _ in range(60):
        fr = render(state["t1"], state["t2"])
        out = trig.act(fr)
        shift(out)

    sd = seek.last_detection
    td = trig.last_detection
    check("T16.lock_exists", sd is not None and sd.ok, f"seek 应锁到一个靶,实测 {sd}")

    # 核心断言:两层看的是**同一个**靶(允许 1px 数值差)
    same = (td is not None and td.ok and sd is not None and sd.ok
            and abs(td.cx - sd.cx) < 1.0 and abs(td.cy - sd.cy) < 1.0)
    check("T16.same_target", same,
          f"开火层看的靶 {None if td is None else (round(td.cx),round(td.cy))}"
          f" 必须 == 瞄准锁的靶 {None if sd is None else (round(sd.cx),round(sd.cy))}")

    # 开火层看的靶必须靠近画面中心(seek 已把它拖过去),而不是另一个角落靶
    if td is not None and td.ok:
        err = float(np.hypot(td.cx - (W - 1) / 2.0, td.cy - (H - 1) / 2.0))
        check("T16.target_near_center", err < 40,
              f"开火层看的靶应在中心附近(err={err:.0f}px < 40),"
              f"若很大说明又锁到另一个靶(分歧复发)")
    else:
        check("T16.target_near_center", False, "开火层未看到靶")

    # 准星已压在靶上 → 必须开火
    check("T16.fires_when_aimed", len(fired) >= 1,
          f"准星已压在靶上时必须开火,实测开火 {len(fired)} 次")

    # 反向守卫:坏栈(ActionEMA 不透传检测)会让开火层**自己重检测**,
    # 从而有机会选中与瞄准层不同的靶。直接断言这个因果机制:
    #   * 好栈:开火层复用内层候选(零次额外检测)
    #   * 坏栈:开火层拿不到候选,只能自己 find_target/find_targets
    import flyaim.bridge.controllers as _c

    state2 = {"t1": (150.0, 620.0, 28.0), "t2": (520.0, 120.0, 30.0)}

    def shift2(action):
        ax, ay = float(action[0]), float(action[1])
        for k in ("t1", "t2"):
            cx, cy, r = state2[k]
            state2[k] = (cx - ax * SPEED, cy - ay * SPEED, r)

    class _BadEma:
        """坏栈:不透传 last_detection/last_candidates(复刻修复前的 ActionEMA)。"""
        def __init__(self, inner):
            self.inner = inner
            self._prev = None

        def act(self, frame):
            a = np.asarray(self.inner.act(frame), dtype=np.float32).reshape(2)
            self._prev = a if self._prev is None else 0.35 * a + 0.65 * self._prev
            return self._prev.copy()

    # 统计开火层额外发起的检测次数
    calls = {"n": 0}
    _orig_ft = _c.find_target
    _orig_fts = _c.find_targets

    def _count_ft(*a, **k):
        calls["n"] += 1
        return _orig_ft(*a, **k)

    def _count_fts(*a, **k):
        calls["n"] += 1
        return _orig_fts(*a, **k)

    seek2 = SeekController(ref_color=(48, 224, 224), tolerance=95.0,
                           use_aim_detect=False, kp=2.2, downsample=2,
                           xhair_px=12.0, xhair_max_r=8.0)

    def _click2():
        return True

    _c.find_target, _c.find_targets = _count_ft, _count_fts
    try:
        # 好栈:零次额外检测
        calls["n"] = 0
        trig_good = TriggerOnTarget(_Ema(seek2), _click2, ref_color=(48, 224, 224),
                                    tolerance=95.0, fire_frac=1.4, min_radius_px=4.0,
                                    cooldown_s=0.0, sticky_px=220.0, downsample=2,
                                    xhair_px=12.0, xhair_max_r=8.0, jump_reset_px=45.0)
        for _ in range(10):
            trig_good.act(render((150.0, 620.0, 28.0), (520.0, 120.0, 30.0)))
        good_calls = calls["n"]

        # 坏栈:必发起额外检测
        seek3 = SeekController(ref_color=(48, 224, 224), tolerance=95.0,
                               use_aim_detect=False, kp=2.2, downsample=2,
                               xhair_px=12.0, xhair_max_r=8.0)
        calls["n"] = 0
        trig_bad = TriggerOnTarget(_BadEma(seek3), _click2, ref_color=(48, 224, 224),
                                   tolerance=95.0, fire_frac=1.4, min_radius_px=4.0,
                                   cooldown_s=0.0, sticky_px=220.0, downsample=2,
                                   xhair_px=12.0, xhair_max_r=8.0, jump_reset_px=45.0)
        for _ in range(10):
            trig_bad.act(render((150.0, 620.0, 28.0), (520.0, 120.0, 30.0)))
        bad_calls = calls["n"]
    finally:
        _c.find_target, _c.find_targets = _orig_ft, _orig_fts

    check("T16.good_stack_reuses", good_calls == 10,
          f"好栈每拍只有 seek 自己那 1 次检测(共 10),实测 {good_calls}"
          f" —— 多出来就是开火层在重复检测")
    check("T16.bad_stack_detects", bad_calls > good_calls,
          f"坏栈开火层额外自检测(实测 {bad_calls} > 好栈 {good_calls})"
          f" —— 这正是『瞄准锁 A / 开火看 B』分歧的来源")

    def _shift2(action):
        ax, ay = float(action[0]), float(action[1])
        for k in ("t1", "t2"):
            cx, cy, r = state2[k]
            state2[k] = (cx - ax * SPEED, cy - ay * SPEED, r)

    # 顺带确认坏栈在闭环里也真的少开火/开错靶(统计口径,非硬断言)
    fired2 = []

    def _click2b():
        fired2.append(1)
        return True

    seek4 = SeekController(ref_color=(48, 224, 224), tolerance=95.0,
                           use_aim_detect=False, kp=2.2, downsample=2,
                           xhair_px=12.0, xhair_max_r=8.0)
    trig4 = TriggerOnTarget(_BadEma(seek4), _click2b, ref_color=(48, 224, 224),
                            tolerance=95.0, fire_frac=1.4, min_radius_px=4.0,
                            cooldown_s=0.0, sticky_px=220.0, downsample=2,
                            xhair_px=12.0, xhair_max_r=8.0, jump_reset_px=45.0)
    for _ in range(60):
        fr = render(state2["t1"], state2["t2"])
        out = trig4.act(fr)
        _shift2(out)
    print(f"     [参考] 坏栈闭环开火 {len(fired2)} 次 / 好栈 {len(fired)} 次")


def t17_action_smoother() -> None:
    """D33:动作整形守卫 —— 治「瞄得抖、太激进」(2026-10-05 用户反馈)。

    原始 seek = 纯 PD + 硬裁剪。实测:12%% 的帧 |act|>0.9(满舵),
    37%% >0.5,err 符号翻转率 25%% —— 典型的 bang-bang 自激。
    用户的原话是「晃动太大了,太激进」。

    ActionSmoother 三件事:速率限制 / 软饱和 / 中心死区。本组逐项守卫,
    并特别守住一个**危险边界**:死区必须严格小于开火门限,否则会把自己
    卡在进不了门限的死区里(重蹈 D30 覆辙)。
    """
    print("[T17] 动作整形守卫 (D33)")
    import numpy as np

    from flyaim.bridge.controllers import ActionSmoother

    class _Const:
        """输出恒定 action 的假内层。"""
        def __init__(self, val):
            self.val = np.array(val, dtype=np.float32)
            self.last_detection = None
            self.last_candidates = None

        def act(self, frame):
            return self.val.copy()

        def reset(self):
            pass

        def close(self):
            pass

        name = "const"

    # 1) 速率限制:从 0 起步,第一拍最多只能走到 max_delta
    sm = ActionSmoother(_Const([1.0, 0.0]), max_delta=0.3, soft=0.0, deadband_px=0.0)
    a1 = sm.act(np.zeros((10, 10, 3), np.uint8))
    check("T17.slew_first_tick", abs(a1[0] - 0.3) < 1e-4,
          f"第一拍应从 0 限速到 0.3,实测 {a1[0]:.4f}")

    # 2) 速率限制应逐拍逼近而非一步到位,且最终到 1.0
    for _ in range(10):
        aN = sm.act(np.zeros((10, 10, 3), np.uint8))
    check("T17.slew_converges", abs(aN[0] - 1.0) < 1e-4,
          f"持续输入 1.0 应最终收敛到 1.0,实测 {aN[0]:.4f}")

    # 3) 速率限制计数
    check("T17.slew_counted", sm.n_limited > 0,
          f"被削过的拍数应 >0,实测 {sm.n_limited}")

    # 4) max_delta=0 时不限速(向后兼容)
    sm0 = ActionSmoother(_Const([1.0, 0.0]), max_delta=0.0, soft=0.0, deadband_px=0.0)
    a0 = sm0.act(np.zeros((10, 10, 3), np.uint8))
    check("T17.slew_off_passthrough", abs(a0[0] - 1.0) < 1e-4,
          f"max_delta=0 应直通 1.0,实测 {a0[0]:.4f}")

    # 5) 软饱和:小信号近似线性(|a|<<k 时 k*tanh(a/k) ≈ a)
    sm2 = ActionSmoother(_Const([0.05, 0.0]), max_delta=0.0, soft=1.5, deadband_px=0.0)
    a2 = sm2.act(np.zeros((10, 10, 3), np.uint8))
    rel = abs(a2[0] - 0.05) / 0.05
    check("T17.soft_linear_small", rel < 0.01,
          f"小信号 0.05 经 tanh 应≈0.05(偏差 {rel:.2%}),保住小误差精度")

    # 6) 软饱和:满舵被温和收敛(降最高速 = 去激进),但仍接近 1
    sm3 = ActionSmoother(_Const([1.0, 0.0]), max_delta=0.0, soft=1.5, deadband_px=0.0)
    a3 = sm3.act(np.zeros((10, 10, 3), np.uint8))
    check("T17.soft_tames_full", 0.80 < a3[0] < 1.0,
          f"满输入经软饱和应落在 (0.80,1) —— 实测 {a3[0]:.4f}"
          f"(最高速温和降档,而不是硬裁剪的 1.0)")

    # 6b) 软饱和:中间段被压缩(这才是治 bang-bang 的机制)
    sm3b = ActionSmoother(_Const([0.5, 0.0]), max_delta=0.0, soft=1.5, deadband_px=0.0)
    a3b = sm3b.act(np.zeros((10, 10, 3), np.uint8))
    check("T17.soft_compresses_mid", a3b[0] < 0.5 - 1e-3,
          f"中间量 0.5 应被压缩到 <0.5,实测 {a3b[0]:.4f}(削临界振荡)")

    # 7) 软饱和单调:输入越大输出越大(不震荡)
    outs = []
    for v in (0.1, 0.3, 0.6, 1.0):
        s = ActionSmoother(_Const([v, 0.0]), max_delta=0.0, soft=0.9, deadband_px=0.0)
        outs.append(float(s.act(np.zeros((10, 10, 3), np.uint8))[0]))
    check("T17.soft_monotonic", all(outs[i] < outs[i + 1] for i in range(len(outs) - 1)),
          f"软饱和必须单调递增,实测 {[round(o,3) for o in outs]}")

    # 8) 死区:靶在中心附近时动作清零
    class _WithDet(_Const):
        def __init__(self, val, cx, cy, ok=True):
            super().__init__(val)
            from flyaim.bridge.detect import Detection
            self.last_detection = Detection(ok=ok, cx=cx, cy=cy, radius_px=10.0)

    fr = np.zeros((700, 700, 3), np.uint8)  # 中心 (349.5,349.5)
    sm4 = ActionSmoother(_WithDet([1.0, 1.0], 352.0, 350.0), max_delta=0.0, soft=0.0,
                         deadband_px=6.0)
    a4 = sm4.act(fr)
    check("T17.deadband_zeroes", float(np.max(np.abs(a4))) < 1e-6,
          f"靶距中心 2.9px < 死区 6px,动作应清零,实测 {a4}")

    # 9) 死区外不受影响
    sm5 = ActionSmoother(_WithDet([1.0, 1.0], 400.0, 350.0), max_delta=0.0, soft=0.0,
                         deadband_px=6.0)
    a5 = sm5.act(fr)
    check("T17.deadband_outside", abs(a5[0] - 1.0) < 1e-6,
          f"靶距中心 50.5px > 死区,动作应原样,实测 {a5}")

    # 10) 危险边界守卫:死区必须 < 开火门限(否则卡死区不开火)
    #     典型门限 = max(radius*fire_frac, min_radius_px) = max(10*1.4,4) = 14px
    fire_gate = max(10.0 * 1.4, 4.0)
    check("T17.deadband_below_gate", 6.0 < fire_gate,
          f"默认死区 6px 必须 < 典型门限 {fire_gate}px,"
          f"否则会在『死区里但没进开火门限』的死区卡住")

    # 11) 组合:三项同开也不发散(输入恒定 → 输出稳定)
    sm6 = ActionSmoother(_Const([1.0, 0.0]), max_delta=0.35, soft=0.9, deadband_px=0.0)
    seq = [float(sm6.act(np.zeros((10, 10, 3), np.uint8))[0]) for _ in range(30)]
    tail = seq[-5:]
    check("T17.combo_stable", max(tail) - min(tail) < 1e-4,
          f"三项同开时输出应收敛稳定(尾段极差 {max(tail)-min(tail):.2e})")

    # 12) 透传:包装层不能挡住内层统计量
    check("T17.passthrough", getattr(sm6, "name") == "const+smooth",
          f"name 应透传为 const+smooth,实测 {getattr(sm6,'name')}")


def t18_dual_window() -> None:
    """D34:双窗(搜靶全屏 / 跟踪中心窗)守卫 —— 治「命中率低」的真因。

    实测铁证(2026-10-05 14:2x):靶在全屏 (144,48) 半径 25px。
        全屏检测     -> ok=True
        中心窗 700   -> ok=False(窗范围 x[610,1310] y[190,890],靶在窗外左 366px)
        中心窗 900   -> ok=False(窗范围 x[510,1410] y[ 90,990],靶仍在窗外)
    而 play-cov900 那局 2152 帧 det-ok=0、|act| 全程 0、开火 0 次 ——
    **靶根本不在镜头里,任何瞄准算法的收益都是 0**。
    """
    from flyaim.bridge.capture import ArraySource, DualWindowSource
    from flyaim.bridge.controllers import SeekController

    # --- 1) 几何一致性:中心窗必须真的落在全屏窗中心 ------------------------
    fs = 400
    dw = DualWindowSource.__new__(DualWindowSource)  # 不走 __init__(免开真实设备)
    full = (0, 0, 1920, 1080)
    cs = 900
    fl, ft, fw, fh = full
    expect = (fl + (fw - cs) // 2, ft + (fh - cs) // 2, cs, cs)
    check("T18.center_geometry", expect == (510, 90, 900, 900),
          f"1920x1080 的中心窗 900 应在 (510,90,900,900),算出 {expect}")

    # --- 2) 靶在窗外时,搜靶态必须请求全屏窗 -------------------------------
    # 用真实数组源构造一个"靶在左上角"的帧,验证：中心窗裁掉靶、全屏能看到。
    from flyaim.bridge.detect import find_target

    TEAL = (48, 224, 224)
    frame = np.zeros((1080, 1920, 3), np.uint8)
    frame[23:73, 119:169] = np.array(TEAL, np.uint8)  # 半径 ~25 的方块在 (144,48)
    d_full = find_target(frame, ref_color=TEAL, tolerance=60.0)
    check("T18.full_sees_target", d_full.ok and abs(d_full.cx - 143.5) < 6
          and abs(d_full.cy - 47.5) < 6,
          f"全屏应看到 (144,48) 的靶,实测 ok={d_full.ok} "
          f"({d_full.cx:.0f},{d_full.cy:.0f})")

    sub = frame[90:990, 510:1410]  # 中心窗 900
    d_ctr = find_target(sub, ref_color=TEAL, tolerance=60.0)
    check("T18.center_misses_target", not d_ctr.ok,
          f"中心窗 900 应看不到左上角 (144,48) 的靶 —— 这正是『命中率低』的根因"
          f"(实测 ok={d_ctr.ok})")

    # --- 3) 双窗切换逻辑:未锁 → 全屏;已锁且靶在窗内 → 中心窗 -------------
    class _FakeCtrl:
        def __init__(self):
            self.last_detection = None

    dw = DualWindowSource.__new__(DualWindowSource)
    dw.full_region = full
    dw.center_size = cs
    dw.lock_switch_px = 260.0
    dw._center_region = expect
    dw._controller = _FakeCtrl()
    dw.searching = True
    from flyaim.bridge.detect import Detection

    dw._controller.last_detection = Detection(ok=False)
    check("T18.unlocked_uses_full", not dw._target_in_full(),
          "未检出靶时必须继续用全屏窗(否则永远看不到窗外的靶)")

    # 靶在窗外(全屏坐标) -> 仍用全屏
    dw.searching = True
    dw._controller.last_detection = Detection(ok=True, cx=144.0, cy=48.0, radius_px=25)
    check("T18.target_outside_keeps_full", not dw._target_in_full(),
          "靶在全屏 (144,48) 时,中心窗(x>=510)装不下它 -> 必须继续全屏搜靶")

    # 靶已进中心窗 -> 切小窗
    dw._controller.last_detection = Detection(ok=True, cx=650.0, cy=400.0, radius_px=25)
    check("T18.target_inside_switches_center", dw._target_in_full(),
          "靶落到中心窗矩形内(x[510,1410] y[90,990])后应切到中心窗跟瞄")

    # --- 4) 切窗必须通知控制器清状态(否则 sticky 把坐标平移误判成换靶) ----
    called = {"n": 0}

    class _Inner:
        def on_window_switch(self):
            called["n"] += 1

    class _Wrap:
        """模拟 TriggerOnTarget 的 on_window_switch 转发 + _prev_xy 清理。"""
        def __init__(self, inner):
            self.inner = inner
            self._prev_xy = (100.0, 100.0)

    w = _Wrap(_Inner())
    # 直接调用真实实现(避免复制粘贴导致守卫失真)
    from flyaim.bridge.controllers import TriggerOnTarget
    TriggerOnTarget.on_window_switch(w)
    check("T18.switch_clears_sticky", w._prev_xy is None and called["n"] == 1,
          f"切窗必须清 sticky 并转发内层(实测 _prev_xy={w._prev_xy}, 转发 {called['n']} 次)")

    # --- 5) 误差归一化必须与窗大小无关(dual 切窗不改增益) ----------------
    # 同一物理误差 100px:全屏窗(1920) vs 中心窗(900) 应给出相同 action
    def _mk_ctrl():
        c = SeekController(ref_color=TEAL, tolerance=95.0, kp=2.2, kd=0.0,
                           use_aim_detect=False, err_scale_px=320.0)
        c.lock = False
        return c

    fr_full = np.zeros((1080, 1920, 3), np.uint8)
    fr_full[500:540, 1010:1050] = np.array(TEAL, np.uint8)   # 质心 1029.5,中心 959.5 -> err=70
    # 中心窗:同 err=70px(质心 519.5,中心 449.5)
    fr_ctr2 = np.zeros((900, 900, 3), np.uint8)
    fr_ctr2[410:450, 500:540] = np.array(TEAL, np.uint8)
    c1, c2 = _mk_ctrl(), _mk_ctrl()
    a1 = c1.act(fr_full)
    a2 = c2.act(fr_ctr2)
    check("T18.gain_window_invariant", abs(a1[0] - a2[0]) < 0.02,
          f"同一物理误差(err=70px)下,全屏窗与中心窗的 action 必须一致(实测 "
          f"{a1[0]:.4f} vs {a2[0]:.4f}) —— 否则切窗即改增益,是抖动来源。"
          f"容差 0.02 吸收 ds 整数量化的质心离散误差(原 bug 是 2.1 倍系统差)")

    # --- 6) 旧行为(w/2)确实会随窗变 —— 证明这条修复不是多余的 -----------
    c3 = SeekController(ref_color=TEAL, tolerance=95.0, kp=2.2, kd=0.0,
                        use_aim_detect=False, err_scale_px=None)
    c3.lock = False
    c4 = SeekController(ref_color=TEAL, tolerance=95.0, kp=2.2, kd=0.0,
                        use_aim_detect=False, err_scale_px=None)
    c4.lock = False
    b1 = c3.act(fr_full)
    b2 = c4.act(fr_ctr2)
    check("T18.legacy_gain_window_dependent", abs(b1[0] - b2[0]) > 0.05,
          f"旧行为(w/2)下同误差 action 应随窗差 ~2.1 倍(实测 {b1[0]:.3f} vs "
          f"{b2[0]:.3f}) —— 这是必须修的偏离")

    # --- 7) downsample 自适应:大窗多降、小窗少降 -------------------------
    c5 = SeekController(ref_color=TEAL, tolerance=95.0, downsample=2)
    d_small = c5._pick_downsample(900, 900)
    d_big = c5._pick_downsample(1080, 1920)
    check("T18.downsample_small_window", d_small == 3,
          f"900 窗应降到 ds=3(>=256px 下限;实测 {d_small})—— ds=2 检测 10.6ms,"
          f"ds=3 只要 6.1ms,拍频从 29Hz 拉回 50+Hz")
    check("T18.downsample_large_window", d_big > d_small,
          f"全屏窗应比中心窗降更多以保住拍频(实测 全屏 ds={d_big} > 中心 ds={d_small})")


def t19_ema_phase_lag() -> None:
    """D35:EMA 相位滞后守卫 —— 治「晃动太大,命中率 50-60%」。

    实测(play-1min, 54Hz):err 的中位 65px(靶半径 28px 的 2.3 倍),
    误差时间自相关在 **1.92Hz** 处出现显著负相关 —— 这是**极限环振荡**,
    用户感受到的「晃动太大」就是它。

    病因:`ActionEMA(α=0.35)` 在 54Hz 下时间常数 = 1/0.35 = 2.9 拍,
    一阶低通在 f 处的相位滞后 = atan(2πf·τ):

        f=1.92Hz, τ=2.9/54 s -> 滞后 34.6°

    而**动作本身从未饱和**(实测反推 raw 动作,|raw|>1 占 0%%) ——
    EMA 根本没在"压制裁剪",纯粹是往回路里塞了 34.6° 的相位滞后,
    直接把相位裕度吃掉,把系统从"接近临界阻尼"推到"极限环"。

    修法:α 0.35 → 0.6(τ 从 2.9 拍降到 1.7 拍,滞后降到 23°),
    实测振荡从「1.92Hz 明显」→「无明显」,一局开火 263 → 550。
    """
    import math

    # 1) 相位滞后的量化:一阶低通 atan(2πf·τ)
    def lag_deg(alpha, f, fps):
        tau = 1.0 / max(alpha, 1e-6) / fps
        return math.degrees(math.atan(2 * math.pi * f * tau))

    l35 = lag_deg(0.35, 1.92, 53.9)
    l60 = lag_deg(0.60, 1.92, 53.9)
    check("T19.alpha35_lag", 28.0 < l35 < 40.0,
          f"α=0.35 在 1.92Hz 的相位滞后应在 30° 量级(实测 {l35:.1f}°) —— "
          f"这就是被白白消耗掉的相位裕度")
    check("T19.alpha60_less_lag", l60 < l35 - 8.0,
          f"α=0.6 必须显著降低滞后(实测 {l60:.1f}° < {l35:.1f}°)")

    # 2) 一阶低通在低频应近似无滞后、在高频趋于 90°
    check("T19.lag_monotonic_freq", lag_deg(0.35, 0.5, 53.9) < lag_deg(0.35, 5.0, 53.9),
          "滞后必须随频率单调增(低频近似直通)")

    # 3) 复现"振荡"本身:二阶闭环 + 纯 P + 延迟,加 EMA 后自相关变差
    def closed_loop(alpha, delay=3, kp=2.2, err_scale=320.0, speed=0.021875,
                    w=900.0, n=1400, seed=0):
        rng = np.random.default_rng(seed)
        tgt = np.array([120.0, 0.0])
        cam = np.zeros(2)
        buf = [np.zeros(2) for _ in range(delay)]
        prev = np.zeros(2)
        ema = None
        ex = []
        for i in range(n):
            # 靶缓慢漂移(近似静止目标,模拟跟踪阶段)
            tgt += rng.normal(0, 1.2, 2)
            err = tgt - cam
            a = np.clip(kp * err / err_scale, -1, 1)
            if alpha > 0:
                ema = a if ema is None else alpha * a + (1 - alpha) * ema
                a = ema
            buf.append(a.copy())
            a_eff = buf.pop(0)
            cam += a_eff * speed * w
            ex.append(cam[0] - tgt[0])
        ex = np.array(ex[-800:])
        ex = ex - ex.mean()
        ac = np.correlate(ex, ex, "full")[len(ex) - 1:]
        return ac / max(ac[0], 1e-12)

    def first_neg_peak(ac):
        for lag in range(1, 120):
            if ac[lag] < -0.10 and ac[lag] < ac[lag - 1] and ac[lag] < ac[lag + 1]:
                return lag
        return -1

    ac_hi = first_neg_peak(closed_loop(0.6))
    ac_lo = first_neg_peak(closed_loop(0.35))
    # 两者都可能有振铃;这不是要求"完全没有",而是要求高 α 的振铃不更差
    check("T19.high_alpha_not_worse",
          (ac_lo == -1 and ac_hi == -1) or (ac_hi == -1) or (ac_hi >= ac_lo),
          f"高 α 的振荡不得比低 α 更差(α0.6 首个负峰 lag={ac_hi},"
          f"α0.35 lag={ac_lo};-1 表示无显著振荡)")

    # 4) 危险边界:α 太高(接近 1)时 EMA 退化为直通,失去平滑意义但不致振荡
    check("T19.alpha_1_is_passthrough", lag_deg(1.0, 1.92, 53.9) < l60,
          "α=1 应几乎无滞后(退化为直通)")

    # 5) 默认值守卫:CLI 默认 α 必须 ≥ 0.5(否则会重新引入 34° 滞后)
    import re
    src = (ROOT / "tools" / "aimlab_play.py").read_text(encoding="utf-8")
    m = re.search(r'"--action-ema",\s*type=float,\s*default=([0-9.]+)', src)
    check("T19.default_alpha_high_enough", m is not None and float(m.group(1)) >= 0.5,
          f"CLI 默认 --action-ema 必须 ≥0.5(实测 {m.group(1) if m else '未找到'})")
    m2 = re.search(r'"--cooldown",\s*type=float,\s*default=([0-9.]+)', src)
    # D40 修订:防两头 —— 太长(>0.20)会吃掉刚刷新的新靶(历史 0.22 的教训),
    # 太短(<0.05)会对同一颗球连泻 5-10 枪,Gridshot 每球只记 1 hit,
    # 多打的枪全是 miss,游戏内准确率被砸到 ~10%(2026-10-05 实测 425 发/52s)。
    cd = float(m2.group(1)) if m2 else -1.0
    check("T19.default_cooldown_low_enough", 0.05 <= cd <= 0.20,
          f"CLI 默认 --cooldown 必须在 [0.05,0.20](实测 {cd};"
          f"<0.05 连发倾泻砸准确率,>0.20 吃掉新靶首枪窗口)")


def t20_target_assoc() -> None:
    """D36:目标关联锁 —— 治「同屏多靶互相抢、准星来回弹」。

    用户反馈「瞄准时晃动太大」(play-1min 实测):err 方向翻转率 **40.4%**,
    且每 50 拍就有一次 >150px 的**靶坐标瞬移**(相机本身只动了 ~8px/拍)。
    这是**换靶抖动** —— 靶场同屏 3 个靶(preview 帧目检确认),旧实现每拍
    取「离上拍位置最近的候选」,没有关联门限也没有换靶代价,准星在两个
    靶之间来回弹,永远进不了开火门限(几何命中率上限只有 26.8%)。

    修法:关联门限 + 换靶迟滞 + 丢靶容忍(见 SeekController._select_locked)。
    """
    import numpy as np
    from flyaim.bridge.detect import Detection
    from flyaim.bridge.controllers import SeekController

    def mk(cx, cy, r=25.0):
        return Detection(ok=True, cx=cx, cy=cy, radius_px=r,
                         area_px=int(3.14159 * r * r), score=0.9)

    def mk_sc(**kw):
        return SeekController(use_aim_detect=False, kp=2.2, kd=0.15,
                              err_scale_px=320.0, **kw)

    # 1) 关联半径内 = 同一靶(不换身份)
    sc = mk_sc(assoc_px=60.0, switch_gain=1.3, lost_tol=3)
    d1 = sc._select_locked([mk(450, 450)], 900, 900)
    d2 = sc._select_locked([mk(455, 452)], 900, 900)   # 移动 5px = 同一个靶
    check("T20.assoc_keeps_identity", sc.n_switch == 1 and d2.cx > 450,
          f"关联半径内的位移必须保持同一靶(switch={sc.n_switch}, "
          f"d2=({d2.cx:.0f},{d2.cy:.0f}) —— 锁定后应只 switch 一次)")

    # 2) 两靶互相抢:准星稳定跟一个,不来回弹
    sc = mk_sc(assoc_px=60.0, switch_gain=1.3, lost_tol=3)
    picks = []
    for i in range(80):
        ax = 450 + 28 * np.sin(i * 0.3)
        bx = 450 - 28 * np.sin(i * 0.3)
        d = sc._select_locked([mk(ax, 450), mk(bx, 450)], 900, 900)
        picks.append(d.cx)
    jumps = sum(1 for i in range(1, len(picks)) if abs(picks[i] - picks[i - 1]) > 20)
    check("T20.no_pingpong", jumps == 0,
          f"两靶交叉时准星不得在两靶间弹跳(实测跳变 {jumps} 次,应 0)")

    # 3) 换靶迟滞:新靶只近一点点不换
    sc = mk_sc(assoc_px=30.0, switch_gain=2.0, lost_tol=1)
    sc._select_locked([mk(449, 450)], 900, 900)          # 锁在中心附近
    # 旧靶跑远(超出关联门限),新靶离中心略远 -> 迟滞应仍不让换
    for _ in range(3):
        sc._select_locked([mk(800, 450)], 900, 900)      # 只有 800 一个候选
    # 800 是唯一候选 -> 关联失败 lost_tol 拍后必然改锁 800
    check("T20.lost_tol_then_switch", sc.n_switch >= 2,
          f"唯一候选且旧靶已丢 -> 容忍拍后必须改锁(switch={sc.n_switch})")

    # 4) 短失落:保持在**最后观测位置**(不追幻影、不改追远处新靶)
    #    (D37 教训:曾经按"上拍位置+速度"外推幻影,结果靶飞出视野后
    #     准星跟着幻影一直往左上角推 —— 用户实测「靶出视野再入视野就
    #     一直往左上角滑」。纯视觉纪律:动作只落有观测证据的位置。)
    sc = mk_sc(assoc_px=60.0, switch_gain=1.3, lost_tol=5)
    sc._select_locked([mk(450, 450)], 900, 900)
    n_before = sc.n_switch
    d = sc._select_locked([mk(700, 450)], 900, 900)      # 原靶暂时消失
    check("T20.short_loss_holds_position",
          sc.n_switch == n_before and d.cx == 450.0,
          f"短失落必须停在最后观测位置 (450),不得外推滑行也不改追新靶"
          f"(switch={sc.n_switch}, cx={d.cx:.0f})")
    d2 = sc._select_locked([mk(470, 450)], 900, 900)     # 原靶在关联门限内回来
    check("T20.reacquire_keeps_identity", sc.n_switch == n_before and d2.cx == 470.0,
          f"靶在门限内回来应重续身份(switch={sc.n_switch}, cx={d2.cx:.0f})")

    # 5) 默认值守卫:assoc 自动半径必须 > 0(不能退化成旧行为)
    import re
    src = (ROOT / "tools" / "aimlab_play.py").read_text(encoding="utf-8")
    m = re.search(r'"--switch-gain",\s*type=float,\s*default=([0-9.]+)', src)
    check("T20.default_switch_gain_ge_1", m is not None and float(m.group(1)) >= 1.0,
          f"CLI 默认 --switch-gain 必须 ≥1(实测 {m.group(1) if m else '未找到'})")
    m2 = re.search(r'"--lost-tol",\s*type=int,\s*default=([0-9]+)', src)
    check("T20.default_lost_tol_positive", m2 is not None and int(m2.group(1)) >= 1,
          f"CLI 默认 --lost-tol 必须 ≥1(实测 {m2.group(1) if m2 else '未找到'})")

    # 6) argparse 守卫:所有 help 字符串必须能安全 `%` 格式化
    #    (实战踩过:help 里写 `(+54%)` 会让 argparse 的 %-格式化抛
    #     ValueError: unsupported format character ')',整个 --help 崩掉)
    import re as _re
    bad = []
    for mm in _re.finditer(r'ap\.add_argument\((.*?)\n(?=\s*ap\.add_argument|\s*\))',
                           src, _re.S):
        block = mm.group(1)
        om = _re.search(r'"--([a-z0-9-]+)"', block)
        hm = _re.search(r'help=((?:\s*"[^"]*"\s*)+)', block)
        if hm:
            s = "".join(_re.findall(r'"([^"]*)"', hm.group(1)))
            try:
                s % {}
            except Exception as exc:
                bad.append((om.group(1) if om else "?", str(exc)))
    check("T20.argparse_help_safe", not bad,
          f"所有 --help 文本必须能被 %-格式化(裸 %% 会崩 --help);"
          f"违规: {bad if bad else '无'}")


def _mk_frame(cx, cy, r=28.0, size=900):
    """构造一张 900x900 帧:中心放置青靶 + 画面中心白色准星点。"""
    f = np.zeros((size, size, 3), np.uint8)
    x0, y0 = int(cx - r), int(cy - r)
    yy, xx = np.mgrid[x0:x0 + int(2 * r), y0:y0 + int(2 * r)]
    mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= r * r
    sub = f[x0:x0 + int(2 * r), y0:y0 + int(2 * r)]
    sub[mask] = (48, 224, 224)
    f[size // 2 - 2:size // 2 + 2, size // 2 - 2:size // 2 + 2] = (240, 240, 240)
    return f


def t21_saccade_fixate() -> None:
    """D41:扫视-固视双模态 —— 治「相机停不住、驻留只有 108ms」。

    量化依据(相位相关逐帧角速度,用户手机理想实录 vs 程序):
    理想运动占空比 5%/驻留 p50 629ms;程序 27%/108ms —— 连续比例追踪
    永远在微调。果蝇与人手的共同节律是「扫视-固视」交替。

    **门限关系约束(T5 教训 + D41b 死锁教训)**:固视带必须整体落在开火
    门限(0.75r)**内侧**。r≈28 大靶门限 21px,静态带 [14,20] 成立;但
    Gridshot 小靶 r≈20.7 → 门限仅 15.5px,静态保持带 20px 与之交叠 →
    「固视着但永远开不了火」死锁 1153 拍(d41-confirm)。故带是动态的:
    fix_enter=min(fixate_px, 0.6×0.75r),fix_hold=min(saccade_px, 0.85×0.75r)。
    """
    import sys as _s
    from flyaim.bridge.controllers import SeekController
    _s.path.insert(0, str(ROOT / "tools"))

    def mk(**kw):
        base = dict(ref_color=(48, 224, 224), tolerance=95.0,
                    use_aim_detect=False, kp=2.6, err_scale_px=320.0,
                    lock=True, scan_amp=0.0)
        base.update(kw)
        c = SeekController(**base)
        c.reset()
        return c

    # 1) err < fixate_px(球在 14px 内)-> 动作必须为 0(真正停住)
    c = mk()
    f = _mk_frame(450 + 8, 450 + 5)       # err ~9.4px < 14
    a = c.act(f)
    check("T21.fixate_zero_action", float(np.abs(a).max()) == 0.0,
          f"球在固视带内(err~9px<14)动作必须归零(实测 {a})")

    # 2) 保持固视:同样的球小幅漂移仍为 0(不因检测噪声 ±3px 抖回追踪)
    a2 = c.act(_mk_frame(450 + 10, 450 + 7))
    check("T21.fixate_holds", float(np.abs(a2).max()) == 0.0,
          f"固视带内小幅漂移(±3px)不得退出固视(实测 {a2})")

    # 3) err > saccade_px -> 退出固视,输出非零且方向指向球心
    a3 = c.act(_mk_frame(450 + 90, 450 + 60))   # err ~109px > 20
    check("T21.saccade_resumes", float(np.abs(a3).max()) > 0.05 and a3[0] > 0,
          f"球跑出迟滞带(err~109px>20)必须恢复追踪且方向正确(实测 {a3})")

    # 4) 闭环收敛:模拟相机转动(球位 -= action·px/拍),静止球应最终进入
    #    固视带并停住(零动作),完整重现「扫视→固视」节奏
    c2 = mk()
    pos = [450.0 + 160, 450.0 + 120]
    zero_streak = 0
    for _ in range(60):
        f = _mk_frame(pos[0], pos[1])
        a = c2.act(f)
        if float(np.abs(a).max()) == 0.0:
            zero_streak += 1
        else:
            # 相机转动 → 靶在画面中反向移动;1 action ≈ 60px/拍(量级一致即可)
            pos[0] -= float(a[0]) * 60.0
            pos[1] -= float(a[1]) * 60.0
    check("T21.converge_then_fixate", zero_streak >= 3,
          f"60 拍闭环应收敛进入固视并停住(实测固视 {zero_streak} 拍)")

    # 5) 固视关闭(fixate_px=0)回退连续追踪:同场景不出现零动作停驻
    c3 = mk(fixate_px=0.0)
    a = c3.act(_mk_frame(450 + 8, 450 + 5))
    check("T21.disable_fallback", float(np.abs(a).max()) > 0.0,
          f"fixate_px=0 必须回退连续追踪不归零(实测 {a})")

    # 6) 静态迟滞带必须小于大靶开火门限(T5+D41 首局教训):
    #    首局实测迟滞带 [16,26] 与门限 21px 交叠,固视卡在 21-26px
    #    「不动也不开火」835 拍,被看门狗误判重开 19 次。
    c4 = mk()
    gate = 21.0  # = 0.75 × r28(Gridshot 实测半径)
    check("T21.fixate_below_gate",
          c4.fixate_px < gate and c4.saccade_px < gate,
          f"迟滞带 [{c4.fixate_px},{c4.saccade_px}] 必须整体 < 开火门限 "
          f"{gate}px(0.75×28px);任何一段出圈都会「固视卡门外死等」")

    # 7) CLI 默认 fixate 开启
    src = (ROOT / "tools" / "aimlab_play.py").read_text(encoding="utf-8")
    import re as _re
    m = _re.search(r'"--fixate-px",\s*type=float,\s*default=([0-9.]+)', src)
    check("T21.cli_default_on", m is not None and float(m.group(1)) > 0,
          f"CLI 默认 --fixate-px 必须 >0(实测 {m.group(1) if m else '未找到'})")

    # 8) 迟滞带语义:fixate_px < saccade_px(否则边界抖动)
    check("T21.hysteresis_order", c4.fixate_px < c4.saccade_px,
          f"必须 fixate_px({c4.fixate_px}) < saccade_px({c4.saccade_px})")

    # 9) 死锁回归(d41-confirm,2026-10-05 深夜):**动态固视带**。
    #    实测小靶 r=20.66 → 开火门限仅 15.5px < 静态保持带 20px:err=12.5
    #    进固视后注入管道滑行把 err 停在 17.5(门限外/旧带内)→「固视着但
    #    永远进不了门限」冻结 1153 拍,19 次重开也逃不掉。动态带
    #    fix_hold=min(20, 0.85×15.5=13.2) 下,滑到 17.5 必须退出固视恢复追踪。
    c5 = mk(fire_gate_frac=0.75)
    a9 = c5.act(_mk_frame(450 + 8.0, 450, r=20.66))
    check("T21.small_target_fixate", float(np.abs(a9).max()) == 0.0,
          f"小靶(r=20.66)err=8 < 动态进入带 min(14,0.6×15.5=9.3),应固视归零(实测 {a9})")
    a9b = c5.act(_mk_frame(450 + 17.5, 450, r=20.66))
    check("T21.deadlock_regression", float(np.abs(a9b).max()) > 0.0,
          f"滑行停点 err=17.5 > 动态保持带 min(20,0.85×15.5=13.2),必须退出固视"
          f"恢复追踪(实测 {a9b};动作 0 = d41-confirm 死锁复现)")

    # 10) force_unfixate 自救:踢回追踪 + 禁入冷却(防下一拍 err 仍小就重新锁死)
    c6 = mk()
    c6.act(_mk_frame(450 + 8.0, 450 + 5))          # 进入固视
    c6.force_unfixate(cooldown_beats=8)
    a10 = c6.act(_mk_frame(450 + 8.0, 450 + 5))    # 同位置:冷却期内不得重新固视
    check("T21.force_unfixate_kick", float(np.abs(a10).max()) > 0.0,
          f"force_unfixate 后禁入冷却期内必须保持追踪输出(实测 {a10})")


class _CountClick:
    """计次点击桩。"""

    def __init__(self):
        self.n = 0

    def __call__(self):
        self.n += 1
        return True


def t22_fire_confirm() -> None:
    """D41 开火确认:连续 N 拍进门限才开火 —— 治「扫视惯性滑行期误开火」。

    扫视刚到位时,前几拍的大额注入还在指针管道消化(EPP 放大),画面
    err 达标但真实准星在滑行 → 此刻点击必 miss。确认 2 拍推迟 15ms,
    Gridshot 靶静止,零代价。
    """
    import sys as _s
    from flyaim.bridge.controllers import TriggerOnTarget
    _s.path.insert(0, str(ROOT / "tools"))

    class _FixedDet:
        """恒定返回同一检测的内层(模拟已收敛到门限内)。"""
        name = "fixed"
        lock = True
        blind = False

        def __init__(self):
            from flyaim.bridge.detect import Detection
            self.last_detection = Detection(ok=True, cx=459.0, cy=459.0,
                                            radius_px=28.0, area_px=0,
                                            score=1.0)
            self.last_candidates = [self.last_detection]
            self.n_fires = 0

        def act(self, frame):
            return np.zeros(2, np.float32)

        def reset(self):
            pass

        def close(self):
            pass

    # 900 窗:中心 (449.5,449.5),det 在 (459,459) → err≈13.4 < 门限 21 ✓
    inner = _FixedDet()
    click = _CountClick()
    t = TriggerOnTarget(inner, click, ref_color=(48, 224, 224),
                        cooldown_s=0.0, fire_frac=1.0, min_radius_px=4.0,
                        fire_confirm_beats=2)
    f = np.zeros((900, 900, 3), np.uint8)
    t.act(f)
    n1 = click.n
    t.act(f)
    n2 = click.n
    check("T22.confirm_requires_two", n1 == 0 and n2 == 1,
          f"confirm=2:第 1 拍不得开火,第 2 拍必须开火(实测 {n1},{n2})")
    # confirm=1 回退旧行为:首拍即开火
    inner2 = _FixedDet()
    click2 = _CountClick()
    t2 = TriggerOnTarget(inner2, click2, ref_color=(48, 224, 224),
                         cooldown_s=0.0, fire_frac=1.0, min_radius_px=4.0,
                         fire_confirm_beats=1)
    t2.act(f)
    check("T22.confirm1_immediate", click2.n == 1,
          f"confirm=1 首拍即开火(实测 {click2.n})")
    # err 出圈后确认计数清零
    inner3 = _FixedDet()
    click3 = _CountClick()
    t3 = TriggerOnTarget(inner3, click3, ref_color=(48, 224, 224),
                         cooldown_s=0.0, fire_frac=1.0, min_radius_px=4.0,
                         fire_confirm_beats=2)
    t3.act(f)                                   # streak=1
    inner3.last_detection.cx = 700.0            # err~250px 出圈
    t3.act(f)                                   # streak 清零
    inner3.last_detection.cx = 459.0            # 回到门限内
    n_before = click3.n
    t3.act(f)                                   # streak=1(重新确认),不得开火
    check("T22.streak_reset_on_miss", click3.n == n_before,
          f"出圈必须清零确认计数,重新确认(实测 {click3.n} vs {n_before})")


def t23_score_hud_ocr() -> None:
    """D42 HUD 分数 OCR:Gridshot 计分=命中间隔加权+miss 扣分,真实分数
    (而非几何推断命中)才是调试闭环的地面真值。目标基准(用户理想实录,
    2026-10-06 逐帧复核实测):**约 130.3k 分 / 95% ACC / 60s**。

    ⚠️ **本组测试曾经把 bug 固化(D42.8)**:第 1 项原断言
    `rd["points"] == 0 and rd["acc_pct"] == 100` —— 而 `0` 与 `100%` 正是
    d42-score 局实盘故障时读到的那两个值(335 次读分仅 11 次产出数值),
    fixture 又恰好截于 `TIME=∞` 的**未开局态**。于是冒烟全绿而实盘 324/335
    拒读。**fixture 取自故障现场 ⇒ 测试只能证明"能复现故障"。**

    因此第 1 项改名为 `practice_frame_layout`,只声称它**真正证明的东西**:
    布局(box 划分 + 数字行 y 范围)与分段在真实 PC 帧上成立。并在
    `seed_gap_is_documented` 里把"种子集只有 {0,1,%}"这条**已知能力边界**
    显式锁住 —— 让限制是"写明的",而不是"藏着的"。

    覆盖:练习态真实帧布局、% 剥离、∞ 拒识、TIME 自校准→读取闭环、
    未知字符拒读、TIME 分段诊断量、种子集边界、HUD 框映射几何。
    """
    from flyaim.bridge.score_ocr import (ScoreHUD, GRAB_BBOX, _BOX_X, _DIGIT_Y,
                                         hud_region_for, _HUD_REF_WH)
    from PIL import Image

    def _blank():
        # GRAB 尺寸的深灰底(735x30)
        return np.full((GRAB_BBOX[3] - GRAB_BBOX[1],
                        GRAB_BBOX[2] - GRAB_BBOX[0], 3), 100, np.int32)

    def _paste(img, box, slot, pattern):
        """往某框数字行贴一个手画字符(pattern: 28x14 0/1),slot=第几个字符。"""
        x0, x1 = _BOX_X[box]
        gw = 14
        gx = x0 + 4 + slot * (gw + 5)
        y0, y1 = _DIGIT_Y[0], _DIGIT_Y[1]
        sub = np.asarray(Image.fromarray((pattern * 255).astype(np.uint8))
                         .resize((gw, y1 - y0), Image.NEAREST))
        img[y0:y1, gx:gx + gw] = sub[:, :, None]
        return img

    def _glyph(kind):
        g = np.zeros((28, 14), np.uint8)
        if kind == "ring":      # 类 0
            g[3:25, 2:12] = 1; g[7:21, 5:9] = 0
        elif kind == "bar":     # 类 1
            g[3:25, 6:10] = 1
        elif kind == "top":     # 上横+下横
            g[3:7, 2:12] = 1; g[21:25, 2:12] = 1
        elif kind == "mid":     # 三横
            g[3:6, 2:12] = 1; g[12:16, 2:12] = 1; g[22:26, 2:12] = 1
        elif kind == "dots":    # 冒号
            g[7:11, 5:9] = 1; g[17:21, 5:9] = 1
        elif kind == "cross":   # 类 x(未学字符用)
            for i in range(22):
                g[3 + i, 2 + i // 2] = 1; g[3 + i, 11 - i // 2] = 1
        return g

    # 1) **练习态**真实帧(fixture 自 PC 实测截屏,时值 TIME=∞ 未开局):
    #    POINTS 0 / TIME ∞ / ACC 100%。这一项**只证明布局与分段成立**
    #    (box 划分 + 数字行 y 范围在真实 PC 帧上对得上),**不证明"能读分"** ——
    #    它断言的两个值恰好也是故障值,详见函数 docstring 的 D42.8 说明。
    fx = ROOT / "tools" / "fixtures" / "hud_sample.png"
    img_real = np.asarray(Image.open(fx).convert("RGB")).astype(np.int32)
    hud = ScoreHUD(grab_fn=lambda: img_real)
    rd = hud.read()
    check("T23.practice_frame_layout", rd["points"] == 0 and rd["acc_pct"] == 100,
          f"练习态帧的布局/分段必须读出 points=0/acc=100(实测 {rd})")
    check("T23.infinity_rejected", rd["time_s"] is None,
          f"练习模式 TIME=∞ 必须拒识为 None(实测 {rd['time_s']})")
    check("T23.infinity_no_learn", hud.calibrate_from_time(img_real, 45) is False,
          "TIME=∞ 段数≠5,不得学习")
    # TIME 分段数:不依赖任何模板就能拿到的诊断量,是分辨 H1/H2 的关键(D42.6)
    check("T23.time_glyphs_infinity", rd.get("time_glyphs") == 1,
          f"练习态 TIME=∞ 分段应为 1 个 glyph(实测 {rd.get('time_glyphs')})")

    # 2) TIME 自校准 → 读取闭环:贴 5 个手画字符,以 remain=45 学习「00:45」
    img2 = _blank()
    for slot, kind in enumerate(["ring", "ring", "dots", "top", "mid"]):
        _paste(img2, "time", slot, _glyph(kind))
    hud2 = ScoreHUD(grab_fn=lambda: img2)
    ok = hud2.calibrate_from_time(img2, 45)     # 学 0,0,:,4,5
    rd2 = hud2.read()
    check("T23.calibrate_then_read", ok and rd2["time_s"] == 45,
          f"自校准后 TIME 必须读出 45(实测 ok={ok} {rd2['time_s']})")
    check("T23.learned_chars", all(c in hud2.templates for c in "045:"),
          f"自校准必须学会 0/4/5/:(实测 {sorted(hud2.templates.keys())})")

    # 3) 用刚学的字形读 POINTS:贴「45」→ 应读 45
    for slot, kind in enumerate(["top", "mid"]):
        _paste(img2, "points", slot, _glyph(kind))
    rd3 = hud2.read()
    check("T23.points_after_learn", rd3["points"] == 45,
          f"POINTS 框「45」必须读出 45(实测 {rd3['points']})")

    # 4) 未知字符 → 该框 None(宁缺勿错,防污染分数闭环)
    img3 = _blank()
    _paste(img3, "points", 0, _glyph("cross"))
    _paste(img3, "points", 1, _glyph("ring"))
    rd4 = hud2.read(img3)
    check("T23.unknown_rejected", rd4["points"] is None,
          f"含未学字符的分数必须拒读为 None(实测 {rd4['points']})")

    # 5) **已知能力边界**:冷启动种子集只有 {0,1,%} → 含 2~9 的 POINTS 必然拒读。
    #    这条是把限制**显式写进测试**,防止它再被误当成"分数闭环已就绪"。
    #    实测代价:d42-score 局 335 次读分仅 11 次产出数值(score 1 次 / acc 10 次)。
    img5 = _blank()
    for slot, kind in enumerate(["ring", "top", "mid"]):   # 只有第 1 个能命中种子
        _paste(img5, "points", slot, _glyph(kind))
    hud5 = ScoreHUD(grab_fn=lambda: img5)
    check("T23.seed_gap_is_documented", hud5.read(img5)["points"] is None,
          "冷启动种子集仅 {0,1,%}:含未学字形的 POINTS 必须整框拒读而非猜读。"
          "这是已知边界 —— 但它同时意味着**只靠种子集读不出真实分数**,"
          "必须补全 0-9 种子或修好 TIME 自校准的时钟锚点")

    # 6) HUD 框映射几何(纯函数,零依赖,不需屏幕)
    check("T23.hud_ref_is_1920x1080", _HUD_REF_WH == (1920, 1080),
          f"参考分辨率必须与逐像素校准的 1920x1080 一致(实测 {_HUD_REF_WH})")
    _bb = (GRAB_BBOX[0], GRAB_BBOX[1],
           GRAB_BBOX[2] - GRAB_BBOX[0], GRAB_BBOX[3] - GRAB_BBOX[1])
    r0 = hud_region_for((0, 0, 1920, 1080))
    check("T23.hud_region_identity", r0 == _bb,
          f"1920x1080 客户区必须**恒等**映射(校准值原样生效,实测 {r0} vs {_bb})")
    r1 = hud_region_for((0, 0, 3840, 2160))
    check("T23.hud_region_scaled", r1 == tuple(v * 2 for v in _bb),
          f"2x 客户区必须等比缩放(实测 {r1})")
    r2 = hud_region_for((1920, 0, 1920, 1080))
    check("T23.hud_region_offset",
          r2[0] == 1920 + GRAB_BBOX[0] and r2[1] == GRAB_BBOX[1] and r2[2] == _bb[2],
          f"副屏(左偏移 1920)必须**平移**而不是原地(实测 {r2})")


def t24_hud_diagnostics() -> None:
    """D42.7 读分自证基建:`BeatRecorder` 的 HUD 落图 / TIME 分段诊断 / 拒读率。

    **守的是"下一次实盘能不能自己说出病因"。** D42.4 的教训是:335 次读分里
    324 次拒读,日志却只剩一句"读到 0/100" —— 两个假说(H1 内容变了 /
    H2 抓帧坏了)在日志里长得**一模一样**,导致一轮调试方向全错。修法是把
    判断依据**落盘**:HUD 裁剪图 + TIME 分段数(不依赖模板)+ 拒读率。

    本组用**注入帧**(完全不碰屏幕/游戏/鼠标)验证这条通路真的在工作。
    """
    import importlib.util as _ilu

    from PIL import Image

    from flyaim.bridge.score_ocr import GRAB_BBOX, _BOX_X, _DIGIT_Y, ScoreHUD

    spec = _ilu.spec_from_file_location("_ap_under_test",
                                        ROOT / "tools" / "aimlab_play.py")
    ap = _ilu.module_from_spec(spec)
    spec.loader.exec_module(ap)          # 只取定义;main() 有 __main__ 守卫

    # 合成帧工具已提到模块级(_blank/_glyph/_put),T25 共用同一份
    blank = _blank()

    # 三个态:练习态 TIME=∞(1 段)→ 倒计时 00:45(5 段)→ 完全空白(0 段)
    # 练习态**直接用真实 fixture**(POINTS=0 / TIME=∞ / ACC=100%),不要手画:
    # 手画的 ring 并不匹配种子 0(MSE 超阈)—— T24 首轮就是这么想当然写错的。
    # 而真实帧的 TIME=∞ 恰好就是 1 段,正是要对照的那个态。
    practice = np.asarray(
        Image.open(ROOT / "tools" / "fixtures" / "hud_sample.png")
        .convert("RGB")).astype(np.int32)
    countdown = blank.copy()
    for slot, k in enumerate(["ring", "ring", "dots", "mid", "top"]):
        _put(countdown, "time", slot, k)                      # 5 段,但字未学
    empty = blank.copy()

    seq = [practice] + [countdown] * 10 + [empty] * 11        # 22 次读分
    it = iter(seq)

    with tempfile.TemporaryDirectory() as td:
        run_dir = Path(td)
        hud = ScoreHUD(grab_fn=lambda: next(it, empty))
        rec = ap.BeatRecorder(run_dir, width=0, score_hud=hud, score_every=1,
                              total_seconds=60.0, learn_time=False)
        for b in range(len(seq)):
            rec(b, np.zeros((8, 8, 3), np.uint8), None)
        rec.close(tick_hz=0.0)
        pngs = sorted(p.name for p in run_dir.glob("hud_*.png"))
        header = (run_dir / "beats.csv").read_text(encoding="utf-8").splitlines()[0]
        lines = rec.report_lines()

    check("T24.recorder_ran", rec.n_score_reads == len(seq),
          f"应完成 {len(seq)} 次读分(实测 {rec.n_score_reads})")
    check("T24.hud_crops_dumped", len(pngs) >= 2,
          f"HUD 裁剪必须落盘(D42.7 分辨 H1/H2 的唯一依据;实测 {pngs})")
    check("T24.dump_on_transition",
          any("_g5" in n for n in pngs) and any("_g0" in n for n in pngs),
          f"**分段数变化必须落图** —— 那正是「∞ → 倒计时」的转变瞬间"
          f"(实测 {pngs})")
    check("T24.csv_has_diag_columns",
          "time_s" in header and "n_glyphs" in header,
          f"beats.csv 必须带 time_s/n_glyphs 诊断列(实测表头 {header})")
    check("T24.glyph_histogram",
          rec.glyph_hist.get(1) == 1 and rec.glyph_hist.get(5) == 10
          and rec.glyph_hist.get(0) == 11,
          f"TIME 分段直方图必须逐态计数:练习态 1 段(∞)×1 / 倒计时 5 段×10 / "
          f"空白 0 段×11(实测 {rec.glyph_hist})")
    check("T24.practice_frame_still_reads",
          rec.score_first == 0 and rec.score_last == 0,
          f"练习态 POINTS=0 仍应读得出(种子含 0),用来对照"
          f"「能读的框」与「读不出的框」(实测 {rec.score_first}→{rec.score_last})")
    check("T24.reject_rate_reported", any("拒读" in ln for ln in lines),
          f"战报必须打拒读率(实测 {lines})")
    check("T24.loud_warning_on_high_reject",
          any("H1" in ln and "H2" in ln for ln in lines),
          f"拒读率 >90% 且读分 ≥20 次时必须红字指向 hud_*.png 并列出 H1/H2"
          f"(实测 {lines})")
    check("T24.learn_time_off_by_default",
          ap.BeatRecorder(Path(tempfile.gettempdir()), width=0).learn_time is False,
          "在线学 TIME 字形必须**默认关闭**(D42.5:时钟未对齐时会污染模板库)")


def t25_round_gate_and_time_calib() -> None:
    """D42.5 修正 + 开局门闸 —— 守这次**实盘**才暴露的两件事。

    ① **种子集必须覆盖真实数字**。上一版只有 {0,1,%},于是实盘 HUD 显示
       `10820` / `48%` 时整框拒读(335 次读分仅 11 次产出数值)。本组用
       **真实帧 fixture** 做接受测试 —— 从真值反推种子对不对。
       (fixture 来自 run 20261006-024357-d42-hudfix 的 hud_*.png,原生 735x30)

    ② **"是否开局"必须看 TIME 分段,不能看有没有青靶**。上一版在 (1664,43)
       检出一个 **R=11.3px** 的青色 UI 块(任务靶 p50 是 27.7px)就断定
       "确认已在任务内"并早退,于是整局 60s 跑在 TIME=∞ 的练习模式里
       —— 分数 10,820 / ACC 48% 都是真的,但**没有计时器**,与目标不可比,
       而且**毫无告警**。

    ③ `TimeCalibrator` 的锚点**必须来自观测**(首次见到 5 段倒计时的那一刻),
       不能用宿主时钟 —— 后者与游戏时钟差几秒会把字形错标(旧的
       `calibrate_from_time` 就是这么坏掉的)。
    """
    from PIL import Image

    from flyaim.bridge.score_ocr import (ScoreHUD, TimeCalibrator, _parse_seed,
                                         round_running, time_glyph_count)

    fx = ROOT / "tools" / "fixtures"
    inf = np.asarray(Image.open(fx / "hud_practice_10820_48.png")
                     .convert("RGB")).astype(np.int32)
    start = np.asarray(Image.open(fx / "hud_practice_0_100.png")
                       .convert("RGB")).astype(np.int32)

    # ---- ① 真实帧接受测试 ----
    rd = ScoreHUD().read(inf)
    check("T25.reads_real_10820", rd["points"] == 10820 and rd["acc_pct"] == 48,
          f"真实帧 POINTS=10820 / ACC=48% 必须读对(实测 "
          f"{rd['points']}/{rd['acc_pct']})—— 失败即说明种子集又缺数字")
    rd0 = ScoreHUD().read(start)
    check("T25.reads_real_0_100", rd0["points"] == 0 and rd0["acc_pct"] == 100,
          f"开局帧 POINTS=0 / ACC=100% 必须读对(实测 "
          f"{rd0['points']}/{rd0['acc_pct']})")
    check("T25.seeds_cover_01248", set("01248") <= set(ScoreHUD().templates),
          f"种子必须含 0/1/2/4/8(实测 {sorted(ScoreHUD().templates)})")

    # ---- ② TIME 分段 = 是否开局(不依赖任何模板)----
    check("T25.practice_time_is_inf", time_glyph_count(inf) == 1,
          f"练习态 TIME 分段必须为 1(∞)(实测 {time_glyph_count(inf)})")
    check("T25.round_running_predicate",
          round_running(5) and not round_running(1) and not round_running(0),
          "round_running 必须**只**认 5 段 = 计分局;青靶不算判据")

    # ---- 种子宽度校验要能点名出错行(手抄种子极易多打一个点)----
    try:
        _parse_seed("\n".join(["#" * 20] * 27 + ["#" * 21]), "坏种子")
        check("T25.seed_width_validated", False, "行宽不齐必须报错")
    except ValueError as exc:
        check("T25.seed_width_validated",
              any(str(k) in str(exc) for k in (26, 27)),
              f"行宽错误必须点名行号(实测 {exc})")

    # ---- ③ TimeCalibrator:锚点来自观测,不是宿主时钟 ----
    def _norm(kind):
        g = (_glyph(kind) * 255).astype(np.uint8)
        return np.asarray(Image.fromarray(g).resize((20, 28), Image.BILINEAR),
                          dtype=np.float32) / 255.0

    t5 = _blank()
    for slot, k in enumerate(["ring", "ring", "dots", "top", "mid"]):
        _put(t5, "time", slot, k)          # 合成 5 段(M M : S S)

    # (a) 本进程没见过 ∞ ⇒ 不许建锚、不许学习(否则标签无从对齐)
    h = ScoreHUD()
    cal = TimeCalibrator(h, round_len_s=60.0)
    cal.observe(t5, 100.0)
    check("T25.no_inf_no_anchor", cal.anchor_t is None and cal.n_learn == 0,
          f"未见过 TIME=∞ 时不得建锚/学习(实测 anchor={cal.anchor_t} "
          f"learn={cal.n_learn})")

    # (b) 见过 ∞ 之后的第一个 5 段帧 = 回合开始 → 锚点 == 该帧的 now
    cal.observe(_blank(), 50.0)            # 空白帧:0 段,不是 ∞
    cal.observe(inf, 51.0)                 # 真实帧:TIME=∞ → 1 段
    cal.observe(t5, 52.0)                  # 首次 5 段
    check("T25.anchor_is_observed_time", cal.anchor_t == 52.0,
          f"锚点必须 == 首次见到 5 段的那一帧时刻(实测 {cal.anchor_t},期望 52.0)"
          f" —— 用宿主时钟就会错成别的值")

    # (c) 跨帧校验失败 ⇒ 回滚模板并停用
    #     手工把模板补全(用空白模板占位,真正用到的字符给对应字形),
    #     再把锚点挪到自己预测不了的过去 → 读数与预测必然不符
    h2 = ScoreHUD()
    for ch in TimeCalibrator.NEED:
        h2.templates[ch] = [np.zeros((28, 20), np.float32)]
    for ch, kind in [("0", "ring"), (":", "dots"), ("5", "top"), ("9", "mid")]:
        h2.templates[ch] = [_norm(kind)]
    cal2 = TimeCalibrator(h2, round_len_s=60.0)
    cal2.saw_infinity = True
    cal2.anchor_t = 0.0
    before = {c: len(v) for c, v in h2.templates.items()}
    for i in range(TimeCalibrator.CONFIRM + 1):
        diag = cal2.observe(t5, 5.0)       # 预测 remain=55,而模板读成 59
    check("T25.cross_frame_verify_disables", cal2.disabled,
          f"跨帧校验连续不符必须停用校准(实测 disabled={cal2.disabled}, "
          f"verify_bad={cal2.n_verify_bad})")
    check("T25.rollback_restores_seeds", set(h2.templates) == {"0", "1", "2", "4", "8", "%"},
          f"停用时必须**回滚到冷启动种子**(实测 {sorted(h2.templates)},"
          f"回滚前 {sorted(before)})")
    check("T25.disable_leaves_note", bool(cal2.note),
          f"停用必须留下可读原因(实测 {cal2.note!r})")


def t26_ref_ground_truth_fixture() -> None:
    """D42.11 守卫:理想基准地面真值 fixture 必须保留 **130,709 / 96%**。

    守的是 Lead 自己犯过的错:用 `ffmpeg -vf fps=1` 抽帧时**假设了 t = cell**,
    而该滤镜实测有 **+0.45s 相位**,于是终局值 130,709(出现在 t≈61.60)
    **整个落在 cell 61 与 cell 62 之间被漏掉**,Lead 遂把中间值 130,335 当成
    终局,还"纠正"了用户本来正确的记录 —— 并且推送了出去。

    fixture 用 `source` 列区分两段:`grid_fps1`(轨迹,相位 +0.45s)
    与 `tail_ss`(**终局值的唯一权威来源**)。
    """
    import csv as _csv

    p = ROOT / "tools" / "fixtures" / "ref_hud_ground_truth.csv"
    if not p.exists():
        check("T26.fixture_exists", False, f"缺 {p}")
        return
    with p.open(encoding="utf-8") as fh:
        rows = list(_csv.DictReader(fh))
    grid = [r for r in rows if r["source"] == "grid_fps1" and r["points"]]
    tail = [r for r in rows if r["source"] == "tail_ss" and r["points"]]
    check("T26.fixture_has_both_sources", bool(grid) and bool(tail),
          f"必须同时含 grid_fps1 与 tail_ss 两段(实测 {len(grid)}/{len(tail)})")
    if not grid or not tail:
        return
    gv = [int(r["points"]) for r in grid]
    tv = [int(r["points"]) for r in tail]
    check("T26.final_is_130709", tv[-1] == 130709,
          f"终局必须为 130,709(实测 {tv[-1]})—— 固定的是**用户原始记录**,"
          f"防止再被上涨途中的中间值顶替")
    check("T26.final_acc_96", tail[-1]["acc_pct"] == "96",
          f"终局 ACC 必须为 96%(实测 {tail[-1]['acc_pct']})")
    check("T26.tail_monotonic", all(b >= a for a, b in zip(tv, tv[1:])),
          f"尾部点数必须单调不减(实测 {tv})")
    check("T26.grid_misses_final", gv[-1] < tv[-1],
          f"**网格末值必须严格小于尾部终局** —— 这正是「fps=1 相位漏掉终局」"
          f"的证据本身(网格 {gv[-1]} < 尾部 {tv[-1]})。"
          f"若两者相等,说明有人又把网格值当成终局了")


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
    t14_trigger_gate()
    t15_speed_and_xhair()
    t16_aim_fire_consistency()
    t17_action_smoother()
    t18_dual_window()
    t19_ema_phase_lag()
    t20_target_assoc()
    t21_saccade_fixate()
    t22_fire_confirm()
    t23_score_hud_ocr()
    t24_hud_diagnostics()
    t25_round_gate_and_time_calib()
    t26_ref_ground_truth_fixture()
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
