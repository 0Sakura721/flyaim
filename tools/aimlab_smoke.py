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
from flyaim.bridge.gain import GainConfig, GainModel  # noqa: E402
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
