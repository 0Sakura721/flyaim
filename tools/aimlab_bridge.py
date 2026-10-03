"""Aim Lab 桥接主入口:屏幕 -> 果蝇网络 -> 鼠标。

===============================================================================
三种运行模式(按准备程度递进,AIMLAB.md 有完整分阶段说明)
===============================================================================
1. 无头彩排(不动鼠标、不需要游戏)—— 验证 fly 链路在桥接闭环里能跑:

       & $py tools/aimlab_bridge.py --controller fly --source arena --frames 50 --device cuda

   source=arena 时自建靶场就是"屏幕",action 回灌靶场(closed_loop),
   语义与正式实验完全一致,只是走了桥接的全部管道。

2. 管道校验(截真实屏幕,不注入)—— 核对捕获区域/分辨率/检测框/延迟:

       & $py tools/aimlab_bridge.py --controller seek --source screen --frames 200

   Aim Lab 放无边框窗口摆在主屏中央,跑完看 runs/<ts>-bridge/live/preview/
   里检测框画得对不对。**不会动鼠标。**

3. 真实闭环(注入真实鼠标增量)—— **会动你的鼠标**:

       & $py tools/aimlab_gain.py --from-cm360 800 40      # 先标定增益
       & $py tools/aimlab_bridge.py --controller fly --source screen \
             --sink sendinput --gain-json flyaim/runs/bridge/gain.json --frames 300

   建议:游戏内自定义任务 + 静止靶;先把 max-counts 调小试方向;
   Ctrl-C 随时优雅停止。seek 控制器 + sendinput 可以先用「经典 PD 会瞄准」
   验证注入方向与灵敏度标定,再换 fly 跑科学问题。
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from flyaim.bridge import (  # noqa: E402
    ArenaSource,
    ArraySource,
    BridgeLoop,
    FlyController,
    GainConfig,
    GainModel,
    NullSink,
    RandomController,
    ScreenCapture,
    SeekController,
    SendInputSink,
    ZeroController,
)
from flyaim.config import BrainConfig, ReadoutConfig, RetinaConfig  # noqa: E402

D = ROOT / "flyaim" / "data" / "build"
NORM = ROOT / "flyaim" / "runs" / "norm"
READOUT = ROOT / "flyaim" / "runs" / "readout" / "readout_weights.npz"

# 与 tools/run_experiment.py 相同的 Phase 2 工作点 —— 不要单独改动
CAL_W_SCALE = 16.0
CAL_INPUT_GAIN = 1.5
CAL_NORM = "indeg"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="FlyAim -> Aim Lab 桥接")
    ap.add_argument("--controller", default="seek",
                    choices=["fly", "seek", "random", "zero"],
                    help="fly=连接组;seek=纯视觉PD(管道校验,非实验臂);random;zero")
    ap.add_argument("--source", default="arena", choices=["arena", "screen", "array"],
                    help="arena=自建靶场(无头彩排);screen=真实屏幕;array=内置合成帧")
    ap.add_argument("--region", default=None,
                    help="screen 捕获区域 L,T,W,H(默认主屏中央 640x480)")
    ap.add_argument("--window", default=None,
                    help="screen 按窗口标题子串自动定位区域(优先于 --region),"
                         "如 --window aimlab")
    ap.add_argument("--out-size", default="640x480",
                    help="捕获后缩放到该尺寸(与训练分辨率对齐);0 表示不缩放")
    ap.add_argument("--sink", default="null", choices=["null", "sendinput"],
                    help="sendinput 会真实移动鼠标!")
    ap.add_argument("--device", default="cuda", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--gain-json", default=None, help="GainConfig JSON(tools/aimlab_gain.py 生成)")
    ap.add_argument("--cm360", type=float, default=None, help="cm/360(配 --dpi 推导增益)")
    ap.add_argument("--dpi", type=float, default=None)
    ap.add_argument("--max-counts", type=float, default=600.0,
                    help="每 tick 注入计数上限(安全钳位)")
    ap.add_argument("--target-color", default="48,224,224",
                    help="seek 控制器的靶色 RGB(默认 Aim Lab 青色靶 48,224,224;"
                         "自建靶场用 235,70,70)")
    ap.add_argument("--target-tolerance", type=float, default=60.0,
                    help="seek 靶色距离阈值(默认 60)")
    ap.add_argument("--no-aim-detect", action="store_true",
                    help="seek 不检测准星标记,直接用画面中心(Aim Lab 全屏捕获时"
                         "准星恒在窗口几何中心,推荐;且红色准星会与红色枪模型混淆)")
    ap.add_argument("--no-focus", action="store_true",
                    help="sendinput 模式下不自动把目标窗口调到前台(默认强制调,"
                         "因为注入只进焦点窗口)")
    ap.add_argument("--invert-y", action="store_true", help="dy 方向翻转")
    ap.add_argument("--invert-x", action="store_true", help="dx 方向翻转")
    ap.add_argument("--frames", type=int, default=None, help="运行帧数上限")
    ap.add_argument("--seconds", type=float, default=None, help="运行秒数上限")
    ap.add_argument("--min-tick-ms", type=float, default=0.0, help="最小拍间隔(ms)")
    ap.add_argument("--live", action="store_true", help="写遥测 JSONL(live_view/live_web 可看)")
    ap.add_argument("--tag", default="bridge", help="运行目录标签")
    ap.add_argument("--array-frames", type=int, default=30, help="array 源的合成帧数")
    return ap.parse_args()


def build_gain(args: argparse.Namespace) -> GainModel:
    """增益来源优先级:--gain-json > --cm360/--dpi > 默认值(打印醒目警告)。"""
    if args.gain_json:
        g = GainModel.load(args.gain_json)
        print(f"  增益: 载入 {args.gain_json} (counts_per_360={g.cfg.counts_per_360:.0f})")
        return g
    if args.cm360 and args.dpi:
        g = GainModel(GainConfig.from_cm360(args.dpi, args.cm360,
                                            max_counts_per_tick=args.max_counts,
                                            invert_x=args.invert_x, invert_y=args.invert_y))
        print(f"  增益: 由 {args.dpi} DPI × {args.cm360} cm/360 推导 "
              f"(counts_per_360={g.cfg.counts_per_360:.0f})")
        return g
    g = GainModel(GainConfig(max_counts_per_tick=args.max_counts,
                             invert_x=args.invert_x, invert_y=args.invert_y))
    print("  ⚠️ 增益未标定:使用默认 counts_per_360=12000(约 800DPI×38cm)。")
    print("     正式跑之前务必用 tools/aimlab_gain.py 生成并核对 gain.json!")
    return g


def build_source(args: argparse.Namespace):
    if args.source == "arena":
        from flyaim.arena.arena import Arena
        from flyaim.config import ArenaConfig

        cfg = ArenaConfig()
        return ArenaSource(Arena(cfg, seed=0), closed_loop=True)
    if args.source == "screen":
        region = None
        if args.window:
            from flyaim.bridge.capture import find_window_region

            region = find_window_region(args.window)
            print(f"  窗口定位: {args.window!r} -> region={region}")
        elif args.region:
            l, t, w, h = (int(x) for x in args.region.split(","))
            region = (l, t, w, h)
        out_size = None
        if args.out_size and args.out_size != "0":
            w, h = (int(x) for x in args.out_size.lower().split("x"))
            out_size = (w, h)
        return ScreenCapture(region=region, out_size=out_size)
    # array:合成帧(亮蓝圆盘匀速移动),给无屏幕环境做最小闭环
    w, h = 640, 480
    frames = []
    for i in range(max(4, int(args.array_frames))):
        f = np.full((h, w, 3), 24, dtype=np.uint8)
        cx = int(80 + (w - 160) * i / max(4, int(args.array_frames)))
        cy = int(120 + 60 * np.sin(i / 5.0))
        yy, xx = np.mgrid[0:h, 0:w]
        m = (xx - cx) ** 2 + (yy - cy) ** 2 <= 20 ** 2
        f[m] = (70, 170, 255)
        frames.append(f)
    return ArraySource(frames)


def build_controller(args: argparse.Namespace):
    if args.controller == "fly":
        wp = NORM / f"connectome_{CAL_NORM}.npz"
        if not wp.exists():
            sys.exit(f"❌ 缺少归一化权重 {wp}(先跑 tools/test_normalization.py)")
        if not READOUT.exists():
            sys.exit(f"❌ 缺少读出权重 {READOUT}(先跑 tools/train_readout.py)")
        from flyaim.pipeline import FlySystem

        rc = RetinaConfig()
        bc = BrainConfig(weight_scale_exc=CAL_W_SCALE, weight_scale_inh=CAL_W_SCALE,
                         input_gain=CAL_INPUT_GAIN)
        ro = ReadoutConfig(mode="trained", weights_path=str(READOUT))
        t0 = time.perf_counter()
        system = FlySystem(D, rc, bc, ro, weights_override=wp, device=args.device)
        print(f"  fly: {type(system.brain).__name__} device={system.device} "
              f"(装配 {time.perf_counter()-t0:.0f}s);读出权重 {READOUT.name}")
        return FlyController(system)
    if args.controller == "seek":
        tc = tuple(int(x) for x in args.target_color.split(","))
        return SeekController(ref_color=tc, tolerance=args.target_tolerance,
                              use_aim_detect=not args.no_aim_detect)
    if args.controller == "random":
        return RandomController(seed=0)
    return ZeroController()


def build_telemetry(args: argparse.Namespace, run_dir: Path):
    if not args.live:
        return None
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from telemetry import TelemetryWriter, make_sample_plan  # type: ignore

        import pandas as pd

        from flyaim.io import load_roles

        idx = pd.read_parquet(D / "neuron_index.parquet")
        roles = load_roles(D / "roles.json")
        plan = make_sample_plan(int(idx.shape[0]), roles, neuron_index=idx)
        out = run_dir / "live"
        writer = TelemetryWriter(out, sample_shape=plan["sample_shape"],
                                 sample_idx=plan["sample_idx"], dn_idx=plan["dn_idx"],
                                 layer_labels=plan["layer_labels"], preview_every=20,  # PNG 编码持 GIL ~50-100ms,每 5 帧一次会把拍频拖低 ~15%
                                 run_tag=f"aimlab-{args.controller}")
        print(f"  遥测 -> {out}(tools/live_web.py 可实时看)")
        return writer
    except Exception as e:
        print(f"  [warn] 遥测不可用({type(e).__name__}: {e}),本次不开")
        return None


def main() -> int:
    args = parse_args()
    print("=" * 84)
    print("FlyAim -> Aim Lab 桥接")
    print(f"  controller={args.controller} source={args.source} sink={args.sink}")
    if args.sink == "sendinput":
        print("  ⚠️ sink=sendinput:本程序将真实移动你的鼠标。Ctrl-C 可随时停止。")
        # 注入只进焦点窗口(2026-10-04 实测:游戏不在前台时注入 100% 静默丢失),
        # 所以注入模式强制把目标窗口调到前台。
        if args.source == "screen" and args.window and not args.no_focus:
            from flyaim.bridge.capture import focus_window

            ok = focus_window(args.window)
            if ok:
                print(f"  已把窗口 {args.window!r} 调到前台(注入目标)")
            else:
                print(f"  ❌ 无法把 {args.window!r} 调到前台 —— 注入会丢失,中止。"
                      "请手动点击游戏窗口后重跑,或用 --no-focus 强行继续。")
                return 1
        print("  请在手离开鼠标/键盘的状态下等待运行结束。")
    print("=" * 84, flush=True)

    from flyaim.bridge.inject import pointer_accel_enabled
    from flyaim.io import new_run_dir, save_json

    run_dir = new_run_dir(ROOT / "flyaim" / "runs", tag=args.tag)
    print(f"  运行目录 {run_dir}")

    source = build_source(args)
    if args.source == "screen":
        print(f"  捕获: {getattr(source, 'backend_name', '?')} region={source.region} "
              f"out={source.out_size}")
    controller = build_controller(args)
    gain = build_gain(args)
    sink = NullSink() if args.sink == "null" else SendInputSink(
        max_counts_per_tick=int(args.max_counts)
    )
    telemetry = build_telemetry(args, run_dir)

    loop = BridgeLoop(
        source=source, sink=sink, controller=controller, gain=gain,
        telemetry=telemetry,
        min_tick_interval_s=args.min_tick_ms / 1000.0,
        max_frames=args.frames, max_seconds=args.seconds,
    )
    summary = loop.run()
    summary["pointer_accel_enabled"] = pointer_accel_enabled()  # 用户可选保留;影响写入留档
    if summary["pointer_accel_enabled"]:
        print("  ⚠️ 指针加速度(提高指针精确度)处于开启状态 —— 增益随注入速度非线性,"
              "Phase C 标定须在网络实际输出幅度附近进行,结论必须连同本标记一起报告。")
    save_json(str(run_dir / "bridge_summary.json"), summary)

    lat = summary["latency_ms"]
    print("\n---- 桥接摘要 ----")
    print(f"  帧数 {summary['frames']}  用时 {summary['wall_s']}s  "
          f"拍频 {summary['tick_hz']} Hz")
    print(f"  完整一拍 p50/p95 = {lat['tick_p50']}/{lat['tick_p95']} ms"
          f"(含等新帧;其中控制器决策 act p50={lat['act_p50']} ms)")
    print(f"  画面年龄 p50/p95 = {lat['capture_age_p50']}/{lat['capture_age_p95']} ms")
    print(f"  注入累计计数 = {summary['counts_total']}")
    print(f"  摘要落盘 {run_dir / 'bridge_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
