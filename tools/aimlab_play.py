"""Aim Lab 任务内会话:自动开始任务 -> 控制器打满一局 -> 命中数入账。

模式:
    collect  seek 当导师打靶(带开火),同时步进真实网络记录 (DN rates, action) 对
    seek     纯 seek + 开火(导师上限参照,也是"能瞄准会开枪"的工程基线)
    fly      果蝇网络 + 开火(被测臂;--readout 指定重训权重)

运行(一局 Gridshot ≈ 60s,工具自动点开始):

    & $py tools/aimlab_play.py --mode collect --npz-out flyaim/runs/bridge/collect_1.npz
    & $py tools/aimlab_play.py --mode seek            # 导师上限
    & $py tools/aimlab_play.py --mode fly --readout flyaim/runs/bridge/readout_ingame.npz

命中入账规则(Gridshot = hitscan 无散布):准星压住靶盘(err <= r*0.9)时
点击 = 必中。inferred_hits 即该口径的命中数。

⚠️ 运行期间会真实移动/点击鼠标。请保持 Aim Lab 前台、手离鼠标。
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from flyaim.bridge.capture import ScreenCapture, focus_window, find_window_region  # noqa: E402
from flyaim.bridge.controllers import (  # noqa: E402
    FlyController,
    TeacherCollectController,
    TriggerOnTarget,
)
from flyaim.bridge.detect import find_target  # noqa: E402
from flyaim.bridge.gain import GainModel  # noqa: E402
from flyaim.bridge.inject import SendInputSink, click_at, press_key  # noqa: E402
from flyaim.bridge.loop import BridgeLoop  # noqa: E402
from flyaim.config import BrainConfig, ReadoutConfig, RetinaConfig  # noqa: E402

D = ROOT / "flyaim" / "data" / "build"
NORM = ROOT / "flyaim" / "runs" / "norm"
READOUT_OFFLINE = ROOT / "flyaim" / "runs" / "readout" / "readout_weights.npz"
TEAL = (48, 224, 224)
CAL_W_SCALE = 16.0
CAL_INPUT_GAIN = 1.5
CAL_NORM = "indeg"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["collect", "seek", "fly", "hybrid"])
    ap.add_argument("--beta", type=float, default=0.25,
                    help="hybrid 模式:action = seek + β·fly(果蝇扰动权重)")
    ap.add_argument("--kp", type=float, default=2.2,
                    help="seek 比例增益(EPP 慢速衰减的补偿,实测标定)")
    ap.add_argument("--npz-out", default=None, help="collect 模式的数据落盘路径")
    ap.add_argument("--readout", default=None, help="fly 模式的读出权重(默认离线权重)")
    ap.add_argument("--gain-json", default=str(ROOT / "flyaim/runs/bridge/gain.json"))
    ap.add_argument("--max-counts", type=float, default=150.0)
    ap.add_argument("--seconds", type=float, default=52.0,
                    help="打靶时长(留出任务开场与收尾余量)")
    ap.add_argument("--no-autostart", action="store_true",
                    help="不自动点「点击开始」(已手动进入任务时用)")
    ap.add_argument("--no-trigger", action="store_true", help="关闭开火层(纯瞄准)")
    ap.add_argument("--action-ema", type=float, default=0.35,
                    help="动作 EMA 平滑系数(0=关;抑制抽搐)")
    ap.add_argument("--tag", default=None)
    return ap.parse_args()


def _task_running(cap_region) -> bool:
    """任务是否在进行:画面里能找到青色靶。"""
    cap = ScreenCapture(region=cap_region, out_size=None)
    frame, _ = cap.read()
    cap.close()
    return bool(find_target(frame, ref_color=TEAL, tolerance=60.0).ok)


def autostart_task(cap_region, tries: int = 3) -> bool:
    """分层尝试进入任务(2026-10-04 教训:大厅也有青靶,且大厅球≠任务):

      1. 「点击开始」文字是屏幕固定 UI(y≈0.28H)—— 见到就点,最能区分大厅;
      2. 结算页的大白按钮(0.3-0.8H 大面积白块)—— 点击重开;
      3. 都没有 -> 认为已在任务内。
    """
    for i in range(tries):
        cap = ScreenCapture(region=cap_region, out_size=None)
        frame, _ = cap.read()
        cap.close()
        l, t, w, h = cap_region
        # 1) 大厅的「点击开始」固定文字(细笔画,面积阈值要小)
        txt = find_target(frame, ref_color=(245, 245, 245), tolerance=45.0, min_area_px=80)
        if txt.ok and 0.18 * h < txt.cy < 0.40 * h and 0.30 * w < txt.cx < 0.70 * w:
            sx, sy = l + int(txt.cx), t + int(txt.cy)
            print(f"  autostart[{i}]: 检测到「点击开始」({sx},{sy}),点击")
            click_at(sx, sy)
            time.sleep(2.0)
            continue
        # 2) 结算页大按钮(大面积白块)
        btn = find_target(frame, ref_color=(245, 245, 245), tolerance=45.0, min_area_px=1500)
        if btn.ok and 0.30 * h < btn.cy < 0.85 * h:
            sx, sy = l + int(btn.cx), t + int(btn.cy)
            print(f"  autostart[{i}]: 检测到大按钮/重开键 ({sx},{sy}),点击")
            click_at(sx, sy)
            time.sleep(2.0)
            continue
        # 3) 结算页/未知页:按 R(再练一次热键),按完看有没有青靶
        from flyaim.bridge.inject import press_key as _pk

        _pk(0x52)  # VK R
        print(f"  autostart[{i}]: 按 R(再练一次)")
        time.sleep(2.0)
    return True


class ActionEMA:
    """动作 EMA 平滑:把 17-30 Hz 的阶跃指令变成连续轨迹(治抽搐,不改语义)。"""

    def __init__(self, inner, alpha: float) -> None:
        self.inner = inner
        self.alpha = float(alpha)  # 新样本权重;0.35 = 温和平滑
        self._prev: np.ndarray | None = None
        self.raw_action = np.zeros(2, dtype=np.float32)

    @property
    def brain(self):
        return getattr(self.inner, "brain", None)

    @property
    def name(self):
        return getattr(self.inner, "name", "?") + "+ema"

    def act(self, frame: np.ndarray) -> np.ndarray:
        a = np.asarray(self.inner.act(frame), dtype=np.float32).reshape(2)
        self.raw_action = a
        if self.alpha <= 0 or self._prev is None:
            self._prev = a
        else:
            self._prev = self.alpha * a + (1 - self.alpha) * self._prev
        return self._prev.copy()

    def reset(self) -> None:
        self.inner.reset()
        self._prev = None

    def close(self) -> None:
        self.inner.close()

    @property
    def n_hits_inferred(self):
        return getattr(self.inner, "n_hits_inferred", 0)

    @property
    def n_fires(self):
        return getattr(self.inner, "n_fires", 0)


def _build_fly(args):
    rp = Path(args.readout) if args.readout else READOUT_OFFLINE
    if not Path(rp).exists():
        sys.exit(f"❌ 读出权重不存在: {rp}")
    ro = ReadoutConfig(mode="trained", weights_path=str(rp))
    core = FlyController(build_system())
    core.system.readout.load(rp)
    print(f"  fly 读出权重: {rp.name}")
    return core


def build_system(device: str = "cuda"):
    from flyaim.pipeline import FlySystem

    rc = RetinaConfig()
    bc = BrainConfig(weight_scale_exc=CAL_W_SCALE, weight_scale_inh=CAL_W_SCALE,
                     input_gain=CAL_INPUT_GAIN)
    ro = ReadoutConfig(mode="trained", weights_path=str(READOUT_OFFLINE))
    return FlySystem(D, rc, bc, ro, weights_override=NORM / f"connectome_{CAL_NORM}.npz",
                     device=device)


def main() -> int:
    args = parse_args()
    print("=" * 84)
    print(f"Aim Lab 任务会话: mode={args.mode} seconds={args.seconds}")
    print("  ⚠️ 将真实控制鼠标。请保持 Aim Lab 前台、手离鼠标键。")
    print("=" * 84, flush=True)

    from flyaim.bridge.capture import primary_monitor_size, centered_region
    from flyaim.io import new_run_dir, save_json

    region = find_window_region("aimlab")
    if not focus_window("aimlab"):
        print("❌ 无法聚焦 Aim Lab 窗口")
        return 1
    print(f"  窗口已聚焦 region={region}")
    if not args.no_autostart:
        autostart_task(region)
        time.sleep(1.2)  # 等任务开场动画
    cap = ScreenCapture(region=region, out_size=(640, 480))
    # 开局握手:注入模式必须实测「相机响应注入」。失灵(锁定丢失/焦点被夺)
    # 时注入会 100% 静默丢失 —— 与其跑一局废数据,不如当场失败。
    if not args.no_trigger or args.mode != "collect":
        from flyaim.bridge.inject import SendInputSink as _S

        probe = _S()
        ok = False
        for i in range(3):
            f0, _ = cap.read()
            probe.send(300, 0)
            time.sleep(0.15)
            f1, _ = cap.read()
            d = float(np.abs(f1.astype(np.int16) - f0.astype(np.int16)).mean())
            if d > 1.5:
                ok = True
                print(f"  握手: 300 计数 -> 画面差 {d:.1f}/255,相机响应正常")
                break
            print(f"  握手[{i}]: 画面差 {d:.1f} —— 相机未响应,点击中央重建锁定...")
            click_at(region[0] + region[2] // 2, region[1] + int(region[3] * 0.28))
            time.sleep(2.0)
        probe.close()
        if not ok:
            print("❌ 三次握手失败:相机对注入无响应。请人工点进游戏窗口后再跑。")
            return 1

    sink = SendInputSink(max_counts_per_tick=int(args.max_counts))
    gain = GainModel.load(args.gain_json)

    if args.mode == "collect":
        out_npz = args.npz_out or str(ROOT / "flyaim/runs/bridge/collect_1.npz")
        Path(out_npz).parent.mkdir(parents=True, exist_ok=True)
        core = TeacherCollectController(build_system(), out_npz=out_npz)
    elif args.mode == "seek":
        from flyaim.bridge.controllers import SeekController

        core = SeekController(ref_color=TEAL, use_aim_detect=False, kp=args.kp)
    elif args.mode == "fly":
        core = _build_fly(args)
    else:  # hybrid
        from flyaim.bridge.controllers import HybridController

        core = HybridController(_build_fly(args), beta=args.beta, kp=args.kp)
        print(f"  hybrid: β={args.beta}(seek + β·fly 加性扰动)")

    stack = core
    if args.action_ema > 0:
        stack = ActionEMA(stack, args.action_ema)
    if not args.no_trigger:
        stack = TriggerOnTarget(stack, sink.click, ref_color=TEAL, err_frac=1.15)
    print(f"  控制栈: {core.name}"
          f"{' +EMA' if args.action_ema > 0 else ''}{' +trigger' if not args.no_trigger else ''}")

    run_dir = new_run_dir(ROOT / "flyaim" / "runs", tag=args.tag or f"play-{args.mode}")
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    try:
        from telemetry import TelemetryWriter, make_sample_plan  # type: ignore

        import pandas as pd

        from flyaim.io import load_roles

        idx = pd.read_parquet(D / "neuron_index.parquet")
        roles = load_roles(D / "roles.json")
        plan = make_sample_plan(int(idx.shape[0]), roles, neuron_index=idx)
        writer = TelemetryWriter(run_dir / "live", sample_shape=plan["sample_shape"],
                                 sample_idx=plan["sample_idx"], dn_idx=plan["dn_idx"],
                                 layer_labels=plan["layer_labels"], preview_every=20,
                                 run_tag=f"play-{args.mode}")
    except Exception as e:
        print(f"  [warn] 遥测不可用({e})")
        writer = None

    loop = BridgeLoop(source=cap, sink=sink, controller=stack, gain=gain,
                      telemetry=writer, max_seconds=args.seconds,
                      focus_watchdog_title="aimlab")
    summary = loop.run()
    summary["mode"] = args.mode
    summary["n_fires"] = int(getattr(stack, "n_fires", 0))
    summary["n_hits_inferred"] = int(getattr(stack, "n_hits_inferred", 0))
    save_json(str(run_dir / "bridge_summary.json"), summary)

    print("\n---- 本局战报 ----")
    print(f"  模式 {args.mode}  帧数 {summary['frames']}  拍频 {summary['tick_hz']} Hz")
    print(f"  开火 {summary['n_fires']} 次  推断命中 {summary['n_hits_inferred']}")
    print(f"  目录 {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
