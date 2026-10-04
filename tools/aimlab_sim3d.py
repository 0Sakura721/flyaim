"""实战彩排:在**复刻真实 FPS 视觉语义**的模拟域里跑复眼伺服闭环。

===============================================================================
为什么要这一步(而不是直接开游戏)
===============================================================================
D18/D19 已经踩过一次:2D 靶场里调通的东西搬到游戏里会失效,原因是**语义变了**
(准星动 vs 相机动、静态背景 vs 全域光流、像素速度 vs 角速度)。
`flyaim/bridge/sim3d.py` 把这三点都实现出来,于是:

  * 可以在**不动真实鼠标**的前提下,把「实战配置」整条链路跑通;
  * 量出投产前必须知道的三个数:锁定角误差、再捕获率、增益标定容差;
  * 真机上剩下的就只是"环境差别",不是"控制律对不对" —— 两者可以分开归因。

本工具做四件事:
  1. 正对照:复眼伺服(青靶 -> R8 通路、准星居中、带扫视)在模拟游戏里闭环;
  2. **负对照**:故意用错色觉通路(R7 找青靶)/ 故意不居中,证明这不是
     "随便配都行";
  3. 增益容差扫描:把 counts_per_360 故意配错 0.5x ~ 2x,看锁定率怎么塌
     —— 直接回答「cm/360 要标多准」;
  4. 像素≈角度 这条线性近似的实测偏差(画面边缘的 px/度 vs 中心)。

真机步骤见 `AIMLAB.md` §10;边界与合规见 §7。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from flyaim.baselines.eye_servo import make_eye_servo  # noqa: E402
from flyaim.bridge import (  # noqa: E402
    BridgeLoop,
    EyeController,
    GainConfig,
    GainModel,
    NullSink,
)
from flyaim.bridge.sim3d import SimFPS3DConfig, SimFPS3DSource  # noqa: E402
from flyaim.config import RetinaConfig  # noqa: E402

DATA_DIR = ROOT / "flyaim" / "data" / "build"


def build_eye(chroma: str, aim_mode: str, search: str, v_px: float,
              eye_rows: int = 24, eye_cols: int = 32, width: int = 640):
    """装配桥接版复眼控制器(与离线 D25 同一份 arm 代码)。

    cell_px 必须跟着复眼网格走:小眼接受角 = 画面宽/eye_cols。
    错配会让"近距限速"和"误差换算成像素"同时错,所以它是装配参数而非调参。
    """
    retina_cfg = RetinaConfig(eye_rows=int(eye_rows), eye_cols=int(eye_cols))
    arm = make_eye_servo(str(DATA_DIR), retina_cfg=retina_cfg, v_px=v_px,
                         cell_px=float(width) / float(eye_cols),
                         chroma=chroma, aim_mode=aim_mode, search=search)
    return EyeController(arm)


class _Saturate:
    """把内层输出归一化到满量程:只保留方向,去掉幅度(不减速)。

    复刻 D12 的 `const` 失败模式(「读出层不看距离就输出满量程」)。
    这是**机制探针**,不是候选控制器 —— 它用来量"增益语义错多少会直接飞出去"。
    """

    name = "saturate"

    def __init__(self, inner) -> None:
        self.inner = inner
        self.last_detection = None

    @property
    def brain(self):
        return getattr(self.inner, "brain", None)

    def act(self, frame) -> np.ndarray:
        a = np.asarray(self.inner.act(frame), dtype=np.float64).reshape(2)
        n = float(np.linalg.norm(a))
        self.last_detection = getattr(self.inner, "last_detection", None)
        if n <= 1e-9:
            return np.zeros(2, dtype=np.float32)
        return (a / n).astype(np.float32)

    def reset(self) -> None:
        self.inner.reset()

    def close(self) -> None:
        self.inner.close()

    def describe(self) -> dict:
        d = dict(self.inner.describe())
        d["mode"] = "saturate(满量程,不减速)"
        return d


def run_once(*, gain: GainModel, sim_cfg: SimFPS3DConfig, chroma: str,
             aim_mode: str, search: str, max_frames: int | None = None,
             max_seconds: float | None = None, v_px: float = 14.0,
             eye_rows: int = 24, eye_cols: int = 32, saturate: bool = False,
             verbose: bool = False) -> dict:
    src = SimFPS3DSource(sim_cfg, gain=gain)
    ctrl = build_eye(chroma, aim_mode, search, v_px, eye_rows, eye_cols,
                     width=sim_cfg.width)
    if saturate:
        ctrl = _Saturate(ctrl)
    loop = BridgeLoop(source=src, sink=NullSink(), controller=ctrl, gain=gain,
                      telemetry=None, max_frames=max_frames,
                      max_seconds=max_seconds)
    summary = loop.run()

    h = src.history
    if not h:
        return {"n_ticks": 0}
    err = np.array([r["err_deg"] for r in h], dtype=np.float64)
    on = np.array([r["on_screen"] for r in h], dtype=bool)
    hit = np.array([r["hit"] for r in h], dtype=bool)
    locked = err <= sim_cfg.target_ang_deg / 2.0
    first = int(np.argmax(locked)) if bool(locked.any()) else -1
    post = err[first:] if first >= 0 else err
    # 像素≈角度:每拍的 (像素误差 / 角误差) 应该接近中心的 f*pi/180
    px = np.array([r["dist_px"] for r in h], dtype=np.float64)
    ratio = np.divide(px, np.maximum(err, 1e-6))
    f = (sim_cfg.width / 2.0) / np.tan(np.radians(sim_cfg.fov_h_deg / 2.0))
    ideal = f * np.pi / 180.0
    out = {
        "n_ticks": len(h),
        "tick_hz": summary.get("tick_hz", 0.0),
        "err_mean_deg": float(err.mean()),
        "err_p50_deg": float(np.percentile(err, 50)),
        "err_p95_deg": float(np.percentile(err, 95)),
        "lock_frac": float(locked.mean()),
        "post_lock_frac": float((post <= sim_cfg.target_ang_deg / 2.0).mean()),
        "post_lock_max_deg": float(post.max()),
        "first_lock_tick": first,
        "on_screen_frac": float(on.mean()),
        "n_respawns": int(src.n_respawns),
        "px_per_deg_center": float(ideal),
        "px_per_deg_mean": float(ratio.mean()),
        "px_per_deg_p95": float(np.percentile(ratio, 95)),
        "act_p50_ms": summary.get("latency_ms", {}).get("act_p50", -1.0),
        "capture_age_p95_ms": summary.get("latency_ms", {}).get("capture_age_p95", -1.0),
        "ctrl": ctrl.describe(),
    }
    if verbose:
        print(f"    {json.dumps(out, ensure_ascii=False)}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="实战彩排:模拟 FPS 里的复眼伺服闭环")
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--cm360", type=float, default=40.0)
    ap.add_argument("--dpi", type=float, default=800.0)
    ap.add_argument("--target-ang-deg", type=float, default=11.0)
    ap.add_argument("--fov", type=float, default=103.0,
                    help="游戏水平 FOV;必须给,否则增益语义差 2.5 倍(见 gain.py)")
    ap.add_argument("--eye", default="24x32",
                    help="复眼网格 rows x cols(默认 24x32 = 项目默认;密度决定角分辨率)")
    ap.add_argument("--respawn", action="store_true",
                    help="命中即换位(Gridshot 语义;默认关,这样才量得到稳靶率)")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    eye_rows, eye_cols = (int(v) for v in args.eye.lower().split("x"))
    base = GainConfig.from_cm360(args.dpi, args.cm360, fov_h_deg=args.fov)
    legacy = GainConfig.from_cm360(args.dpi, args.cm360)          # 旧语义(未给 FOV)
    gain = GainModel(base)
    sim = SimFPS3DConfig(target_ang_deg=args.target_ang_deg,
                         respawn_on_hit=bool(args.respawn))
    EYE = dict(eye_rows=eye_rows, eye_cols=eye_cols)
    print("=" * 100)
    print("实战彩排:模拟 FPS(针孔投影 / 准星居中 / 青色靶 / 世界点阵光流)")
    print("=" * 100)
    print(f"  画面 {sim.width}x{sim.height}  FOV {sim.fov_h_deg}°  "
          f"靶角直径 {sim.target_ang_deg}°(= 命中判定角半径 {sim.target_ang_deg/2:.2f}°)")
    print(f"  增益 counts_per_360={base.counts_per_360:.0f}  FOV={args.fov}°  "
          f"每拍满量程 {gain.deg_per_action():.2f}°(旧语义 "
          f"{GainModel(legacy).deg_per_action():.2f}°,差 "
          f"{GainModel(legacy).deg_per_action()/gain.deg_per_action():.2f} 倍)")
    print(f"  复眼 {eye_rows}x{eye_cols}(小眼接受角 {sim.width/eye_cols:.1f}px)  "
          f"respawn_on_hit={sim.respawn_on_hit}  时长 {args.seconds}s")
    print(f"  中心 px/度(理论)= "
          f"{(sim.width/2)/np.tan(np.radians(sim.fov_h_deg/2))*np.pi/180:.2f}\n", flush=True)

    # ---------------- 1) 正对照 ----------------
    print("-" * 100)
    print("### 1) 正对照:复眼伺服(chroma=r8 青靶 / aim_mode=center / search=scan)")
    print(f"{'':<28}{'角度误差 mean/p95':>22}{'锁定率':>10}{'锁定后':>10}"
          f"{'首达拍':>8}{'靶在画面内':>12}{'拍频':>8}")
    print("-" * 100, flush=True)
    pos = run_once(gain=GainModel(base), sim_cfg=sim, chroma="r8",
                   aim_mode="center", search="scan", max_seconds=args.seconds, **EYE)
    print(f"{'eye-servo(correct)':<28}"
          f"{pos['err_mean_deg']:>9.2f}/{pos['err_p95_deg']:<9.2f}°"
          f"{pos['lock_frac']*100:>9.1f}%{pos['post_lock_frac']*100:>9.1f}%"
          f"{pos['first_lock_tick']:>8}{pos['on_screen_frac']*100:>11.1f}%"
          f"{pos['tick_hz']:>8.1f}", flush=True)

    # ---------------- 2) 负对照 ----------------
    print("\n" + "-" * 100)
    print("### 2) 负对照:证明「配错了就是不行」(不是随便配都行)")
    print("-" * 100, flush=True)
    negs = [
        ("chroma=r7(用长波找青靶)", dict(chroma="r7")),
        ("search=none(看不到靶就停住)", dict(search="none")),
        ("旧增益语义(不指定 FOV)", dict(legacy=True)),
    ]
    neg_rows = {}
    for label, kw in negs:
        r = run_once(gain=GainModel(legacy if kw.get("legacy") else base),
                     sim_cfg=sim, chroma=kw.get("chroma", "r8"),
                     aim_mode="center", search=kw.get("search", "scan"),
                     max_seconds=args.seconds, **EYE)
        neg_rows[label] = r
        print(f"{label:<28}{r['err_mean_deg']:>9.2f}/{r['err_p95_deg']:<9.2f}°"
              f"{r['lock_frac']*100:>9.1f}%{r['post_lock_frac']*100:>9.1f}%"
              f"{r['first_lock_tick']:>8}{r['on_screen_frac']*100:>11.1f}%"
              f"{r['tick_hz']:>8.1f}", flush=True)

    # ---------------- 3) 增益容差 ------------
    print("\n" + "-" * 100)
    print("### 3) 增益标定容差(counts_per_360 乘错多少会塌)")
    print("    它决定「cm/360 要标多准」;EPP(指针加速)开启时等效于此列随机漂移")
    print("-" * 100)
    print(f"{'增益倍率':>10}{'锁定率':>10}{'锁定后':>10}{'角度误差 mean':>16}{'首达拍':>8}")
    print("-" * 100, flush=True)
    gain_rows = []
    for mult in (0.5, 0.75, 1.0, 1.5, 2.0):
        gg = GainModel(GainConfig(counts_per_360=base.counts_per_360 * mult,
                                  speed_fraction_per_action=base.speed_fraction_per_action,
                                  fov_h_deg=args.fov,
                                  max_counts_per_tick=base.max_counts_per_tick))
        r = run_once(gain=gg, sim_cfg=sim, chroma="r8", aim_mode="center",
                     search="scan", max_seconds=args.seconds, **EYE)
        gain_rows.append({"mult": mult, **r})
        print(f"{mult:>9.2f}x{r['lock_frac']*100:>9.1f}%{r['post_lock_frac']*100:>9.1f}%"
              f"{r['err_mean_deg']:>14.2f}°{r['first_lock_tick']:>8}", flush=True)

    # ---------------- 4) 靶大小 × 复眼网格 ----------------
    print("\n" + "-" * 100)
    print("### 4) 能锁住多小的靶?(命中判定角半径 = 靶角直径/2)")
    print("    这一列直接换算成「实战能接什么任务」:")
    print("    Aim Lab Gridshot 的球约 2~3° 角直径(半径 1~1.5°)—— 比复眼网格的")
    print("    角分辨率下限还小,所以**不是所有任务都能接**,必须在任务内先量。")
    print("-" * 100)
    print(f"{'复眼网格':>12}{'靶角直径':>10}{'判定半径':>10}{'锁定率':>10}"
          f"{'锁定后':>10}{'稳态角误差':>14}")
    print("-" * 100, flush=True)
    size_rows = []
    for (er, ec) in ((24, 32), (48, 64)):
        for ang in (4.0, 7.0, 11.0, 16.0):
            s2 = SimFPS3DConfig(target_ang_deg=ang,
                               respawn_on_hit=bool(args.respawn))
            r = run_once(gain=GainModel(base), sim_cfg=s2, chroma="r8",
                         aim_mode="center", search="scan",
                         max_seconds=max(6.0, args.seconds / 2.0),
                         eye_rows=er, eye_cols=ec)
            size_rows.append({"eye": f"{er}x{ec}", "ang_deg": ang, **r})
            print(f"{f'{er}x{ec}':>12}{ang:>9.1f}°{ang/2:>9.1f}°"
                  f"{r['lock_frac']*100:>9.1f}%{r['post_lock_frac']*100:>9.1f}%"
                  f"{r['err_mean_deg']:>12.2f}°", flush=True)

    # ---------------- 4) 再捕获(靶一开始不在画面里) ----------------
    print("\n" + "-" * 100)
    print("### 4b) 增益语义对「恒定量程」控制器的杀伤(旧 fly 臂的失败模式)")
    print("    D12 的 const 模式:只出方向、不出幅度(永远满量程,不减速)。")
    print("    这是当时 fly 臂的行为特征 —— 它没有刹停律,增益错多少就直接飞多少")
    print("-" * 100)
    print(f"{'增益':<26}{'锁定率':>10}{'锁定后':>10}{'角度误差 mean':>16}")
    print("-" * 100, flush=True)
    bang_rows = {}
    for label, gg in (("正确(FOV=103°)", base), ("旧语义(过转 2.5x)", legacy)):
        r = run_once(gain=GainModel(gg), sim_cfg=sim, chroma="r8",
                     aim_mode="center", search="scan", max_seconds=args.seconds,
                     saturate=True, **EYE)
        bang_rows[label] = r
        print(f"{label:<26}{r['lock_frac']*100:>9.1f}%{r['post_lock_frac']*100:>9.1f}%"
              f"{r['err_mean_deg']:>14.2f}°", flush=True)

    # ---------------- 5) 再捕获 ----------------
    print("\n" + "-" * 100)
    print("### 5) 再捕获:靶一开始就不在画面里(D19 的 staring-at-wall 场景)")
    print("    初始角距 70~110°(视野半宽仅 51.5°)—— 这正是 D19 里 fly 臂"
          "「一转出去就再也回不来」的复现条件")
    print("-" * 100)
    print(f"{'搜索策略':<26}{'首次锁定':>10}{'锁定率':>10}{'锁定后':>10}"
          f"{'靶在画面内':>12}")
    print("-" * 100, flush=True)
    rec_rows = {}
    for label, srch in (("search=none(停在原地)", "none"),
                        ("search=scan(扫掠)", "scan")):
        s3 = SimFPS3DConfig(target_ang_deg=11.0,
                            respawn_on_hit=bool(args.respawn),
                            spawn_ang_deg=(70.0, 110.0))
        r = run_once(gain=GainModel(base), sim_cfg=s3, chroma="r8",
                     aim_mode="center", search=srch, max_seconds=args.seconds * 2.0,
                     **EYE)
        rec_rows[label] = r
        tick = r["first_lock_tick"]
        print(f"{label:<26}{(tick if tick >= 0 else -1):>10}"
              f"{r['lock_frac']*100:>9.1f}%{r['post_lock_frac']*100:>9.1f}%"
              f"{r['on_screen_frac']*100:>11.1f}%", flush=True)

    # ---------------- 6) 像素≈角度 ----------------
    print("\n" + "-" * 100)
    print("### 6) 「屏幕像素 ≈ 角度」这条线性近似的偏差(直接量投影,不跑闭环)")
    print("-" * 100)
    print(f"  {'角偏移':>8}{'实测 px':>12}{'局部 px/度':>14}{'vs 中心':>10}")
    print("-" * 100, flush=True)
    probe = SimFPS3DSource(sim, gain=GainModel(base))
    f_px = (sim.width / 2.0) / np.tan(np.radians(sim.fov_h_deg / 2.0))
    center_ratio = f_px * np.pi / 180.0
    proj_rows = []
    prev = 0.0
    for deg in (5.0, 10.0, 20.0, 30.0, 40.0):
        probe.target_yaw, probe.target_pitch = deg, 0.0
        u, _ = probe._project(probe.target_yaw, probe.target_pitch)
        px = u - sim.width / 2.0
        local = (px - prev) / (deg - (0.0 if deg == 5.0 else deg - 5.0))
        proj_rows.append({"deg": deg, "px": round(px, 2),
                          "px_per_deg": round(px / deg, 3)})
        print(f"{deg:>7.0f}°{px:>11.1f}{px/deg:>13.2f}"
              f"{px/deg/center_ratio:>9.2f}x", flush=True)
        prev = px
    print(f"  中心(小角度)理论 px/度 = {center_ratio:.2f};"
          f"40° 处已是 {proj_rows[-1]['px_per_deg']:.2f} "
          f"= 中心的 {proj_rows[-1]['px_per_deg']/center_ratio:.2f} 倍")
    print("  ⇒ 同一个 action 在画面边缘转过的角度**小于**中心,"
          "即 GainModel 的线性假设在最坏情况下有这个量级的系统偏差。")
    print("     本项目的对准动作几乎只发生在画面中心附近,所以它主要表现为"
          "「接近阶段稍微冲过头」,不是致命项 —— 但**必须随结果一起报**。")

    out_dir = Path(args.out) if args.out else (
        ROOT / "flyaim" / "runs"
        / (time.strftime("%Y%m%d-%H%M%S") + "-sim3d"))
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "sim3d_summary.json").write_text(json.dumps(
        {"config": vars(args), "sim": vars(sim), "gain": gain.describe(),
         "gain_legacy": GainModel(legacy).describe(),
         "positive": pos, "negative": neg_rows, "gain_sweep": gain_rows,
         "bangbang": bang_rows, "size_sweep": size_rows,
         "recapture": rec_rows, "projection": proj_rows},
        ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n产物 -> {out_dir / 'sim3d_summary.json'}")
    print("=" * 100)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
