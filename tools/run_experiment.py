"""Phase 3:完整对照实验(果蝇 / 打乱接线 / PID / 随机)。

Lead 持有。使用 Phase 2 标定出的工作点:
    - 权重:行入度归一化(indeg),w_scale ≈ 16 → DN 14.7 Hz / 572 活跃
    - 读出:在冻结连接组上训练岭回归(训练 seed 与评估 seed 分离)

流程:
    1. 训练读出层(导师 PID 驱动靶场,记录 DN 放电率 → 期望方向)
    2. 冻结校验(assert_connectome_frozen)
    3. 四臂同 seed 同帧预算评估
    4. 预注册判定 + 出报告

运行::

    <捆绑 python> tools/run_experiment.py --seeds 10 --frames 900
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from flyaim.config import ArenaConfig, BrainConfig, ExperimentConfig, ReadoutConfig, RetinaConfig  # noqa: E402
from flyaim.fit_readout import assert_connectome_frozen, collect_training_set, fit_readout  # noqa: E402
from flyaim.io import (  # noqa: E402
    ConnectomeArtifacts,
    Manifest,
    load_roles,
    new_run_dir,
    save_json,
)
from flyaim.report import compute_stats, write_report  # noqa: E402
from flyaim.runner import run_episode  # noqa: E402

D = ROOT / "flyaim" / "data" / "build"
NORM = ROOT / "flyaim" / "runs" / "norm"

# ---- Phase 2 标定出的工作点 ----
CAL_W_SCALE = 16.0
CAL_INPUT_GAIN = 1.5
CAL_NORM = "indeg"


class RandomArm:
    name = "random"

    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)

    def reset(self, seed: int) -> None:
        self.rng = np.random.default_rng(seed)

    def act(self, frame, state=None):
        del frame, state
        return self.rng.uniform(-1.0, 1.0, size=2).astype(np.float32)


class PidArm:
    name = "pid"

    def __init__(self):
        from flyaim.baselines.pid import PIDBaseline

        self.inner = PIDBaseline()
        self.arena = None

    def attach_arena(self, arena) -> None:
        self.arena = arena

    def reset(self, seed: int) -> None:
        del seed

    def act(self, frame, state=None):
        del frame, state
        if self.arena is None:
            return np.zeros(2, dtype=np.float32)
        return self.inner.act(self.arena.get_state())


class FrozenReadoutArm:
    """用**同一套已训练好的读出权重**驱动果蝇臂。

    weights_override 用于 shuffle 零模型:前端与读出完全一致,只有连接组拓扑不同,
    因此任何差异都可归因于接线图本身 —— 这正是零模型对照要控制的唯一变量。
    """

    def __init__(self, name: str, weights_path: Path, readout_weights: dict,
                 retina_cfg, brain_cfg, readout_cfg, device: str = "cpu"):
        self.name = name
        self.weights_path = weights_path
        self.ro_weights = readout_weights
        self.retina_cfg = retina_cfg
        self.brain_cfg = brain_cfg
        self.readout_cfg = readout_cfg
        self.device = device
        self.system = None
        self.state_access_log: list = []

    @property
    def cfg(self):
        return self.readout_cfg

    def reset(self, seed: int) -> None:
        del seed
        from flyaim.pipeline import FlySystem

        self.system = FlySystem(D, self.retina_cfg, self.brain_cfg, self.readout_cfg,
                                weights_override=self.weights_path, device=self.device)
        self.system.readout.set_linear_weights(**self.ro_weights)
        self.system.reset()
        self.state_access_log.clear()

    def act(self, frame, state=None):
        # 架构级约束:记录并丢弃环境状态(果蝇只能看像素)
        if state:
            self.state_access_log.append(sorted(state.keys()))
        return self.system.act(frame)

    def assert_state_blind(self) -> None:
        if self.state_access_log:
            raise AssertionError(
                f"{self.name} 被传入环境状态 {len(self.state_access_log)} 次,违反禁止事项 1"
            )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--frames", type=int, default=900)
    ap.add_argument("--train-seeds", type=int, default=4)
    ap.add_argument("--train-frames", type=int, default=1200)
    ap.add_argument("--ridge", type=float, default=10.0)
    ap.add_argument("--record", action="store_true",
                    help="录制各臂的画面(APNG),产出 CONTRACT §4 要求的录屏")
    ap.add_argument("--record-every", type=int, default=4,
                    help="抽帧率:每 N 帧存 1 帧(默认 4,配合 900 帧 => ~225 帧动画)")
    ap.add_argument("--live", action="store_true",
                    help="写遥测到 <run>/live/<arm>/,供 tools/live_view.py 实时查看")
    ap.add_argument("--live-every", type=int, default=1,
                    help="遥测采样间隔(帧)")
    ap.add_argument("--live-preview-every", type=int, default=5,
                    help="遥测预览图保存间隔(帧)")
    ap.add_argument("--render", action="store_true", default=True,
                    help="评估时直接实时渲染窗口(默认开)")
    ap.add_argument("--no-render", dest="render", action="store_false",
                    help="关闭实时渲染")
    ap.add_argument("--window-size", default="1360x820",
                    help="渲染窗口大小,如 1600x900(默认 1360x820)")
    ap.add_argument("--refresh-every", type=int, default=2,
                    help="每 N 帧重绘一次(默认 2)")
    ap.add_argument("--render-backend", default="web", choices=["web", "tk"],
                    help="渲染后端:web=浏览器 Canvas(默认,不卡不抢焦点);tk=matplotlib")
    ap.add_argument("--device", default="cuda", choices=["auto", "cpu", "cuda"],
                    help="LIF 引擎:默认 cuda(GPU,实测 6.1x);cpu 复现历史结果")
    args = ap.parse_args()

    t_start = time.perf_counter()
    print("=" * 84)
    print("Phase 3 对照实验")
    print(f"  工作点: norm={CAL_NORM} w_scale={CAL_W_SCALE} input_gain={CAL_INPUT_GAIN}")
    print(f"  评估: {args.seeds} seeds × {args.frames} 帧")
    print("=" * 84, flush=True)

    # ---- 准备归一化权重 ----
    norm_wp = NORM / f"connectome_{CAL_NORM}.npz"
    if not norm_wp.exists():
        print(f"❌ 缺少归一化权重 {norm_wp},请先跑 tools/test_normalization.py")
        return 1
    roles_src = D / "roles.json"
    roles_dst = norm_wp.parent / "roles.json"
    if not roles_dst.exists():
        shutil.copy(roles_src, roles_dst)
    # 补齐 neuron_index(context/manifest 读取需要)
    for extra in ("neuron_index.parquet", "manifest.json"):
        dst = norm_wp.parent / extra
        if not dst.exists():
            shutil.copy(D / extra, dst)

    from flyaim.arena.arena import Arena
    from flyaim.arena.metrics import Metrics

    rc = RetinaConfig()
    bc = BrainConfig(weight_scale_exc=CAL_W_SCALE, weight_scale_inh=CAL_W_SCALE,
                     input_gain=CAL_INPUT_GAIN)
    ro_train = ReadoutConfig(mode="trained", ridge_lambda=args.ridge)
    ro_eval = ReadoutConfig(mode="trained", ridge_lambda=args.ridge)

    # ---- 1. 训练读出层 ----
    print("\n[1/4] 训练读出层(连接组冻结)", flush=True)
    before = ConnectomeArtifacts.load(norm_wp)
    from flyaim.pipeline import FlySystem

    model = FlySystem(D, rc, bc, ro_train, weights_override=norm_wp,
                      device=args.device)
    print(f"  引擎: {type(model.brain).__name__} (device={args.device})", flush=True)
    print(f"  复眼 {model.retina.describe()['n_input']} 个真实感光细胞;"
          f"读出源 {model.roles.descending.size} 个 DN", flush=True)

    from flyaim.baselines.pid import PIDBaseline

    pid = PIDBaseline()
    train_sets = []
    for s in range(args.train_seeds):
        t0 = time.perf_counter()
        ts = collect_training_set(
            system=model, arena=Arena(ArenaConfig(), seed=9000 + s), seed=9000 + s,
            n_frames=args.train_frames, leader_policy=lambda st: pid.act(st),
            log_interval=1, progress=print,
        )
        train_sets.append(ts)
        print(f"    seed {9000+s}: {time.perf_counter()-t0:.0f}s", flush=True)

    diag = fit_readout(model.readout, train_sets, ridge_lambda=args.ridge)
    print(f"  r2 = {diag['r2']:.4f}  rmse = {diag['rmse']:.4f}  "
          f"死特征 = {diag['n_dead_features']}/{diag['n_features']}", flush=True)
    if diag["r2"] <= 0.01:
        print("  ⚠️ r2 <= 0.01:读出层几乎没学到东西。DN 信号可能仍不可用。", flush=True)

    assert_connectome_frozen(before, norm_wp)
    print("  ✅ 连接组冻结校验通过", flush=True)
    ro_weights = {k: np.asarray(v) for k, v in model.readout.get_linear_weights().items()}

    # ---- 2. 生成 shuffle 零模型 ----
    print("\n[2/4] 生成打乱接线零模型", flush=True)
    from flyaim.baselines.shuffle import shuffle_connectome

    shuffle_paths: dict[int, Path] = {}
    base_art = ConnectomeArtifacts.load(norm_wp)
    sd = NORM / "shuffle"
    sd.mkdir(parents=True, exist_ok=True)
    for extra in ("roles.json", "neuron_index.parquet", "manifest.json"):
        dst = sd / extra
        if not dst.exists():
            shutil.copy(norm_wp.parent / extra, dst)
    for s in range(args.seeds):
        p = sd / f"connectome_shuffle_seed{s}.npz"
        if not p.exists():
            t0 = time.perf_counter()
            shuffle_connectome(base_art, s).save(p)
            print(f"  seed {s}: {time.perf_counter()-t0:.0f}s", flush=True)
        shuffle_paths[s] = p

    # ---- 3. 四臂评估 ----
    print("\n[3/4] 四臂评估", flush=True)
    run_dir = new_run_dir(ROOT / "flyaim" / "runs", tag="phase3")
    print(f"  运行目录 {run_dir}", flush=True)

    arena_cfg = ArenaConfig()
    arena_cfg.max_frames = args.frames
    results: dict[str, list[dict]] = {a: [] for a in ("fly", "shuffle", "pid", "random")}
    timings: dict[str, float] = {}
    record_log: list[dict] = []
    telemetry_plan = None
    if args.live:
        try:
            import sys as _sys
            _sys.path.insert(0, str(Path(__file__).resolve().parent))
            from telemetry import TelemetryWriter, make_sample_plan

            idx = pd.read_parquet(D / "neuron_index.parquet")
            from flyaim.io import load_roles

            _roles = load_roles(D / "roles.json")
            telemetry_plan = (TelemetryWriter, make_sample_plan,
                              idx, _roles)
            print(f"  [live] 遥测开启 -> {run_dir / 'live'}", flush=True)
        except Exception as e:
            print(f"  [warn] 遥测不可用({type(e).__name__}: {e}),本次不开", flush=True)
            telemetry_plan = None

    # --- 实时渲染 ---
    renderer_factory = None
    if args.render:
        try:
            import sys as _sys
            _sys.path.insert(0, str(Path(__file__).resolve().parent))
            import pandas as _pd

            _idx_r = _pd.read_parquet(D / "neuron_index.parquet")
            from flyaim.io import load_roles

            _roles_r = load_roles(D / "roles.json")
            _w, _h = (float(x) / 100.0 for x in args.window_size.lower().split("x"))
            _refresh = max(1, int(args.refresh_every))
            _backend = args.render_backend
            _run_dir = run_dir

            def renderer_factory(title):
                if _backend == "web":
                    from live_web import make_viewer

                    return make_viewer(_roles_r, _idx_r,
                                       out_dir=_run_dir / "live-web",
                                       title=title, every=_refresh)
                from live_render import make_renderer

                return make_renderer(_roles_r, _idx_r, window_title=title,
                                     figsize=(_w, _h), refresh_every=_refresh)

            print(f"  [render] 后端={_backend}  每 {_refresh} 帧刷新", flush=True)
        except Exception as e:
            print(f"  [warn] 渲染不可用({type(e).__name__}: {e})", flush=True)
            renderer_factory = None

    def evaluate(arm_name: str, builder) -> None:
        t_arm = time.perf_counter()
        for seed in range(args.seeds):
            arm = builder(seed)
            t0 = time.perf_counter()

            # 录屏(CONTRACT §4 交付物)。recorder 若不可用则明确告警,不静默跳过。
            recorder = None
            if args.record:
                try:
                    from flyaim.arena.recorder import FrameRecorder

                    recorder = FrameRecorder(
                        run_dir / f"episode_{arm_name}_seed{seed}.png",
                        every=args.record_every,
                    )
                except Exception as e:
                    print(f"  [warn] 录屏不可用({type(e).__name__}: {e}),本次不录", flush=True)
                    recorder = None

            # 实时遥测(独立进程可查看)
            telem = None
            observer = None
            live = None
            if renderer_factory is not None:
                live = renderer_factory(f"FlyAim — {arm_name} (seed {seed})")
            if telemetry_plan is not None or live is not None:
                TelemetryWriter, make_sample_plan, idx, _roles = (
                    telemetry_plan if telemetry_plan is not None
                    else (None, None, None, None))
                if telemetry_plan is not None:
                    plan = make_sample_plan(int(idx.shape[0]), _roles, idx)
                    telem = TelemetryWriter(
                        run_dir / "live" / f"{arm_name}_seed{seed}",
                        run_tag=f"{arm_name}/seed{seed}",
                        preview_every=args.live_preview_every,
                        **plan,
                    )
                _arm_ref, _telem_ref, _live_ref = arm, telem, live
                _every = max(1, int(args.live_every))

                def observer(i, frame, res, arena, _a=_arm_ref, _t=_telem_ref,
                             _e=_every, _name=arm_name, _sd=seed, _l=_live_ref):
                    if i % _e:
                        return
                    st = arena.get_state()
                    ch = np.asarray(st["crosshair"], np.float64).reshape(2)
                    tg = np.asarray(st["targets"], np.float64).reshape(-1, 2)
                    dist = float(np.linalg.norm(tg - ch, axis=1).min()) if tg.size else float("nan")
                    brain = getattr(_a, "system", None)
                    b = getattr(brain, "brain", None)
                    meta = {"target_dist": dist,
                            "hit": bool(getattr(res, "hit", False)),
                            "hits": int(getattr(res, "info", {}).get("n_respawns", 0)),
                            "arm": _name, "seed": _sd}
                    if b is None:
                        if _t is not None:
                            _t.step(t_ms=float(i), frame=frame,
                                    spikes=np.zeros(1, np.float32),
                                    rates=np.zeros(1, np.float32),
                                    meta={**meta, "no_brain": True})
                        if _l is not None:
                            _l.update(frame=frame, spikes=None, rates=None, meta=meta)
                        return
                    if _t is not None:
                        _t.step(t_ms=float(i), frame=frame,
                                spikes=b.spikes, rates=b.rates, meta=meta)
                    if _l is not None:
                        _l.update(frame=frame, spikes=b.spikes, rates=b.rates, meta=meta)

            res = run_episode(arm, Arena, arena_cfg, Metrics, seed=seed,
                              frames=args.frames, on_frame=recorder, observer=observer)
            if hasattr(arm, "assert_state_blind"):
                arm.assert_state_blind()

            if telem is not None:
                info = telem.close()
                print(f"     遥测 -> {info['jsonl']}  ({info['frames']} 帧, "
                      f"{info['bytes']/1024:.0f} KB, 预览 {info['previews']})", flush=True)
            if live is not None:
                try:
                    live.close()
                except Exception:
                    pass

            if recorder is not None:
                try:
                    info = recorder.close()
                    status = info.get("status", "ok")
                    if status != "ok":
                        # 空 episode 是合法情形,但必须显式记录,
                        # 否则会出现"录屏缺了但实验报成功"(recorder 的 RuntimeWarning)。
                        print(f"     [warn] 录屏 status={status}: {info.get('warnings')}", flush=True)
                    else:
                        print(f"     录屏 -> {info.get('path')} "
                              f"({info.get('frames')} 帧, {info.get('bytes', 0)/1024:.1f} KB)",
                              flush=True)
                    record_log.append({"arm": arm_name, "seed": seed, **info})
                except Exception as e:
                    print(f"     [warn] 录屏写盘失败: {type(e).__name__}: {e}", flush=True)

            results[arm_name].append(res.to_dict())
            print(f"  [{arm_name:8s}] seed={seed:2d} hit_rate={res.summary['hit_rate']:.4f} "
                  f"dist={res.summary['mean_target_dist_px']:6.1f}px "
                  f"({time.perf_counter()-t0:.0f}s)", flush=True)
        timings[arm_name] = time.perf_counter() - t_arm

    evaluate("fly", lambda s: FrozenReadoutArm(
        "fly", norm_wp, ro_weights, rc, bc, ro_eval, device=args.device))
    evaluate("shuffle", lambda s: FrozenReadoutArm(
        "shuffle", shuffle_paths[s], ro_weights, rc, bc, ro_eval, device=args.device))
    evaluate("pid", lambda s: PidArm())
    evaluate("random", lambda s: RandomArm(s))

    # ---- 4. 统计与报告 ----
    print("\n[4/4] 统计与报告", flush=True)
    payload = {"config": ExperimentConfig(
        seeds=tuple(range(args.seeds)), frames_per_episode=args.frames,
        arms=("fly", "shuffle", "pid", "random"), arena=arena_cfg,
        retina=rc, brain=bc, readout=ro_eval).to_dict(),
        "results": results, "timings": timings}
    save_json(run_dir / "raw_results.json", payload)
    save_json(run_dir / "readout_diag.json", diag)
    if record_log:
        save_json(run_dir / "recording_log.json", record_log)
        n_ok = sum(1 for r in record_log if r.get("status") == "ok")
        print(f"  录屏 {n_ok}/{len(record_log)} 个 episode 成功", flush=True)

    stats = compute_stats(payload, metric="hit_rate")
    save_json(run_dir / "stats.json", stats)

    manifest = None
    mp = D / "manifest.json"
    if mp.exists():
        import dataclasses

        manifest = dataclasses.asdict(Manifest.load(mp))

    notes = [
        f"工作点为 Phase 2 标定结果:权重行入度归一化(indeg)+ weight_scale={CAL_W_SCALE} + input_gain={CAL_INPUT_GAIN}。",
        f"读出层为岭回归(lambda={args.ridge}),训练 {args.train_seeds} seeds × {args.train_frames} 帧,"
        f"训练 seed 为 9000+,评估 seed 为 0..{args.seeds-1},二者分离。",
        f"读出训练诊断:r2={diag['r2']:.4f}, 死特征 {diag['n_dead_features']}/{diag['n_features']}。",
        "读出监督目标为**比例控制语义**(方向 × 随距离衰减的幅度,参考距离 = 2 个 action 步长),"
        "而非纯单位方向;后者会导致全速过冲振荡。",
        "连接组权重在训练前后逐位校验未变(assert_connectome_frozen 通过)。",
    ]
    text = write_report(run_dir / "REPORT.md", payload, stats, manifest, extra_notes=notes)
    print("\n" + "=" * 84)
    if stats.get("primary"):
        print(stats["primary"]["verdict"])
    print("=" * 84)
    print(f"报告: {run_dir / 'REPORT.md'} ({len(text.encode('utf-8')):,} B)")
    print(f"总耗时 {(time.perf_counter()-t_start)/60:.1f} 分钟")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
