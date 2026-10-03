"""Phase 2:训练读出层(冻结连接组)。

流程(CONTRACT 禁止事项 2 允许的唯一学习路径):
    1. 用导师策略(PID)驱动靶场,产生有覆盖度的靶位序列
    2. 每一帧把画面喂给「真实感光细胞 → 真实连接组」,记录 1,360 个下行神经元的放电率
    3. 用岭回归拟合 DN 放电率 → 期望鼠标方向
    4. **训练前后逐位校验连接组权重未变**(assert_connectome_frozen)
    5. 把读出权重存盘,供 fly/shuffle 两臂共用

注意:训练用 seed 与评估用 seed 严格分离(ReadoutConfig.train_seeds / eval_seeds)。

运行::

    <捆绑 python> tools/train_readout.py --seeds 6 --frames 2000
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from flyaim.config import ArenaConfig, BrainConfig, ReadoutConfig, RetinaConfig  # noqa: E402
from flyaim.fit_readout import assert_connectome_frozen, collect_training_set, fit_readout  # noqa: E402
from flyaim.io import ConnectomeArtifacts, save_json  # noqa: E402
from flyaim.pipeline import FlySystem  # noqa: E402

DATA = ROOT / "flyaim" / "data" / "build"
OUT = ROOT / "flyaim" / "runs" / "readout"
WEIGHTS = DATA / "connectome.npz"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=6)
    ap.add_argument("--frames", type=int, default=2000)
    ap.add_argument("--ridge", type=float, default=1.0)
    ap.add_argument("--subnet", type=int, default=0, help="0 = 全量")
    ap.add_argument("--live", action="store_true",
                    help="写遥测到 <OUT>/live/,供 tools/live_view.py 实时查看训练")
    ap.add_argument("--live-preview-every", type=int, default=8)
    ap.add_argument("--tag", default="", help="run 目录后缀,便于区分")
    ap.add_argument("--render", action="store_true", default=True,
                    help="直接实时渲染窗口(默认开)")
    ap.add_argument("--no-render", dest="render", action="store_false",
                    help="关闭实时渲染")
    ap.add_argument("--window-size", default="1360x820",
                    help="渲染窗口大小,如 1600x900(默认 1360x820)")
    ap.add_argument("--refresh-every", type=int, default=2,
                    help="每 N 帧重绘一次(默认 2;脑仿真慢,无需每帧重绘)")
    ap.add_argument("--render-backend", default="web", choices=["web", "tk"],
                    help="渲染后端:web=浏览器 Canvas(默认,不卡不抢焦点);tk=matplotlib")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"],
                    help="LIF 引擎:auto(默认,有 GPU 就用)/ cpu / cuda")
    ap.add_argument("--dt-ms", type=float, default=None,
                    help="LIF 步长(默认 4.0;传 1.0 复现历史结果,但慢 4.8 倍)")
    ap.add_argument("--steps-per-frame", type=int, default=None,
                    help="每帧步数(默认 8;传 33 复现历史结果)")
    args = ap.parse_args()

    print("=" * 74)
    print("训练读出层(连接组冻结)")
    print("=" * 74)

    # --- 遥测(独立进程可实时查看) ---
    telem = None
    if args.live:
        try:
            sys.path.insert(0, str(Path(__file__).resolve().parent))
            from telemetry import TelemetryWriter, make_sample_plan
            import pandas as _pd

            _idx = _pd.read_parquet(DATA / "neuron_index.parquet")
            _roles = load_roles(DATA / "roles.json")
            live_dir = OUT / ("live" + (f"-{args.tag}" if args.tag else ""))
            plan = make_sample_plan(int(_idx.shape[0]), _roles, _idx)
            telem = TelemetryWriter(live_dir, run_tag="train",
                                    preview_every=args.live_preview_every, **plan)
            print(f"[live] 遥测 -> {telem.jsonl}")
            print(f"[live] 另开一窗运行:")
            print(f"       {sys.executable} tools/live_view.py --dir \"{live_dir}\"")
        except Exception as e:
            print(f"[warn] 遥测不可用({type(e).__name__}: {e})")
            telem = None

    # --- 事前权重指纹(用于冻结校验) ---
    before = ConnectomeArtifacts.load(WEIGHTS)
    fp_before = tuple(
        (m.nnz, int(m.data.sum()), int(m.indptr[-1]))
        for m in (before.W_exc, before.W_inh)
    )
    print(f"连接组: N={before.n:,} 边={before.n_edges:,}")
    print(f"  事前指纹: {fp_before}")

    rc = RetinaConfig()
    # 标定工作点(见 tools/test_normalization.py):
    #   weight_norm="indeg" + weight_scale=16 -> DN 有效活动
    # 若用默认 weight_scale=0.02,网络一步都不发放(1360 个 DN 全为死特征)。
    #
    # 时间步:默认用 BrainConfig 的 dt=4ms / 8 步(实测 4.79x 提速,保真度不变)。
    # 传 --dt-ms 1.0 --steps-per-frame 33 可复现早期结果(慢 4.8 倍)。
    bc = BrainConfig(
        weight_scale_exc=16.0,
        weight_scale_inh=16.0,
        input_gain=1.5,
        weight_norm="indeg",
        subnet_size=(args.subnet or None),
        **({"dt_ms": args.dt_ms} if args.dt_ms is not None else {}),
        **({"steps_per_frame": args.steps_per_frame}
           if args.steps_per_frame is not None else {}),
    )
    print(f"  时间步: dt={bc.dt_ms}ms × {bc.steps_per_frame} 步 "
          f"= {bc.dt_ms*bc.steps_per_frame:.0f}ms 模拟/帧")
    ro = ReadoutConfig(mode="trained", ridge_lambda=args.ridge)

    from flyaim.arena.arena import Arena
    from flyaim.baselines.pid import PIDBaseline

    arena_cfg = ArenaConfig()
    model = FlySystem(DATA, rc, bc, ro, device=args.device)
    print(f"  引擎: {type(model.brain).__name__} (device={args.device})")
    if hasattr(model.brain, "info"):
        print(f"    {model.brain.info()}")
    print(f"  复眼: {model.retina.describe()['n_input']} 个真实感光细胞")
    print(f"  读出源: {model.roles.descending.size} 个下行神经元")

    # --- 实时渲染窗口(进程内,直接画) ---
    renderer = None
    if args.render:
        try:
            sys.path.insert(0, str(Path(__file__).resolve().parent))
            import pandas as _pd

            _idx = _pd.read_parquet(DATA / "neuron_index.parquet")
            if args.render_backend == "web":
                from live_web import make_viewer

                renderer = make_viewer(
                    model.roles, _idx,
                    out_dir=OUT / ("live" + (f"-{args.tag}" if args.tag else "")),
                    title="FlyAim — 读出层训练(实时)",
                    every=max(1, args.refresh_every),
                )
            else:
                from live_render import make_renderer

                _w, _h = (float(x) / 100.0 for x in args.window_size.lower().split("x"))
                renderer = make_renderer(
                    model.roles, _idx,
                    window_title="FlyAim — 读出层训练(实时)",
                    figsize=(_w, _h),
                    refresh_every=args.refresh_every,
                )
            if getattr(renderer, "is_live", False):
                print(f"  [render] 已启动 ({args.render_backend})  每 {args.refresh_every} 帧刷新")
            else:
                print("  [render] 已降级为无输出模式(详见 stderr)")
                renderer = None
        except Exception as e:
            print(f"  [warn] 渲染不可用({type(e).__name__}: {e})")
            renderer = None

    train_sets = []
    pid = PIDBaseline()
    for s in range(args.seeds):
        arena = Arena(arena_cfg, seed=9000 + s)  # 训练 seed 段:9000+
        t0 = time.perf_counter()
        ts = collect_training_set(
            system=model,
            arena=arena,
            seed=9000 + s,
            n_frames=args.frames,
            leader_policy=lambda st: pid.act(st),
            log_interval=1,
            progress=print,
            telemetry=telem,
            diag_every=50,
            renderer=renderer,
        )
        train_sets.append(ts)
        print(f"    {time.perf_counter()-t0:.1f}s")
        if telem is not None:
            telem.diag(r2=float("nan"), n_samples=sum(t.n_samples for t in train_sets),
                       seed_done=9000 + s)

    print("\n--- 拟合岭回归 ---")
    diag = fit_readout(model.readout, train_sets, ridge_lambda=args.ridge)
    for k, v in diag.items():
        print(f"  {k:20s}: {v}")
    if telem is not None:
        telem.diag(r2=float(diag.get("r2", float("nan"))),
                   n_samples=int(diag.get("n_samples", 0)),
                   rmse=float(diag.get("rmse", float("nan"))),
                   final=True)

    if renderer is not None:
        try:
            renderer.update(frame=None, spikes=None, rates=None,
                            meta={"r2": float(diag.get("r2", float("nan"))),
                                  "n_samples": int(diag.get("n_samples", 0)),
                                  "final": True})
        except Exception:
            pass

    if telem is not None:
        info = telem.close()
        print(f"\n[live] 遥测 {info['frames']} 帧 / {info['bytes']/1024:.0f} KB -> {info['jsonl']}")

    # --- 冻结校验:这是防造假的关键检查 ---
    print("\n--- 连接组冻结校验 ---")
    assert_connectome_frozen(before, WEIGHTS)
    after = ConnectomeArtifacts.load(WEIGHTS)
    fp_after = tuple(
        (m.nnz, int(m.data.sum()), int(m.indptr[-1]))
        for m in (after.W_exc, after.W_inh)
    )
    print(f"  事后指纹: {fp_after}")
    assert fp_before == fp_after, "连接组指纹变化——违反 CONTRACT 禁止事项 2"
    print("  ✅ 连接组权重逐位未变(断言通过)")

    # --- 存盘 ---
    OUT.mkdir(parents=True, exist_ok=True)
    wp = OUT / "readout_weights.npz"
    w = model.readout.get_linear_weights()
    np.savez(wp, **{k: np.asarray(v) for k, v in w.items()})
    print(f"\n读出权重 -> {wp} ({wp.stat().st_size:,} B)")

    save_json(OUT / "train_diag.json", {**diag, "frozen_check": "passed",
                                        "fingerprint": str(fp_before),
                                        "n_train_frames": args.seeds * args.frames})
    print(f"诊断     -> {OUT / 'train_diag.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
