"""FlyAim CLI 入口。

用法::

    python -m flyaim.main check                      # 检查 A 线数据是否就绪
    python -m flyaim.main run --seeds 10 --frames 900 # 跑完整对照实验
    python -m flyaim.main report <run_dir>            # 由已有结果重生成报告

设计原则:所有臂在同一批 seed、同一帧预算、同一评分口径下运行(CONTRACT 第 3 节)。
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

import numpy as np

from flyaim.config import ExperimentConfig
from flyaim.io import load_json, new_run_dir, save_json
from flyaim.pipeline import FlyArm, check_data_ready
from flyaim.report import build_report, compute_stats
from flyaim.runner import Arm, run_experiment

# ---------------------------------------------------------------- 常量

DEFAULT_DATA_DIR = "flyaim/data/build"
DEFAULT_RUNS_DIR = "flyaim/runs"


# ---------------------------------------------------------------- 辅助臂


class RandomArm:
    """均匀随机下界。"""

    name = "random"

    def __init__(self, cfg=None, seed: int = 0):
        del cfg
        self.rng = np.random.default_rng(seed)

    def reset(self, seed: int) -> None:
        self.rng = np.random.default_rng(seed)

    def act(self, frame, state=None) -> np.ndarray:
        del frame, state
        return self.rng.uniform(-1.0, 1.0, size=2).astype(np.float32)


def build_arms(cfg: ExperimentConfig, data_dir: Path, log=print) -> dict:
    """构造所有控制臂。

    返回 {arm_name: builder(seed) -> Arm}
    """
    builders: dict = {}

    # ---- fly:果蝇连接组 ----
    def fly_builder(seed: int):
        from flyaim.pipeline import FlySystem

        def factory(_s: int):
            return FlySystem(data_dir, cfg.retina, cfg.brain, cfg.readout)

        return FlyArm(factory, name="fly")

    # ---- shuffle:打乱接线零模型 ----
    def shuffle_builder(seed: int):
        from flyaim.pipeline import build_shuffle_override, FlySystem

        weights_path = build_shuffle_override(data_dir, seed, Path(cfg.out_dir) / "_shuffle")

        def factory(_s: int):
            return FlySystem(data_dir, cfg.retina, cfg.brain, cfg.readout,
                             weights_override=weights_path)

        return FlyArm(factory, name="shuffle")

    # ---- pid:性能上限参照(偷看靶位) ----
    def pid_builder(seed: int):
        try:
            from flyaim.baselines.pid import PIDBaseline
        except Exception as e:  # pragma: no cover
            log(f"[warn] PIDBaseline 不可用: {e}")
            return RandomArm(seed=seed)

        class _Pid:
            """PID 参照臂。允许偷看靶位;若 arena 不提供 `get_state()` 则退化为不动。

            注意:它**只用于给出性能上限刻度**,不参与"果蝇是否有效"的判定。
            """

            name = "pid"

            def __init__(self, s: int):
                self.seed = s
                self.inner = PIDBaseline()
                self.arena = None

            def attach_arena(self, arena) -> None:
                self.arena = arena

            def reset(self, s: int) -> None:
                self.inner = PIDBaseline()

            def act(self, frame, state=None) -> np.ndarray:
                del frame, state
                if self.arena is None or not hasattr(self.arena, "get_state"):
                    return np.zeros(2, dtype=np.float32)
                return self.inner.act(self.arena.get_state())

        return _Pid(seed)

    builders["fly"] = fly_builder
    builders["shuffle"] = shuffle_builder
    builders["pid"] = pid_builder
    builders["random"] = lambda seed: RandomArm(seed=seed)

    return {a: builders[a] for a in cfg.arms if a in builders}


# ---------------------------------------------------------------- state 供应

# 只有这些臂被允许看到环境状态(靶位/准星坐标)
STATE_ALLOWED = frozenset({"pid", "random", "oracle"})


def make_state_router(arm_name: str):
    """返回 state_fn。

    CONTRACT 禁止事项 1:果蝇臂**绝不允许**看到靶位坐标。
    这里在编排层再做一道闸:非白名单臂一律返回 None。
    """

    def state_fn(_res) -> dict | None:
        return None

    return state_fn


# ---------------------------------------------------------------- 子命令


def cmd_check(args) -> int:
    data_dir = Path(args.data_dir)
    ready, missing = check_data_ready(data_dir)
    print(f"数据目录: {data_dir.resolve()}")
    if ready:
        from flyaim.io import Manifest, ConnectomeArtifacts

        m = Manifest.load(data_dir / "manifest.json")
        art = ConnectomeArtifacts.load(data_dir / "connectome.npz")
        print("状态: ✅ 就绪")
        print(f"  神经元 N     : {m.n_neurons:,}")
        print(f"  边数         : {m.n_edges:,}")
        print(f"  npz 校验     : N={art.n:,}  nnz={art.n_edges:,}")
        print(f"  视觉输入降级 : {m.visual_input_fallback}  (策略: {m.visual_input_strategy})")
        for k, v in (m.roles or {}).items():
            print(f"  {k:22s}: {v}")
        if m.notes:
            print("  备注:")
            for n in m.notes:
                print(f"    - {n}")
        if m.n_neurons != art.n:
            print(f"  ❌ manifest N({m.n_neurons}) 与 npz N({art.n}) 不一致")
            return 2
        return 0
    print(f"状态: ❌ 缺失 {len(missing)} 个产物")
    for x in missing:
        print(f"  - {x}")
    print("\n(A 线仍在生成中;可先跑 `--dry-run` 验证编排)")
    return 1


def cmd_run(args) -> int:
    data_dir = Path(args.data_dir)
    ready, missing = check_data_ready(data_dir)
    if not ready:
        print(f"❌ 数据未就绪,缺失: {missing}")
        print("   先等 A 线完成,或运行 `python -m flyaim.main check` 查看进度。")
        return 1

    cfg = ExperimentConfig(
        seeds=tuple(range(args.seeds)),
        frames_per_episode=args.frames,
        arms=tuple(args.arms.split(",")),
        data_dir=str(data_dir),
        out_dir=args.out_dir,
    )
    cfg.arena.max_frames = args.frames

    run_dir = new_run_dir(args.out_dir, tag=args.tag)
    print(f"运行目录: {run_dir}")
    save_json(run_dir / "config.json", cfg.to_dict())

    from flyaim.arena.arena import Arena
    from flyaim.arena.metrics import Metrics

    builders = build_arms(cfg, data_dir, log=print)

    payload = run_experiment(
        cfg=cfg,
        arm_builders=builders,
        arena_factory=Arena,
        metrics_factory=Metrics,
        out_dir=run_dir,
        progress=print,
    )

    stats = compute_stats(payload, metric=args.metric)
    save_json(run_dir / "stats.json", stats)

    from flyaim.io import Manifest

    manifest = None
    mp = data_dir / "manifest.json"
    if mp.exists():
        import dataclasses

        manifest = dataclasses.asdict(Manifest.load(mp))

    text = build_report(run_dir / "REPORT.md", payload, stats, manifest)
    print("\n" + "=" * 70)
    primary = stats.get("primary")
    if primary:
        print(primary["verdict"])
    print("=" * 70)
    print(f"报告: {run_dir / 'REPORT.md'}  ({len(text.encode('utf-8')):,} 字节)")
    return 0


def cmd_report(args) -> int:
    run_dir = Path(args.run_dir)
    payload = load_json(run_dir / "raw_results.json")
    stats = compute_stats(payload, metric=args.metric)
    save_json(run_dir / "stats.json", stats)

    data_dir = Path(args.data_dir)
    manifest = None
    mp = data_dir / "manifest.json"
    if mp.exists():
        import dataclasses

        from flyaim.io import Manifest

        manifest = dataclasses.asdict(Manifest.load(mp))

    build_report(run_dir / "REPORT.md", payload, stats, manifest)
    print(f"报告已生成: {run_dir / 'REPORT.md'}")
    if stats.get("primary"):
        print(stats["primary"]["verdict"])
    return 0


# ---------------------------------------------------------------- main


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="flyaim", description="果蝇连接组闭环瞄准控制")
    p.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("check", help="检查 A 线数据是否就绪")
    sp.set_defaults(func=cmd_check)

    sp = sub.add_parser("run", help="跑完整对照实验")
    sp.add_argument("--seeds", type=int, default=10, help="种子数(契约要求 >= 10)")
    sp.add_argument("--frames", type=int, default=900, help="每 episode 帧数")
    sp.add_argument("--arms", default="fly,shuffle,pid,random")
    sp.add_argument("--out-dir", default=DEFAULT_RUNS_DIR)
    sp.add_argument("--metric", default="hit_rate")
    sp.add_argument("--tag", default="")
    sp.set_defaults(func=cmd_run)

    sp = sub.add_parser("report", help="由已有结果重生成报告")
    sp.add_argument("run_dir")
    sp.add_argument("--metric", default="hit_rate")
    sp.set_defaults(func=cmd_report)

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except KeyboardInterrupt:
        print("\n[中断]", file=sys.stderr)
        return 130
    except Exception:
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
