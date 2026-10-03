"""实验编排:把「控制臂」放进「靶场」跑完,采集指标。

设计要点
--------
- 编排器**不关心**控制臂内部是什么(果蝇连接组 / PID / 随机),只依赖
  `Arm.act(frame, state)` 这一个协议 → 换臂不改编排。
- `state` 只传给允许偷看环境的基线。果蝇臂在 `FlyArm` 内部**显式丢弃** state,
  以保证 CONTRACT 禁止事项 1(视觉输入只能用像素)在架构上不可能被绕过。
- 同一 seed 下所有臂必须使用**完全相同**的靶场序列 → 配对比较有效。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Protocol

import numpy as np

from flyaim.config import ExperimentConfig
from flyaim.io import save_json

# ---------------------------------------------------------------- Arm 协议


class Arm(Protocol):
    """控制臂协议。

    act(frame, state) -> (2,) float32,值域约 [-1, 1]
        frame: (H, W, 3) uint8 靶场画面 —— 唯一被允许的视觉信息
        state: 含靶位/准星坐标的调试信息。**仅允许 pid/random/oracle 类基线使用。**
    """

    name: str

    def reset(self, seed: int) -> None: ...

    def act(self, frame: np.ndarray, state: dict | None = None) -> np.ndarray: ...


class StateBlindMixin:
    """显式丢弃 state 的混入,用于果蝇臂。

    这不是形式主义:它让"果蝇只能看像素"成为**类型级约束**,
    而不是靠开发者自觉。
    """

    _state_access_log: list

    def guarded_act(self, frame: np.ndarray, state: dict | None) -> np.ndarray:
        if state:
            self._state_access_log.append(sorted(state.keys()))
        raise NotImplementedError


# ---------------------------------------------------------------- 单次运行


@dataclass
class EpisodeResult:
    arm: str
    seed: int
    summary: dict
    wall_s: float
    frames: int
    config: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "arm": self.arm,
            "seed": self.seed,
            "wall_s": self.wall_s,
            "frames": self.frames,
            "summary": self.summary,
        }


def run_episode(
    arm: Arm,
    arena_factory: Callable[[object, int], object],
    arena_cfg,
    metrics_factory: Callable[[], object],
    seed: int,
    frames: int,
    state_fn: Callable[[object], dict] | None = None,
    on_frame: Callable[[int, np.ndarray, object], None] | None = None,
    observer: Callable[[int, np.ndarray, object, object, object], None] | None = None,
) -> EpisodeResult:
    """跑一个 episode。

    arena_factory(cfg, seed) -> Arena
    metrics_factory() -> Metrics
    state_fn(step_result) -> dict   仅对允许看环境的臂提供;
                                    果蝇臂会收到 None(见 flyaim.pipeline.FlyArm)
    on_frame(i, frame, step_result) 可选回调,用于录屏/可视化
    observer(i, frame, step_result, arena, arm) **观测者**回调,能拿到 arena 与 arm
        的完整状态(含靶位与脑活动)。用于遥测落盘。

        **边界声明:observer 的输出只进文件,绝不回流给 arm。**
        这不违反 CONTRACT 禁止事项 1 —— arm 本身仍然只看像素。
    """
    arena = arena_factory(arena_cfg, seed)
    metrics = metrics_factory()
    frame = arena.reset()

    # 可选钩子:允许需要环境状态的臂(pid 等)拿到 arena 引用。
    # 果蝇臂**没有**这个钩子,FlyArm 也不定义它 → 架构上无法偷看靶位。
    if hasattr(arm, "attach_arena"):
        arm.attach_arena(arena)

    # 必须重置控制臂:否则有状态的臂(如 FlyArm,构造 lazy 且要求先 reset)
    # 会在第一次 act 就抛错。顺序要在 attach_arena 之后,
    # 因为部分臂的 reset 依赖已绑定的 arena。
    arm.reset(seed)

    t0 = time.perf_counter()
    for i in range(frames):
        state = state_fn(None) if state_fn is not None else None
        action = np.asarray(arm.act(frame, state), dtype=np.float32).reshape(2)
        action = np.clip(action, -1.0, 1.0)

        res = arena.step(action)
        frame = res.frame
        metrics.update(res)

        if on_frame is not None:
            on_frame(i, frame, res)
        if observer is not None:
            observer(i, frame, res, arena, arm)

        if getattr(res, "done", False):
            break

    wall = time.perf_counter() - t0
    n_frames = i + 1
    if hasattr(metrics, "set_fps"):
        metrics.set_fps(n_frames / max(wall, 1e-9))

    summary = metrics.summary()
    return EpisodeResult(
        arm=getattr(arm, "name", arm.__class__.__name__),
        seed=seed,
        summary=summary,
        wall_s=wall,
        frames=n_frames,
    )


# ---------------------------------------------------------------- 全实验


def run_experiment(
    cfg: ExperimentConfig,
    arm_builders: dict[str, Callable[[int], Arm]],
    arena_factory: Callable[[object, int], object],
    metrics_factory: Callable[[], object],
    out_dir,
    progress: Callable[[str], None] | None = None,
) -> dict:
    """跑完整对照实验。

    arm_builders: {arm_name: builder(seed) -> Arm}
    返回结果字典,并落盘到 out_dir。
    """
    say = progress or (lambda _m: None)
    results: dict[str, list[dict]] = {a: [] for a in cfg.arms}
    timings: dict[str, float] = {}

    for arm_name in cfg.arms:
        if arm_name not in arm_builders:
            say(f"[skip] 臂 {arm_name} 未提供 builder")
            continue
        t_arm = time.perf_counter()
        for seed in cfg.seeds:
            builder = arm_builders[arm_name]
            arm = builder(seed)
            arena_cfg = cfg.arena
            # 保证帧预算一致:所有臂用同一 frames
            res = run_episode(
                arm=arm,
                arena_factory=arena_factory,
                arena_cfg=arena_cfg,
                metrics_factory=metrics_factory,
                seed=seed,
                frames=cfg.frames_per_episode,
            )
            results[arm_name].append(res.to_dict())
            say(
                f"[{arm_name}] seed={seed} hit_rate={res.summary.get('hit_rate', float('nan')):.3f} "
                f"mean_dist={res.summary.get('mean_target_dist_px', float('nan')):.1f}px "
                f"{res.wall_s:.2f}s"
            )
        timings[arm_name] = time.perf_counter() - t_arm

    payload = {
        "config": cfg.to_dict(),
        "results": results,
        "timings": timings,
    }
    out_dir = str(out_dir)
    save_json(f"{out_dir}/raw_results.json", payload)
    return payload
