"""闭环 episode 驱动的便利封装(可选工具,不改变任何契约)。

Phase 3 的编排层由 Lead 持有(``flyaim/runner.py``);本模块只是给 C 线的自检、
基线对比和快速实验提供一个人人都能用的最小驱动,避免各处重复写 while 循环。

两种控制器协议
--------------
1. 两参数回调 ``act_fn(frame, state) -> (2,)``:真实闭环(需要像素)用它。
2. 单参数控制器对象 ``controller.act(state)``(PID / random 这类"偷看型"基线)
   用 :func:`as_act_fn` 包一层即可。
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from flyaim.arena.arena import Arena
from flyaim.arena.metrics import Metrics

__all__ = ["run_episode", "as_act_fn"]

ActFn = Callable[[np.ndarray, dict], np.ndarray]


def as_act_fn(controller: Any, kind: str = "state") -> ActFn:
    """把控制器对象适配成 ``act_fn(frame, state)``。

    Parameters
    ----------
    controller:
        任意对象。
    kind:
        ``"state"``(默认)—— 调 ``controller.act(state)``,用于 PID/random;
        ``"frame"`` —— 调 ``controller.act(frame)``,用于只看像素的控制器;
        ``"frame_state"`` —— 调 ``controller.act(frame, state)``。
        若对象本身就是可调用的,直接返回它(此时 ``kind`` 被忽略)。
    """
    if callable(controller) and not hasattr(controller, "act"):
        return controller

    if kind == "state":
        return lambda frame, state: controller.act(state)
    if kind == "frame":
        return lambda frame, state: controller.act(frame)
    if kind == "frame_state":
        return lambda frame, state: controller.act(frame, state)
    raise ValueError(f"kind 必须是 'state'/'frame'/'frame_state',收到 {kind!r}")


def run_episode(
    arena: Arena,
    act_fn: ActFn,
    max_frames: int | None = None,
    metrics: Metrics | None = None,
    n_bins: int = 10,
    fps: float | None = None,
    collect_frames: bool = False,
    on_step: Callable | None = None,
) -> tuple[Metrics, list[np.ndarray] | None]:
    """跑完一个 episode。

    Parameters
    ----------
    arena:
        已构造的 :class:`Arena`(本函数会调用 ``reset()``)。
    act_fn:
        ``act_fn(frame, state) -> (2,)``;``state`` 是 ``arena.get_state()``,
        在每帧 step **之前** 读取(即控制器看到的当前状态)。
    max_frames:
        覆盖帧预算(``None`` = 用 ``arena.cfg.max_frames``)。
    metrics:
        传入已有 Metrics 则复用,否则新建(``n_bins``/``fps`` 生效)。
    collect_frames:
        是否把每帧存进列表返回(900 帧 640x480x3 ≈ 830 MB,**默认关闭**)。
    on_step:
        每帧回调 ``on_step(res)``,用于流式记录/录屏。

    Returns
    -------
    (metrics, frames)
        ``frames`` 为 ``None``(未收集)或 ``list[np.ndarray]``。
    """
    budget = int(max_frames if max_frames is not None else arena.cfg.max_frames)
    if metrics is None:
        metrics = Metrics(fps=fps, n_bins=n_bins, frames_budget=budget)

    frames: list[np.ndarray] | None = [] if collect_frames else None
    frame = arena.reset()
    state = arena.get_state()
    n = 0
    while n < budget:
        action = np.asarray(act_fn(frame, state), dtype=np.float32).reshape(-1)
        res = arena.step(action)
        metrics.update(res)
        if frames is not None:
            frames.append(res.frame)
        if on_step is not None:
            on_step(res)
        n += 1
        frame = res.frame
        state = arena.get_state()
        if bool(getattr(res, "done", False)):
            break
    return metrics, frames
