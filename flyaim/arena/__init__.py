"""C 线:自建 2D 瞄准靶场与指标。契约见 CONTRACT.md 2.4。"""

from __future__ import annotations

from flyaim.arena.arena import Arena, StepResult
from flyaim.arena.metrics import CORE_KEYS, Metrics
from flyaim.arena.recorder import FrameRecorder
from flyaim.arena.rollout import as_act_fn, run_episode

__all__ = [
    "Arena",
    "StepResult",
    "Metrics",
    "CORE_KEYS",
    "run_episode",
    "as_act_fn",
    "FrameRecorder",
]
