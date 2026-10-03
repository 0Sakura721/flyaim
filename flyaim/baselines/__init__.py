"""C 线:对照实验的三个基线臂(pid / random / shuffle 零模型)。"""

from __future__ import annotations

from flyaim.baselines.pid import PIDBaseline
from flyaim.baselines.random import RandomBaseline
from flyaim.baselines.shuffle import shuffle_connectome, verify_shuffle

__all__ = [
    "PIDBaseline",
    "RandomBaseline",
    "shuffle_connectome",
    "verify_shuffle",
]
