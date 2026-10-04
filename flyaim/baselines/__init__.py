"""C 线:对照实验的三个基线臂(pid / random / shuffle 零模型)+ 复眼视觉伺服。

注意 `EyeServoArm` 的性质与三个基线都不同:它**不偷看**环境(只用帧),
但**绕过了连接组** —— 它是"果蝇的眼睛 + 一个 2 维读出",用来界定任务难度,
不得替代 `fly` 进入 CONTRACT 第 3 节的判定。见 DECISIONS.md D25。
"""

from __future__ import annotations

from flyaim.baselines.eye_servo import EyeServoArm, make_eye_servo
from flyaim.baselines.pid import PIDBaseline
from flyaim.baselines.random import RandomBaseline
from flyaim.baselines.shuffle import shuffle_connectome, verify_shuffle

__all__ = [
    "EyeServoArm",
    "make_eye_servo",
    "PIDBaseline",
    "RandomBaseline",
    "shuffle_connectome",
    "verify_shuffle",
]
