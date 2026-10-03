"""均匀随机基线 —— 对照实验的**性能下界**。

每个 action 从 ``[-1, 1]^2`` 均匀独立采样,不看任何环境状态(连靶场都不需要)。
它回答的问题是:"什么都不学、纯抖动,能得到多少命中率?"
任何真实臂(``fly``/``shuffle``)如果不显著优于本基线,说明它连"随机乱动"都不如。

确定性:同 seed -> 同一条 action 序列。
"""

from __future__ import annotations

import numpy as np

__all__ = ["RandomBaseline"]


class RandomBaseline:
    """均匀随机鼠标增量。

    Parameters
    ----------
    seed:
        PRNG 种子;相同 seed 产生完全相同的 action 序列。
    scale:
        输出缩放(默认 1.0 = 满幅)。采样后仍会 clip 到 ``[-1, 1]``。
    """

    def __init__(self, seed: int = 0, scale: float = 1.0) -> None:
        self.seed = int(seed)
        self.scale = float(scale)
        self._rng = np.random.default_rng(self.seed)
        self._arena = None

    def reset(self) -> None:
        """重置 PRNG 到构造时的 seed(episode 边界调用,保证可复现)。"""
        self._rng = np.random.default_rng(self.seed)

    def attach_arena(self, arena) -> "RandomBaseline":
        """兼容编排层的可选钩子。随机基线**不使用**环境信息,仅为接口统一。"""
        self._arena = arena  # noqa: SLF001 —— 仅保存引用,绝不读取
        return self

    def act(self, arena_state: dict | None = None) -> np.ndarray:
        """返回 ``(2,) float32`` 的 ``[dx, dy]``,均匀分布于 ``[-1, 1]``。

        ``arena_state`` 被完全忽略(传 ``None`` / dict / ``Arena`` 都行),
        以明确表达"本基线不看环境"。
        """
        out = self._rng.uniform(-1.0, 1.0, size=2) * self.scale
        return np.clip(out, -1.0, 1.0).astype(np.float32)

    def __repr__(self) -> str:
        return f"RandomBaseline(seed={self.seed}, scale={self.scale})"
