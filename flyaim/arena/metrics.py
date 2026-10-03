"""Episode 指标汇总。契约见 CONTRACT.md 2.4。

``Metrics.summary()`` 必含键(逐字)::

    hits, shots, hit_rate, mean_target_dist_px, time_to_first_hit_ms,
    frames, fps_loop, acq_curve

其中 ``acq_curve`` 是本项目的核心诊断量:把整段 episode 等分成 ``n_bins`` 个箱,
给出每箱内的命中率,用来回答 **"它有没有越玩越准(acquired the skill)"**,
而不只是"总共打中几次"。箱子按 ``frames_budget``(= ``ArenaConfig.max_frames``)
划分,这样不同臂/不同 seed 即使提前结束也共享同一套横轴,曲线可直接平均。

设计约束(集成层要求):``update(res)`` **只读 ``res.hit`` 与 ``res.target_dist``**
(``res.done`` 用 getattr 安全读取),不依赖 ``StepResult.info`` 的任何键,
这样编排层可以自由往 info 里塞调试字段。

``Metrics()`` 必须能无参构造。
"""

from __future__ import annotations

import numpy as np

__all__ = ["Metrics", "CORE_KEYS"]

# summary() 的必含键(契约逐字要求的 8 个,顺序即契约顺序)
CORE_KEYS = (
    "hits",
    "shots",
    "hit_rate",
    "mean_target_dist_px",
    "time_to_first_hit_ms",
    "frames",
    "fps_loop",
    "acq_curve",
)


class Metrics:
    """累积 ``StepResult`` 并汇总。

    Parameters
    ----------
    fps:
        实测回路帧率(Hz),用于把帧数换算成毫秒(``time_to_first_hit_ms``)
        以及输出 ``fps_loop``。可以是 ``None``(未知),之后用 :meth:`set_fps` 注入。
    n_bins:
        ``acq_curve`` 的箱数,默认 10。
    frames_budget:
        episode 的帧预算(通常传 ``cfg.max_frames``)。给定时 ``acq_curve`` 的箱
        边界按它划分,便于跨臂逐箱平均;``None`` 时按实际观测帧数划分。

    用法::

        m = Metrics(n_bins=10, frames_budget=900)
        m.set_fps(29.8)                     # 由外部实测耗时换算后注入
        ... m.update(res) ...
        m.add_extra("arm", "pid")
        summary = m.summary()               # 纯 Python 类型,可直接 json.dumps
    """

    def __init__(
        self,
        fps: float | None = None,
        n_bins: int = 10,
        frames_budget: int | None = None,
    ) -> None:
        if int(n_bins) < 1:
            raise ValueError(f"n_bins 必须 >= 1,收到 {n_bins}")
        if fps is not None and float(fps) <= 0.0:
            raise ValueError(f"fps 必须 > 0 或 None,收到 {fps}")
        if frames_budget is not None and int(frames_budget) < 1:
            raise ValueError(f"frames_budget 必须 >= 1 或 None,收到 {frames_budget}")
        self.n_bins = int(n_bins)
        self.frames_budget = None if frames_budget is None else int(frames_budget)
        self._fps: float | None = None if fps is None else float(fps)
        self.reset()

    # ------------------------------------------------------------------ 生命周期

    def reset(self) -> None:
        """清空所有累计量(保留 fps / n_bins / frames_budget 配置与 extras)。"""
        self._hits: int = 0
        self._frames: int = 0
        self._first_hit_frame: int | None = None
        self._dists: list[float] = []
        self._hit_flags: list[bool] = []
        self._hit_dists: list[float] = []
        self._done: bool = False
        self._extras: dict = {}

    def set_fps(self, fps: float | None) -> None:
        """注入实测回路帧率(Hz)。``None`` 表示未知。"""
        if fps is not None and float(fps) <= 0.0:
            raise ValueError(f"fps 必须 > 0 或 None,收到 {fps}")
        self._fps = None if fps is None else float(fps)

    def add_extra(self, key: str, value) -> None:
        """塞入附加指标,会原样合并进 ``summary()``。

        为避免覆盖契约要求的核心键,``key`` 与 :data:`CORE_KEYS` 冲突时抛错。
        """
        if key in CORE_KEYS:
            raise ValueError(f"add_extra 的 key {key!r} 与契约核心键冲突,请改名")
        self._extras[key] = value

    # ------------------------------------------------------------------ 累积

    def update(self, res) -> None:
        """吃一个 ``StepResult``(鸭子类型:只要求 ``hit`` 与 ``target_dist``)。"""
        idx = self._frames  # 本帧的 0-based 下标
        self._frames += 1

        dist = float(getattr(res, "target_dist"))
        self._dists.append(dist)

        hit = bool(getattr(res, "hit"))
        self._hit_flags.append(hit)
        if hit:
            self._hits += 1
            self._hit_dists.append(dist)
            if self._first_hit_frame is None:
                self._first_hit_frame = idx
        if bool(getattr(res, "done", False)):
            self._done = True

    # ------------------------------------------------------------------ 汇总

    @property
    def hits(self) -> int:
        return int(self._hits)

    @property
    def shots(self) -> int:
        """射击次数 == 帧数(每帧自动开火,见 arena.py 的设计决定)。"""
        return int(self._frames)

    @property
    def frames(self) -> int:
        return int(self._frames)

    def acq_curve(self) -> list[float]:
        """分箱命中率(学习曲线)。空箱记 0.0,空箱数见 ``summary()["acq_curve_counts"]``。"""
        flags = np.asarray(self._hit_flags, dtype=np.float64)
        if flags.size == 0:
            return [0.0] * self.n_bins
        total = self.frames_budget if self.frames_budget else flags.size
        total = max(int(total), int(flags.size))
        idx = np.arange(flags.size, dtype=np.int64)
        bins = np.minimum(idx * self.n_bins // total, self.n_bins - 1)
        out: list[float] = []
        for b in range(self.n_bins):
            sel = bins == b
            out.append(float(flags[sel].mean()) if bool(np.any(sel)) else 0.0)
        return out

    def acq_curve_counts(self) -> list[int]:
        """每个箱里实际有多少帧(用于识别空箱)。"""
        if self._frames == 0:
            return [0] * self.n_bins
        total = self.frames_budget if self.frames_budget else self._frames
        total = max(int(total), int(self._frames))
        idx = np.arange(self._frames, dtype=np.int64)
        bins = np.minimum(idx * self.n_bins // total, self.n_bins - 1)
        return [int(np.count_nonzero(bins == b)) for b in range(self.n_bins)]

    def summary(self) -> dict:
        """返回汇总 dict(纯 Python 类型,可直接 ``json.dumps``)。"""
        frames = int(self._frames)
        hits = int(self._hits)
        hit_rate = (hits / frames) if frames else 0.0
        dists = np.asarray(self._dists, dtype=np.float64)
        mean_dist = float(dists.mean()) if dists.size else 0.0

        ttff: float | None = None
        if self._first_hit_frame is not None and self._fps:
            # 第 i 帧(0-based)结束时刻 == (i+1)/fps 秒
            ttff = float(self._first_hit_frame + 1) * 1000.0 / float(self._fps)

        out = {
            "hits": hits,
            "shots": frames,
            "hit_rate": float(hit_rate),
            "mean_target_dist_px": mean_dist,
            "time_to_first_hit_ms": ttff,
            "frames": frames,
            "fps_loop": (None if self._fps is None else float(self._fps)),
            "acq_curve": self.acq_curve(),
            # ---- 以下为附加诊断键(契约允许超出必含集合) ----
            "n_bins": int(self.n_bins),
            "acq_curve_counts": self.acq_curve_counts(),
            "frames_budget": self.frames_budget,
            "mean_target_dist_std_px": (float(dists.std()) if dists.size else 0.0),
            "min_target_dist_px": (float(dists.min()) if dists.size else 0.0),
            "mean_target_dist_at_hit_px": (
                float(np.mean(self._hit_dists)) if self._hit_dists else 0.0
            ),
            "time_to_first_hit_frames": self._first_hit_frame,
            "done": bool(self._done),
            "first_half_hit_rate": _half_rate(self._hit_flags, 0),
            "second_half_hit_rate": _half_rate(self._hit_flags, 1),
        }
        out.update(self._extras)
        return out

    def __repr__(self) -> str:
        return (
            f"Metrics(frames={self._frames}, hits={self._hits}, "
            f"hit_rate={self.hits / self._frames if self._frames else 0.0:.4f}, "
            f"n_bins={self.n_bins}, fps={self._fps})"
        )


def _half_rate(flags: list[bool], which: int) -> float:
    """前后半段命中率(which=0 前半 / 1 后半),用于快速看"是否越玩越准"。"""
    n = len(flags)
    if n == 0:
        return 0.0
    mid = n // 2
    seg = flags[:mid] if which == 0 else flags[mid:]
    if not seg:
        return 0.0
    return float(sum(1 for x in seg if x) / len(seg))
