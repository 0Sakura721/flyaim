"""自建 2D 瞄准靶场(Arena)。契约见 CONTRACT.md 2.4。

=========================== 关键设计决定:"开火"是每帧自动的 ===========================
本项目把"何时开火"这个自由度从控制问题里彻底拿掉:**每一帧结束时自动判定一次**,
只要准星的瞄准点(准星中心)落在任一靶的半径内,就计一次命中 ``hit=True``。
因此::

    shots == 帧数                       # 每帧都是一次射击机会
    hit_rate == 平均每帧命中率           # Metrics 里的定义
    控制器唯一需要学的事 == "把准星移到靶上并保持"

为什么这样设计是合理的(而不是偷懒):
  1. MaleCNS v1.0 是"脑 + 腹神经索",**没有复眼/视叶**,也**没有已知的"扣扳机"
     运动程序**对应的下行通路标注。把开火时序也交给连接组,等于凭空引入一个
     没有数据支撑的自由度,实验结果无法归因。
  2. 每帧自动开火把任务约化为一个纯粹的**视觉伺服(visual servoing)**问题:
     画面 -> Retina -> Connectome -> DN 读出 -> 鼠标增量,与真实实验要检验的
     那条链路一一对应,使"连接组对本任务有没有因果贡献"可以被干净地检验
     (CONTRACT 第 3 节的 fly vs shuffle 判定规则)。
  3. 判定规则是纯几何的,零时序抖动、零随机性:同一 seed + 同一 action 序列
     -> 逐位相同的帧序列,可确定性复现,消除了 cross-run 方差的一个主要来源。
  4. 对"作弊"基线(PID)也公平:它同样不能靠"提前扣扳机"取巧,只能靠瞄得准。

=====================================================================================

坐标系与渲染
------------
* 像素坐标 ``(x, y)``,``x`` 向右、``y`` 向下;``frame[y, x]``。
* ``action = (2,) [dx, dy]``,取值 clip 到 ``[-1, 1]``,映射为
  ``speed_px_per_action`` 像素位移;准星被裁剪在画面内(见 ``step``)。
* 靶:``n_targets`` 个半径 ``target_radius`` 的圆,随机散布、互不重叠、且
  初始不与准星重叠;可选匀速运动(碰壁按镜面反射反弹)。
* 渲染:纯 numpy 画进 ``(H, W, 3) uint8``。背景 ``bg_color`` 与靶 ``target_color``
  的亮度差 > 5 倍,靶外圈再加一道更亮的描边,保证人工复眼编码器(只看像素)
  有足够信噪比。

确定性
------
``Arena(cfg, seed)`` 内部持有 ``np.random.default_rng(seed)``;``reset()`` 会用
同一个 seed **重新播种**,所以 reset 是幂等的、可重复的。所有随机性只来自这个
PRNG,所有数值运算都是确定性的 numpy 表达式 -> 逐位可复现。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict

import numpy as np

from flyaim.config import ArenaConfig

__all__ = ["Arena", "StepResult"]

# 靶之间 / 靶与准星之间的最小间隙(像素),避免初始就重叠或贴边
_SPAWN_MARGIN = 4.0
# 拒绝采样最大次数(每个靶)
_SPAWN_TRIES_PER_TARGET = 400
# 靶出生点距离画面边缘的额外留白
_SPAWN_EDGE_PAD = 1.0


@dataclass
class StepResult:
    """``Arena.step`` 的返回结构(字段名/顺序与 CONTRACT.md 2.4 一致)。

    Attributes
    ----------
    frame:
        ``(H, W, 3) uint8``,**判定 hit 那一刻的场景**(命中时靶仍在画面里,
        并以更亮的颜色闪烁标记)。每次调用都是新分配的数组,不与内部缓冲共享。
    hit:
        本帧准星是否落在某个靶的半径内(= 每帧自动开火是否命中)。
    target_dist:
        本帧准星中心到**最近靶心**的像素距离(float)。判定的就是它。
    done:
        episode 是否结束(``frame_idx >= cfg.max_frames``)。
    info:
        dict,调试/可视化/基线偷看用。**是当前环境状态的超集**:
        含 ``crosshair``、``targets``、``target_vel``、``target_radius``、
        ``width``/``height``、``speed_px_per_action``、``frame_idx`` 等
        (与 ``Arena.get_state()`` 的键一致),外加本帧判定字段
        ``frame_index``、``hit_target_index``、``respawned``、
        ``nearest_target``、``target_dist``。
        注意 ``info["targets"]`` 是 **respawn 之后**的位置(下一步该瞄的目标),
        而 ``info["nearest_target"]`` 是本帧判定时那个靶的位置。
    """

    frame: np.ndarray
    hit: bool
    target_dist: float
    done: bool
    info: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.hit = bool(self.hit)
        self.done = bool(self.done)
        self.target_dist = float(self.target_dist)
        if self.info is None:
            self.info = {}


class Arena:
    """一个确定性的 2D 瞄准靶场。

    用法::

        arena = Arena(ArenaConfig(), seed=0)
        frame = arena.reset()                 # (H,W,3) uint8
        res = arena.step(np.array([0.3, -0.1], dtype=np.float32))
        res.frame, res.hit, res.target_dist, res.done, res.info

    Parameters
    ----------
    cfg:
        :class:`flyaim.config.ArenaConfig`。只读使用,不会被修改。
    seed:
        决定靶的初始位置与每次 respawn 的位置。相同 seed + 相同 action 序列
        -> 逐位相同的帧序列。
    """

    def __init__(self, cfg: ArenaConfig, seed: int) -> None:
        self.cfg = cfg
        self.seed = int(seed)

        w, h = int(cfg.width), int(cfg.height)
        r = int(cfg.target_radius)
        if int(cfg.n_targets) < 1:
            raise ValueError(f"n_targets 必须 >= 1,收到 {cfg.n_targets}")
        if w < 2 * r + 4 or h < 2 * r + 4:
            raise ValueError(
                f"画面 {w}x{h} 太小,放不下半径 {r} 的靶(需要 >= {2 * r + 4} 像素)"
            )
        if float(cfg.speed_px_per_action) < 0:
            raise ValueError("speed_px_per_action 必须 >= 0")

        self._rng = np.random.default_rng(self.seed)

        # 准星 / 靶的状态(全部 float64,便于确定性比较)
        self.crosshair: np.ndarray = np.zeros(2, dtype=np.float64)
        self.targets: np.ndarray = np.zeros((int(cfg.n_targets), 2), dtype=np.float64)
        self.target_vel: np.ndarray = np.zeros_like(self.targets)
        self.frame_idx: int = 0
        self._last_action: np.ndarray = np.zeros(2, dtype=np.float64)
        self._n_respawns: int = 0

        # 准星中心的可达范围
        self._ch_lo = np.array([float(_SPAWN_EDGE_PAD), float(_SPAWN_EDGE_PAD)])
        self._ch_hi = np.array([w - 1.0 - _SPAWN_EDGE_PAD, h - 1.0 - _SPAWN_EDGE_PAD])

        self.reset()

    # ------------------------------------------------------------------ 公共 API

    def reset(self) -> np.ndarray:
        """重置 episode(重新播种 PRNG -> 幂等)并返回首帧 ``(H,W,3) uint8``。"""
        cfg = self.cfg
        self._rng = np.random.default_rng(self.seed)
        self.frame_idx = 0
        self._last_action = np.zeros(2, dtype=np.float64)
        self._n_respawns = 0

        w, h = int(cfg.width), int(cfg.height)
        if cfg.start_centered:
            self.crosshair = np.array([(w - 1) / 2.0, (h - 1) / 2.0], dtype=np.float64)
        else:
            lo = np.array([cfg.target_radius + 1.0, cfg.target_radius + 1.0])
            hi = np.array(
                [w - 1.0 - cfg.target_radius - 1.0, h - 1.0 - cfg.target_radius - 1.0]
            )
            self.crosshair = self._rng.uniform(lo, np.maximum(lo, hi))

        self.targets = self._sample_positions(
            int(cfg.n_targets), exclude_center=self.crosshair,
            exclude_radius=float(cfg.crosshair_radius),
        )
        self.target_vel = self._sample_velocities(int(cfg.n_targets))
        return self._render()

    def step(self, action: np.ndarray) -> StepResult:
        """推进一帧。

        顺序(固定,保证确定性):
          1. 准星按 ``action * speed_px_per_action`` 位移,并裁剪到画面内;
          2. 靶按速度前进(碰壁镜面反射);
          3. 判定:准星中心到最近靶心的距离 <= ``target_radius`` 即 ``hit=True``;
          4. 渲染 **判定时刻** 的场景(命中靶高亮闪烁);
          5. 若 ``respawn_on_hit`` 且命中,命中靶在本帧渲染之后重采样到新位置
             (所以 ``info["targets"]`` 是下一步该瞄的位置)。

        Parameters
        ----------
        action:
            ``(2,) [dx, dy]``,已归一化到 ``[-1, 1]``;越界值会被 clip。

        Returns
        -------
        StepResult
        """
        cfg = self.cfg
        a = np.asarray(action, dtype=np.float64).reshape(-1)
        if a.size != 2:
            raise ValueError(f"action 必须是 (2,) [dx, dy],收到 shape={np.shape(action)}")
        a = np.clip(a, -1.0, 1.0)
        self._last_action = a

        # 1. 准星位移 + 画面裁剪
        self.crosshair = np.clip(
            self.crosshair + a * float(cfg.speed_px_per_action), self._ch_lo, self._ch_hi
        )

        # 2. 靶运动
        self._advance_targets()

        # 3. 判定(每帧自动开火)
        d = np.linalg.norm(self.targets - self.crosshair[None, :], axis=1)
        j = int(np.argmin(d))
        dist = float(d[j])
        hit = bool(dist <= float(cfg.target_radius))

        frame = self._render(hit_index=j if hit else -1)

        # 4. 命中后重采样(渲染之后,所以 frame 仍是"命中那一刻"的画面)
        respawned = False
        hit_target = self.targets[j].copy()
        if hit and bool(cfg.respawn_on_hit):
            self._respawn_target(j)
            respawned = True

        self.frame_idx += 1
        done = bool(self.frame_idx >= int(cfg.max_frames))

        info: Dict[str, Any] = self.get_state()
        info.update(
            frame_index=self.frame_idx - 1,
            t_frame=self.frame_idx - 1,
            hit=hit,
            target_dist=dist,
            hit_target_index=j,
            nearest_target=hit_target,
            respawned=respawned,
            n_respawns=self._n_respawns,
            action=a.copy(),
        )
        return StepResult(frame=frame, hit=hit, target_dist=dist, done=done, info=info)

    def render(self) -> np.ndarray:
        """按 **当前** 状态重新渲染一帧 ``(H,W,3) uint8``(契约要求的别名,无副作用)。

        与 ``StepResult.frame`` 的唯一差别:step 返回的是"判定 hit 那一刻"的画面,
        本方法返回的是"当前(可能已 respawn)"的画面。
        """
        return self._render()

    def get_state(self) -> dict:
        """返回当前环境状态 dict(给需要偷看环境的基线,如 PID 用)。

        键(集成层约定,不要改名):

        ==========================  ==========================================
        ``crosshair``               (2,) float64,准星中心像素坐标 (x, y)
        ``targets``                 (K,2) float64,所有靶心像素坐标 (x, y)
        ``target_vel``              (K,2) float64,靶的速度(px/帧);静止时为 0
        ``target_radius``           int,靶半径(命中判定阈值,像素)
        ``crosshair_radius``        int,准星绘制半径(像素)
        ``width`` / ``height``      int,画面尺寸
        ``speed_px_per_action``     float,action=1.0 对应的像素位移(归一化用)
        ``frame_idx``               int,已推进的帧数(下一帧的下标)
        ``max_frames``              int,帧预算
        ``seed``                    int,本 episode 的 seed
        ==========================  ==========================================

        返回的数组都是副本,外部修改不会影响靶场状态。
        """
        cfg = self.cfg
        return {
            "crosshair": self.crosshair.copy(),
            "targets": self.targets.copy(),
            "target_vel": self.target_vel.copy(),
            "target_radius": int(cfg.target_radius),
            "crosshair_radius": int(cfg.crosshair_radius),
            "width": int(cfg.width),
            "height": int(cfg.height),
            "speed_px_per_action": float(cfg.speed_px_per_action),
            "frame_idx": int(self.frame_idx),
            "max_frames": int(cfg.max_frames),
            "seed": int(self.seed),
        }

    # ------------------------------------------------------------------ 便捷只读属性

    @property
    def state(self) -> dict:
        """``get_state()`` 的属性别名。"""
        return self.get_state()

    @property
    def last_action(self) -> np.ndarray:
        """上一帧实际生效的 action(clip 后,副本)。"""
        return self._last_action.copy()

    @property
    def done(self) -> bool:
        """是否已达帧预算。"""
        return bool(self.frame_idx >= int(self.cfg.max_frames))

    def target_dist(self) -> float:
        """当前准星中心到最近靶心的像素距离。"""
        if self.targets.size == 0:
            return float("inf")
        return float(np.min(np.linalg.norm(self.targets - self.crosshair[None, :], axis=1)))

    # ------------------------------------------------------------------ 采样(确定性)

    def _bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """靶心可取范围(保证整个圆在画面内)。"""
        cfg = self.cfg
        r = float(cfg.target_radius) + _SPAWN_EDGE_PAD
        lo = np.array([r, r], dtype=np.float64)
        hi = np.array([cfg.width - 1.0 - r, cfg.height - 1.0 - r], dtype=np.float64)
        return lo, np.maximum(lo, hi)

    def _sample_positions(
        self, count: int, exclude_center: np.ndarray | None, exclude_radius: float
    ) -> np.ndarray:
        """拒绝采样 ``count`` 个互不重叠、且不与 exclude 圆重叠的靶心。

        完全由 ``self._rng`` 驱动 -> 确定性。极端配置(画面塞不下)下退化为
        确定性环形排布,并在 docstring 中说明该退化行为。
        """
        lo, hi = self._bounds()
        r = float(self.cfg.target_radius)
        min_sep = 2.0 * r + _SPAWN_MARGIN
        min_excl_sep = r + float(exclude_radius) + _SPAWN_MARGIN

        placed: list[np.ndarray] = []
        for i in range(count):
            ok = False
            p = np.zeros(2, dtype=np.float64)
            for _ in range(_SPAWN_TRIES_PER_TARGET):
                p = self._rng.uniform(lo, hi)
                if exclude_center is not None and (
                    float(np.linalg.norm(p - exclude_center)) < min_excl_sep
                ):
                    continue
                if any(float(np.linalg.norm(p - q)) < min_sep for q in placed):
                    continue
                ok = True
                break
            if not ok:
                p = self._fallback_position(i, count, lo, hi)
                if exclude_center is not None and (
                    float(np.linalg.norm(p - exclude_center)) < min_excl_sep
                ):
                    center = np.array(
                        [(self.cfg.width - 1) / 2.0, (self.cfg.height - 1) / 2.0]
                    )
                    p = np.clip(2.0 * center - p, lo, hi)
            placed.append(p)
        return np.asarray(placed, dtype=np.float64).reshape(count, 2)

    def _fallback_position(
        self, i: int, count: int, lo: np.ndarray, hi: np.ndarray
    ) -> np.ndarray:
        """拒绝采样失败时的确定性环形排布(不消耗 PRNG)。"""
        center = np.array([(self.cfg.width - 1) / 2.0, (self.cfg.height - 1) / 2.0])
        radius = 0.42 * min(float(self.cfg.width), float(self.cfg.height))
        angle = 2.0 * np.pi * (i / max(count, 1)) + 0.37
        p = center + radius * np.array([np.cos(angle), np.sin(angle)])
        return np.clip(p, lo, hi)

    def _sample_velocities(self, count: int) -> np.ndarray:
        """每个靶一个随机方向的匀速速度(px/帧);speed<=0 时全零。"""
        speed = float(self.cfg.target_speed_px_per_frame)
        if speed <= 0.0 or count == 0:
            return np.zeros((count, 2), dtype=np.float64)
        ang = self._rng.uniform(0.0, 2.0 * np.pi, size=count)
        return np.stack([speed * np.cos(ang), speed * np.sin(ang)], axis=1)

    def _respawn_target(self, index: int) -> None:
        """命中后把第 index 个靶重采样到新位置(确定性,复用同一个 PRNG 流)。"""
        others = [self.targets[k] for k in range(self.targets.shape[0]) if k != index]
        lo, hi = self._bounds()
        r = float(self.cfg.target_radius)
        min_sep = 2.0 * r + _SPAWN_MARGIN
        min_excl_sep = r + float(self.cfg.crosshair_radius) + _SPAWN_MARGIN

        p = None
        for _ in range(_SPAWN_TRIES_PER_TARGET):
            cand = self._rng.uniform(lo, hi)
            if float(np.linalg.norm(cand - self.crosshair)) < min_excl_sep:
                continue
            if any(float(np.linalg.norm(cand - q)) < min_sep for q in others):
                continue
            p = cand
            break
        if p is None:
            p = self._fallback_position(index, self.targets.shape[0], lo, hi)
            if float(np.linalg.norm(p - self.crosshair)) < min_excl_sep:
                center = np.array(
                    [(self.cfg.width - 1) / 2.0, (self.cfg.height - 1) / 2.0]
                )
                p = np.clip(2.0 * center - p, lo, hi)
        self.targets[index] = p
        if float(self.cfg.target_speed_px_per_frame) > 0.0:
            ang = float(self._rng.uniform(0.0, 2.0 * np.pi))
            self.target_vel[index] = float(self.cfg.target_speed_px_per_frame) * np.array(
                [np.cos(ang), np.sin(ang)]
            )
        self._n_respawns += 1

    def _advance_targets(self) -> None:
        """靶按速度前进,碰壁镜面反射(向量化、确定性)。"""
        if float(self.cfg.target_speed_px_per_frame) <= 0.0 or self.targets.size == 0:
            return
        self.targets = self.targets + self.target_vel
        r = float(self.cfg.target_radius)
        for axis, limit in ((0, float(self.cfg.width) - 1.0), (1, float(self.cfg.height) - 1.0)):
            lo, hi = r, limit - r
            under = self.targets[:, axis] < lo
            over = self.targets[:, axis] > hi
            if np.any(under):
                self.targets[under, axis] = lo + (lo - self.targets[under, axis])
            if np.any(over):
                self.targets[over, axis] = hi - (self.targets[over, axis] - hi)
            flip = under | over
            if np.any(flip):
                self.target_vel[flip, axis] = -self.target_vel[flip, axis]
            np.clip(self.targets[:, axis], lo, hi, out=self.targets[:, axis])

    # ------------------------------------------------------------------ 渲染(纯 numpy)

    def _render(self, hit_index: int = -1) -> np.ndarray:
        """把当前状态画进新的 ``(H,W,3) uint8`` 数组。"""
        cfg = self.cfg
        frame = np.empty((int(cfg.height), int(cfg.width), 3), dtype=np.uint8)
        frame[:, :] = np.asarray(cfg.bg_color, dtype=np.uint8)

        base = np.asarray(cfg.target_color, dtype=np.uint8)
        outline = _brighten(base, 0.45) if bool(getattr(cfg, "render_outline", True)) else None
        r = int(cfg.target_radius)
        for i in range(self.targets.shape[0]):
            color = _brighten(base, 0.60) if i == hit_index else base
            self._draw_disc(frame, self.targets[i, 0], self.targets[i, 1], r, color, outline)

        self._draw_crosshair(frame)
        return frame

    @staticmethod
    def _draw_disc(
        frame: np.ndarray,
        cx: float,
        cy: float,
        radius: float,
        color: np.ndarray,
        outline_color: np.ndarray | None = None,
    ) -> None:
        """在 frame 上画一个实心圆(可选一圈更亮的外描边)。"""
        h, w = frame.shape[0], frame.shape[1]
        x0 = max(0, int(np.floor(cx - radius)))
        x1 = min(w, int(np.ceil(cx + radius)) + 1)
        y0 = max(0, int(np.floor(cy - radius)))
        y1 = min(h, int(np.ceil(cy + radius)) + 1)
        if x1 <= x0 or y1 <= y0:
            return
        yy = np.arange(y0, y1, dtype=np.float64)[:, None] + 0.5
        xx = np.arange(x0, x1, dtype=np.float64)[None, :] + 0.5
        d2 = (xx - cx) ** 2 + (yy - cy) ** 2
        inside = d2 <= radius * radius
        patch = frame[y0:y1, x0:x1]
        patch[inside] = color
        if outline_color is not None and radius >= 3.0:
            inner = max(1.0, radius - max(1.0, radius * 0.25))
            patch[inside & (d2 > inner * inner)] = outline_color

    def _draw_crosshair(self, frame: np.ndarray) -> None:
        """画一个加号准星(亮度最高,便于人工复眼在降采样后仍能看到)。"""
        h, w = frame.shape[0], frame.shape[1]
        color = np.asarray(self.cfg.crosshair_color, dtype=np.uint8)
        cx = int(round(float(self.crosshair[0])))
        cy = int(round(float(self.crosshair[1])))
        half = int(max(4, self.cfg.crosshair_radius * 3))
        th = int(max(1, self.cfg.crosshair_radius // 2))

        ry0 = min(max(cy - th // 2, 0), h)
        ry1 = min(max(ry0 + th, 0), h)
        rx0 = min(max(cx - half, 0), w)
        rx1 = min(max(cx + half + 1, 0), w)
        if ry1 > ry0 and rx1 > rx0:
            frame[ry0:ry1, rx0:rx1] = color

        rx0 = min(max(cx - th // 2, 0), w)
        rx1 = min(max(rx0 + th, 0), w)
        ry0 = min(max(cy - half, 0), h)
        ry1 = min(max(cy + half + 1, 0), h)
        if ry1 > ry0 and rx1 > rx0:
            frame[ry0:ry1, rx0:rx1] = color


def _brighten(color: np.ndarray, t: float) -> np.ndarray:
    """把颜色朝白色插值 t 比例(确定性整数运算)。"""
    c = np.asarray(color, dtype=np.float64)
    out = c + (255.0 - c) * float(t)
    return np.clip(np.round(out), 0, 255).astype(np.uint8)
