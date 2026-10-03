"""PID 基线 —— **作弊参照系(性能上限刻度)**,不是果蝇的成绩。

.. warning::

   **此基线使用了目标坐标,仅作性能上限参照,不得与果蝇成绩混为一谈。**

   它直接读取 ``Arena.get_state()`` 里的靶心像素坐标(以及 ``width``/``height``/
   ``speed_px_per_action``),完全没有经过 Retina(像素)/Connectome(DN 读出)
   这条真实链路。CONTRACT.md 第 3 节明确规定:``pid`` 只提供"好好瞄准能到什么水平"
   的刻度,**不参与**"该接线图对本任务是否有因果贡献"的判定
   (禁止事项 4:禁止把 pid 的成绩当作"果蝇的成绩"对外表述)。

控制律
------
取最近靶心,误差 ``e = target_center - crosshair``(像素),归一化
``u = e / speed_px_per_action``(``|u| = 1`` 表示"恰好一帧全速能走完的位移"),然后::

    action = clip(kp * u + kd * (u - u_prev) + ki * integral, -1, 1)

* ``kp = 1`` 时是 deadbeat(一帧内消除误差),这就是"理想化上限"的含义;
* ``kd`` 抑制接近时的过冲;
* ``ki`` 默认 0(积分项对这个纯伺服任务只有坏处,但保留以名副其实);
* ``deadband_px`` 内的误差视为 0,避免在靶心附近抖动。

接口与果蝇控制器一致(``act(state) -> (2,)``),另外提供 ``attach_arena(arena)``,
便于编排层把环境引用交给"偷看型"基线。
"""

from __future__ import annotations

import numpy as np

from flyaim.config import ArenaConfig

__all__ = ["PIDBaseline"]


class PIDBaseline:
    """比例 + 微分(可选积分)控制器,直接偷看靶心坐标。

    Parameters
    ----------
    cfg:
        可选的 :class:`ArenaConfig`,用于缺省 ``speed_px_per_action``(仅当 state 里
        没给这个键时使用)。不会修改它。
    kp, kd, ki:
        归一化误差的比例 / 微分 / 积分增益。默认 ``kp=1.0, kd=0.15, ki=0.0``。
    deadband_px:
        误差绝对值小于该值时输出 0(像素),默认 0.5。
    target_index:
        ``None`` = 追最近靶(默认);给定整数 = 固定追第 k 个靶。
    normalize_derivative:
        是否对微分项做限幅(``|du| <= 2``),防止靶 respawn 的跳变把微分项打爆。

    Notes
    -----
    状态通过 :meth:`attach_arena` 绑定后,``act()`` 可以不带参数调用;
    也可以显式传 ``act(state)``。
    """

    def __init__(
        self,
        cfg: ArenaConfig | None = None,
        kp: float = 1.0,
        kd: float = 0.15,
        ki: float = 0.0,
        deadband_px: float = 0.5,
        target_index: int | None = None,
        normalize_derivative: bool = True,
        seed: int = 0,
    ) -> None:
        self.cfg = cfg if cfg is not None else ArenaConfig()
        self.kp = float(kp)
        self.kd = float(kd)
        self.ki = float(ki)
        self.deadband_px = float(deadband_px)
        self.target_index = target_index
        self.normalize_derivative = bool(normalize_derivative)
        self.seed = int(seed)

        self._arena = None
        self._u_prev = np.zeros(2, dtype=np.float64)
        self._integral = np.zeros(2, dtype=np.float64)
        self._initialized = False
        self.last_error_px = np.zeros(2, dtype=np.float64)
        self.last_target_dist: float = float("inf")
        self.last_state: dict | None = None
        self.tuning_result: dict | None = None

    # ------------------------------------------------------------------ 环境绑定

    def attach_arena(self, arena) -> "PIDBaseline":
        """绑定 ``Arena`` 引用(编排层的可选钩子 ``attach_arena``)。

        绑定后 ``act()``/``act(None)`` 会实时读取 ``arena.get_state()``,
        这样基线永远看到"最新"的环境状态。
        """
        self._arena = arena
        return self

    # ------------------------------------------------------------------ 控制

    def reset(self) -> None:
        """清空微分/积分记忆(每个 episode 开始时调用)。"""
        self._u_prev = np.zeros(2, dtype=np.float64)
        self._integral = np.zeros(2, dtype=np.float64)
        self._initialized = False
        self.last_error_px = np.zeros(2, dtype=np.float64)
        self.last_target_dist = float("inf")

    def act(self, arena_state: dict | None = None) -> np.ndarray:
        """返回 ``(2,) float32`` 的鼠标增量 ``[dx, dy]``,已 clip 到 ``[-1, 1]``。

        Parameters
        ----------
        arena_state:
            环境状态。**期望(且只期望)以下键**(逐字来自 ``Arena.get_state()``):

            ========================  ==================================================
            ``crosshair``             (2,) 准星像素坐标 ``(x, y)`` —— 必需
            ``targets``               (K,2) 所有靶心像素坐标 —— 必需
            ``target_radius``         靶半径,像素 —— 可选(仅用于诊断记录)
            ``speed_px_per_action``   action=1.0 对应的像素位移 —— 借用它把像素误差
                                      归一化成 action;缺失则回退到构造时的 ``cfg``
            ``width`` / ``height``    画面尺寸 —— 可选,本控制器不用
            ========================  ==================================================

            三种写法都接受:① ``dict``;② ``Arena`` 对象(自动 ``get_state()``);
            ③ ``StepResult``(自动用 ``.info``,它含上述键)。传 ``None`` 时使用
            :meth:`attach_arena` 绑定的 arena。
        """
        state = _as_state(arena_state, self._arena)
        self.last_state = state

        crosshair = np.asarray(state["crosshair"], dtype=np.float64).reshape(2)
        targets = np.asarray(state["targets"], dtype=np.float64).reshape(-1, 2)
        if targets.shape[0] == 0:
            self.last_error_px = np.zeros(2, dtype=np.float64)
            return np.zeros(2, dtype=np.float32)

        if self.target_index is None:
            d = np.linalg.norm(targets - crosshair[None, :], axis=1)
            j = int(np.argmin(d))
            self.last_target_dist = float(d[j])
        else:
            j = int(self.target_index) % targets.shape[0]
            self.last_target_dist = float(np.linalg.norm(targets[j] - crosshair))

        err = targets[j] - crosshair  # 像素
        self.last_error_px = err.copy()

        speed = state.get("speed_px_per_action", None)
        speed = float(self.cfg.speed_px_per_action if speed is None else speed)
        if speed <= 0.0:
            # 无法移动的靶场:给出 0 而不是除零
            return np.zeros(2, dtype=np.float32)

        # 死区:足够近就保持不动
        if float(np.linalg.norm(err)) <= self.deadband_px:
            u = np.zeros(2, dtype=np.float64)
        else:
            u = err / speed

        if not self._initialized:
            self._u_prev = u.copy()
            self._initialized = True

        du = u - self._u_prev
        if self.normalize_derivative:
            du = np.clip(du, -2.0, 2.0)
        self._u_prev = u.copy()

        out = self.kp * u + self.kd * du
        if self.ki != 0.0:
            self._integral = np.clip(self._integral + u, -2.0, 2.0)
            out = out + self.ki * self._integral

        out = np.clip(out, -1.0, 1.0).astype(np.float32)
        return out

    # ------------------------------------------------------------------ 调参

    def tune(
        self,
        seed: int = 0,
        frames: int | None = None,
        grid: tuple[tuple[float, float], ...] | None = None,
        cfg: ArenaConfig | None = None,
        verbose: bool = False,
    ) -> "PIDBaseline":
        """在给定 seed 的靶场上网格搜索 ``(kp, kd)``,把本对象的增益设为最优值。

        完全确定性:每个候选增益都在 **同一个 seed、同一帧预算** 的靶场上评测,
        没有任何额外随机性。返回 ``self``(便于链式调用),结果存在
        ``self.tuning_result``(含 ``kp``/``kd``/``hit_rate``)。

        注意:该 seed 只用于**选增益**,不用于对照实验的评测 seed
        (CONTRACT 第 3 节:不得在对比不同条件时更换靶场 seed / 帧预算)。
        """
        from flyaim.arena.arena import Arena
        from flyaim.arena.metrics import Metrics

        arena_cfg = cfg if cfg is not None else self.cfg
        n_frames = int(frames if frames is not None else min(300, arena_cfg.max_frames))
        cand = grid if grid is not None else ((0.6, 0.0), (0.8, 0.0), (1.0, 0.0),
                                              (0.8, 0.15), (1.0, 0.15), (1.0, 0.3))

        best_key: tuple[float, float, float] | None = None  # (hit_rate, -kd, -kp)
        best_gains: tuple[float, float] = (self.kp, self.kd)
        best_rate = float("-inf")
        results = []
        for kp, kd in cand:
            probe = PIDBaseline(
                cfg=arena_cfg,
                kp=kp,
                kd=kd,
                ki=self.ki,
                deadband_px=self.deadband_px,
                target_index=self.target_index,
                normalize_derivative=self.normalize_derivative,
                seed=seed,
            )
            arena = Arena(arena_cfg, seed=seed)
            arena.reset()
            probe.attach_arena(arena)
            m = Metrics(frames_budget=n_frames)
            for _ in range(n_frames):
                res = arena.step(probe.act())
                m.update(res)
                if res.done:
                    break
            hr = float(m.summary()["hit_rate"])
            results.append({"kp": kp, "kd": kd, "hit_rate": hr})
            if verbose:
                print(f"[tune] kp={kp} kd={kd} hit_rate={hr:.4f}")
            # 平手时偏好更小的 kd、再偏好更小的 kp(更稳、更少依赖微分)
            key = (hr, -float(kd), -float(kp))
            if best_key is None or key > best_key:
                best_key = key
                best_gains = (float(kp), float(kd))
                best_rate = hr

        assert best_key is not None
        self.kp, self.kd = best_gains
        self.tuning_result = {
            "seed": int(seed),
            "frames": n_frames,
            "kp": self.kp,
            "kd": self.kd,
            "hit_rate": float(best_rate),
            "grid": results,
        }
        self.reset()
        return self

    def __repr__(self) -> str:
        return (
            f"PIDBaseline(kp={self.kp}, kd={self.kd}, ki={self.ki}, "
            f"deadband_px={self.deadband_px}, target_index={self.target_index})"
        )


def _as_state(arena_state, attached_arena) -> dict:
    """把 dict / Arena / StepResult / None 统一成 state dict。"""
    if arena_state is None:
        if attached_arena is None:
            raise ValueError(
                "act() 没有拿到环境状态:请显式传入 state dict,"
                "或先用 attach_arena(arena) 绑定靶场"
            )
        return attached_arena.get_state()
    if isinstance(arena_state, dict):
        if "crosshair" not in arena_state or "targets" not in arena_state:
            raise KeyError(
                "state dict 必须含 'crosshair' 和 'targets'(见 Arena.get_state())"
            )
        return arena_state
    get_state = getattr(arena_state, "get_state", None)
    if callable(get_state):
        return get_state()
    info = getattr(arena_state, "info", None)
    if isinstance(info, dict):
        return info
    raise TypeError(
        f"无法把 {type(arena_state)!r} 解释为 arena state;"
        "请传 dict / Arena / StepResult"
    )
