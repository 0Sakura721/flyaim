"""监督训练读出层(`ReadoutConfig.mode == "trained"`)。

方法学说明(重要,别跳过)
------------------------
`CONTRACT.md` 禁止事项 2 规定:**学习只能发生在读出层,绝不能改连接组权重**。
本模块实现该约束下的唯一合法训练路径。

训练数据的产生方式:

    靶场(由导师策略驱动,如 PID) ──► 帧序列
                                        │
                       ┌────────────────┴────────────────┐
                       ▼                                 ▼
              Retina → Connectome                  靶位真值(arena state)
                       │                                 │
                  DN 放电率记录                    期望鼠标增量
                       └────────────┬────────────────────┘
                                    ▼
                          岭回归 (DN rates → action)
                                    │
                          冻结为 Readout.weights

关键点与诚实边界
----------------
1. **训练时靶位由导师策略驱动,不由果蝇驱动。** 否则果蝇的随机游走会让靶子
   永远停在同一位置,训练数据没有覆盖度。
2. **果蝇自身不学习**。被学习的只有读出层那一个线性映射(参数量 = n_DN × 2)。
   连接组权重在整个训练与评估过程中逐位不变,由 `assert_connectome_frozen()` 校验。
3. **评估必须用分离的 seed**(`ReadoutConfig.eval_seeds`),否则是数据泄漏。
4. 因此正确表述是:「**在固定接线图上训练一个线性读出器**」,而不是「果蝇学会了瞄准」。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from flyaim.config import ReadoutConfig
from flyaim.io import ConnectomeArtifacts, save_json

# ---------------------------------------------------------------- 记录容器


@dataclass
class TrainingSet:
    """训练/评估用的 (DN 放电率, 期望动作) 配对数据集。"""

    X: np.ndarray  # (n_samples, n_dn) float32,DN 放电率
    Y: np.ndarray  # (n_samples, 2) float32,期望鼠标增量 [-1,1]
    seed: int
    n_frames: int
    meta: dict

    def __post_init__(self) -> None:
        if self.X.ndim != 2 or self.Y.ndim != 2 or self.X.shape[0] != self.Y.shape[0]:
            raise ValueError(f"TrainingSet 形状不匹配: X={self.X.shape}, Y={self.Y.shape}")

    @property
    def n_samples(self) -> int:
        return int(self.X.shape[0])


# ---------------------------------------------------------------- 采集


def collect_training_set(
    system,
    arena,
    seed: int,
    n_frames: int,
    leader_policy: Callable[[dict], np.ndarray],
    log_interval: int = 1,
    progress: Callable[[str], None] | None = None,
    telemetry=None,
    diag_every: int = 50,
    renderer=None,
) -> TrainingSet:
    """采集「DN 放电率 → 期望鼠标增量」配对样本。

    参数
    ----
    system    : FlySystem(提供 retina / brain / readout;此处只用前两者)
    arena     : Arena 实例(**不会被果蝇控制**,由 leader_policy 驱动)
    leader_policy : state -> (2,) 导师动作。用 PID 即可。
    log_interval  : 每多少帧记录一次(降低样本自相关)。
    telemetry : 可选 `TelemetryWriter`。每帧调用其 `step()`,
                `diag_every` 帧额外 `diag(r2=...)`,供 `tools/live_view.py` 实时显示。
                **可视化关闭不影响训练。**

    返回
    ----
    TrainingSet

    注意:果蝇的 brain 会因输入帧而变化,**这是真实的前向传播**,不是模拟数据。
    """
    say = progress or (lambda _m: None)
    frame = arena.reset()
    system.reset()

    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []

    dn_idx = system.roles.descending
    if dn_idx.size == 0:
        raise RuntimeError("roles.descending 为空,无法训练读出层(无合法控制来源)")

    speed = float(getattr(arena.cfg, "speed_px_per_action", 1.0))
    n_img = 0

    for i in range(n_frames):
        # --- 果蝇前向传播(仅用于读 DN 活动) ---
        drive = system.retina.frame_to_spikes(frame)
        for _ in range(system.brain.cfg.steps_per_frame):
            system.brain.step(drive[:, 0], drive[:, 1], system.brain.cfg.dt_ms)

        want = None
        action = None
        if i % log_interval == 0:
            rates = np.asarray(system.brain.rates, dtype=np.float32)
            dn_rates = rates[dn_idx]

            # --- 期望动作:从 arena 真值算"朝靶心该走多少" ---
            state = arena.get_state() if hasattr(arena, "get_state") else {}
            want = _desired_action(state, speed)
            action = np.clip(np.asarray(leader_policy(state), dtype=np.float32).reshape(2), -1, 1)
            if want is not None:
                xs.append(dn_rates.copy())
                ys.append(want)
                n_img += 1

        # --- 遥测(独立进程可实时查看) ---
        if telemetry is not None or renderer is not None:
            st = arena.get_state() if hasattr(arena, "get_state") else {}
            ch = st.get("crosshair")
            tg = st.get("targets")
            dist = float("nan")
            if ch is not None and tg is not None:
                _c = np.asarray(ch, np.float64).reshape(2)
                _t = np.asarray(tg, np.float64).reshape(-1, 2)
                if _t.size:
                    dist = float(np.linalg.norm(_t - _c, axis=1).min())
            meta = {"target_dist": dist}
            if want is not None:
                meta["want"] = want
            if action is not None:
                meta["action"] = action
            if diag_every > 0 and i > 0 and i % diag_every == 0 and len(xs) >= 10:
                meta["r2"] = _quick_r2(np.stack(xs), np.stack(ys))
                meta["n_samples"] = len(xs)
            if telemetry is not None:
                telemetry.step(t_ms=float(i), frame=frame,
                               spikes=system.brain.spikes, rates=system.brain.rates,
                               meta=meta)
                if "r2" in meta:
                    telemetry.diag(r2=meta["r2"], n_samples=len(xs), frame=i)
            if renderer is not None:
                renderer.update(frame=frame, spikes=system.brain.spikes,
                                rates=system.brain.rates, meta=meta)

        # --- 导师驱动靶场 ---
        if action is None:
            state = arena.get_state() if hasattr(arena, "get_state") else {}
            action = np.clip(np.asarray(leader_policy(state), dtype=np.float32).reshape(2), -1, 1)
        res = arena.step(action)
        frame = res.frame

    if not xs:
        raise RuntimeError("未采集到任何训练样本——检查 arena.get_state() 是否可用")

    X = np.stack(xs).astype(np.float32)
    Y = np.stack(ys).astype(np.float32)
    say(f"  seed={seed}: 采集 {X.shape[0]} 样本, DN 维度={X.shape[1]}")

    return TrainingSet(
        X=X,
        Y=Y,
        seed=seed,
        n_frames=n_frames,
        meta={"speed_px_per_action": speed, "log_interval": log_interval, "n_logged": n_img},
    )


def _quick_r2(X: np.ndarray, Y: np.ndarray, lam: float = 1.0) -> float:
    """轻量岭回归 R²(仅用于实时诊断,不产出可用权重)。"""
    Xn = X.astype(np.float64)
    mu = Xn.mean(axis=0)
    sd = Xn.std(axis=0)
    sd_safe = np.where(sd < 1e-8, 1.0, sd)
    Xn = np.hstack([(Xn - mu) / sd_safe, np.ones((Xn.shape[0], 1))])
    n_feat = Xn.shape[1]
    A = Xn.T @ Xn + lam * np.eye(n_feat)
    A[-1, -1] -= lam
    try:
        W = np.linalg.solve(A, Xn.T @ Y.astype(np.float64))
    except np.linalg.LinAlgError:
        return float("nan")
    pred = Xn @ W
    resid = Y.astype(np.float64) - pred
    ss_res = float((resid ** 2).sum())
    ss_tot = float(((Y - Y.mean(axis=0)) ** 2).sum())
    return 1.0 - ss_res / max(ss_tot, 1e-12)


def _desired_action(state: dict, speed: float) -> np.ndarray | None:
    """由 arena 状态算出「朝最近靶心」的归一化鼠标增量。

    **幅度必须随距离衰减** —— 这是仿 PID 的比例控制语义,不是可选项:
    若只给单位方向(幅度恒为 1),读出层学到的策略会"永远全速冲",
    靠近靶时刹不住、在靶周围来回振荡,命中率反而接近 0。
    实测教训:仅用单位方向时,果蝇臂平均靶距 83.8px(比 PID 的 204.6px 更近)
    却一次都打不中 —— 典型的过冲振荡。

    归一化尺度取 `2 × speed`:距离 >= 2 个 action 步长时给满幅度,
    更近时线性减速,与 PID 的 P 项行为一致。
    """
    if not state:
        return None
    cross = state.get("crosshair")
    targets = state.get("targets")
    if cross is None or targets is None:
        return None
    cross = np.asarray(cross, dtype=np.float64).reshape(2)
    targets = np.asarray(targets, dtype=np.float64).reshape(-1, 2)
    if targets.size == 0 or speed <= 0:
        return None
    d = targets - cross
    dist = np.linalg.norm(d, axis=1)
    k = int(np.argmin(dist))
    r = float(dist[k])
    if r < 1e-9:
        return np.zeros(2, dtype=np.float32)
    # 比例增益:以 2 个 action 步长为满幅度的参考距离
    mag = float(np.clip(r / (2.0 * speed), 0.0, 1.0))
    return ((d[k] / r) * mag).astype(np.float32)


# ---------------------------------------------------------------- 冻结校验


def assert_connectome_frozen(before: ConnectomeArtifacts, after_path: str | Path) -> None:
    """校验连接组权重在训练前后**逐位不变**(CONTRACT 禁止事项 2)。

    这是防造假的关键检查:若有人偷偷改权重来提升成绩,这里会失败。
    """
    after = ConnectomeArtifacts.load(after_path)
    if before.n != after.n:
        raise AssertionError(f"连接组规模变化: {before.n} -> {after.n}")
    for name in ("W_exc", "W_inh"):
        a = getattr(before, name)
        b = getattr(after, name)
        if a.nnz != b.nnz:
            raise AssertionError(f"{name} 非零元数变化: {a.nnz} -> {b.nnz}")
        if not np.array_equal(a.indptr, b.indptr):
            raise AssertionError(f"{name} 行指针变化——权重被修改过")
        if not np.array_equal(a.indices, b.indices):
            raise AssertionError(f"{name} 列索引变化——权重被修改过")
        if not np.array_equal(a.data, b.data):
            raise AssertionError(f"{name} 权重值变化——违反 CONTRACT 禁止事项 2")
    if not np.array_equal(before.neuron_ids, after.neuron_ids):
        raise AssertionError("neuron_ids 变化——索引空间被破坏")


# ---------------------------------------------------------------- 拟合


def fit_readout(readout, train_sets: list[TrainingSet], ridge_lambda: float | None = None) -> dict:
    """用岭回归拟合读出层。返回训练诊断信息。"""
    if not train_sets:
        raise ValueError("没有训练集")

    X = np.concatenate([t.X for t in train_sets], axis=0).astype(np.float64)
    Y = np.concatenate([t.Y for t in train_sets], axis=0).astype(np.float64)

    lam = readout.cfg.ridge_lambda if ridge_lambda is None else ridge_lambda

    # 标准化特征(否则不同 DN 的量纲差异会让岭惩罚失衡)
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd_safe = np.where(sd < 1e-8, 1.0, sd)
    Xn = (X - mu) / sd_safe
    Xn = np.hstack([Xn, np.ones((Xn.shape[0], 1))])  # 偏置项

    n_feat = Xn.shape[1]
    A = Xn.T @ Xn + lam * np.eye(n_feat)
    A[-1, -1] -= lam  # 不惩罚偏置
    B = Xn.T @ Y
    W = np.linalg.solve(A, B)

    pred = Xn @ W
    resid = Y - pred
    ss_res = float((resid**2).sum())
    ss_tot = float(((Y - Y.mean(axis=0)) ** 2).sum())
    r2 = 1.0 - ss_res / max(ss_tot, 1e-12)
    rmse = float(np.sqrt((resid**2).mean()))

    diag = {
        "n_samples": int(X.shape[0]),
        "n_features": int(n_feat - 1),
        "ridge_lambda": float(lam),
        "r2": r2,
        "rmse": rmse,
        "train_seeds": [t.seed for t in train_sets],
        "feature_mean_abs": float(np.abs(mu).mean()),
        "feature_std_mean": float(sd.mean()),
        "n_dead_features": int((sd < 1e-8).sum()),
    }

    # 交给 Readout 自己持有参数(它负责推理)
    readout.set_linear_weights(W=W, mu=mu, sd=sd_safe)
    return diag


def save_training_diagnostics(path: str | Path, diag: dict) -> None:
    save_json(path, diag)
