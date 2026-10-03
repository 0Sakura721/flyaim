"""读出层(`Readout`):下行神经元(DN)放电率 -> 鼠标增量 [dx, dy]。

===============================================================================
1. 输入来源(契约 §2.3 + CONTRACT 第 1 节)
===============================================================================
**唯一合法的控制读出群是下行神经元(DN)**,MN 只作辅助:
    - `roles.descending` 非空 -> 特征 = DN 群;
    - DN 为空 -> 回退到 `roles.motor`(辅助路径,`describe()` 会标注);
    - `cfg.source == "dn+mn"` 时显式拼接两组(DN 优先)。
两组都为空时不猜、不用其它神经元(会违反"只允许 DN/MN"),`act()` 返回 0 并在
`describe()` 里置 `readout_source_fallback=True`。

===============================================================================
2. mode="fixed":手写固定映射(无学习)
===============================================================================
真实 DN 与"左右/上下"的对应关系未知(数据集也没有功能标注),因此这里采用
**数据驱动的固定分组**:把特征群按 (左右侧 × 索引两半) 拆成 **4 个互不相交的组**,
组间均值两两做差:

    设 A/B = 每一侧内部按特征向量中的**索引顺序**对半拆出的两组
       (A = 前半,B = 后半),L/R = 左右侧
    四组:G_LA, G_LB, G_RA, G_RB   (均值记为 m_LA, m_LB, m_RA, m_RB)
    dx = 0.5*(m_RA + m_RB) - 0.5*(m_LA + m_LB)      # 右 - 左
    dy = 0.5*(m_LA + m_RA) - 0.5*(m_LB + m_RB)      # 前半 - 后半

左右侧信息来自 neuron_index 的 `side` 列(感光细胞的 side 几乎全 NaN,但 DN 的
side 可能可用)。**只有当特征群中 L/R 标注覆盖率 >= 60% 时才使用侧向分组**;
否则退化为纯索引四分位:

    把特征群按索引顺序切成 4 个连续四分位 Q0..Q3
    dx = 0.5*(m_Q1 + m_Q3) - 0.5*(m_Q0 + m_Q2)
    dy = 0.5*(m_Q0 + m_Q2) - 0.5*(m_Q1 + m_Q3)

最后 `raw = (dx, dy) * gain / rate_norm_hz` 并裁剪到 [-1,1]。
**诚实声明:这个映射是任意的** —— 没有任何证据表明 DN 的索引顺序或 side 标注
与"向上/向下"的鼠标方向对应。它的作用是提供一个**无学习的对照基线**,
真正有意义的映射由 `mode="trained"` 学出来;`fixed` 模式不能当作"果蝇会瞄准"的证据。

===============================================================================
3. mode="trained":岭回归线性层(学习只发生在这里)
===============================================================================
**学习只发生在 Readout,绝不写回连接组权重**(CONTRACT 禁止事项 5.2;
`fit_readout.assert_connectome_frozen()` 会在训练前后逐位比对 W_exc/W_inh)。

前向(必须与 `flyaim/fit_readout.py` 的约定严格一致):

    z = (x - mu) / sd              # x = rates[feature_ids],(n_features,)
    z_aug = concat([z, 1.0])       # (n_features + 1,)
    y = z_aug @ W                  # W: (n_features + 1, 2) -> [dx, dy]

权重由 `set_linear_weights(W, mu, sd)` 注入(外部训练器或本类的 `fit()`),
`save/load` 用 `.npz` 持久化。输出再经低通滤波(`cfg.smoothing`)并裁剪到 [-1,1]。

训练/评估种子必须分离:`cfg.train_seeds` 与 `cfg.eval_seeds` 若相交则直接抛
ValueError(在 `__init__` 与 `fit` 都会检查)。
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from flyaim.config import ReadoutConfig

logger = logging.getLogger(__name__)

_ROLE_RELPATH = Path("flyaim") / "data" / "build" / "roles.json"
_INDEX_RELPATH = Path("flyaim") / "data" / "build" / "neuron_index.parquet"

_DEFAULT_RATE_NORM_HZ = 50.0
"""fixed 模式:DN 群均值之差达到 50 Hz 即映射到满偏(±1)。可用 cfg.rate_norm_hz 覆盖。"""

_MIN_SIDE_COVERAGE = 0.6
"""侧向分组所需的最低 L/R 标注覆盖率;低于此值退化为索引四分位。"""

_LEFT_LABELS = frozenset({"left", "l", "lhs", "lft", "leftside"})
_RIGHT_LABELS = frozenset({"right", "r", "rhs", "rgt", "rightside"})


class Readout:
    """DN(可选 +MN)放电率 -> 鼠标增量 [dx, dy]。"""

    def __init__(
        self,
        cfg: ReadoutConfig,
        roles: Any | None = None,
        neuron_index: Any | None = None,
        side: np.ndarray | None = None,
    ) -> None:
        """构造读出层。

        cfg: ReadoutConfig。
        roles: `flyaim.neuron_index.RoleSelection`。为空时尝试按
            `cfg.roles_path` -> `cfg.data_dir/roles.json` -> `flyaim/data/build/roles.json` 读取。
        neuron_index: `flyaim.neuron_index.NeuronIndex` 或 parquet 路径(用于取 side 列)。
        side: (n_features,) 特征群的左右标注(可选,优先级最高),取值 'left'/'right'/
            其它(未知)。
        """
        self.cfg = cfg
        if set(cfg.train_seeds) & set(cfg.eval_seeds):
            raise ValueError(
                f"训练/评估种子必须分离,当前交集={sorted(set(cfg.train_seeds) & set(cfg.eval_seeds))}"
            )
        self.mode = str(cfg.mode)
        if self.mode not in ("fixed", "trained"):
            raise ValueError(f"未知 mode: {self.mode}")
        self.smoothing = float(np.clip(cfg.smoothing, 0.0, 1.0))
        self.gain = float(cfg.gain)
        self.rate_norm_hz = float(getattr(cfg, "rate_norm_hz", _DEFAULT_RATE_NORM_HZ))

        if roles is None:
            roles = self._load_roles(cfg)
        self.roles = roles

        dn = self._ids_of(roles, "descending")
        mn = self._ids_of(roles, "motor")
        self.dn_ids, self.mn_ids = dn, mn
        self.readout_source_fallback = False
        src_cfg = str(getattr(cfg, "source", "dn"))

        if src_cfg == "dn+mn" and dn.size and mn.size:
            ids = np.union1d(dn, mn)
            self.source_kind = "dn+mn"
        elif dn.size:
            ids = dn
            self.source_kind = "dn"
            if src_cfg == "dn+mn":
                logger.warning("Readout: source='dn+mn' 但 MN 为空,退化为纯 DN")
        elif mn.size:
            ids = mn
            self.source_kind = "mn(fallback)"
            self.readout_source_fallback = True
            logger.warning("Readout: DN 群为空,回退到 MN 群(辅助读出);报告须标注")
        else:
            ids = np.empty(0, dtype=np.int64)
            self.source_kind = "none"
            self.readout_source_fallback = True
            logger.warning(
                "Readout: DN 与 MN 均为空 -> 无合法读出源,act() 恒返回 0。"
                "报告必须标注该降级,不能用其它神经元冒充控制读出群"
            )
        self.feature_ids = np.asarray(ids, dtype=np.int64).reshape(-1)
        self.feature_index = {int(g): k for k, g in enumerate(self.feature_ids)}
        self.n_features = int(self.feature_ids.size)

        # ---------------------------------------------------------- 固定模式分组
        side_arr, self.side_source = self._resolve_side(side, neuron_index, cfg)
        self._build_groups(side_arr)

        # ---------------------------------------------------------- 训练模式状态
        self._W: np.ndarray | None = None
        self._mu: np.ndarray | None = None
        self._sd: np.ndarray | None = None
        self._prev_out = np.zeros(2, dtype=np.float32)
        self.last_raw = np.zeros(2, dtype=np.float32)
        self.train_info: dict[str, Any] = {}

        if self.mode == "trained" and cfg.weights_path:
            p = Path(cfg.weights_path)
            if p.exists():
                self.load(p)
            else:
                logger.warning("Readout: weights_path=%s 不存在,trained 模式尚无权重", p)

    # ================================================================== 构造辅助

    @staticmethod
    def _load_roles(cfg: ReadoutConfig) -> Any | None:
        from flyaim.io import load_roles

        cand: list[Path] = []
        rp = getattr(cfg, "roles_path", None)
        if rp:
            cand.append(Path(str(rp)))
        dd = getattr(cfg, "data_dir", None)
        if dd:
            cand.append(Path(str(dd)) / "roles.json")
        cand.append(_ROLE_RELPATH)
        cand.append(Path(__file__).resolve().parents[2] / _ROLE_RELPATH)
        for p in cand:
            try:
                if p.exists():
                    return load_roles(p)
            except Exception as exc:
                logger.warning("Readout: 读取 roles(%s) 失败: %s", p, exc)
        logger.warning("Readout: 未找到 roles.json,读出源为空")
        return None

    @staticmethod
    def _ids_of(roles: Any | None, name: str) -> np.ndarray:
        if roles is None:
            return np.empty(0, dtype=np.int64)
        v = np.asarray(getattr(roles, name, np.empty(0)), dtype=np.int64).reshape(-1)
        return np.unique(v[v >= 0])

    def _resolve_side(
        self, side: np.ndarray | None, neuron_index: Any, cfg: ReadoutConfig
    ) -> tuple[np.ndarray | None, str]:
        """解析特征群的左右标注;返回 (labels 或 None, 来源说明)。"""
        n = self.n_features
        if n == 0:
            return None, "none"
        raw: np.ndarray | None = None
        src = "none"
        if side is not None:
            a = np.asarray(side, dtype=object).reshape(-1)
            if a.size == n:
                raw, src = a, "kwarg"
            else:
                logger.warning("Readout: side 长度 %d != n_features %d,忽略", a.size, n)
        if raw is None:
            cfg_side = getattr(cfg, "side", None)
            if cfg_side is not None:
                a = np.asarray(cfg_side, dtype=object).reshape(-1)
                if a.size == n:
                    raw, src = a, "cfg.side"
        if raw is None:
            ni = neuron_index if neuron_index is not None else getattr(cfg, "neuron_index", None)
            if ni is None:
                for p in (
                    Path(str(getattr(cfg, "data_dir", ""))) / "neuron_index.parquet"
                    if getattr(cfg, "data_dir", None)
                    else None,
                    _INDEX_RELPATH,
                    Path(__file__).resolve().parents[2] / _INDEX_RELPATH,
                ):
                    if p is None or not Path(p).exists():
                        continue
                    try:
                        from flyaim.neuron_index import NeuronIndex

                        ni = NeuronIndex.load(p)
                        break
                    except Exception as exc:
                        logger.warning("Readout: 加载 neuron_index 失败(%s): %s", p, exc)
            df = getattr(ni, "df", None)
            if df is not None and "side" in df.columns and df.shape[0] > int(self.feature_ids.max()):
                try:  # type: ignore[union-attr]
                    raw = df["side"].astype(str).to_numpy()[self.feature_ids]
                    src = "neuron_index.side"
                except Exception as exc:
                    logger.warning("Readout: 读取 side 列失败: %s", exc)
        if raw is None:
            return None, "none"
        lab = np.array([self._norm_side(v) for v in raw], dtype=object)
        cov = float(np.mean((lab == "left") | (lab == "right")))
        if cov < _MIN_SIDE_COVERAGE:
            logger.warning(
                "Readout: 左右标注覆盖率 %.0f%% < %.0f%%,退化为索引四分位分组(来源=%s)",
                cov * 100,
                _MIN_SIDE_COVERAGE * 100,
                src,
            )
            return None, f"{src}(coverage={cov:.2f}->index_quartiles)"
        return lab, src

    @staticmethod
    def _norm_side(v: Any) -> str:
        s = str(v).strip().lower()
        if s in _LEFT_LABELS:
            return "left"
        if s in _RIGHT_LABELS:
            return "right"
        return "unknown"

    def _build_groups(self, side: np.ndarray | None) -> None:
        """构造 fixed 模式的 4 个互不相交组(特征空间位置索引)。"""
        n = self.n_features
        self.group_names: list[str] = []
        self.groups: dict[str, np.ndarray] = {}
        self.grouping = "index_quartiles"
        if n == 0:
            return
        pos = np.arange(n, dtype=np.int64)

        if side is not None:
            left = pos[side == "left"]
            right = pos[side == "right"]
            if left.size > 0 and right.size > 0:
                for name, sel in (("L", left), ("R", right)):
                    h = sel.size // 2
                    a, b = sel[: max(h, 0)], sel[max(h, 0) :]
                    if a.size:
                        self.groups[f"{name}A"] = a
                    if b.size:
                        self.groups[f"{name}B"] = b
                self.grouping = "side_x_half"
                self.group_names = list(self.groups.keys())
                return

        # 退化:4 个连续四分位
        self.groups = {}
        for qi, sel in enumerate(np.array_split(pos, 4)):
            if sel.size:
                self.groups[f"Q{qi}"] = sel
        self.group_names = list(self.groups.keys())

    # ================================================================== 前向

    def act(self, brain: Any) -> np.ndarray:
        """返回 (2,) float32 鼠标增量 [dx, dy],已归一化到约 [-1, 1]。

        只读取 brain.rates(全局索引空间);不读写连接组权重。
        """
        if self.n_features == 0:
            return np.zeros(2, dtype=np.float32)
        x = self.features(brain)
        if self.mode == "trained":
            raw = self._forward_trained(x)
        else:
            raw = self._forward_fixed(x)
        self.last_raw = raw.astype(np.float32)
        if self.smoothing > 0.0:
            y = self.smoothing * self._prev_out + (1.0 - self.smoothing) * raw
        else:
            y = raw
        y = np.clip(y, -1.0, 1.0).astype(np.float32)
        self._prev_out = y
        return y

    def features(self, brain: Any) -> np.ndarray:
        """取特征向量 = rates[feature_ids](DN 群;合约只允许 DN/MN)。"""
        rates = np.asarray(brain.rates, dtype=np.float32)
        if rates.size <= int(self.feature_ids.max()):
            raise ValueError(
                f"brain.rates 长度 {rates.size} 小于特征最大索引 {int(self.feature_ids.max())}"
            )
        return rates[self.feature_ids].astype(np.float32)

    def _forward_fixed(self, x: np.ndarray) -> np.ndarray:
        m = {k: float(np.mean(x[sel])) for k, sel in self.groups.items() if sel.size}
        if self.grouping == "side_x_half":
            la = m.get("LA")
            lb = m.get("LB")
            ra = m.get("RA")
            rb = m.get("RB")
            l_all = np.mean([v for v in (la, lb) if v is not None]) if (la is not None or lb is not None) else 0.0
            r_all = np.mean([v for v in (ra, rb) if v is not None]) if (ra is not None or rb is not None) else 0.0
            a_all = np.mean([v for v in (la, ra) if v is not None]) if (la is not None or ra is not None) else 0.0
            b_all = np.mean([v for v in (lb, rb) if v is not None]) if (lb is not None or rb is not None) else 0.0
            dx, dy = r_all - l_all, a_all - b_all
        else:  # 索引四分位
            q = [m.get(f"Q{i}", 0.0) for i in range(4)]
            dx = 0.5 * (q[1] + q[3]) - 0.5 * (q[0] + q[2])
            dy = 0.5 * (q[0] + q[2]) - 0.5 * (q[1] + q[3])
        scale = self.gain / max(self.rate_norm_hz, 1e-6)
        return np.array([dx * scale, dy * scale], dtype=np.float32)

    def _forward_trained(self, x: np.ndarray) -> np.ndarray:
        if self._W is None or self._mu is None or self._sd is None:
            raise RuntimeError(
                "trained 模式尚无权重:请先 fit() 或 set_linear_weights()/load()"
            )
        z = (x.astype(np.float64) - self._mu) / self._sd
        z_aug = np.concatenate([z, np.ones(1, dtype=np.float64)])
        y = z_aug @ self._W
        return (y * self.gain).astype(np.float32)

    def reset(self) -> None:
        """清空低通滤波状态(每个 episode 开始时调用)。"""
        self._prev_out = np.zeros(2, dtype=np.float32)
        self.last_raw = np.zeros(2, dtype=np.float32)

    # ================================================================== 训练

    def set_linear_weights(self, W: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> None:
        """由外部训练器注入线性层参数。

        形状约定(必须与 `act()` 的前向严格一致):
            W  : (n_features + 1, 2) float64/float32
                 W[:n_features, :] 对应标准化后的特征,W[n_features, :] 是**偏置行**;
                 输出列顺序为 [dx, dy]。
            mu : (n_features,) 特征均值。
            sd : (n_features,) 特征标准差(**调用方保证已做零值保护**,即 sd>0)。
        前向:x -> z=(x-mu)/sd -> z_aug=concat([z, 1]) -> y = z_aug @ W。
        """
        W = np.asarray(W, dtype=np.float64)
        mu = np.asarray(mu, dtype=np.float64).reshape(-1)
        sd = np.asarray(sd, dtype=np.float64).reshape(-1)
        d = self.n_features
        if W.shape != (d + 1, 2):
            raise ValueError(f"W 形状应为 ({d + 1}, 2),得到 {W.shape}")
        if mu.size != d or sd.size != d:
            raise ValueError(f"mu/sd 长度应为 ({d},),得到 {mu.shape}/{sd.shape}")
        if np.any(sd <= 0) or not np.all(np.isfinite(sd)):
            raise ValueError("sd 必须为有限正值(零方差特征请由调用方置 1)")
        self._W, self._mu, self._sd = W.copy(), mu.copy(), sd.copy()

    def get_linear_weights(self) -> dict:
        """返回 {'W':(d+1,2), 'mu':(d,), 'sd':(d,)},供 save/load 与冻结校验。"""
        if self._W is None:
            raise RuntimeError("尚无线性权重(未 fit / 未注入)")
        return {"W": self._W.copy(), "mu": self._mu.copy(), "sd": self._sd.copy()}

    @property
    def has_weights(self) -> bool:
        return self._W is not None

    @property
    def n_dn(self) -> int:
        """下行神经元(DN)群规模。"""
        return int(self.dn_ids.size)

    @property
    def n_mn(self) -> int:
        """运动神经元(MN)群规模。"""
        return int(self.mn_ids.size)

    def fit(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        sample_weight: np.ndarray | None = None,
    ) -> "Readout":
        """岭回归拟合线性读出。

        X: (n_samples, n_features) DN 放电率。
        Y: (n_samples, 2) 期望鼠标增量(≈[-1,1])。
        sample_weight: (n_samples,) 非负权重(可选)。

        回归在**标准化特征 + 偏置行**上求解:
            A = Z^T diag(w) Z + lambda * diag([1]*d + [0])   # 偏置不正则化
            beta = A^{-1} Z^T diag(w) Y                       # (d+1, 2)
        即 `set_linear_weights(beta, mu, sd)`,与 fit_readout.py 的约定一致。
        """
        X = np.asarray(X, dtype=np.float64)
        Y = np.asarray(Y, dtype=np.float64)
        if X.ndim != 2 or Y.ndim != 2 or Y.shape[1] != 2:
            raise ValueError(f"X/Y 形状应为 (n,d)/(n,2),得到 {X.shape}/{Y.shape}")
        if X.shape[0] != Y.shape[0]:
            raise ValueError(f"X/Y 样本数不一致: {X.shape[0]} vs {Y.shape[0]}")
        d = self.n_features
        if X.shape[1] != d:
            raise ValueError(f"X 特征维 {X.shape[1]} != n_features {d}")
        if set(self.cfg.train_seeds) & set(self.cfg.eval_seeds):
            raise ValueError("训练/评估种子必须分离")
        if X.shape[0] == 0:
            raise ValueError("X 为空")

        w = np.ones(X.shape[0], dtype=np.float64) if sample_weight is None else np.asarray(
            sample_weight, dtype=np.float64
        ).reshape(-1)
        if w.size != X.shape[0]:
            raise ValueError(f"sample_weight 长度 {w.size} != 样本数 {X.shape[0]}")
        if np.any(w < 0):
            raise ValueError("sample_weight 不能为负")
        wsum = float(w.sum())
        if wsum <= 0:
            raise ValueError("sample_weight 之和必须为正")

        mu = (w[:, None] * X).sum(axis=0) / wsum
        var = (w[:, None] * (X - mu) ** 2).sum(axis=0) / wsum
        sd = np.sqrt(np.maximum(var, 0.0))
        sd = np.where(sd > 1e-12, sd, 1.0)  # 零方差保护(契约要求)
        Z = (X - mu) / sd
        Za = np.concatenate([Z, np.ones((X.shape[0], 1))], axis=1)

        lam = float(self.cfg.ridge_lambda)
        if lam <= 0:
            logger.warning("Readout: ridge_lambda=%.3g <= 0,退化为最小二乘", lam)
            beta, *_ = np.linalg.lstsq(Za * np.sqrt(w)[:, None], Y * np.sqrt(w)[:, None], rcond=None)
        else:
            D = np.ones(d + 1, dtype=np.float64)
            D[d] = 0.0  # 偏置不正则化
            A = Za.T @ (w[:, None] * Za) + lam * np.diag(D)
            rhs = Za.T @ (w[:, None] * Y)
            try:
                beta = np.linalg.solve(A, rhs)
            except np.linalg.LinAlgError:
                logger.warning("Readout: 岭回归矩阵奇异,改用 lstsq")
                beta, *_ = np.linalg.lstsq(A, rhs, rcond=None)

        self.set_linear_weights(beta, mu, sd)
        info = {
            "n_samples": int(X.shape[0]),
            "n_features": d,
            "ridge_lambda": lam,
            "train_seeds": list(self.cfg.train_seeds),
            "eval_seeds": list(self.cfg.eval_seeds),
        }
        info.update(self.evaluate(X, Y, sample_weight=w, prefix="train_"))
        self.train_info = info
        logger.info("Readout: 拟合完成 train_r2=%s", info.get("train_r2"))
        return self

    def evaluate(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        sample_weight: np.ndarray | None = None,
        prefix: str = "",
    ) -> dict:
        """在该数据集上评估(用当前权重);返回 r2(逐轴 + 整体)与 rmse。"""
        X = np.asarray(X, dtype=np.float64)
        Y = np.asarray(Y, dtype=np.float64)
        if self._W is None:
            raise RuntimeError("尚无权重")
        Z = (X - self._mu) / self._sd
        P = np.concatenate([Z, np.ones((X.shape[0], 1))], axis=1) @ self._W
        w = np.ones(X.shape[0]) if sample_weight is None else np.asarray(sample_weight, float)
        wsum = float(w.sum())
        wmean = (w[:, None] * Y).sum(0) / wsum
        ss_res = (w[:, None] * (Y - P) ** 2).sum(0)
        ss_tot = (w[:, None] * (Y - wmean) ** 2).sum(0)
        r2 = 1.0 - ss_res / np.maximum(ss_tot, 1e-12)
        rmse = np.sqrt(ss_res / wsum)
        return {
            f"{prefix}r2_dx": float(r2[0]),
            f"{prefix}r2_dy": float(r2[1]),
            f"{prefix}r2_all": float(1.0 - ss_res.sum() / max(ss_tot.sum(), 1e-12)),
            f"{prefix}rmse": float(np.sqrt(np.mean(rmse**2))),
            f"{prefix}n": int(X.shape[0]),
        }

    # ================================================================== 持久化

    def save(self, weights_path: str | Path | None = None) -> Path:
        """保存权重到 .npz(含 W / mu / sd 与元数据)。"""
        p = Path(weights_path) if weights_path is not None else (
            Path(str(self.cfg.weights_path)) if self.cfg.weights_path else None
        )
        if p is None:
            raise ValueError("未提供 weights_path(cfg.weights_path 也为空)")
        if self._W is None:
            raise RuntimeError("尚无权重可保存")
        p.parent.mkdir(parents=True, exist_ok=True)
        meta = {
            "mode": self.mode,
            "source_kind": self.source_kind,
            "n_features": self.n_features,
            "feature_ids_head": [int(v) for v in self.feature_ids[:16]],
            "grouping": self.grouping,
            "side_source": self.side_source,
            "train_info": {k: v for k, v in self.train_info.items()},
        }
        np.savez(
            p,
            W=self._W,
            mu=self._mu,
            sd=self._sd,
            feature_ids=self.feature_ids.astype(np.int64),
            meta=np.asarray(json.dumps(meta, ensure_ascii=False)),
        )
        return p

    def load(self, weights_path: str | Path | None = None) -> "Readout":
        """从 .npz 载入权重(形状校验:与当前特征维度一致)。"""
        p = Path(weights_path) if weights_path is not None else (
            Path(str(self.cfg.weights_path)) if self.cfg.weights_path else None
        )
        if p is None:
            raise ValueError("未提供 weights_path(cfg.weights_path 也为空)")
        z = np.load(p, allow_pickle=False)
        W, mu, sd = z["W"], z["mu"], z["sd"]
        d = self.n_features
        if W.shape != (d + 1, 2):
            raise ValueError(f"权重文件 W 形状 {W.shape} 与当前 n_features={d} 不匹配")
        self.set_linear_weights(W, mu, sd)
        if "meta" in z.files:
            try:
                self.train_info = json.loads(str(z["meta"]))
            except Exception:
                pass
        logger.info("Readout: 已载入权重 %s", p)
        return self

    # ================================================================== 诊断

    def describe(self) -> dict:
        """读出层配置与分组明细(供 manifest / 报告)。"""
        return {
            "mode": self.mode,
            "source_kind": self.source_kind,
            "readout_source_fallback": bool(self.readout_source_fallback),
            "n_dn": int(self.dn_ids.size),
            "n_mn": int(self.mn_ids.size),
            "n_features": self.n_features,
            "dn_head": [int(v) for v in self.feature_ids[:8]],
            "grouping": self.grouping,
            "side_source": self.side_source,
            "groups": {k: int(v.size) for k, v in self.groups.items()},
            "gain": self.gain,
            "rate_norm_hz": self.rate_norm_hz,
            "smoothing": self.smoothing,
            "has_weights": bool(self.has_weights),
            "ridge_lambda": float(self.cfg.ridge_lambda),
            "train_seeds": list(self.cfg.train_seeds),
            "eval_seeds": list(self.cfg.eval_seeds),
            "fixed_mapping_is_arbitrary": True,
            "note": (
                "fixed 模式的分组是任意的数据驱动基线(DN 的功能方向未知);"
                "学习只发生在 Readout,连接组权重从不被修改"
            ),
        }

    def __repr__(self) -> str:  # pragma: no cover - 诊断用
        return (
            f"Readout(mode={self.mode}, source={self.source_kind}, n_features={self.n_features}, "
            f"grouping={self.grouping}, has_weights={self.has_weights})"
        )
