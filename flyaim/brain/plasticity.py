"""三因子可塑性外挂(CONTRACT_PLASTIC.md 附录 P,预注册)。

===============================================================================
设计
===============================================================================
可塑性**不重写引擎**:作为外挂层,每帧对引擎有效合成矩阵 W(GPU CSR)
的 data 数组做一次原地更新。引擎的 SpMV 每步都读 W.data,所以下一帧
立即用新权重 —— 零拷贝、零引擎改动。

规则(P2 修订 v2,预注册):

    e_ij ← λ_e · e_ij + post_i · (pre_j − θ_pre)
    W_ij ← clip(W_ij + η · m · e_ij, 极性边界)

    post_i = s[rows](行=突触后),pre_j = s[cols](列=突触前),
    s 为本帧最后一步的脉冲向量(0/1,引擎自带)。
    极性边界:兴奋项 [0, cap_e],抑制项 [−cap_i, 0],cap = 2×初始极值,
    极性归属以首次调用时的符号快照为准(之后冻结,过零不改变归属)。

η 标定(P2 v2 程序):η=1 空跑 probe_frames 帧(m=1,真实闭环),
记录每帧行漂移 |Σ_j ΔW_ij| 的运行最大值 D;冻结
    η = 0.25 × median(|W.data|初) / D
三因子中的调制因子 m 由训练器提供(oracle 距离或内部新颖度,见 P3)。

保存:训练后的有效权重反演回 artifact 尺度(indeg 归一化 × we_eff 的
逆映射),落成标准 npz,供普通引擎以 weights_override 复用。
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


class PlasticOverlayGPU:
    """GPU 引擎的可塑性外挂。所有数组驻留显存,更新零同步。"""

    def __init__(self, brain, cfg, artifact_path: str | Path) -> None:
        v = brain.plastic_view()
        self.cp = v["cp"]
        self.data = v["data"]          # cupy float32 (nnz,) —— 引擎 W.data 本体
        self.cols = v["cols"]
        self.rows = v["rows"]
        self.is_inh = v["is_inh"]      # 极性快照(冻结)
        self.nnz = v["nnz"]
        self.n = v["n"]
        self.we_eff = float(v["we_eff"])
        self.wi_eff = float(v["wi_eff"])
        self.s = v["s"]               # 引擎脉冲向量(cupy 引用)
        self.cfg = cfg

        cap = float(cfg.cap_multiple)
        d0 = self.data
        pos_max = float(self.cp.max(d0[d0 > 0])) if bool((d0 > 0).any()) else 1.0
        neg_min = float(self.cp.min(d0[d0 < 0])) if bool((d0 < 0).any()) else -1.0
        self.hi = self.cp.where(self.is_inh, self.cp.float32(0.0),
                                self.cp.float32(cap * pos_max))
        self.lo = self.cp.where(self.is_inh,
                                self.cp.float32(cap * neg_min), self.cp.float32(0.0))
        self.e = self.cp.zeros(self.nnz, dtype=self.cp.float32)
        self.eta = float(cfg.eta)
        self._w_typ = float(self.cp.median(self.cp.abs(d0)))
        self._probe_drift_rows = None   # 逐行全期最大漂移(探针期累计)
        self.updates_done = 0
        self.last_abs_dW_mean = 0.0
        self._artifact_path = Path(artifact_path)
        logger.info("PlasticOverlay: nnz=%d w_typ=%.4g cap_e=%.3g cap_i=%.3g",
                    self.nnz, self._w_typ, cap * pos_max, abs(cap * neg_min))

    # -- η 标定(P2 v2 程序) --------------------------------------------

    def probe_frame(self, m: float = 1.0) -> float:
        """η=1 的探针帧(P2 v3):率基资格迹 + 逐行全期最大漂移累计。"""
        cp = self.cp
        pre = self._rate_pre()          # cupy (nnz,) = rate[cols]/100 − θ
        post = self._rate_post()        # cupy (nnz,) = rate[rows]/100
        self.e = self.cfg.lambda_e * self.e + post * pre
        dW = m * self.e
        if self._probe_drift_rows is None:
            self._probe_drift_rows = cp.zeros(self.n, dtype=cp.float32)
        import cupyx

        cpyx = cupyx
        stepdrift = cp.zeros(self.n, dtype=cp.float32)
        cpyx.scatter_add(stepdrift, self.rows, dW)
        self._probe_drift_rows = cp.maximum(self._probe_drift_rows,
                                            cp.abs(stepdrift))
        return float(cp.max(cp.abs(stepdrift)))

    def _rate_pre(self):
        cp = self.cp
        r = cp.asarray(self._rates)     # 由训练器每帧 set_rates 上传
        return r[self.cols] * self.cp.float32(0.01) - self.cp.float32(self.cfg.theta_pre)

    def _rate_post(self):
        cp = self.cp
        r = cp.asarray(self._rates)
        return r[self.rows] * self.cp.float32(0.01)

    def set_rates(self, rates: np.ndarray) -> None:
        """训练器每帧传入 brain.rates(numpy,反正要同步做读出)。"""
        self._rates = rates

    def freeze_eta(self) -> float:
        """按预注册程序冻结 η = 0.25 × w_typ / D95(P2 v3)。"""
        cp = self.cp
        d = self._probe_drift_rows
        pos = d[d > 0]
        D = float(cp.percentile(pos, 95)) if pos.size else 1e-12
        D = max(D, 1e-12)
        self.eta = 0.25 * self._w_typ / D
        self.e[:] = 0.0  # 探针期的资格迹作废,训练从零开始
        self._probe_drift_rows = None
        logger.info("PlasticOverlay: η 冻结 = %.4g (w_typ=%.4g, D95=%.4g)",
                    self.eta, self._w_typ, D)
        return self.eta

    # -- 训练更新 ----------------------------------------------------------

    def update(self, m: float) -> None:
        """每帧一次(或每 plastic_every 帧,由训练器控制)。率基资格迹(P2 v3)。"""
        cp = self.cp
        pre = self._rate_pre()
        post = self._rate_post()
        self.e = self.cfg.lambda_e * self.e + post * pre
        dW = self.eta * self.cp.float32(m) * self.e
        newd = self.cp.clip(self.data + dW, self.lo, self.hi)
        self.data[...] = newd  # 原地写引擎 W.data
        self.updates_done += 1
        # 快速失败口径(P2 v3):活跃边(e≠0)上的平均 |ΔW|,
        # 全 nnz 均值会被海量零资格迹边稀释 3 个数量级,失去意义。
        act = self.cp.abs(self.e) > 0
        n_act = int(cp.count_nonzero(act))
        self.last_active_edges = n_act
        self.last_abs_dW_active = (float(cp.mean(cp.abs(dW[act])))
                                   if n_act else 0.0)  # 会同步

    # -- 保存(反演归一化与尺度,回 artifact 口径) -------------------------

    def save_artifacts(self, out_npz: str | Path) -> Path:
        """有效权重 → artifact npz(indeg 口径),供 weights_override 复用。

        反演:data_eff = we_eff × We_art(引擎 weight_norm="none",
        norm npz 已预归一化)→ We_art = data_eff / we_eff。抑制侧同理。
        """
        from flyaim.io import ConnectomeArtifacts

        cp = self.cp
        d = cp.asnumpy(self.data)
        rows = cp.asnumpy(self.rows)
        is_inh = cp.asnumpy(self.is_inh)

        art = ConnectomeArtifacts.load(self._artifact_path)
        # npz 是预归一化口径(indeg 已烘进权重,引擎 weight_norm="none"),
        # 反演只需除掉 w_scale×dt/tau 的尺度,无行逆变换。
        exc = np.where(is_inh, 0.0, d) / self.we_eff
        inh = np.where(is_inh, -d, 0.0) / self.wi_eff
        # 合成矩阵的模式 = W_exc ∪ W_inh(按边不相交划分,每个 (i,j) 恰属一侧)。
        # 用 COO 重建两个源矩阵(另一侧的项写显式 0,无害)。
        import scipy.sparse as sp

        cols = cp.asnumpy(self.cols)
        W_exc_new = sp.csr_matrix(
            (exc.astype(np.float32), (rows.astype(np.int64), cols.astype(np.int64))),
            shape=(self.n, self.n), dtype=np.float32)
        W_inh_new = sp.csr_matrix(
            (inh.astype(np.float32), (rows.astype(np.int64), cols.astype(np.int64))),
            shape=(self.n, self.n), dtype=np.float32)
        W_exc_new.sort_indices()
        W_inh_new.sort_indices()
        out = ConnectomeArtifacts(W_exc=W_exc_new, W_inh=W_inh_new,
                                  neuron_ids=art.neuron_ids,
                                  soma_pos=art.soma_pos)
        out_p = Path(out_npz)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out.save(out_p)
        logger.info("PlasticOverlay: 训练权重已保存 %s(边界内 clip 若干次)", out_p)
        return out_p
