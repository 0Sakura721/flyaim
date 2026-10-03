"""LIF 稀疏连接组仿真引擎(`Connectome`)。

===============================================================================
1. 模型
===============================================================================
每个神经元一个漏积分放电(LIF)单元,离散时间步进(默认 dt = 1 ms):

    I_syn(t)  = w_exc * (W_exc @ s(t-1)) - w_inh * (W_inh @ s(t-1))
    I_ext(t)  = in_gain * (in_exc - in_inh)          # 只加到 input_neuron_ids
    v(t+1)    = v(t) + (dt/tau_m) * ( -(v(t) - v_rest) + I_syn(t) + I_ext(t) )
    s(t+1)    = 1  iff  v(t+1) >= v_thresh  且  不在不应期
    v         = v_reset (发放神经元);不应期内 v 保持并被钳住

`W_exc[i, j]` 是 j -> i 的投射强度(行=突触后,列=突触前),与
`flyaim.io.ConnectomeArtifacts` 的约定一致。两张矩阵都用 scipy CSR,**从不 densify**。

**`step()` 只推进一个 dt。** 集成层负责"每帧调用 steps_per_frame 次"
(见 BRAIN/契约补充:`cfg.steps_per_frame` 只作为配置被集成层读取,引擎内部不循环)。

读出率 `rates`:脉冲经指数滑动平均(时间常数 `rate_tau_ms`,默认 50 ms)转成 Hz:

    rate <- rate + (1000/dt * s - rate) * min(1, dt / rate_tau_ms)

===============================================================================
2. 子网模式(subnet_size)与"全局索引空间"契约
===============================================================================
`cfg.subnet_size` 非 None 时,只仿真"子网"以提速:
    - **必含**全部角色神经元(visual_input / descending / motor)与外部输入群;
    - 其余名额用 `subnet_seed` 随机补齐(可复现);
    - 若必含集合本身已超过 subnet_size,则子网 = 必含集合(并在 `stats()` 里标注)。
子网被**重排**为紧凑索引 [0, m),但对外暴露的 `rates` / `spikes` **永远是全局
索引空间、长度 N 的数组**(未仿真位置为 0),符合 CONTRACT.md 第 1 节。
需要紧凑空间时用 `rates_compact` / `spikes_compact`。

子网取的是**诱导子图**(两端都在子网内的边),因此每个神经元的入度会按
子网比例下降。若需要保持期望输入驱动,可设 `cfg.subnet_renormalize = True`
(按 N/m 放大子网权重);默认 False(忠实于原接线图,不做任何缩放)。
注意这是**只读**的工程缩放:本引擎绝不原地修改连接组权重
(见 `fit_readout.assert_connectome_frozen` 的冻结校验要求)。

===============================================================================
3. 实测性能(i5-9300H 4C8T + 15.8 GB,scipy 1.18.1,float32;合成连接组
   N=166,700 / 密度 0.546% / 151,863,700 条边;由 selftest_synth.py 第 3 节输出)
===============================================================================
  配置            m         nnz         矩阵内存  单步 p50   33 步/帧  FPS 上限  发放率
  全量 166.7k   166,700  151,863,700   1216 MB   3.085 ms   101.8 ms    9.8    0.27%
  子网 8,166      8,166      362,997      3 MB   0.526 ms    17.4 ms   57.6    7.35%
  子网 12,000    12,000      784,470      6 MB   0.815 ms    26.9 ms   37.2    5.00%
  子网 20,000    20,000    2,180,204     18 MB   0.591 ms    19.5 ms   51.3    3.00%
  子网 30,000    30,000    4,907,447     39 MB   0.734 ms    24.2 ms   41.3    2.00%
  子网 50,000    50,000   13,628,200    109 MB   0.865 ms    28.5 ms   35.0    1.20%

**结论**
  - **全量 166,700 支撑不了 30 FPS**(3.1 ms/步 -> 102 ms/帧 -> 约 10 FPS),
    而且这一档对活跃度极其敏感:实测在发放率 1% 时单步劣化到 44 ms(见下);
  - **8k-50k 子网都能支撑 30 FPS**(0.5-0.9 ms/步 -> 17-29 ms/帧 -> 35-58 FPS),
    推荐 20k 子网:只占 18 MB,比"仅角色"的 8,166 多保留 2.4 倍递归结构,
    仍留 1.7 倍帧预算余量;
  - 单步耗时 ≈ **0.5 ms 固定开销**(每步约 20 次 numpy/scipy 调用)
    + (发放神经元数 x 出度) 的稀疏传播。因此子网缩到 8k 以下几乎不再变快
    (固定开销占主导),而放大到 50k 也仍够 30 FPS;
  - 活跃度是最大变量:同样的全量网络,发放率 0.27% -> 3.1 ms/步,
    发放率 ~1% -> 44 ms/步(spike-driven 代价 ∝ 活跃列的非零数)。
    真实数据上的活跃度以 selftest_real.py 的实测为准。
  - 内存:子网 20k = 矩阵 18 MB + 状态 0.3 MB;全量 = 矩阵 1216 MB,
    **外加首次稀疏传播时惰性构建的 CSC 视图(同样 ~1216 MB)**,合计约 2.4 GB。
    子网模式没有这个问题。
  - 计时口径:预热 >= 30 步后取 p50(预热步会包含一次性的 `W.tocsc()` 构建,
    约 0.3 s/13.65M 非零)。本机同时运行 A/C 两条工作线,数值偏保守。

===============================================================================
5. 外部驱动的量纲与 input_gain 标定(重要,标定结论见 selftest_real.py 第 6/7 节)
===============================================================================
`in_exc` / `in_inh` 的量纲是 **Hz 等效驱动**(RetinaConfig.max_drive_hz),不是突触
计数,因此**不能**复用 `weight_scale_exc` 这个突触电导尺度。本引擎的换算关系是:

    I_ext = input_gain * (dt/tau) * drive          # 折进 dt/tau 后加进 v
    v_ss  = I_ext / (dt/tau) = input_gain * drive  # 稳态膜电位

即 **input_gain = 每 1 Hz 驱动对应的稳态膜电位**,发放门槛是 drive >= v_thresh/input_gain:

    input_gain = 1.5   -> 0.67 Hz 就饱和发放(感光层退化为二值探测器,且全局 140 Hz)
    input_gain = 0.2   -> 5 Hz 起发放(有梯度)
    input_gain = 0.02  -> 50 Hz 起发放(只有驱动峰值区发)
    input_gain = 0.01  -> 100 Hz 起

**递归权重的量纲则完全不同**:`v_ss = weight_scale * (W_exc @ s)`,即
`weight_scale >= v_thresh / (f * S_i)`,S_i = 该神经元入权重总和,f = 同时活跃的
输入比例。实测(A-A 子图)S_exc 中位数 266、S_inh 中位数 69,因此默认
weight_scale_exc=0.02 要求 f>=18.8% 的输入同时发放才能发放 —— 这就是
"默认参数下整个网络一步都不发"的根本原因。两层标度必须分别标定。

**实测结论(真实 MaleCNS A-A 子图,N=166,700,25,582,938 条边):**
  - weight_scale_exc <= 0.10:全网络静默,DN = 0 Hz(引擎很快,~0.4-1 ms/步);
  - weight_scale_exc = 0.12(ws_inh 同量级)+ input_gain 1.5:网络被点燃,
    逐层均值率 ol_sensory 10 Hz / ol_intrinsic 16 Hz / visual_projection 44 Hz /
    descending 98 Hz(87% 非零)/ vnc_motor 98 Hz,**靶位可分辨 80.7 Hz**,
    但每步 4.8% 神经元发放 -> p50 15 ms/步(33 步/帧 ≈ 2 FPS);
  - 两者之间**没有稳定的稀疏活动区**(双稳态):抑制侧入权重和只有兴奋侧的 27%,
    且确定性 LIF 无适应,活动一旦越过传播阈值就自持到不应期上限。
    膜噪声(`noise_sigma`,默认 0)实测无法打开稀疏区。

**行归一化(`cfg.weight_norm`)是解决双稳态的正确手段**(默认 "none" 保持向后兼容):
hub 神经元的入度比低入度神经元高 2-3 个数量级(DN 入度中位 321、最大 5355,
感光细胞出度中位 4),同一全局 weight_scale 下 hub 先着火并自持 —— 归一化后每个
神经元的总驱动只与"上游有多少**比例**在发放"成比例,与入度无关。实测(真实数据,
`weight_norm="indeg"`,ws_inh = 2*ws_exc,input_gain=1.5):

    ws    每步发放   DN 均值   DN 非零  逐层 ol_sensory/ol_intrinsic/VP/DN/vMN    单步
    8.0    0.51%     0.04 Hz    0.1%    44 /  3.2 /  2.3 /  0.04 /  0.0 Hz      1.7 ms
   10.0    1.24%     1.36 Hz    3.6%    55 / 11.2 /  4.7 /  1.36 /  0.0 Hz      3.0 ms
   12.0    2.07%     9.00 Hz   19.1%    60 / 16.8 /  9.3 /  9.00 /  0.33 Hz     4.8 ms
   14.0    3.16%    18.08 Hz   38.0%    64 / 20.7 / 15.4 / 18.08 / 10.5 Hz      7.2 ms

  即:归一化**把"全静默<->全饱和"的二元开关变成连续可调的梯度**(DN 从 0.04 Hz
  连续升到 18 Hz),这是唯一能标定到生理区间的方式。raw(不归一化)在任何
  weight_scale 下都只有 0 Hz 或 >90 Hz 两种结局。

**但在该生理工作点上,视觉信息不可用(必须诚实记录,见 selftest_real.py 第 7d 节):**
DN 输出确实被画面调制(完全相同的历史下换画面 |ΔDN| = 14-56% of mean),但网络自身
历史涨落造成的变异同样大 —— "刺激间差异 / 同刺激重复差异" 比值实测 0.96-1.04
(判据 > 3),即 **DN 输出无法与网络自发状态区分**,监督读出的 r2 会是 ~0。
这是把"果蝇不会瞄准"与"该 LIF 简化模型不承载视觉信号"区分开的关键证据。

**真实数据上的实测(见 selftest_real.py):** 真实 A-A 子图只有 25,582,938 条边
(密度 0.092%,平均入度 153,而不是合成压测里的 911),所以**全量 166,700 只要
0.45-1.6 ms/步(33 步/帧 = 15-52 ms -> 19-67 FPS)**,真实数据下不需要子网也能跑
30 FPS。子网 30,043(含全部 22,122 个抑制性神经元,仅 11 MB)为 0.20-0.54 ms/步。
注意:单步耗时随**发放率**线性增长(活跃列 x 出度),网络被点燃后(每步约 5% 发放)
会劣化到 15 ms/步,此时 30 FPS 不再可能 —— 见第 5 节。

===============================================================================
4. 数值与内存
===============================================================================
- 状态数组: v / 不应期 / 上一步脉冲 / 集中率共 4 个 (m,) float32 = 16m 字节
  (m = 50,000 时约 0.8 MB);子网矩阵之外没有 (N, N) 稠密分配。
- `spikes` / `rates` 每次访问**新建数组**(不复用缓冲区),因此外部拿到的是独立
  副本,不会被后续 `step()` 覆盖;代价是每次访问 N 个元素的分配与写零。
- 电位在每步后被裁剪到 [-1e3, 1e3],避免数值爆炸;权重在构造时一次性校验有限性,
  外部驱动在写入前做有限性检查(NaN 不会进入状态)。
"""

from __future__ import annotations

import logging
import time
from collections import deque
from typing import Any

import numpy as np
import scipy.sparse as sp

from flyaim.config import BrainConfig, RetinaConfig
from flyaim.io import ARTIFACT_ROLES, ConnectomeArtifacts, load_roles

logger = logging.getLogger(__name__)

# 回退输入群规模约定:与 Retina 的回退(arange(eye_rows*eye_cols))保持一致,
# 这样集成层把同一个(空的)roles.visual_input 传给两端时,两端会落到同一组索引。
_FALLBACK_N_INPUT = int(RetinaConfig().eye_rows * RetinaConfig().eye_cols)

_SPIKE_DRIVEN_MAX_ACTIVE_FRAC = 0.10
"""活跃神经元比例低于此值时改用"脉冲驱动"稀疏传播(数值等价,只是求和顺序不同)。"""

_V_CLAMP = 1.0e3
_EXTRACT_BLOCK_ROWS = 4096


class Connectome:
    """稀疏 LIF 连接组仿真器。

    典型用法(集成层)::

        brain = Connectome("flyaim/data/build/connectome.npz", cfg,
                           input_neuron_ids=roles.visual_input)
        for _ in range(cfg.steps_per_frame):        # 集成层负责这个循环
            brain.step(in_exc, in_inh, cfg.dt_ms)
        rates = brain.rates                        # (N,) float32, 全局索引空间
    """

    def __init__(
        self,
        path: str,
        cfg: BrainConfig,
        input_neuron_ids: np.ndarray | None = None,
        roles: Any | None = None,
    ) -> None:
        """加载连接组并建立(可选的)子网。

        path: connectome.npz 路径(`ConnectomeArtifacts.format`),同目录若有
            roles.json 会被自动读取,用于子网必含集合。
        cfg: BrainConfig。
        input_neuron_ids: (n_input,) 全局索引,外部驱动落点,通常 =
            roles.visual_input。为空/None 时回退到 `arange(768)`(见模块常量),
            并在 `stats()` 里置 `input_neuron_ids_fallback=True`。
        roles: RoleSelection(可选)。缺省时自动从 path 同目录的 roles.json 读取。
        """
        self.path = str(path)
        self.cfg = cfg
        t0 = time.perf_counter()
        self.art = ConnectomeArtifacts.load(path)
        self._load_sec = time.perf_counter() - t0

        n = int(self.art.n)
        self._n = n
        if n <= 0:
            raise ValueError(f"连接组为空: {path}")

        self.roles = roles if roles is not None else self._load_roles_sibling(path)
        self.role_ids: dict[str, np.ndarray] = {}
        if self.roles is not None:
            for r in ("visual_input", "descending", "motor", "inhibitory"):
                v = np.asarray(getattr(self.roles, r, np.empty(0)), dtype=np.int64).reshape(-1)
                v = v[(v >= 0) & (v < n)]
                self.role_ids[r] = np.unique(v)

        # ---------------------------------------------------------- 输入群解析
        self.input_neuron_ids_fallback = False
        ids = np.asarray([] if input_neuron_ids is None else input_neuron_ids).reshape(-1)
        if ids.size == 0 and self.roles is not None:
            ids = np.asarray(getattr(self.roles, "visual_input", np.empty(0))).reshape(-1)
        cfg_ids = getattr(cfg, "in_neuron_ids", None)
        if ids.size == 0 and cfg_ids is not None:
            ids = np.asarray(cfg_ids).reshape(-1)
        self._input_ids_explicit = ids.size > 0
        if ids.size == 0:
            # 优雅回退(不抛异常):与 Retina 的回退集合一致
            ids = np.minimum(np.arange(_FALLBACK_N_INPUT, dtype=np.int64), n - 1)
            self.input_neuron_ids_fallback = True
            logger.warning(
                "Connectome: 未提供 input_neuron_ids(空/None),回退到 arange(%d);"
                "manifest 必须标记 visual_input_fallback=true",
                _FALLBACK_N_INPUT,
            )
        ids = np.asarray(ids, dtype=np.int64).reshape(-1)
        ids = ids[(ids >= 0) & (ids < n)]
        self.in_neuron_ids = np.unique(ids)

        # ---------------------------------------------------------- 子网选择
        must: list[np.ndarray] = [self.in_neuron_ids]
        for v in self.role_ids.values():
            must.append(v)
        must_arr = np.unique(np.concatenate([m for m in must if m.size])) if must else np.empty(0, np.int64)

        subnet_size = cfg.subnet_size
        if subnet_size is not None and int(subnet_size) < n:
            size = max(int(subnet_size), 1)
            if must_arr.size > size:
                logger.warning(
                    "Connectome: 必含角色神经元 %d 个 > subnet_size=%d,子网扩展到 %d",
                    must_arr.size,
                    size,
                    must_arr.size,
                )
                sim = must_arr
            else:
                rng = np.random.default_rng(int(cfg.subnet_seed))
                pool = np.setdiff1d(np.arange(n, dtype=np.int64), must_arr, assume_unique=False)
                extra = rng.choice(pool, size=size - must_arr.size, replace=False)
                sim = np.sort(np.concatenate([must_arr, extra]))
            self.subnet_applied = True
        else:
            sim = np.arange(n, dtype=np.int64)
            self.subnet_applied = False
        self._c2g = sim.astype(np.int64)
        self._m = int(sim.size)
        g2c = np.full(n, -1, dtype=np.int64)
        g2c[sim] = np.arange(self._m, dtype=np.int64)
        self._g2c = g2c

        # ---------------------------------------------------------- 权重抽取
        t0 = time.perf_counter()
        if self.subnet_applied:
            self._Wexc = self._extract(self.art.W_exc, sim, g2c)
            self._Winh = self._extract(self.art.W_inh, sim, g2c)
            renorm = float(getattr(cfg, "subnet_renormalize", False)) and (n / max(self._m, 1))
            if renorm:
                self._Wexc = (self._Wexc * renorm).tocsr()
                self._Winh = (self._Winh * renorm).tocsr()
                logger.warning("Connectome: 子网权重按 N/m=%.3f 缩放(subnet_renormalize=True)", renorm)
        else:
            # 只读引用:绝不在此对象上调用原地方法(冻结校验会逐位比对)
            self._Wexc = self.art.W_exc
            self._Winh = self.art.W_inh
            renorm = 1.0
        # 行归一化(hub 抑制):入度在神经元间差几个数量级(DN 入度中位 321、最大 5355,
        # 感光细胞出度中位 4),单一全局 weight_scale 下 hub 会先着火并自持,导致
        # "全静默 <-> 癫痫"双稳态。行归一化让每个神经元的总驱动只与"上游活跃比例"
        # 成比例,与入度无关。**返回新矩阵,绝不原地修改 art.W_exc/W_inh**(冻结校验)。
        self.weight_norm = str(getattr(cfg, "weight_norm", "none"))
        if self.weight_norm != "none":
            self._Wexc, self._Winh, self._norm_den_stats = self._normalize_rows(
                self._Wexc, self._Winh, self.weight_norm
            )
            logger.info(
                "Connectome: 行归一化 mode=%s(分母 p50=%.2f),weight_scale 需重新标定",
                self.weight_norm,
                float(np.median(self._norm_den_stats)),
            )
        else:
            self._norm_den_stats = np.empty(0, dtype=np.float32)
        self._subnet_sec = time.perf_counter() - t0

        # 外部输入在紧凑空间的落点(keep 掩码保证驱动器与落点一一对应)
        self._in_c, self._in_keep = self._map_ids(self.in_neuron_ids)
        self._in_keep_idx = np.flatnonzero(self._in_keep).astype(np.int64)
        self._in_buf = np.zeros(max(self._in_c.size, 1), dtype=np.float32)
        self._in_buf2 = np.zeros(max(self._in_c.size, 1), dtype=np.float32)
        self._I_ext = np.zeros(self._m, dtype=np.float32)
        self._last_in_size: int | None = int(self.in_neuron_ids.size)

        # ---------------------------------------------------------- 缓存常量
        self.dt = float(cfg.dt_ms)
        self.tau_m = float(cfg.tau_m_ms)
        if self.dt <= 0 or self.tau_m <= 0:
            raise ValueError(f"dt_ms/tau_m_ms 必须为正: {self.dt}/{self.tau_m}")
        self.v_rest = float(cfg.v_rest)
        self.v_thresh = float(cfg.v_thresh)
        self.v_reset = float(cfg.v_reset)
        self.w_exc = float(cfg.weight_scale_exc)
        self.w_inh = float(cfg.weight_scale_inh)
        self.dtype = np.float32 if cfg.dtype == "float32" else np.float64
        self._refr_steps = int(max(0, round(float(cfg.refractory_ms) / max(self.dt, 1e-9))))
        self._rate_alpha = float(
            np.clip(self.dt / max(float(getattr(cfg, "rate_tau_ms", 50.0)), 1e-9), 0.0, 1.0)
        )
        # 快速 LIF 路径:v_rest == 0 时把 dt/tau 折进权重尺度,每步只需
        #   v *= (1 - dt/tau); v += I     (2 次 O(m) 融合操作,无临时数组)
        self._dt_over_tau = min(self.dt / self.tau_m, 1.0)
        self._decay = float(np.clip(1.0 - self._dt_over_tau, 0.0, 1.0))
        self._fast_lif = abs(self.v_rest) < 1e-12
        self._w_exc_eff = self.w_exc * self._dt_over_tau
        self._w_inh_eff = self.w_inh * self._dt_over_tau
        self._rate_gain = (1000.0 / self.dt) * self._rate_alpha
        self._rate_instant = self._rate_alpha <= 0.0
        # 外部驱动缩放:默认与"一个突触的电导尺度"一致,即 200 Hz 驱动 -> I=4.0。
        # 可用 cfg.input_gain 覆盖(标定用,见 selftest_real 的驱动统计)。
        self._input_gain = 1.0
        self._input_gain_eff = 1.0
        self.input_gain = float(getattr(cfg, "input_gain", self.w_exc))
        self._use_neuromod = bool(cfg.use_neuromodulation)
        self._sparse_mode = str(getattr(cfg, "sparse_mode", "auto"))
        # 膜噪声(可选,默认 0 = 关闭):无噪声的确定性 LIF + 无适应,会让网络呈现
        # "全静默 or 全饱和"的双稳态,缺少中间稀疏活动区(实测,见 selftest_real.py
        # 第 6/7 节)。加入高斯膜噪声后发放率才会随亚阈值输入**连续**变化。
        self.noise_sigma = float(getattr(cfg, "noise_sigma", 0.0))
        self._noise_sigma_eff = self.noise_sigma * self._dt_over_tau
        self._noise_rng = np.random.default_rng(int(getattr(cfg, "noise_seed", cfg.subnet_seed)))
        self._wt_exc: sp.csr_matrix | None = None
        self._wt_inh: sp.csr_matrix | None = None
        self._validate_weights()

        mem = self._matrix_bytes()
        logger.info(
            "Connectome: N=%d m=%d exc_nnz=%d inh_nnz=%d 内存≈%.1f MB (载荷 %.1fs, 抽取 %.1fs)",
            n,
            self._m,
            self._Wexc.nnz,
            self._Winh.nnz,
            mem / 1e6,
            self._load_sec,
            self._subnet_sec,
        )

        self.reset()

    # ================================================================== 构造辅助

    @staticmethod
    def _load_roles_sibling(path: str) -> Any | None:
        from pathlib import Path

        p = Path(path).parent / ARTIFACT_ROLES
        if not p.exists():
            logger.warning("Connectome: 未找到 %s,子网必含集合将只包含输入群", p)
            return None
        try:
            return load_roles(p)
        except Exception as exc:
            logger.warning("Connectome: 读取 roles.json 失败: %s", exc)
            return None

    @staticmethod
    def _extract(W: sp.spmatrix, sim: np.ndarray, g2c: np.ndarray) -> sp.csr_matrix:
        """抽取诱导子图 W[sim][:, sim],分块进行以限制峰值内存。

        只读取输入矩阵,绝不原地修改(W 在冻结校验中被逐位比对)。
        """
        Wc = W.tocsr()
        m = int(sim.size)
        acc: sp.csr_matrix | None = None
        for start in range(0, m, _EXTRACT_BLOCK_ROWS):
            rows = sim[start : start + _EXTRACT_BLOCK_ROWS]
            A = Wc[rows]  # C 层行切片(拷贝)
            cols = g2c[A.indices]
            keep = cols >= 0
            if not keep.any():
                continue
            counts = np.diff(A.indptr)
            rr = np.repeat(np.arange(start, start + rows.size, dtype=np.int32), counts)[keep]
            cc = cols[keep].astype(np.int32)
            dd = A.data[keep].astype(np.float32)
            blk = sp.csr_matrix((dd, (rr, cc)), shape=(m, m), dtype=np.float32)
            acc = blk if acc is None else (acc + blk)
        if acc is None:
            return sp.csr_matrix((m, m), dtype=np.float32)
        acc.sum_duplicates()
        acc.sort_indices()
        return acc.tocsr()

    @staticmethod
    def _normalize_rows(
        W_e: sp.csr_matrix, W_i: sp.csr_matrix, mode: str
    ) -> tuple[sp.csr_matrix, sp.csr_matrix, np.ndarray]:
        """按行归一化兴奋/抑制权重矩阵(**返回新矩阵,绝不原地修改输入**)。

        mode:
          "indeg"     : 分母 = 该神经元的总突触前个数(nnz_exc + nnz_inh)。
                        两侧共用分母 -> **每个神经元的 E/I 比例保持不变**。
          "wsum"      : 分母 = 该神经元的总入权重和(exc+inh),同样保持 E/I 比例。
          "wsum_sep"  : 两张矩阵各自按自己的行和归一化(会改变每神经元 E/I 比例,
                        但两条通路的总强度各自均匀)。
        零入度的神经元分母置 1(保持全 0 行)。

        为什么需要它:入度在神经元间相差几个数量级(hub),同一全局 weight_scale 下
        hub 先着火并自持,而低入度神经元永远静默 —— 归一化后每个神经元的总驱动只与
        "上游有多少**比例**在发放"成比例,网络才可能落在均匀、可标定的工作点。
        """
        de = np.diff(W_e.indptr).astype(np.float64)
        di = np.diff(W_i.indptr).astype(np.float64)
        if mode == "indeg":
            den = de + di
        elif mode == "wsum":
            se = np.asarray(W_e.sum(axis=1)).reshape(-1).astype(np.float64)
            si = np.asarray(W_i.sum(axis=1)).reshape(-1).astype(np.float64)
            den = se + si
        elif mode == "wsum_sep":
            se = np.asarray(W_e.sum(axis=1)).reshape(-1).astype(np.float64)
            si = np.asarray(W_i.sum(axis=1)).reshape(-1).astype(np.float64)
            inv_e = 1.0 / np.where(se > 0, se, 1.0)
            inv_i = 1.0 / np.where(si > 0, si, 1.0)
            We = (sp.diags(inv_e.astype(np.float32)) @ W_e).tocsr() if W_e.nnz else W_e.copy()
            Wi = (sp.diags(inv_i.astype(np.float32)) @ W_i).tocsr() if W_i.nnz else W_i.copy()
            return We, Wi, np.maximum(se, si).astype(np.float32)
        else:
            raise ValueError(
                f"未知 weight_norm={mode!r};可选 none / indeg / wsum / wsum_sep"
            )
        inv = (1.0 / np.where(den > 0, den, 1.0)).astype(np.float32)
        D = sp.diags(inv)
        We = (D @ W_e).tocsr() if W_e.nnz else W_e.copy()
        Wi = (D @ W_i).tocsr() if W_i.nnz else W_i.copy()
        return We, Wi, den.astype(np.float32)

    def _map_ids(self, ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """全局索引 -> (紧凑索引, keep 掩码)。

        keep 掩码记录哪些输入位置真的落在子网内:驱动数组按原顺序取
        `exc[keep]` 才对得上落点(丢掉的索引不会错位)。
        """
        if ids.size == 0:
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=bool)
        c = self._g2c[ids]
        keep = c >= 0
        dropped = int(np.sum(~keep))
        if dropped:
            logger.warning("Connectome: %d/%d 个指定索引不在子网内,已丢弃", dropped, ids.size)
        return c[keep].astype(np.int64), keep

    def _matrix_bytes(self) -> int:
        tot = 0
        for W in (self._Wexc, self._Winh):
            tot += int(W.data.nbytes) + int(W.indices.nbytes) + int(W.indptr.nbytes)
        return tot

    def _validate_weights(self) -> None:
        """构造时一次性校验权重有限性(只读,不修改数据 —— 冻结校验会逐位比对)。"""
        for name, W in (("W_exc", self._Wexc), ("W_inh", self._Winh)):
            if W.nnz and not np.all(np.isfinite(W.data)):
                raise ValueError(f"{name} 含非有限权重,拒绝仿真(数据文件可能损坏)")
            if W.nnz and float(W.data.min()) < 0:
                raise ValueError(f"{name} 含负权重;兴奋/抑制必须分行存储")

    @property
    def input_gain(self) -> float:
        """外部驱动增益标定参数(默认 = weight_scale_exc)。"""
        return self._input_gain

    @input_gain.setter
    def input_gain(self, value: float) -> None:
        self._input_gain = float(value)
        self._input_gain_eff = float(value) * self._dt_over_tau

    # ================================================================== 仿真

    def reset(self) -> None:
        """把所有状态清零(电位、不应期、上一步脉冲、集中率、计时)。

        同时预分配全部工作缓冲区,使 `step()` 不再每步 malloc
        (实测这是单步耗时的主要来源之一)。
        """
        m = self._m
        self._v = np.zeros(m, dtype=np.float32)
        self._refr = np.zeros(m, dtype=np.int16)
        self._spk = np.zeros(m, dtype=bool)          # 本步脉冲(内部缓冲)
        self._spk_f = np.zeros(m, dtype=np.float32)  # 上一步脉冲(突触源)
        self._fire = np.zeros(m, dtype=bool)         # 阈值比较缓冲
        self._bool_tmp = np.zeros(m, dtype=bool)     # 不应期门控缓冲
        self._I = np.zeros(m, dtype=np.float32)      # 电流累加缓冲(无脉冲时用)
        self._rate = np.zeros(m, dtype=np.float32)
        self._I_ext = np.zeros(m, dtype=np.float32)
        self._in_buf = np.zeros(max(self._in_c.size, 1), dtype=np.float32)
        self._in_buf2 = np.zeros(max(self._in_c.size, 1), dtype=np.float32)
        self._active_idx = np.empty(0, dtype=np.int64)
        self._refr_timer = 0            # >0 表示群内仍有神经元在不应期
        self._n_fired = 0               # 上一步发放数(用于选稀疏路径)
        self._rate_pending = 0          # 集中率的惰性衰减步数
        self.last_path = "init"
        self.t_ms = 0.0
        self._steps = 0
        self._t_hist: deque[float] = deque(maxlen=200)
        self._last_step_ms = float("nan")
        self._neuromod_ignored = 0

    def step(
        self,
        in_exc: np.ndarray,
        in_inh: np.ndarray,
        t_ms: float,
        neuromod: np.ndarray | None = None,
    ) -> None:
        """推进**一个** dt(cfg.dt_ms)的仿真,保持内部状态。

        in_exc / in_inh: (n_input,) 外部兴奋/抑制驱动,加到 `input_neuron_ids`
            指定的神经元上(`I_ext = input_gain * (in_exc - in_inh)`)。长度必须与
            `input_neuron_ids` 一致(回退输入群例外,见 `_remap_input_group`)。
        t_ms: 当前时刻,仅记录(`self.t_ms`),**不参与积分** —— 积分恒为 dt。
            集成层按帧循环调用本方法 `cfg.steps_per_frame` 次。
        neuromod: (N,) / (m,) / 标量 神经调质增益(乘在突触电流上)。仅当
            `cfg.use_neuromodulation=True` 时生效,否则忽略并计数。

        实现要点(全部为性能服务,已实测):
            1. `_n_fired == 0` 时跳过全部稀疏乘法(零脉冲 -> 零突触输入);
            2. 活跃神经元少时用**脉冲驱动**传播(活跃列的非零数),而不是 `W @ s`
               (全非零数)—— 实测全量 16.67 万时 445 ms vs 2.5 ms;
            3. v_rest==0 时把 dt/tau 折进权重尺度,膜电位更新只有 2 次融合操作;
            4. 所有缓冲区预分配,`step()` 内不做 malloc;
            5. 集中率用惰性衰减:无脉冲的步只加一个计数器。
        """
        t0 = time.perf_counter()
        m = self._m
        dt_over_tau = self._dt_over_tau

        # ---------------- 外部输入(回退输入群可能在这一步才确定长度)
        have_input = in_exc is not None or in_inh is not None
        if have_input:
            n_in = self._prepare_input(in_exc, in_inh)

        # ---------------- 突触电流(上一步的脉冲)
        n_prev = self._n_fired
        if n_prev == 0:
            I = self._I
            I.fill(0.0)
            self.last_path = "silent"
        elif self._use_spike_driven(n_prev):
            # 活跃列只算一次:上一步已经存好活跃索引,这里不再扫描整个向量
            idx = self._active_idx
            vals = np.ones(idx.size, dtype=np.float32)  # 脉冲恒为 1
            I = self._gather(self._ensure_transpose("exc"), idx, vals)
            np.multiply(I, self._w_exc_eff, out=I)
            if self._Winh.nnz:
                I2 = self._gather(self._ensure_transpose("inh"), idx, vals)
                np.multiply(I2, self._w_inh_eff, out=I2)
                np.subtract(I, I2, out=I)
            self.last_path = "spike-driven"
        else:
            np.copyto(self._spk_f, self._spk, casting="unsafe")
            I = self._Wexc.dot(self._spk_f)
            np.multiply(I, self._w_exc_eff, out=I)
            if self._Winh.nnz:
                I -= self._Winh.dot(self._spk_f) * self._w_inh_eff
            self.last_path = "matvec"
        if have_input and self._in_c.size:
            # 外部驱动直接加到输入神经元(覆盖写:位置固定,无需先清零)
            np.add(I, self._I_ext, out=I)

        if self._use_neuromod and neuromod is not None:
            np.multiply(I, self._neuromod_gain(neuromod), out=I)
        elif neuromod is not None:
            self._neuromod_ignored += 1

        # ---------------- LIF 积分
        v = self._v
        if self._fast_lif:
            np.multiply(v, self._decay, out=v)   # 漏电
            np.add(v, I, out=v)                  # I 已含 dt/tau 因子
        else:
            v += dt_over_tau * (-(v - self.v_rest) + I)
        if self.noise_sigma > 0.0:               # 可选膜噪声(默认关闭)
            v += self._noise_sigma_eff * self._noise_rng.standard_normal(m, dtype=np.float32)
        np.clip(v, -_V_CLAMP, _V_CLAMP, out=v)

        # ---------------- 阈值判定 + 不应期
        fire = self._fire
        np.greater_equal(v, self.v_thresh, out=fire)
        if self._refr_timer > 0:
            r = self._refr
            np.less_equal(r, 0, out=self._bool_tmp)
            np.logical_and(fire, self._bool_tmp, out=fire)
            np.subtract(r, 1, out=r, where=r > 0)
        idx_fire = np.flatnonzero(fire)  # bool 数组,memchr 级扫描;顺带得到发放数
        n_fired = int(idx_fire.size)
        if n_fired:
            if self._refr_steps > 0:
                self._refr[idx_fire] = self._refr_steps
                self._refr_timer = self._refr_steps
            v[idx_fire] = self.v_reset
        elif self._refr_timer > 0:
            self._refr_timer -= 1
        np.copyto(self._spk, fire)
        self._active_idx = idx_fire  # 下一步的稀疏乘法直接复用,无需再扫描
        self._n_fired = n_fired

        # ---------------- 集中率 EMA(rate = d*rate + g*spike)
        # 惰性衰减:连续无脉冲的步只累加计数器,读 rates 时才补算 d^pending。
        if self._rate_instant:
            if n_fired:
                self._rate.fill(0.0)
                self._rate[idx_fire] = 1000.0 / self.dt
        elif n_fired:
            self._rate *= self._decay ** (self._rate_pending + 1)
            if self._rate_gain:
                self._rate[idx_fire] += self._rate_gain
            self._rate_pending = 0
        else:
            self._rate_pending += 1

        self.t_ms = float(t_ms)
        self._steps += 1
        dt_meas = (time.perf_counter() - t0) * 1000.0
        self._last_step_ms = dt_meas
        self._t_hist.append(dt_meas)

    # ------------------------------------------------------------------ 稀疏传播

    def _prepare_input(self, in_exc: np.ndarray | None, in_inh: np.ndarray | None) -> int:
        """把外部驱动写进 `_I_ext` 的输入落点(位置固定,无需整向量清零)。

        返回驱动长度。`_I_ext` 除 `_in_c` 之外恒为 0(初始化后从不写入),
        因此 `step()` 里可以直接 `I += _I_ext` 而不必先清零。
        """
        exc = None if in_exc is None else np.asarray(in_exc, dtype=np.float32).reshape(-1)
        inh = None if in_inh is None else np.asarray(in_inh, dtype=np.float32).reshape(-1)
        size = exc.size if exc is not None else (inh.size if inh is not None else 0)
        if size != self.in_neuron_ids.size:
            self._remap_input_group(size)
        n = int(self._in_c.size)
        if n == 0:
            return size
        idx = self._in_keep_idx
        buf = self._in_buf[:n]
        g = self._input_gain_eff
        if exc is not None and exc.size >= int(idx.max(initial=-1)) + 1:
            np.take(exc, idx, out=buf)
            np.multiply(buf, g, out=buf)
        else:
            buf.fill(0.0)
        if inh is not None and inh.size >= int(idx.max(initial=-1)) + 1:
            b2 = self._in_buf2[:n]
            np.take(inh, idx, out=b2)
            np.multiply(b2, g, out=b2)
            np.subtract(buf, b2, out=buf)
        self._I_ext[self._in_c] = buf
        return size

    def _remap_input_group(self, size: int) -> None:
        """外部驱动长度与输入群不一致时的处理。

        - 显式输入群:长度不匹配是配置错误 -> 抛 ValueError(早失败优于静默错配);
        - 回退输入群:按实际驱动长度重建 `arange(size)`(回退路径允许自适应)。
        """
        if self._input_ids_explicit:
            raise ValueError(
                f"in_exc/in_inh 长度 {size} 与 input_neuron_ids 长度 "
                f"{self.in_neuron_ids.size} 不一致(显式输入群不允许自动重建)"
            )
        if size == self.in_neuron_ids.size:
            return
        n = self._n
        new_ids = np.minimum(np.arange(size, dtype=np.int64), n - 1)
        new_ids = np.unique(new_ids)
        self.in_neuron_ids = new_ids
        self._in_c, self._in_keep = self._map_ids(new_ids)
        self._in_keep_idx = np.flatnonzero(self._in_keep).astype(np.int64)
        self._in_buf = np.zeros(max(self._in_c.size, 1), dtype=np.float32)
        self._in_buf2 = np.zeros(max(self._in_c.size, 1), dtype=np.float32)
        self._I_ext = np.zeros(self._m, dtype=np.float32)
        self._last_in_size = size
        logger.warning(
            "Connectome: 回退输入群按驱动长度重建为 arange(%d)(子网内可用 %d 个)",
            size,
            self._in_c.size,
        )

    def _use_spike_driven(self, n_prev: int) -> bool:
        """活跃比例低 -> 用脉冲驱动的稀疏传播(数值等价,只算活跃列)。"""
        if self._sparse_mode == "matvec":
            return False
        if self._sparse_mode == "spike":
            return True
        if self._Wexc.nnz == 0:
            return False
        return (n_prev / max(self._m, 1)) < _SPIKE_DRIVEN_MAX_ACTIVE_FRAC

    def _ensure_transpose(self, which: str) -> sp.csr_matrix:
        """按列访问用的 CSR 视图(共享 CSC 数据,不复制元素)。

        用 `W.tocsc().T` 得到的 csr 矩阵与 CSC 共享 data/indices/indptr,
        于是"取活跃列" = C 层行切片,代价正比于活跃列的非零数,
        数值上与 `W @ s` 完全等价(仅浮点求和顺序不同)。
        """
        cur = self._wt_exc if which == "exc" else self._wt_inh
        if cur is None:
            W = self._Wexc if which == "exc" else self._Winh
            cur = W.tocsc().T if W.nnz else W
            if which == "exc":
                self._wt_exc = cur
            else:
                self._wt_inh = cur
        return cur

    @staticmethod
    def _gather(Wt: sp.csr_matrix, idx: np.ndarray, vals: np.ndarray) -> np.ndarray:
        """脉冲驱动的稀疏传播:sum_j W[:, j] * s_j(等价于 W @ s)。

        `Wt = W.T`(CSR,行 = W 的列)。取活跃行 `Wt[idx]` 后转置回 (m, k) 的 CSC
        (共享数组,不复制元素)再乘权重向量 —— 代价只正比于**活跃列**的非零数,
        而 `W @ s` 的代价正比于全部非零数。数值上完全等价,只是浮点求和顺序不同。
        """
        m = Wt.shape[0]
        if idx.size == 0:
            return np.zeros(m, dtype=np.float32)
        sub = Wt[idx]  # CSR (k, m),C 层行切片
        return np.asarray(sub.T.dot(vals), dtype=np.float32).reshape(-1)

    def _neuromod_gain(self, neuromod: Any) -> np.ndarray:
        a = np.asarray(neuromod, dtype=np.float32)
        if a.ndim == 0:
            return np.full(self._m, float(a), dtype=np.float32)
        a = a.reshape(-1)
        if a.size == self._n:
            a = a[self._c2g]
        elif a.size != self._m:
            raise ValueError(f"neuromod 长度 {a.size} 既不是 N={self._n} 也不是 m={self._m}")
        return (1.0 + a).astype(np.float32)

    # ================================================================== 对外读出

    def _flush_rate(self) -> None:
        """把集中率的惰性衰减补齐(无脉冲的步只累加计数器,读的时候才算)。"""
        if self._rate_pending:
            self._rate *= self._decay ** self._rate_pending
            self._rate_pending = 0

    @property
    def rates(self) -> np.ndarray:
        """(N,) float32,Hz,全局索引空间的瞬时放电率(脉冲的指数滑动平均)。

        每次访问返回**新数组**;未仿真(子网外)位置为 0。
        """
        self._flush_rate()
        out = np.zeros(self._n, dtype=np.float32)
        if self._m == self._n:
            out[:] = self._rate
        else:
            out[self._c2g] = self._rate
        return out

    @property
    def spikes(self) -> np.ndarray:
        """(N,) bool,本步是否发放(全局索引空间,未仿真位置 False)。

        每次访问返回**新数组**,因此不会被后续 `step()` 覆盖。
        """
        out = np.zeros(self._n, dtype=bool)
        if self._m == self._n:
            out[:] = self._spk
        else:
            out[self._c2g] = self._spk
        return out

    @property
    def rates_compact(self) -> np.ndarray:
        """(m,) float32,紧凑(子网)索引空间的放电率,只读视图的副本。"""
        self._flush_rate()
        return self._rate.copy()

    @property
    def spikes_compact(self) -> np.ndarray:
        """(m,) bool,紧凑索引空间的本步脉冲。"""
        return self._spk.copy()

    @property
    def membrane(self) -> np.ndarray:
        """(m,) float32,紧凑空间的膜电位(诊断用)。"""
        return self._v.copy()

    @property
    def n_neurons(self) -> int:
        """全局神经元数 N。"""
        return self._n

    @property
    def n_sim(self) -> int:
        """实际仿真的神经元数 m。"""
        return self._m

    @property
    def sim_indices(self) -> np.ndarray:
        """(m,) int64,紧凑索引 -> 全局索引。"""
        return self._c2g.copy()

    def compact_index_of(self, global_ids: np.ndarray) -> np.ndarray:
        """全局索引 -> 紧凑索引,不在子网内返回 -1。"""
        ids = np.asarray(global_ids, dtype=np.int64).reshape(-1)
        return self._g2c[ids].copy()

    def set_input_neurons(self, ids: np.ndarray) -> None:
        """替换外部输入落点(全局索引)。显式设置后长度不匹配会报错。"""
        ids = np.unique(np.asarray(ids, dtype=np.int64).reshape(-1))
        ids = ids[(ids >= 0) & (ids < self._n)]
        self.in_neuron_ids = ids
        self._in_c, self._in_keep = self._map_ids(ids)
        self._in_keep_idx = np.flatnonzero(self._in_keep).astype(np.int64)
        self._in_buf = np.zeros(max(self._in_c.size, 1), dtype=np.float32)
        self._in_buf2 = np.zeros(max(self._in_c.size, 1), dtype=np.float32)
        self._I_ext = np.zeros(self._m, dtype=np.float32)
        self._input_ids_explicit = True
        self.input_neuron_ids_fallback = False

    # ================================================================== 诊断

    def stats(self) -> dict:
        """结构/性能统计(供 manifest 与报告)。"""
        return {
            "path": self.path,
            "n_neurons": int(self._n),
            "n_sim": int(self._m),
            "subnet_applied": bool(self.subnet_applied),
            "subnet_size_cfg": self.cfg.subnet_size,
            "subnet_seed": int(self.cfg.subnet_seed),
            "n_edges_exc": int(self._Wexc.nnz),
            "n_edges_inh": int(self._Winh.nnz),
            "n_edges_total": int(self._Wexc.nnz + self._Winh.nnz),
            "density_sim": float((self._Wexc.nnz + self._Winh.nnz) / max(self._m * self._m, 1)),
            "matrix_bytes": int(self._matrix_bytes()),
            "state_bytes": int(
                self._v.nbytes + self._refr.nbytes + self._spk_f.nbytes + self._rate.nbytes
            ),
            "input_group": {
                "n": int(self.in_neuron_ids.size),
                "n_mapped_in_subnet": int(self._in_c.size),
                "fallback": bool(self.input_neuron_ids_fallback),
                "input_gain": float(self.input_gain),
                "head": [int(x) for x in self.in_neuron_ids[:8]],
            },
            "roles": {
                k: int(v.size) for k, v in self.role_ids.items()
            },
            "use_neuromodulation": bool(self._use_neuromod),
            "sparse_mode": self._sparse_mode,
            "weight_norm": self.weight_norm,
            "weight_norm_den_p50": (
                float(np.median(self._norm_den_stats)) if self._norm_den_stats.size else None
            ),
            "noise_sigma": float(self.noise_sigma),
            "load_seconds": round(float(self._load_sec), 3),
            "subnet_seconds": round(float(self._subnet_sec), 3),
        }

    def timing_stats(self) -> dict:
        """近 200 步的单步耗时统计(毫秒)。"""
        h = np.asarray(self._t_hist, dtype=np.float64)
        if h.size == 0:
            return {"steps": 0}
        return {
            "steps": int(self._steps),
            "window": int(h.size),
            "mean_ms": float(h.mean()),
            "p50_ms": float(np.percentile(h, 50)),
            "p95_ms": float(np.percentile(h, 95)),
            "max_ms": float(h.max()),
            "last_ms": float(self._last_step_ms),
        }

    def describe(self) -> dict:
        """stats + timing 合并(供 manifest 落盘)。"""
        d = self.stats()
        d["timing"] = self.timing_stats()
        d["steps_per_frame"] = int(self.cfg.steps_per_frame)
        d["note"] = "step() 只推进一个 dt;每帧的 steps_per_frame 次循环由集成层负责"
        return d

    def __repr__(self) -> str:  # pragma: no cover - 诊断用
        return (
            f"Connectome(N={self._n}, m={self._m}, exc_nnz={self._Wexc.nnz}, "
            f"inh_nnz={self._Winh.nnz}, subnet={self.subnet_applied})"
        )
