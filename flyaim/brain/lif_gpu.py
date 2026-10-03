"""GPU (CuPy) LIF 引擎 —— 与 `flyaim.brain.lif.Connectome` 数值等价。

为什么要 GPU
------------
CPU 版的单步成本几乎全在**稀疏传播**上,而 scipy 的 CSR SpMV 是**单线程**的。
实测工作点(全量 166,700,82% 神经元活跃):单步 42 ms,占整帧成本的 99.1%。
GPU 有 288 GB/s 显存带宽 + 1536 个 CUDA 核心,同样规模的 SpMV 应该是亚毫秒级。

与 CPU 版的两处**有意差异**(都是为了 GPU 效率,数学等价)
--------------------------------------------------------
1. **单一带符号矩阵**:CPU 每步做两次 SpMV(`W_exc@s*w_e - W_inh@s*w_i`)。
   GPU 版预先合成 `W = W_exc*w_e - W_inh*w_i`,每步只做**一次** SpMV。
   若某对 (i,j) 同时存在于两张矩阵,合并后正是 `(w_e*we - w_i*wi)*s_j`,
   与 CPU 的减法结果**逐位等价**。省掉一半 kernel 与一半显存带宽。
2. **不使用脉冲驱动的列收集**:CPU 版靠"只收集活跃列"把稀疏性变现;
   GPU 上列收集会退化成不规则的 gather,反而不如直接全量 SpMV(带宽换规整)。

⚠️ 已知的 CPU 端行为,GPU 版**照抄以保持可比性**
------------------------------------------------
`lif.py` 的集中率 EMA 用的是**膜衰减** `self._decay`(= 1 - dt/tau),
而不是 `1 - dt/rate_tau_ms`。即 `_rate_alpha` 只影响了 `rate_gain`,
衰减时间常数实际是 `tau_m`(20ms)而非 `rate_tau_ms`(50ms)。
GPU 版刻意复现该行为,否则同一份权重在 CPU/GPU 上会得到不同的 rates。
(这本身像是 CPU 端的一个 bug,已记录待确认;此处不擅自"修正"，
 否则会破坏与既有实验结果的数值可比性。)

用法::

    from flyaim.brain.lif_gpu import ConnectomeGPU, gpu_available

    if gpu_available():
        brain = ConnectomeGPU(path, cfg, input_neuron_ids=roles.visual_input)
        brain.step(in_exc, in_inh, t_ms)
        r = brain.rates      # -> numpy (H2D 一次/帧)
"""

from __future__ import annotations

import glob
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np

from flyaim.io import ConnectomeArtifacts

_V_CLAMP = 1.0e3

# ---------------------------------------------------------------------------
# 融合 LIF kernel
# ---------------------------------------------------------------------------
# 为什么要融合:优化掉每步的同步之后,瓶颈从"CPU↔GPU 往返"变成
# **kernel 发射开销** —— 每步原本要 9 个 kernel(乘/加/clip/比较/掩码/
# 重置/转 float/两次 rate 运算),8 步/帧就是 72 次发射。
# 这些 kernel 每次都只处理 166,700 个元素(远不足以喂满 GPU),
# 于是整段被发射延迟主导,利用率卡在 ~46%。
#
# 融合后每步只剩 **2 个 kernel**:cuSPARSE 的 SpMV + 这一个全元素 kernel。
# 输入相加也塞进来(has_ext 决定),省掉一次独立发射。
_LIF_SRC = r"""
extern "C" __global__
void lif_step(const float* __restrict__ I,
              const float* __restrict__ Iext,
              float* __restrict__ v,
              float* __restrict__ s,
              float* __restrict__ rate,
              short* __restrict__ refr,
              const float decay, const float thresh, const float vreset,
              const float rdecay, const float rgain,
              const int n, const int has_ext, const int refr_steps)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    // 1) 电流 = SpMV 结果 + 外部驱动
    float ii = I[i];
    if (has_ext) ii += Iext[i];

    // 2) LIF 积分(v_rest = 0):v = v*decay + I
    float vi = v[i] * decay + ii;
    vi = fminf(fmaxf(vi, -1000.0f), 1000.0f);

    // 3) 阈值
    float f = (vi >= thresh) ? 1.0f : 0.0f;

    // 4) 不应期(与 CPU 同序:先门控、再递减)
    if (refr_steps > 0) {
        if (refr[i] > 0) {
            f = 0.0f;
            refr[i] = refr[i] - 1;
        }
    }

    // 5) 重置发放神经元
    v[i] = (f > 0.5f) ? vreset : vi;

    // 6) 脉冲 -> float32(下一步 SpMV 的源)
    s[i] = f;

    // 7) 集中率 EMA
    rate[i] = rate[i] * rdecay + rgain * f;

    // 8) 为刚发放的神经元装填不应期(放在最后,与 CPU 顺序一致)
    if (refr_steps > 0 && f > 0.5f) refr[i] = (short)refr_steps;
}
"""


def _setup_cuda_dll_dirs() -> list[str]:
    """把 pip 装的 nvidia-* CUDA 库目录注册进 DLL 搜索路径。

    本机**没有 CUDA toolkit**,CUDA 库来自 pip 包 `nvidia-*-cu12`,
    它们把 DLL 放在 `site-packages/nvidia/<lib>/bin/`。
    CuPy 靠 `cuda-pathfinder` 也能找到,但显式注册更稳,并顺带消掉
    "CUDA path could not be detected" 这条噪音警告
    (它写 stderr,会让 PowerShell 把成功运行误报成 NativeCommandError)。
    """
    sp = os.path.join(sys.prefix, "Lib", "site-packages", "nvidia")
    if not os.path.isdir(sp):
        return []
    dirs: list[str] = []
    for d in sorted(glob.glob(os.path.join(sp, "*", "bin"))):
        if os.path.isdir(d):
            dirs.append(d)
            try:
                os.add_dll_directory(d)
            except Exception:
                pass
    if dirs and not os.environ.get("CUDA_PATH"):
        # 指向 cuda_runtime,让 pathfinder 直接命中
        rt = os.path.join(sp, "cuda_runtime")
        if os.path.isdir(rt):
            os.environ["CUDA_PATH"] = rt
    warnings.filterwarnings("ignore", message=".*CUDA path could not be detected.*")
    return dirs


_CUDA_DLL_DIRS = _setup_cuda_dll_dirs()


def gpu_available() -> bool:
    """CuPy 与 CUDA 设备是否可用(不抛异常)。"""
    try:
        import cupy as cp  # noqa: F401

        return cp.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


def gpu_info() -> dict:
    try:
        import cupy as cp

        p = cp.cuda.runtime.getDeviceProperties(0)
        free, total = cp.cuda.runtime.memGetInfo()
        return {
            "available": True,
            "name": p["name"].decode() if isinstance(p["name"], bytes) else str(p["name"]),
            "sm": f"{p['major']}.{p['minor']}",
            "mem_total_gb": total / 1e9,
            "mem_free_gb": free / 1e9,
            "cupy": cp.__version__,
        }
    except Exception as e:
        return {"available": False, "error": f"{type(e).__name__}: {e}"}


class ConnectomeGPU:
    """CuPy 版 LIF,接口与 `Connectome` 对齐(step / rates / spikes / reset)。"""

    def __init__(self, path: str | Path, cfg, input_neuron_ids=None,
                 roles=None, neuron_index=None, device: int = 0):
        import cupy as cp
        import cupyx.scipy.sparse as cpsp

        self._cp = cp
        cp.cuda.Device(int(device)).use()
        self.cfg = cfg
        self.device = int(device)
        t0 = time.perf_counter()

        art = ConnectomeArtifacts.load(path)
        self.n = int(art.n)
        m = self.n
        self._m = m

        W_e = art.W_exc.astype(np.float32)
        W_i = art.W_inh.astype(np.float32)

        # ---- 行归一化(与 CPU `_normalize_rows` 完全一致) ----
        self.weight_norm = str(getattr(cfg, "weight_norm", "none"))
        if self.weight_norm != "none":
            W_e, W_i = self._normalize_rows(W_e, W_i, self.weight_norm)

        dt = float(cfg.dt_ms)
        tau = float(cfg.tau_m_ms)
        self.dt = dt
        self.tau_m = tau
        if dt <= 0 or tau <= 0:
            raise ValueError(f"dt_ms/tau_m_ms 必须为正: {dt}/{tau}")
        self._dt_over_tau = min(dt / tau, 1.0)
        self._decay = float(np.clip(1.0 - self._dt_over_tau, 0.0, 1.0))
        self.v_thresh = float(cfg.v_thresh)
        self.v_reset = float(cfg.v_reset)
        self.v_rest = float(cfg.v_rest)
        self._refr_steps = int(max(0, round(float(cfg.refractory_ms) / max(dt, 1e-9))))

        # ---- 电导尺度(先把 dt/tau 折进权重,与 CPU fast_lif 一致) ----
        self._w_exc_eff = float(cfg.weight_scale_exc) * self._dt_over_tau
        self._w_inh_eff = float(cfg.weight_scale_inh) * self._dt_over_tau
        self._input_gain = float(getattr(cfg, "input_gain", 1.5))
        self._input_gain_eff = self._input_gain * self._dt_over_tau

        # ---- 合成单一带符号矩阵(一次 SpMV) ----
        t_sign = time.perf_counter()
        if W_i.nnz:
            W = (W_e * np.float32(self._w_exc_eff)
                 - W_i * np.float32(self._w_inh_eff)).tocsr()
        else:
            W = (W_e * np.float32(self._w_exc_eff)).tocsr()
        W.sort_indices()
        self._sign_sec = time.perf_counter() - t_sign

        # ---- 上传 ----
        t_up = time.perf_counter()
        W = W.astype(np.float32)
        self._W = cpsp.csr_matrix(
            (cp.asarray(W.data), cp.asarray(W.indices.astype(np.int32)),
             cp.asarray(W.indptr.astype(np.int32))),
            shape=W.shape,
        )
        self._upload_sec = time.perf_counter() - t_up
        self.nnz = int(self._W.nnz)

        # ---- 集中率 EMA 常量(照抄 CPU:用膜衰减) ----
        self._rate_alpha = float(np.clip(dt / max(float(getattr(cfg, "rate_tau_ms", 50.0)), 1e-9),
                                         0.0, 1.0))
        self._rate_gain = (1000.0 / dt) * self._rate_alpha
        self._rate_decay = np.float32(self._decay)   # ← 照抄 CPU,非 1-rate_alpha

        # ---- 输入落点 ----
        if input_neuron_ids is None:
            self.in_neuron_ids = np.empty(0, dtype=np.int64)
        else:
            self.in_neuron_ids = np.asarray(input_neuron_ids, dtype=np.int64).ravel()
        self._in_c = self.in_neuron_ids
        if self._in_c.size:
            bad = int(((self._in_c < 0) | (self._in_c >= m)).sum())
            if bad:
                raise ValueError(f"{bad} 个 input_neuron_ids 越界 (N={m})")

        # ---- 编译融合 LIF kernel(nvrtc 运行时 JIT)----
        # 失败不致命:回退到逐元素多 kernel 路径(功能一致,只是慢)
        self._k_lif = None
        self._block = 256
        try:
            t_k = time.perf_counter()
            self._k_lif = cp.RawKernel(_LIF_SRC, "lif_step")
            self._kernel_sec = time.perf_counter() - t_k
        except Exception as e:
            self._kernel_sec = float("nan")
            warnings.warn(f"融合 kernel 编译失败,回退逐元素路径: {type(e).__name__}: {e}")

        self._load_sec = time.perf_counter() - t0
        self.reset()

    # ------------------------------------------------------------------ 归一化

    @staticmethod
    def _normalize_rows(W_e, W_i, mode: str):
        import scipy.sparse as sp

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
            We = (sp.diags((1.0 / np.where(se > 0, se, 1.0)).astype(np.float32)) @ W_e).tocsr() \
                if W_e.nnz else W_e.copy()
            Wi = (sp.diags((1.0 / np.where(si > 0, si, 1.0)).astype(np.float32)) @ W_i).tocsr() \
                if W_i.nnz else W_i.copy()
            return We, Wi
        else:
            raise ValueError(f"未知 weight_norm={mode!r}")
        inv = (1.0 / np.where(den > 0, den, 1.0)).astype(np.float32)
        D = sp.diags(inv)
        We = (D @ W_e).tocsr() if W_e.nnz else W_e.copy()
        Wi = (D @ W_i).tocsr() if W_i.nnz else W_i.copy()
        return We, Wi

    # ------------------------------------------------------------------ 状态

    def reset(self) -> None:
        cp = self._cp
        m = self._m
        self._v = cp.zeros(m, dtype=cp.float32)
        self._s = cp.zeros(m, dtype=cp.float32)     # 上一步脉冲(0/1)
        self._fire = cp.zeros(m, dtype=cp.bool_)
        self._I_ext = cp.zeros(m, dtype=cp.float32)
        self._rate = cp.zeros(m, dtype=cp.float32)
        if self._refr_steps > 0:
            self._refr = cp.zeros(m, dtype=cp.int16)
        else:
            self._refr = None
        # 融合 kernel 需要非空指针;refr_steps=0 时传一个 1 元素哑数组
        self._refr_arg = self._refr if self._refr is not None else cp.zeros(1, dtype=cp.int16)
        # 热路径用的预转标量(避免每步新建 np.float32(...)
        # 以及 CuPy 的标量类型推断开销)
        self._decay_f = np.float32(self._decay)
        self._thresh_f = np.float32(self.v_thresh)
        self._vreset_f = np.float32(self.v_reset)
        self._rate_decay_f = np.float32(self._rate_decay)
        self._rate_gain_f = np.float32(self._rate_gain)
        # 预分配临时缓冲(热路径不做 malloc)
        self._rtmp = cp.empty(m, dtype=cp.float32)
        self._n_fired = 0
        self._steps = 0
        self.t_ms = 0.0
        self._cpu_spikes = None
        self._cpu_rates = None
        self.last_step_ms = float("nan")
        self.n_syncs = 0        # 诊断:实际发生了多少次 device 同步

    # ------------------------------------------------------------------ 输入

    def set_input(self, in_exc: np.ndarray, in_inh: np.ndarray) -> None:
        """设置外部驱动(每帧一次即可;之后 8 个 step 复用)。

        `I_ext = input_gain_eff * (in_exc - in_inh)`,落在 `input_neuron_ids` 上。
        """
        cp = self._cp
        if self._in_c.size == 0:
            return
        exc = np.asarray(in_exc, dtype=np.float32).ravel()
        inh = np.asarray(in_inh, dtype=np.float32).ravel()
        if exc.size != self._in_c.size or inh.size != self._in_c.size:
            raise ValueError(
                f"驱动长度 {exc.size}/{inh.size} 与 input_neuron_ids "
                f"{self._in_c.size} 不一致")
        val = (exc - inh) * np.float32(self._input_gain_eff)
        self._I_ext.fill(0.0)
        # in_c 唯一 -> 可直接赋值;用 cp.asarray 一次 H2D
        self._I_ext[cp.asarray(self._in_c)] = cp.asarray(val)

    # ------------------------------------------------------------------ 单步

    def step(self, in_exc=None, in_inh=None, t_ms: float = 0.0,
             neuromod=None) -> None:
        """推进一个 dt。签名与 CPU `Connectome.step` 兼容。

        与 `step_many` 走同一条无同步热路径;区别只在这里会(按需)上传外部驱动。
        集成层若要走快路径,应当用 `step_many`(驱动只上一次)。
        """
        if in_exc is not None or in_inh is not None:
            if in_exc is None:
                in_exc = np.zeros(self._in_c.size, dtype=np.float32)
            if in_inh is None:
                in_inh = np.zeros(self._in_c.size, dtype=np.float32)
            self.set_input(in_exc, in_inh)
        self._step_no_input()
        self.t_ms = float(t_ms)

    def step_many(self, n_steps: int, in_exc=None, in_inh=None,
                  dt_ms: float | None = None) -> None:
        """连推 n_steps 步(热路径)。

        两处关键优化,都是为了**不打断 GPU 流水线**:

        1. 外部驱动只上传**一次**,之后 8 步复用(不必每步 H2D);
        2. **全程零 device 同步、零 Python 分支**。

        ⚠️ 第 2 点为什么重要:第一版每步都写
        `n_fired = int(cp.count_nonzero(fire))` 再 `if n_fired:` ——
        `int()` 会**强制同步**,即每步一次 CPU↔GPU 往返(每帧 8 次)。
        GPU 于是在"发射 kernel → 等 CPU 发下一个 → 再发射"之间空转,
        利用率上不去(实测只有 40~79%)。

        改法:把所有数据相关的分支改成**无条件的逐元素操作**
        (掩码运算在"没有神经元发放"时也是正确的空操作),kernel 全部异步入队,
        只在真正需要读结果时(`rates` / `spikes`)同步一次。
        """
        if in_exc is not None or in_inh is not None:
            self.set_input(in_exc, in_inh)
        for _ in range(int(n_steps)):
            self._step_no_input(dt_ms)

    def _step_no_input(self, dt_ms=None) -> None:
        """单个 dt 的推进。**热路径:零同步、零数据相关分支。**

        每步只有 **2 个 kernel**:
            1. cuSPARSE 的 SpMV(唯一的带宽大户)
            2. 融合 LIF kernel(积分/阈值/不应期/重置/脉冲转换/rate EMA 全在里面)
        """
        cp = self._cp
        # 1) 稀疏传播(W 已含电导尺度,单次 SpMV)
        I = self._W.dot(self._s)

        if self._k_lif is not None:
            # 2) 融合 kernel:把 8 个逐元素操作压成 1 次发射
            m = self._m
            grid = ((m + self._block - 1) // self._block,)
            self._k_lif(
                grid, (self._block,),
                (I, self._I_ext, self._v, self._s, self._rate, self._refr_arg,
                 self._decay_f, self._thresh_f, self._vreset_f,
                 self._rate_decay_f, self._rate_gain_f,
                 np.int32(m), np.int32(1 if self._in_c.size else 0),
                 np.int32(self._refr_steps)),
            )
        else:
            self._step_elementwise(I)

        self._steps += 1
        self._cpu_spikes = None
        self._cpu_rates = None

    def _step_elementwise(self, I) -> None:
        """回退路径:逐元素多 kernel(与融合 kernel 数学等价,只是发射次数多)。"""
        cp = self._cp
        if self._in_c.size:
            cp.add(I, self._I_ext, out=I)
        v = self._v
        cp.multiply(v, self._decay_f, out=v)
        cp.add(v, I, out=v)
        cp.clip(v, -_V_CLAMP, _V_CLAMP, out=v)
        fire = self._fire
        cp.greater_equal(v, self._thresh_f, out=fire)
        if self._refr is not None:
            cp.logical_and(fire, cp.less_equal(self._refr, 0), out=fire)
            cp.subtract(self._refr, 1, out=self._refr, where=self._refr > 0)
            self._refr[fire] = np.int16(self._refr_steps)
        cp.copyto(v, self._vreset_f, where=fire)
        cp.copyto(self._s, fire, casting="unsafe")
        cp.multiply(self._rate, self._rate_decay_f, out=self._rate)
        cp.multiply(self._s, self._rate_gain_f, out=self._rtmp)
        cp.add(self._rate, self._rtmp, out=self._rate)

    def plastic_view(self) -> dict:
        """可塑性外挂(CONTRACT 附录 P)用的有效权重视图。

        返回 cupy 数组**引用**(非副本):data 原地修改后,后续 step 的
        SpMV 立即看到新权重。is_inh 是首次调用时的极性快照,之后冻结
        (权重过零也不改变其归属)。rows 为每个非零元的行号(惰性缓存)。
        """
        if getattr(self, "_plastic_view", None) is not None:
            return self._plastic_view
        cp = self._cp
        W = self._W
        rows = cp.repeat(cp.arange(self._m, dtype=cp.int32),
                         cp.diff(W.indptr).astype(cp.int32))
        self._plastic_view = {
            "data": W.data,
            "cols": W.indices,
            "rows": rows,
            "is_inh": cp.asarray(W.data < 0),
            "we_eff": self._w_exc_eff,
            "wi_eff": self._w_inh_eff,
            "s": self._s,   # 脉冲向量(cupy,引擎每步原地更新,引用恒有效)
            "cp": cp,
            "nnz": int(W.nnz),
            "n": int(self._m),
        }
        return self._plastic_view

    def fired_count(self) -> int:
        """当前步发放数(**会同步**)。仅诊断用,不要放进热路径。"""
        return int(self._cp.count_nonzero(self._s > 0.5))

    # ------------------------------------------------------------------ 读出

    @property
    def spikes(self) -> np.ndarray:
        """本步发放(bool,CPU numpy)。

        ⚠️ 从 `_s`(float32 0/1)派生,**不是**从 `_fire` 读 ——
        融合 kernel 只写 `s`/`v`/`rate` 三个数组(这是融合的一部分:
        少写一个数组就少一次显存写),`_fire` 只有逐元素回退路径才更新。
        早期版本 `spikes` 读 `_fire`,导致融合路径下"发放数恒为 0"
        (而 rates 正确)—— 被 bench_gpu 的等价性校验当场抓到。
        """
        if self._cpu_spikes is None:
            self._cpu_spikes = self._cp.asnumpy(self._s) > 0.5
        return self._cpu_spikes

    @property
    def rates(self) -> np.ndarray:
        self.n_syncs += 1     # 诊断:这里会发生唯一一次 device 同步
        if self._cpu_rates is None:
            self._cpu_rates = self._cp.asnumpy(self._rate)
        return self._cpu_rates

    def sync(self) -> None:
        """等 GPU 完成(计时/正确性校验用)。"""
        self._cp.cuda.Stream.null.synchronize()

    def info(self) -> dict:
        return {
            "backend": "gpu",
            "device": self.device,
            "n": self.n,
            "nnz": self.nnz,
            "weight_norm": self.weight_norm,
            "dt_ms": self.dt,
            "load_sec": self._load_sec,
            "sign_sec": self._sign_sec,
            "upload_sec": self._upload_sec,
        }
