"""FlyAim 配置数据类。

所有可调参数集中在此,便于把 config 原样写进 runs/<ts>/config.json 以保证可复现。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Literal, Tuple

# ---------------------------------------------------------------- 视网膜前端


@dataclass
class RetinaConfig:
    """人工复眼参数。

    MaleCNS 不含复眼/视叶,因此本前端是**人为设计**的,不是生物真实接线。
    这一事实会写进 manifest.json 的 visual_input_fallback 字段。
    """

    # 视野降采样:靶场帧 -> 小眼阵列
    eye_rows: int = 24
    eye_cols: int = 32
    # ON/OFF 双通路
    use_on_off: bool = True
    # 侧抑制强度(模拟 lamina 的邻近抑制),0 表示关闭
    lateral_inhibition: float = 0.25
    # 时间差分系数:0=纯静态照度,1=纯运动敏感
    temporal_diff: float = 0.7
    # 编码方式
    encoding: Literal["rate", "spike"] = "rate"
    # 每帧最大驱动强度(Hz 等效)
    max_drive_hz: float = 200.0
    # 灰度化权重
    luminance_weights: Tuple[float, float, float] = (0.299, 0.587, 0.114)


# ---------------------------------------------------------------- 神经仿真


@dataclass
class BrainConfig:
    """LIF 连接组仿真参数。

    ⚠️ 关于 `dt_ms` 与 `steps_per_frame`(实测标定,见 tools/bench_timestep.py)
    ------------------------------------------------------------------
    `tau_m_ms = 20`,所以 `dt_ms = 1.0` 相当于每步只推进 tau 的 **1/20**
    —— 对 LIF 是严重过采样,而单步成本几乎全在**脉冲驱动稀疏传播**上
    (随活跃神经元数增长),于是过采样直接变成 CPU 开销。

    实测(全量 166,700,工作点,每帧模拟 32 ms 不变):

        | dt / steps      | 墙钟 ms/帧 | DN Hz | DN 活跃 | 提速  |
        |-----------------|-----------|-------|--------|-------|
        | 1.0 ms / 33 步  |  1695.5   | 94.21 |  1212  | 1.00x |
        | 2.0 ms / 16 步  |   724.9   | 83.84 |  1259  | 2.34x |
        | 4.0 ms /  8 步  |   353.9   | 90.76 |  1302  | 4.79x |
        | 10.0 ms / 3 步  |   143.0   | 37.34 |  1312  | 11.85x(保真度崩) |

    → 默认取 **dt=4ms / 8 步**:4.79x 提速,每帧模拟时间不变(32 ms),
      DN 放电率偏差仅 -3.7%。`dt=10ms` 时 dt/tau=0.5,Euler 积分过粗,
      DN 掉到 37 Hz(-60%),**不可用**。

    注意:`dt_ms` 默认值改动会影响与早期实验记录的数值可比性,
    复现历史结果请显式传 `dt_ms=1.0, steps_per_frame=33`。
    """

    dt_ms: float = 4.0
    tau_m_ms: float = 20.0
    v_rest: float = 0.0
    v_thresh: float = 1.0
    v_reset: float = 0.0
    refractory_ms: float = 2.0
    # 突触权重缩放:把突触计数压到合理电导范围
    weight_scale_exc: float = 0.02
    weight_scale_inh: float = 0.02
    # 权重归一化策略。默认 "none" 保持向后兼容(原始突触计数)。
    # **实测关键**(见 tools/test_normalization.py):
    #   raw(= none)下网络只有「全静默」或「癫痫饱和」两个极端,无可用中间态;
    #   "indeg"(每行除以入度)能把工作点变成连续可调,
    #   weight_scale=16 时 DN 达 14.7 Hz / 572 活跃,属生理区间。
    # "wsum" = 除以行权重和;"wsum_sep" = 兴奋/抑制两张矩阵各自按行和归一化。
    weight_norm: str = "none"   # none | indeg | wsum | wsum_sep
    # 外部驱动(感光细胞注入)缩放。
    # **必须与 weight_scale_* 分开设置**:外部驱动来自视网膜编码器,
    # 其量纲是"Hz 等效驱动"(RetinaConfig.max_drive_hz),不是突触计数。
    # 实测标定(全量 166,700,dv/step = dt/tau = 0.05):
    #     驱动峰值 ~68 Hz 时,若要稳态 v ≈ 3×v_thresh 以保证可靠发放,
    #     需要 input_gain ≈ 3.0/v_thresh 量级 → 取 1.5 使 68 Hz 驱动给出 v_ss≈2.0。
    # 若取成 weight_scale_exc(0.02),I = 0.02*0.05*68 = 0.068,
    # 稳态 v = 1.36 仅逼近阈值,神经元永不发放(曾导致 1360 个 DN 全为死特征)。
    input_gain: float = 1.5
    # 仿真时间步/帧:每帧推进多少 dt。
    # 与 dt_ms 一起决定「每帧模拟多少毫秒」。默认 4ms × 8 步 = 32 ms/帧,
    # 与 30 fps 的实时帧长(33.3 ms)基本一致。
    steps_per_frame: int = 8
    # 子网选择:None=全量 166700;给定则只仿真这些索引(compact 模式)
    subnet_size: int | None = None
    subnet_seed: int = 0
    # 数值精度
    dtype: Literal["float32", "float64"] = "float32"
    # 是否启用多巴胺/神经调质注入
    use_neuromodulation: bool = False


# ---------------------------------------------------------------- 读出


@dataclass
class ReadoutConfig:
    """下行神经元 -> 鼠标增量 的读出层。

    学习只能发生在这里,绝不能改连接组权重(见 CONTRACT.md 禁止事项 2)。
    """

    mode: Literal["fixed", "trained"] = "fixed"
    # 参与读出的神经元来源
    source: Literal["dn", "dn+mn"] = "dn"
    # 固定模式的增益
    gain: float = 1.0
    # 输出低通滤波(0=无滤波),抑制抖动
    smoothing: float = 0.5
    # trained 模式
    train_seeds: Tuple[int, ...] = (0, 1, 2, 3, 4)
    eval_seeds: Tuple[int, ...] = (100, 101, 102, 103, 104)
    ridge_lambda: float = 1.0
    weights_path: str | None = None


# ---------------------------------------------------------------- 靶场


@dataclass
class ArenaConfig:
    """自建 2D 瞄准靶场参数。

    刻意保持纯几何、可确定性复现,避免依赖真实游戏的不可控因素。
    """

    width: int = 640
    height: int = 480
    # 靶
    target_radius: int = 22
    n_targets: int = 1
    target_speed_px_per_frame: float = 0.0
    # 准星
    crosshair_radius: int = 4
    # 每次 action 的移动像素(把 [-1,1] 映射到像素增量)
    speed_px_per_action: float = 14.0
    # 帧预算
    max_frames: int = 900
    # 命中后是否重置靶位
    respawn_on_hit: bool = True
    # 中心起始
    start_centered: bool = True
    # 可视背景/靶颜色
    bg_color: Tuple[int, int, int] = (18, 18, 22)
    target_color: Tuple[int, int, int] = (235, 70, 70)
    crosshair_color: Tuple[int, int, int] = (240, 240, 240)


# ---------------------------------------------------------------- 可塑性(CONTRACT 附录 P)


@dataclass
class PlasticityConfig:
    """三因子可塑性参数(CONTRACT_PLASTIC.md P2 修订 v2,预注册)。

    e_ij ← λ_e·e_ij + post_i·(pre_j − θ_pre)
    W_ij ← clip(W_ij + η·m·e_ij, 极性边界)
    """

    lambda_e: float = 0.90
    theta_pre: float = 0.10
    eta: float = 0.0          # 0 = 按预注册程序自动标定后冻结
    m_sigma_px: float = 60.0  # oracle 调制 m = exp(−err/σ)
    cap_multiple: float = 2.0  # 极性边界 = ±2×初始极值
    plastic_every: int = 1    # 每几帧更新一次
    probe_frames: int = 200   # η 与读出 g 的标定帧数
    readout_seed: int = 20261004
    readout_rate_norm: float = 50.0


# ---------------------------------------------------------------- 实验编排


@dataclass
class ExperimentConfig:
    """一次对照实验的完整设定(Phase 3 使用)。"""

    seeds: Tuple[int, ...] = tuple(range(10))
    frames_per_episode: int = 900
    arms: Tuple[str, ...] = ("fly", "shuffle", "pid", "random")
    arena: ArenaConfig = field(default_factory=ArenaConfig)
    retina: RetinaConfig = field(default_factory=RetinaConfig)
    brain: BrainConfig = field(default_factory=BrainConfig)
    readout: ReadoutConfig = field(default_factory=ReadoutConfig)
    data_dir: str = "flyaim/data/build"
    out_dir: str = "flyaim/runs"

    def to_dict(self) -> dict:
        return asdict(self)
