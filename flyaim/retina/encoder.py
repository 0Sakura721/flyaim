"""人工复眼前端(Retina):靶场像素 -> 真实感光细胞群的驱动。

===============================================================================
1. 定位:不是"假眼睛",而是"真实小眼阵列的照度驱动器"
===============================================================================
MaleCNS v1.0 **确实包含 6,098 个复眼感光细胞**(`superclass == 'ol_sensory'`,
见 CONTRACT.md 第 0 节):R1-R6 3,377 个(运动/明暗通路)、R7* 1,385、R8* 1,329
(色觉)、R7R8_unclear 85、HBeyelet 7。因此本模块的职责不是"虚构一个视觉系统",
而是**把靶场画面映射到真实感光细胞群的照度驱动**:

    R1-R6 群(运动/明暗)  <- 亮度通道(本模块计算)
    R7 / R8 群(色觉)      <- 红-绿对立通道(本模块计算)

未标注 type 的输入神经元(或类型信息不可用时)统一按亮度通道驱动。

**注意:数据集不含 lamina / medulla。** 感光细胞的下游(L1/L2/L3、T4/T5)与
真实视拓扑都不在数据里,因此:
    - ON/OFF 分离**不可能**在感光细胞层面复刻(真实 R1-R6 全是去极化型,
      分 ON/OFF 的是下游 L1/L2/L3);
    - 本实现按契约把 ON(照度上升)放在 `[:,0]`(兴奋驱动)、OFF(照度下降)放在
      `[:,1]`(抑制驱动),让 OFF 通路由连接组内部的抑制性中间神经元自然形成。
      这是数据可用性下的妥协,不是复刻 lamina 分层。

===============================================================================
2. 流水线(每帧)与生物依据
===============================================================================
(1) 小眼阵列采样:复眼是空间采样阵列,每个小眼通过晶锥/感光小体对视野一小块积分
    (接受角约 1-5 度),构成空间低通 + 抗混叠。
    -> 分块均值得到 (eye_rows, eye_cols) 小眼阵列。
    默认 24x32 = 768 个小眼,与真实果蝇复眼约 750-800 个小眼**同量级**
    (数据集 6098/8 = 762 个小眼,每个小眼 6 个 R1-R6 + 2 个 R7/R8),因此默认
    分辨率不是随意取的。
(2) 视拓扑:每个感光细胞 -> 一个小眼。若 neuron_index 提供 `assignedOlHex1/2`
    (小眼六边形坐标),用**排序归一化**把六边形坐标映射到小眼网格(真实视拓扑);
    否则退化为按索引块的确定性映射(无生物学含义,会在 describe() 里标注)。
    朝向/翻转无法标定:数据集不含语义朝向信息,任何朝向假设都是任意的。
(3) 光电转换压缩:感光细胞响应近似 Weber-Fechner 幂律压缩 -> gamma=0.5
    (模块常量 `_GAMMA`),在**每个感光通道上分别压缩后再做对立**(生理上对立即
     receptor adaptation 之后产生)。
(4) 色觉对立:R7 为 UV/绿敏感、R8 为绿/蓝敏感,真实光谱敏感曲线不在数据集里。
    本实现用靶场的 R/G 通道构造红-绿对立 `C = <R**g> - <G**g>`(小眼内均值),
    R7 群取正极(红兴奋)、R8 群取负极(绿兴奋),构成一对反向对立通道。
    **这是面向靶场配色的工程约定,不是生理测量值。**
(5) 侧抑制(lamina 中心-周边拮抗):LMC 经 L2/L4 接收邻近小眼抑制,呈 DoG 样拮抗
    -> `lateral_inhibition` 用 3x3 邻域(不含中心)均值做差。
(6) 广域抑制 / 光适应:复眼后存在宽场抑制池,响应围绕场景平均亮度归一化
    -> 减去整幅视野的空间均值(**空间直流去除**)。它让暗背景不产生驱动,
    只有目标/准星这类局部对比结构才有响应。功能性近似,非逐突触复刻。
(7) 时间差分 / 运动敏感:lamina/medulla 存在瞬变(时间高通)通路,T4/T5 是方向
    选择性运动检测器 -> `temporal_diff` 在静态拮抗响应与相邻帧差分之间混合。
(8) 对比度增益控制 -> `gain_mode`("fixed_ref" 默认/"unit"/"frame_rms"),
    再裁剪到 [0, `max_drive_hz`]。

===============================================================================
3. 生物真实性自评(诚实版,勿夸大)
===============================================================================
**像真的部分(比"手搓假眼睛"实质更强):**
    - 驱动的是 **6,098 个真实感光细胞索引**,且按 type 分成 R1-R6(运动/明暗)与
      R7/R8(色觉)两条真实存在的通道;
    - 小眼数(768)与真实复眼小眼数量级一致(数据 6098/8≈762);
    - 空间采样 + 接受角低通、中心-周边侧抑制、广域适应、时间高通的存在性;
    - 有 `assignedOlHex1/2` 时使用真实小眼六边形坐标(真实视拓扑)。
**明显不真实的部分:**
    - **没有方向选择性**:真实 T4/T5 是 4 方向 EMD,本实现只有无方向的时间差分,
      任何方向的运动都激励同一批神经元(T4/T5 在数据集里存在,但本模块不直接驱动
      它们——驱动它们需要真实 lamina 接线,数据里没有);
    - **没有真实光谱敏感曲线**:R7/R8 的红-绿对立是用靶场 RGB 直接构造的;
    - 没有 lamina 的六角形 cartridge 结构与分层延迟,ON/OFF 只是同一神经元的双通道;
    - 没有 lamina 单极细胞层:真实感光细胞的下游是组胺能抑制性 LMC,本实现把
      感光细胞直接当作驱动源,等于假设了"感光细胞 -> 中央脑"的直连(不真实);
    - 层数/时间常数/增益都是工程标定值,不是生理测量。
**结论:** 这是一个"驱动真实感光细胞群、带真实小眼拓扑(若可用)的生物启发前端",
比虚构视觉系统强得多,但**仍不是复蝇眼仿真**:无方向选择性、无真实光谱、无 lamina。

===============================================================================
4. 契约
===============================================================================
- `frame_to_spikes` **只接收像素**(CONTRACT 禁止事项 5.1)。本模块不 import arena,
  不接收目标坐标/标签/准星位置,没有任何 target 参数。
- 签名:`Retina(cfg, input_neuron_ids=None)`;`input_neuron_ids` 为空/None 时
  优雅回退到 `arange(eye_rows*eye_cols)` 并在 `describe()` 里置
  `input_neuron_ids_fallback=True`。
- 仅在本模块内计算,不修改任何连接组权重(禁止事项 5.2)。
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
from scipy import ndimage

from flyaim.config import RetinaConfig

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------ 固定设计常量
_GAMMA: float = 0.5
"""光电转换压缩指数(幂律),作用于每个感光通道。"""

_REFERENCE_CONTRAST: float = 0.35
"""gain_mode="fixed_ref" 时亮度通道的参考对比度(把 0.35 的拮抗响应映射到满量程)。

取值依据:靶场深色背景(亮度约 0.07)+ 明亮目标(约 0.47),经 gamma 压缩与
侧抑制后目标处响应幅度约 0.3-0.4。**这是面向本靶场的标定常数,不是生理测量值**,
可用 `cfg.reference_contrast` 覆盖。
"""

_REFERENCE_CONTRAST_COLOR: float = 0.70
"""色觉通道的参考对比度。靶场红靶 (235,70,70) 与暗背景 (18,18,22) 的
红-绿对立幅度约 0.65,故取 0.70 让目标映射到接近满量程。同样可用
`cfg.reference_contrast_color` 覆盖。"""

_RMS_FLOOR: float = 0.02
"""gain_mode="frame_rms" 时 RMS 下限,防止空白帧放大噪声。"""

_FRAME_MS_NOMINAL: float = 1000.0 / 30.0
"""encoding="spike" 时假定的帧间隔(30 FPS)。"""

_NEIGHBOR_KERNEL = np.array(
    [[1.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 1.0]], dtype=np.float32
) / 8.0
"""3x3 邻域均值核(不含中心),用于侧抑制。"""

# type 前缀 -> 通道。顺序敏感:先匹配 R1-R6,再 R7/R8。
_LUM_PREFIXES = ("R1-R6", "R1_R6", "R1R6", "HB", "R1-6")
_R7_PREFIX = "R7"
_R8_PREFIX = "R8"

_DEFAULT_INDEX_RELPATH = Path("flyaim") / "data" / "build" / "neuron_index.parquet"


def _is_luminance_type(t: str) -> bool:
    u = t.strip().upper()
    if not u:
        return True  # 未知类型 -> 亮度通道
    return any(u.startswith(p) for p in _LUM_PREFIXES)


def _is_r7_type(t: str) -> bool:
    return t.strip().upper().startswith(_R7_PREFIX)


def _is_r8_type(t: str) -> bool:
    return t.strip().upper().startswith(_R8_PREFIX)


class Retina:
    """复眼前端:靶场帧 -> (n_input, 2) 的 ON/OFF 驱动。

    典型用法(集成层)::

        retina = Retina(cfg, input_neuron_ids=roles.visual_input)  # 6098 个感光细胞
        drive = retina.frame_to_spikes(frame)   # (n_input, 2) float32
        # drive[:, 0] -> Connectome.step 的 in_exc(兴奋/ON)
        # drive[:, 1] -> Connectome.step 的 in_inh(抑制/OFF)

    行顺序与 `self.input_neuron_ids` **严格一一对应**;类型分组在构造时确定。
    """

    def __init__(
        self,
        cfg: RetinaConfig,
        input_neuron_ids: np.ndarray | None = None,
        types: np.ndarray | None = None,
        ol_hex: np.ndarray | None = None,
        neuron_index: Any | None = None,
    ) -> None:
        """构造编码器。

        cfg: RetinaConfig。
        input_neuron_ids: (n_input,) 全局索引,通常 = roles.visual_input(6098 个
            真实感光细胞)。为空/None 时回退到 arange(eye_rows*eye_cols) 并置
            input_neuron_ids_fallback=True。
        types: (n_input,) 细胞 type 字符串(可选)。缺省时按顺序从
            `cfg.input_types` -> `cfg.neuron_index` -> `neuron_index` 参数 ->
            `flyaim/data/build/neuron_index.parquet` 自动解析;都拿不到就全部按
            亮度通道驱动并在 describe() 标注。
        ol_hex: (n_input, 2) 小眼六边形坐标(可选,= assignedOlHex1/2)。
            缺省时自动从 neuron_index 的 hex 列解析;拿不到就退化为索引块映射。
        neuron_index: flyaim.neuron_index.NeuronIndex 对象或 parquet 路径(可选)。
        """
        self.cfg = cfg

        self.eye_rows = int(cfg.eye_rows)
        self.eye_cols = int(cfg.eye_cols)
        if self.eye_rows <= 0 or self.eye_cols <= 0:
            raise ValueError(f"eye_rows/eye_cols 必须为正: {self.eye_rows}x{self.eye_cols}")
        self.n_ommatidia = self.eye_rows * self.eye_cols

        w = np.asarray(cfg.luminance_weights, dtype=np.float32).reshape(-1)
        if w.size != 3:
            raise ValueError(f"luminance_weights 必须是 3 元组,得到 {w.shape}")
        s = float(w.sum())
        if s <= 0:
            raise ValueError("luminance_weights 之和必须为正")
        self._lum_w = (w / s).astype(np.float32)

        self.encoding = str(cfg.encoding)
        if self.encoding not in ("rate", "spike"):
            raise ValueError(f"未知 encoding: {self.encoding}")
        self.max_drive_hz = float(cfg.max_drive_hz)
        self.temporal_diff = float(np.clip(cfg.temporal_diff, 0.0, 1.0))
        self.lateral_inhibition = float(max(cfg.lateral_inhibition, 0.0))
        self.use_on_off = bool(cfg.use_on_off)
        self._ref_lum = float(getattr(cfg, "reference_contrast", _REFERENCE_CONTRAST))
        self._ref_col = float(
            getattr(cfg, "reference_contrast_color", _REFERENCE_CONTRAST_COLOR)
        )
        self._gain_mode = str(getattr(cfg, "gain_mode", "fixed_ref"))
        if self._gain_mode not in ("fixed_ref", "unit", "frame_rms"):
            raise ValueError(f"未知 gain_mode: {self._gain_mode}")

        # ---------------------------------------------------------- 输入索引
        self.input_neuron_ids_fallback = False
        ids = np.asarray([] if input_neuron_ids is None else input_neuron_ids)
        if ids.size == 0:
            # MaleCNS 有 6098 个真实感光细胞,正常情况下不会走到这里;
            # 空输入是**必须优雅处理的降级路径**(见 CONTRACT 第 1 节降级要求)。
            ids = np.arange(self.n_ommatidia, dtype=np.int64)
            self.input_neuron_ids_fallback = True
            logger.warning(
                "Retina: input_neuron_ids 为空/None,回退到 arange(%d);"
                "manifest 必须标记 visual_input_fallback=true",
                self.n_ommatidia,
            )
        ids = np.asarray(ids).reshape(-1)
        if not np.issubdtype(ids.dtype, np.integer):
            if not np.all(np.isfinite(ids)):
                raise ValueError("input_neuron_ids 含非有限值")
            ids = ids.astype(np.int64)
        ids = ids.astype(np.int64, copy=True)
        if ids.min() < 0:
            raise ValueError("input_neuron_ids 含负索引")
        if np.unique(ids).size != ids.size:
            raise ValueError("input_neuron_ids 含重复索引(角色选择有 bug?)")
        self.input_neuron_ids = ids
        n_input = int(ids.size)

        # -------------------------------------------------- 类型 / 视拓扑解析
        ni = self._resolve_neuron_index(cfg, neuron_index)
        types_arr, self._types_source = self._resolve_types(types, ni, n_input)
        hex_arr, self._hex_source = self._resolve_hex(ol_hex, ni, n_input, cfg)

        is_lum = np.array([_is_luminance_type(t) for t in types_arr], dtype=bool)
        is_r7 = np.array([_is_r7_type(t) for t in types_arr], dtype=bool) & ~is_lum
        is_r8 = np.array([_is_r8_type(t) for t in types_arr], dtype=bool) & ~is_lum & ~is_r7
        rest = ~(is_lum | is_r7 | is_r8)
        if rest.any():
            # 既不是 R1-R6 也不是 R7/R8(如 HBeyelet 之外的未知类型)-> 并入亮度通道
            is_lum = is_lum | rest
            self._types_source += "+unknown_as_luminance"

        self._rows_lum = np.flatnonzero(is_lum).astype(np.int64)
        self._rows_r7 = np.flatnonzero(is_r7).astype(np.int64)
        self._rows_r8 = np.flatnonzero(is_r8).astype(np.int64)

        self._cell_lum, hx_lum = self._make_cells(self._rows_lum.size, hex_arr, self._rows_lum)
        self._cell_r7, hx_r7 = self._make_cells(self._rows_r7.size, hex_arr, self._rows_r7)
        self._cell_r8, hx_r8 = self._make_cells(self._rows_r8.size, hex_arr, self._rows_r8)
        self._retinotopy_used = (
            "hex_coords(assignedOlHex 排序归一化到小眼网格)"
            if (hx_lum or hx_r7 or hx_r8)
            else "block_index(无可用小眼坐标的确定性块映射,无生物学含义)"
        )

        # ---------------------------------------------------------- 运行时状态
        self._prev_opp: dict[str, np.ndarray] = {
            "lum": np.zeros((self.eye_rows, self.eye_cols), dtype=np.float32),
            "col": np.zeros((self.eye_rows, self.eye_cols), dtype=np.float32),
        }
        self._has_prev = False
        self._cell_cache: dict[tuple[int, int], np.ndarray] = {}
        self._size_cache: dict[tuple[int, int], np.ndarray] = {}
        self._frames = 0
        self._warned_gray = False
        self._seed = int(getattr(cfg, "seed", 0))
        self._rng = np.random.default_rng(self._seed)

    # ------------------------------------------------------------------ 元数据解析

    @staticmethod
    def _resolve_neuron_index(cfg: RetinaConfig, neuron_index: Any) -> Any | None:
        """按 参数 -> cfg.neuron_index -> cfg.data_dir/neuron_index.parquet -> 默认路径 解析。"""
        from flyaim.neuron_index import NeuronIndex  # 局部 import,避免无谓依赖

        cand: list[Any] = []
        if neuron_index is not None:
            cand.append(neuron_index)
        cfg_ni = getattr(cfg, "neuron_index", None)
        if cfg_ni is not None:
            cand.append(cfg_ni)
        data_dir = getattr(cfg, "data_dir", None)
        if data_dir:
            cand.append(Path(str(data_dir)) / "neuron_index.parquet")
        cand.append(_DEFAULT_INDEX_RELPATH)
        cand.append(Path(__file__).resolve().parents[2] / _DEFAULT_INDEX_RELPATH)

        for c in cand:
            try:
                if isinstance(c, NeuronIndex):
                    return c
                p = Path(str(c))
                if p.exists():
                    return NeuronIndex.load(p)
            except Exception as exc:  # 索引不可用不应阻断编码器构造
                logger.warning("Retina: 加载 neuron_index 失败(%s): %s", c, exc)
        return None

    def _resolve_types(
        self, types: np.ndarray | None, ni: Any | None, n_input: int
    ) -> tuple[np.ndarray, str]:
        if types is not None:
            a = self._as_str_array(types, n_input)
            if a is not None:
                return a, "kwarg"
        cfg_types = getattr(self.cfg, "input_types", None)
        if cfg_types is not None:
            a = self._as_str_array(cfg_types, n_input)
            if a is not None:
                return a, "cfg.input_types"
        if ni is not None and "type" in getattr(ni, "df", None).columns:
            try:
                t = ni.df["type"].astype(str).to_numpy()
                if t.size > int(self.input_neuron_ids.max()):
                    return t[self.input_neuron_ids], "neuron_index.type"
            except Exception as exc:
                logger.warning("Retina: 读取 type 列失败: %s", exc)
        return np.array([""] * n_input, dtype=object), "unavailable"

    def _resolve_hex(
        self,
        ol_hex: np.ndarray | None,
        ni: Any | None,
        n_input: int,
        cfg: RetinaConfig,
    ) -> tuple[np.ndarray | None, str]:
        if ol_hex is not None:
            a = self._as_hex_array(ol_hex, n_input)
            if a is not None:
                return a, "kwarg"
        cfg_hex = getattr(cfg, "ol_hex", None)
        if cfg_hex is not None:
            a = self._as_hex_array(cfg_hex, n_input)
            if a is not None:
                return a, "cfg.ol_hex"
        df = getattr(ni, "df", None)
        if df is not None:
            cols = {str(c).lower(): str(c) for c in df.columns}
            c1 = cols.get("assignedolhex1") or cols.get("assigned_ol_hex1")
            c2 = cols.get("assignedolhex2") or cols.get("assigned_ol_hex2")
            if c1 and c2:
                try:
                    arr = df[[c1, c2]].to_numpy(dtype=np.float64)
                    if arr.shape[0] > int(self.input_neuron_ids.max()):
                        return arr[self.input_neuron_ids], "neuron_index.assignedOlHex1/2"
                except Exception as exc:
                    logger.warning("Retina: 读取 assignedOlHex 列失败: %s", exc)
        return None, "unavailable"

    @staticmethod
    def _as_str_array(x: Any, n: int) -> np.ndarray | None:
        try:
            a = np.asarray(x, dtype=object).reshape(-1)
        except Exception:
            return None
        if a.size != n:
            logger.warning("Retina: types 长度 %d != n_input %d,忽略", a.size, n)
            return None
        return np.array(["" if v is None else str(v) for v in a], dtype=object)

    @staticmethod
    def _as_hex_array(x: Any, n: int) -> np.ndarray | None:
        try:
            a = np.asarray(x, dtype=np.float64)
        except Exception:
            return None
        if a.ndim != 2 or a.shape[1] < 2 or a.shape[0] != n:
            logger.warning("Retina: ol_hex 形状 %s 与 n_input=%d 不符,忽略", a.shape, n)
            return None
        return a[:, :2]

    # ------------------------------------------------------------------ 视拓扑映射

    def _make_cells(
        self, count: int, hex_arr: np.ndarray | None, rows: np.ndarray
    ) -> tuple[np.ndarray, bool]:
        """把一组感光细胞映射到小眼网格 (eye_rows, eye_cols) 的展平单元格索引。

        优先用真实小眼六边形坐标(排序归一化 -> 网格),否则退化为索引块映射。
        块映射:`cell = (i * n_om // count)`,完全确定,每个小眼分到
        count/n_om 个感光细胞(冗余),无生物学含义。

        返回 (单元格索引, 是否真的用了六边形坐标)。注意:即使 neuron_index 里
        存在 assignedOlHex 列,感光细胞自身通常是 NaN(实测 MaleCNS:该列只对
        ol_intrinsic 有值),此时会**逐组回退**到块映射,describe() 会如实标注。
        """
        if count <= 0:
            return np.empty(0, dtype=np.int64), False
        block = (np.arange(count, dtype=np.int64) * self.n_ommatidia) // count
        if hex_arr is None:
            return block, False
        c1 = hex_arr[rows, 0]
        c2 = hex_arr[rows, 1]
        ok = np.isfinite(c1) & np.isfinite(c2)
        if ok.mean() < 0.9:
            return block, False
        if np.unique(c1[ok]).size < 2 or np.unique(c2[ok]).size < 2:
            return block, False
        out = block.copy()
        r = self._rank01(c1[ok]) * (self.eye_rows - 1) if self.eye_rows > 1 else np.zeros(ok.sum())
        c = self._rank01(c2[ok]) * (self.eye_cols - 1) if self.eye_cols > 1 else np.zeros(ok.sum())
        rr = np.clip(np.rint(r).astype(np.int64), 0, self.eye_rows - 1)
        cc = np.clip(np.rint(c).astype(np.int64), 0, self.eye_cols - 1)
        out[ok] = rr * self.eye_cols + cc
        return out, True

    @staticmethod
    def _rank01(v: np.ndarray) -> np.ndarray:
        """排序归一化到 [0,1],对任意量纲/取值范围的坐标都稳健。"""
        if v.size <= 1:
            return np.zeros(v.size, dtype=np.float64)
        order = np.argsort(np.argsort(v, kind="stable"), kind="stable").astype(np.float64)
        return order / float(v.size - 1)

    # ------------------------------------------------------------------ 主接口

    def frame_to_spikes(self, frame: np.ndarray) -> np.ndarray:
        """靶场帧 -> 感光细胞的驱动。

        frame: (H, W, 3) uint8(也容忍 (H,W) 灰度与 float 帧,自动归一化)。
        return: (n_input, 2) float32
                [:, 0] = excitatory drive (>=0),ON / 照度上升通道
                [:, 1] = inhibitory drive (>=0),OFF / 照度下降通道
        行顺序与 self.input_neuron_ids 严格对应。

        只使用像素,不使用任何目标坐标/标签(禁止事项 5.1)。
        """
        gray, chroma = self._to_channels(frame)
        h, w = gray.shape
        center_l = np.power(np.clip(self._ommatidial_mean(gray, h, w), 0.0, None), _GAMMA)
        if chroma is None:
            signed_c = np.zeros((self.eye_rows, self.eye_cols), dtype=np.float32)
        else:
            rc, gc = chroma
            center_r = np.power(
                np.clip(self._ommatidial_mean(rc, h, w), 0.0, None), _GAMMA
            )
            center_g = np.power(
                np.clip(self._ommatidial_mean(gc, h, w), 0.0, None), _GAMMA
            )
            signed_c = (center_r - center_g).astype(np.float32)

        s_lum = self._channel_response(center_l, "lum")
        exc_l, inh_l = self._rectify(s_lum, self._ref_lum)
        s_col = self._channel_response(signed_c, "col")
        exc_c, inh_c = self._rectify(s_col, self._ref_col)

        out = np.zeros((int(self.input_neuron_ids.size), 2), dtype=np.float32)
        exc_l_f, inh_l_f = exc_l.reshape(-1), inh_l.reshape(-1)
        exc_c_f, inh_c_f = exc_c.reshape(-1), inh_c.reshape(-1)
        if self._rows_lum.size:
            out[self._rows_lum, 0] = exc_l_f[self._cell_lum]
            out[self._rows_lum, 1] = inh_l_f[self._cell_lum]
        if self._rows_r7.size:  # R7 群:红/长波兴奋
            out[self._rows_r7, 0] = exc_c_f[self._cell_r7]
            out[self._rows_r7, 1] = inh_c_f[self._cell_r7]
        if self._rows_r8.size:  # R8 群:绿/短波兴奋(对立极性翻转)
            out[self._rows_r8, 0] = inh_c_f[self._cell_r8]
            out[self._rows_r8, 1] = exc_c_f[self._cell_r8]

        if self.encoding == "spike":
            out = self._poisson_gate(out)

        self._frames += 1
        return out

    def reset(self) -> None:
        """清空时间差分状态(每个 episode 开始时调用)。"""
        for k in self._prev_opp:
            self._prev_opp[k] = np.zeros((self.eye_rows, self.eye_cols), dtype=np.float32)
        self._has_prev = False
        self._frames = 0
        self._rng = np.random.default_rng(self._seed)

    def describe(self) -> dict:
        """编码器配置、输入规模与真实性自评(供 manifest / 报告使用)。"""
        n_lum = int(self._rows_lum.size)
        n_r7 = int(self._rows_r7.size)
        n_r8 = int(self._rows_r8.size)
        return {
            "eye_rows": self.eye_rows,
            "eye_cols": self.eye_cols,
            "n_ommatidia": int(self.n_ommatidia),
            "n_input": int(self.input_neuron_ids.size),
            "channel_map": {
                "luminance_R1R6": n_lum,
                "color_R7_red_excited": n_r7,
                "color_R8_green_excited": n_r8,
            },
            "photoreceptors_per_ommatidium": (
                round(float(self.input_neuron_ids.size) / self.n_ommatidia, 3)
            ),
            "types_source": self._types_source,
            "retinotopy": self._retinotopy_used,
            "retinotopy_source": self._hex_source,
            "use_on_off": bool(self.use_on_off),
            "lateral_inhibition": self.lateral_inhibition,
            "temporal_diff": self.temporal_diff,
            "encoding": self.encoding,
            "max_drive_hz": self.max_drive_hz,
            "luminance_weights": [float(x) for x in self._lum_w],
            "gain_mode": self._gain_mode,
            "reference_contrast": self._ref_lum,
            "reference_contrast_color": self._ref_col,
            "gamma": _GAMMA,
            "spatial_dc_removal": True,
            "uses_target_coordinates": False,  # 禁止事项 5.1:仅像素输入
            "input_neuron_ids_fallback": bool(self.input_neuron_ids_fallback),
            "input_neuron_ids_head": [int(x) for x in self.input_neuron_ids[:8]],
            "frames_processed": int(self._frames),
            "biological_fidelity": (
                "驱动的是真实感光细胞群(R1-R6 运动/明暗 + R7/R8 色觉对立),"
                "小眼数 768 与真实复眼同量级;但**无方向选择性**(无 EMD,数据不含 lamina 接线)、"
                "无真实光谱敏感曲线、无 lamina 分层与 LMC 中间层。详见模块 docstring 第 3 节。"
            ),
        }

    # ------------------------------------------------------------------ 内部流水线

    def _to_channels(
        self, frame: np.ndarray
    ) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray] | None]:
        """帧 -> (亮度 [0,1], (R [0,1], G [0,1]) 或 None)。"""
        a = np.asarray(frame)
        if a.ndim == 2:
            g = a.astype(np.float32)
            return self._normalize(g, a.dtype), None
        if a.ndim != 3:
            raise ValueError(f"frame 必须是 (H,W) 或 (H,W,3),得到 shape={a.shape}")
        if a.shape[2] < 3:
            g = a[..., 0].astype(np.float32)
            if not self._warned_gray:
                self._warned_gray = True
                logger.warning("Retina: 帧缺少 RGB 通道,色觉通道(R7/R8)驱动置零")
            return self._normalize(g, a.dtype), None
        rgb = a[..., :3].astype(np.float32)
        gray = rgb @ self._lum_w
        gray = self._normalize(gray, a.dtype)
        r = self._normalize(rgb[..., 0], a.dtype)
        g = self._normalize(rgb[..., 1], a.dtype)
        return gray, (r, g)

    @staticmethod
    def _normalize(g: np.ndarray, src_dtype: Any) -> np.ndarray:
        """整数帧按 255 归一化;浮点帧按值域判断。裁剪到 [0,1] 并去 NaN。"""
        if not np.issubdtype(np.dtype(src_dtype), np.floating):
            g = g / 255.0
        elif g.size and float(np.nanmax(g)) > 1.5:
            g = g / 255.0
        return np.nan_to_num(np.clip(g, 0.0, 1.0), nan=0.0, posinf=1.0, neginf=0.0)

    def _ommatidial_mean(self, gray: np.ndarray, h: int, w: int) -> np.ndarray:
        """每个小眼的接受角积分(分块均值 = 空间低通 + 降采样)。"""
        r, c = self.eye_rows, self.eye_cols
        if h % r == 0 and w % c == 0:  # 快路径(靶场 480x640 / 24x32 走这里)
            bh, bw = h // r, w // c
            return gray.reshape(r, bh, c, bw).mean(axis=(1, 3), dtype=np.float32).astype(
                np.float32
            )
        key = (h, w)
        idx = self._cell_cache.get(key)
        if idx is None:
            rows = (np.arange(h, dtype=np.int64) * r) // h
            cols = (np.arange(w, dtype=np.int64) * c) // w
            idx = (rows[:, None] * c + cols[None, :]).astype(np.int64)
            if len(self._cell_cache) > 4:
                self._cell_cache.clear()
            self._cell_cache[key] = idx
        cnt = self._size_cache.get(key)
        if cnt is None:
            cnt = np.bincount(idx.reshape(-1), minlength=self.n_ommatidia).astype(np.float32)
            cnt[cnt == 0] = 1.0
            self._size_cache[key] = cnt
        sums = np.bincount(
            idx.reshape(-1),
            weights=gray.reshape(-1).astype(np.float64),
            minlength=self.n_ommatidia,
        )
        return (sums.astype(np.float32) / cnt).reshape(r, c)

    def _channel_response(self, center: np.ndarray, key: str) -> np.ndarray:
        """小眼通道照度 -> 有符号响应 s(已做侧抑制、广域直流去除、时间差分)。"""
        x = center.astype(np.float32)
        if self.lateral_inhibition > 0.0:  # 侧抑制 / 中心-周边拮抗
            surround = ndimage.convolve(x, _NEIGHBOR_KERNEL, mode="nearest")
            opp = x - self.lateral_inhibition * surround
        else:
            opp = x
        opp = (opp - opp.mean(dtype=np.float32)).astype(np.float32)  # 广域抑制(适应)
        if self._has_prev and self.temporal_diff > 0.0:  # 时间高通(运动敏感)
            d = opp - self._prev_opp[key]
        else:
            d = np.zeros_like(opp)
        self._prev_opp[key] = opp
        return ((1.0 - self.temporal_diff) * opp + self.temporal_diff * d).astype(np.float32)

    def _rectify(self, s: np.ndarray, ref: float) -> tuple[np.ndarray, np.ndarray]:
        """有符号响应 -> (ON 驱动 Hz, OFF 驱动 Hz)。

        `use_on_off=False` 时只保留 ON(照度上升)通道,OFF 通道恒 0 —— 对应
        "没有明暗双通路"的消融条件。
        """
        g = self._gain_factor(s, ref)
        on = np.clip(np.maximum(s, 0.0) * g, 0.0, 1.0) * self.max_drive_hz
        if self.use_on_off:
            off = np.clip(np.maximum(-s, 0.0) * g, 0.0, 1.0) * self.max_drive_hz
        else:
            off = np.zeros_like(on)
        return on.astype(np.float32), off.astype(np.float32)

    def _gain_factor(self, s: np.ndarray, ref: float) -> float:
        if self._gain_mode == "unit":
            return 1.0
        if self._gain_mode == "frame_rms":
            rms = float(np.sqrt(np.mean(np.square(s, dtype=np.float64))))
            return float(0.5 / max(rms, _RMS_FLOOR))
        return float(1.0 / max(ref, 1e-6))

    def _poisson_gate(self, drive: np.ndarray) -> np.ndarray:
        """encoding="spike":按 p = rate * frame_dt 伯努利发放,否则输出 0。

        帧间隔是固定常量(30 FPS,RetinaConfig 不含 dt);默认 encoding="rate" 不走这里。
        """
        p = np.clip(drive.astype(np.float32) * (_FRAME_MS_NOMINAL / 1000.0), 0.0, 1.0)
        fired = self._rng.random(p.shape) < p
        return (drive * fired).astype(np.float32)

    # ------------------------------------------------------------------ 诊断

    def drive_stats(self, drive: np.ndarray) -> dict:
        """单帧驱动统计量,便于标定与报告。"""
        d = np.asarray(drive, dtype=np.float32)
        if d.size == 0:
            return {}
        return {
            "exc_nonzero_frac": float(np.mean(d[:, 0] > 0)),
            "inh_nonzero_frac": float(np.mean(d[:, 1] > 0)),
            "exc_mean_hz": float(d[:, 0].mean()),
            "exc_max_hz": float(d[:, 0].max()),
            "inh_mean_hz": float(d[:, 1].mean()),
            "inh_max_hz": float(d[:, 1].max()),
        }

    def __repr__(self) -> str:  # pragma: no cover - 诊断用
        return (
            f"Retina({self.eye_rows}x{self.eye_cols} ommatidia, n_input={self.input_neuron_ids.size}, "
            f"lum={self._rows_lum.size}, R7={self._rows_r7.size}, R8={self._rows_r8.size}, "
            f"retinotopy={self._hex_source}, fallback={self.input_neuron_ids_fallback})"
        )


def make_retina(cfg: RetinaConfig, roles: Any | None = None, **kwargs: Any) -> Retina:
    """便捷构造:从 RoleSelection(或 ids 数组)取视觉输入群。"""
    ids = None
    if roles is not None:
        ids = getattr(roles, "visual_input", roles)
    return Retina(cfg, input_neuron_ids=ids, **kwargs)
