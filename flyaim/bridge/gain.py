"""增益层(GainModel):归一化 action [-1,1] -> 鼠标计数。

===============================================================================
为什么需要这一层(2D 语义 -> 3D 语义的转换点)
===============================================================================
离线靶场里,action 的语义是环境内建的::

    action = 1.0  ->  准星移动 14 px(ArenaConfig.speed_px_per_action)
                  ->  即扫过画面宽度的 14/640 = 2.19%

真实游戏里,注入的是**鼠标计数**,准星转过的角度由游戏灵敏度决定::

    注入 C 个计数  ->  视线旋转 C / counts_per_360 * 360 度
    视野里的靶移动 ~ 度数 / FOV * 画面宽度(小角度近似)

GainModel 保留靶场语义的**比例含义**:action=1.0 每拍扫过画面宽度同样的
比例(默认 2.19%),折算成计数就是::

    counts_per_action = speed_fraction_per_action * counts_per_360

⚠️ 两个必须诚实标注的近似:
1. 「屏幕像素比例 ≈ 角度比例」只在小角度/小位移时近似成立(透视投影的
   边缘畸变、FOV 与窗口纵横比不匹配时更差)。
2. 「训练时的每拍速度语义在游戏里保持」依赖 tick 频率与训练闭环频率一致
   (~4-10 Hz)。tick 更快时同样的 action 序列会转得更快 —— 这是语义漂移,
   调 GainConfig.speed_fraction_per_action 时必须连同 tick_hz 一起报。
这两个近似本身就是「读出语义能否跨域迁移」研究对象的一部分,不要悄悄修掉。

参数获取(tools/aimlab_gain.py)有两条等价路线,**都不必真的拿尺子**:

    路线 A(桌面量):  counts_per_360 = dpi * cm360 / 2.54
        例:800 DPI × 40 cm/360 -> 12598 counts/360°

    路线 B(灵敏度推):counts_per_360 = 360 / (sens * 引擎 yaw 系数)
        例:CS2 sens=1.5, yaw_coef=0.022 -> 10909 counts/360°
        ⚠️ DPI 不在路线 B 的式子里 —— 它只决定"手滑 1 厘米产生多少计数"。
        这也是为什么 Aim Lab 的 DPI 缩放会毁掉路线 B(见 SensModel)。
    → 两条路线互为校验:先算一遍,再用另一条对拍。见 tools/aimlab_gain.py
      `--engine/--sens` 与 `--verify-counts`。
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

# 离线靶场的默认语义(ArenaConfig 默认值),不要在这里悄悄改
_ARENA_SPEED_PX = 14.0
_ARENA_WIDTH_PX = 640.0


@dataclass(frozen=True)
class SensModel:
    """游戏灵敏度模型:把「DPI + 游戏内灵敏度」换算成 counts_per_360。

    ===========================================================================
    为什么需要它:cm/360 不必在桌面上量
    ===========================================================================
    引擎里真正决定转角的是一行代码::

        yaw_deg_per_count = sens * yaw_coef        # 每计数的偏航角(度)

    于是::

        counts_per_360 = 360 / (sens * yaw_coef)

    **DPI 不在这个式子里** —— DPI 只决定"手滑 1 厘米产生多少计数"。只要
    counts_per_360 对,DPI 与 sens 怎么组合都无所谓(400DPI×2sens 与
    800DPI×1sens 完全等价,这就是 eDPI 的含义)。所以路线 B 的本质是:
    **用两个游戏内可读的数字换掉一次桌面测量。**

    ===========================================================================
    ⚠️ 三个必须知道的坑
    ===========================================================================
    1. **yaw_coef 因引擎而异**(见 ENGINE_COEFS)。同一个 sens=1,Source 是
       0.022°/count,Quake 风格是 0.022 但走 m_yaw,UE4 的 FOV 缩放是另一套。
       给错系数 = 系统性偏差,且**看起来"能跑"**(闭环会把误差吸收成速度差),
       这正是 D19「staring at wall」那种标定缺陷的典型形态。

    2. **标称 DPI ≠ 真实 DPI**。鼠标驱动里的 800 常有 ±3~5% 误差;更麻烦的是
       **游戏内 DPI 缩放**:Aim Lab 的 3200 DPI 模式会把计数归一化回 800 基准
       (见 `dpi_scale`),此时 360/(0.022·sens) **不成立**,要乘回缩放。

    3. **鼠标加速/角度捕捉必须关闭**。开启时位移→计数不再线性,本文所有公式
       作废 —— 且这种破坏是**状态相关**的,标定一次不够。

    ===========================================================================
    实机校验(比公式更可信)
    ===========================================================================
    在游戏里把准星贴住一个远处竖边,注入已知计数,量准星在屏幕上走过了
    画面宽度的百分之几 -> 反推 deg_per_count -> 与公式对拍。见
    tools/aimlab_gain.py `--verify-counts`。
    """

    sens: float
    yaw_coef: float = 0.022
    pitch_coef: float | None = None       # 通常 = yaw_coef,Source/Quake 皆然
    dpi_scale: float = 1.0                # 计数进入引擎前的缩放(见坑 2)

    def __post_init__(self) -> None:
        if self.sens <= 0:
            raise ValueError(f"sens 必须为正: {self.sens}")
        if self.yaw_coef <= 0:
            raise ValueError(f"yaw_coef 必须为正: {self.yaw_coef}")
        if self.dpi_scale <= 0:
            raise ValueError(f"dpi_scale 必须为正: {self.dpi_scale}")

    @property
    def yaw_deg_per_count(self) -> float:
        return float(self.sens) * float(self.yaw_coef)

    def counts_per_360(self) -> float:
        """准入引擎的计数口径(已含 dpi_scale)。"""
        return 360.0 / (self.yaw_deg_per_count * float(self.dpi_scale))

    def describe(self) -> dict:
        return {
            "sens": self.sens,
            "yaw_coef_deg_per_count": self.yaw_coef,
            "yaw_deg_per_count_at_sens": round(self.yaw_deg_per_count, 6),
            "dpi_scale": self.dpi_scale,
            "counts_per_360": round(self.counts_per_360(), 2),
            "note": "counts_per_360 = 360/(sens·yaw_coef·dpi_scale);DPI 不参与",
        }


#: 各引擎的 sens 语义(每 1 sens 每 1 计数转过的度)。全部为公开约定值。
ENGINE_COEFS: dict[str, dict[str, float]] = {
    "source":  {"yaw_coef": 0.022},      # CS/CS2/Apex/Portal/TF2(默认 m_yaw 0.022)
    "quake":   {"yaw_coef": 0.022},      # Quake 家族同源
    "unreal":  {"yaw_coef": 0.0703125},  # UE 默认 Sensitivity * 0.0703 近似
    "unity":   {"yaw_coef": 0.022},      # 多数 Unity FPS 克隆沿用 Source 值
    "aimlab":  {"yaw_coef": 0.022},      # Aim Lab 默认(Source 血统)
    "raw":     {"yaw_coef": 1.0},        # "sens" 已直接是 °/计数
}


@dataclass
class GainConfig:
    """增益参数(全部可 JSON 持久化,保证可复现)。

    Attributes
    ----------
    counts_per_360:
        鼠标转一整圈(360°)需要的计数。由 cm/360 与 DPI 推导或实测。
    speed_fraction_per_action:
        action=1.0 时每拍扫过画面宽度的比例。默认 = 靶场 14px/640px。
        ⚠️ 它**必须**与 `fov_h_deg` 一起解释,否则差 2.5 倍 —— 见下面 fov_h_deg。
    fov_h_deg:
        游戏的水平 FOV(度)。**给了它,才谈得上"比例语义"**:
            action=1 -> 靶在画面中心移动 speed_fraction × W 像素
            <=> 转 deg_per_action = speed_fraction × linFOV
            linFOV = 2·tan(FOV/2)   (中心处 px/度 = f·π/180 的线性化)
        不给(None)时退回**旧语义**:把 speed_fraction 当成整圈的 360 分之一,
        即 deg_per_action = speed_fraction × 360。
        🔴 **实测这是本项目一处真实的标定缺陷**(2026-10-04,由模拟 FPS 复查发现):
        FOV=103° 时 linFOV=144.06°,旧语义 7.875°/拍 vs 正确 3.151°/拍 ——
        **过转 2.50 倍**,靶一拍就飞过中心 35px(而不是 14px)。
        这解释了 D19/D20 里"fly 臂一开跑视角就转离靶区、全程 staring at wall":
        不是控制器不收敛,是**每一拍都注入了 2.5 倍的转动**。
        为不静默改变既有记录,本字段默认 None(行为不变),但 `describe()` 会报警。
    max_counts_per_tick:
        每 tick 注入计数的绝对值上限(安全钳位,防止增益配错时鼠标甩飞)。
    invert_x / invert_y:
        方向翻转。读出的左右/上下语义是人为定义的(readout.py 第 2 节),
        跨域接入时方向对不对只能实测后用这两个开关修。
    deadzone:
        |action| 低于该值的分量按 0 处理(抑制读出层噪声抖动)。
    """

    counts_per_360: float = 12000.0
    speed_fraction_per_action: float = _ARENA_SPEED_PX / _ARENA_WIDTH_PX
    fov_h_deg: float | None = None
    max_counts_per_tick: float = 600.0
    invert_x: bool = False
    invert_y: bool = False
    deadzone: float = 0.0

    def __post_init__(self) -> None:
        if not (0.0 < self.counts_per_360 < 10_000_000):
            raise ValueError(f"counts_per_360 不合理: {self.counts_per_360}")
        if self.speed_fraction_per_action <= 0 or self.speed_fraction_per_action > 1:
            raise ValueError(f"speed_fraction_per_action 必须在 (0,1]: {self.speed_fraction_per_action}")
        if self.fov_h_deg is not None and not (5.0 <= self.fov_h_deg < 179.0):
            raise ValueError(f"fov_h_deg 必须在 [5,179): {self.fov_h_deg}")
        if self.max_counts_per_tick < 0:
            raise ValueError("max_counts_per_tick 不能为负")
        if self.deadzone < 0 or self.deadzone >= 1:
            raise ValueError(f"deadzone 必须在 [0,1): {self.deadzone}")

    # -- 比例语义的换算量 --------------------------------------------------

    @property
    def lin_fov_deg(self) -> float | None:
        """把透视 FOV 线性化成"中心等价的视野宽度"(度)。None = 未指定 FOV。"""
        if self.fov_h_deg is None:
            return None
        return float(2.0 * math.tan(math.radians(self.fov_h_deg) / 2.0)
                     * 180.0 / math.pi)

    @classmethod
    def from_cm360(
        cls, dpi: float, cm360: float, **kwargs
    ) -> "GainConfig":
        """(路径 A)由鼠标 DPI 与 cm/360 推导 counts_per_360。

        cm/360 实测法:游戏里让准星原地转一整圈,量鼠标在桌面上滑过的厘米数。
        量一次就够,精度取决于尺子 + 是否关掉鼠标加速。
        """
        if dpi <= 0 or cm360 <= 0:
            raise ValueError(f"dpi/cm360 必须为正: {dpi}/{cm360}")
        cfg = cls(**kwargs)
        cfg.counts_per_360 = float(dpi) * float(cm360) / 2.54
        return cfg

    @classmethod
    def from_sens(
        cls, sens: float, engine: str = "source", dpi_scale: float = 1.0,
        yaw_coef: float | None = None, **kwargs,
    ) -> "GainConfig":
        """(路径 B)由游戏内灵敏度推导 counts_per_360 —— **不需要桌面测量**。

        Parameters
        ----------
        sens:
            游戏内灵敏度数值(取引擎实际吃到的那个;UI 上可能是 ×0.01 显示)。
        engine:
            ENGINE_COEFS 的键(source/quake/unreal/unity/aimlab/raw)。
        yaw_coef:
            直接给"每计数转过多少度"。给了它则忽略 engine。
        dpi_scale:
            计数进入引擎前的缩放。Aim Lab 高 DPI 模式要填(见 SensModel 坑 2)。

        例:CS2 sens=1.5 -> counts_per_360 = 360/(1.5×0.022) ≈ 10909。
        """
        if yaw_coef is None:
            key = engine.strip().lower()
            if key not in ENGINE_COEFS:
                raise ValueError(
                    f"未知 engine={engine!r};可选 {sorted(ENGINE_COEFS)}"
                    f",或用 yaw_coef= 直接指定"
                )
            yaw_coef = ENGINE_COEFS[key]["yaw_coef"]
        sm = SensModel(sens=sens, yaw_coef=yaw_coef, dpi_scale=dpi_scale)
        cfg = cls(**kwargs)
        cfg.counts_per_360 = sm.counts_per_360()
        return cfg

    @staticmethod
    def counts_per_360_from_sens(
        sens: float, engine: str = "source", dpi_scale: float = 1.0,
        yaw_coef: float | None = None,
    ) -> float:
        """换算器(不构造 cfg),供工具层做交叉校验。"""
        if yaw_coef is None:
            key = engine.strip().lower()
            if key not in ENGINE_COEFS:
                raise ValueError(f"未知 engine={engine!r}")
            yaw_coef = ENGINE_COEFS[key]["yaw_coef"]
        return SensModel(sens=sens, yaw_coef=yaw_coef, dpi_scale=dpi_scale).counts_per_360()

    @classmethod
    def from_edpi(
        cls, edpi: float, dpi_mouse: float, engine: str = "source",
        dpi_scale: float = 1.0, yaw_coef: float | None = None, **kwargs,
    ) -> "GainConfig":
        """(路径 B 的另一种报法)由 eDPI 与鼠标 DPI 推 counts_per_360。

        什么时候用:你只记得 "eDPI = 800" 这种社区口径的数,手边没有原始 sens。
        **必须连 dpi_mouse 一起给** —— eDPI 是乘法,反解需要知道被乘的那个数。

        等价关系(都可互相验证):
            edpi=1600, dpi_mouse=800  ->  sens=2.0  ->  counts/360 = 8182
            edpi=800,  dpi_mouse=400  ->  sens=2.0  ->  counts/360 = 8182
        """
        if dpi_mouse <= 0:
            raise ValueError(f"dpi_mouse 必须为正: {dpi_mouse}")
        if edpi <= 0:
            raise ValueError(f"edpi 必须为正: {edpi}")
        if yaw_coef is None:
            key = engine.strip().lower()
            if key not in ENGINE_COEFS:
                raise ValueError(f"未知 engine={engine!r};可选 {sorted(ENGINE_COEFS)}")
            yaw_coef = ENGINE_COEFS[key]["yaw_coef"]
        sens = float(edpi) / float(dpi_mouse)
        sm = SensModel(sens=sens, yaw_coef=yaw_coef, dpi_scale=dpi_scale)
        cfg = cls(**kwargs)
        cfg.counts_per_360 = sm.counts_per_360()
        return cfg

    @staticmethod
    def counts_per_360_from_edpi(
        edpi: float, engine: str = "source", yaw_coef: float | None = None,
    ) -> float:
        """由 eDPI(= 游戏内灵敏度的等效 DPI)推 counts_per_360。

        eDPI 是社区常用口径:`edpi = 鼠标DPI × 游戏内灵敏度`。它出现的唯一理由
        是"把两个数合成一个数",**所以它当然也消掉了 DPI** —— eDPI 本身已经
        等价于"灵敏度",与 counts_per_360 成反比::

            counts_per_360 = 360 / (edpi × yaw_coef / dpi_mouse)

        ⚠️ 这个式子**仍然含 dpi_mouse**,因为 eDPI 定义里含它。所以**只报 eDPI
        是不够的** —— 必须连鼠标 DPI 一起报(见 `from_edpi`)。
        本函数假定 edpi 已经是"引擎基准 DPI 口径"下的值(= 800 时等价于 sens)。
        """
        if edpi <= 0:
            raise ValueError(f"edpi 必须为正: {edpi}")
        if yaw_coef is None:
            key = engine.strip().lower()
            if key not in ENGINE_COEFS:
                raise ValueError(f"未知 engine={engine!r}")
            yaw_coef = ENGINE_COEFS[key]["yaw_coef"]
        sens_equiv = float(edpi) / 800.0     # 社区 eDPI 以 800 DPI 为基准口径
        return SensModel(sens=sens_equiv, yaw_coef=yaw_coef).counts_per_360()

    @staticmethod
    def sens_from_edpi(edpi: float, dpi_mouse: float) -> float:
        """eDPI -> 该鼠标 DPI 下的游戏内灵敏度:sens = edpi / dpi_mouse。"""
        if dpi_mouse <= 0:
            raise ValueError(f"dpi_mouse 必须为正: {dpi_mouse}")
        return float(edpi) / float(dpi_mouse)

    @staticmethod
    def cm360_from_edpi(
        edpi: float, dpi_mouse: float, engine: str = "source",
        yaw_coef: float | None = None,
    ) -> float:
        """eDPI + 鼠标 DPI -> cm/360(社区最常用的那个数)。"""
        sens = GainConfig.sens_from_edpi(edpi, dpi_mouse)
        return GainConfig.cm360_from_sens(sens, dpi_mouse, engine, 1.0, yaw_coef)

    @staticmethod
    def cm360_from_sens(
        sens: float, dpi: float, engine: str = "source",
        dpi_scale: float = 1.0, yaw_coef: float | None = None,
    ) -> float:
        """路径 B -> 路径 A 的换算式:由 (sens, DPI) 反推等效 cm/360。

        注意 DPI 在这里**只用于换算显示**,不参与 counts_per_360 本身。
        """
        if dpi <= 0:
            raise ValueError(f"dpi 必须为正: {dpi}")
        c = GainConfig.counts_per_360_from_sens(sens, engine, dpi_scale, yaw_coef)
        return c * 2.54 / float(dpi)

    @staticmethod
    def sens_for_cm360(
        dpi: float, cm360: float, engine: str = "source",
        dpi_scale: float = 1.0, yaw_coef: float | None = None,
    ) -> float:
        """路径 A -> 路径 B 的反解:要把 sens 设成多少才能达到该 cm/360。"""
        if yaw_coef is None:
            key = engine.strip().lower()
            if key not in ENGINE_COEFS:
                raise ValueError(f"未知 engine={engine!r}")
            yaw_coef = ENGINE_COEFS[key]["yaw_coef"]
        c = float(dpi) * float(cm360) / 2.54
        return 360.0 / (c * yaw_coef * dpi_scale)


class GainModel:
    """action [-1,1] -> (dx_counts, dy_counts)。无状态,可随便复用。"""

    def __init__(self, cfg: GainConfig) -> None:
        self.cfg = cfg

    # -- 派生量(供遥测与文档) --------------------------------------------

    @property
    def counts_per_action(self) -> float:
        """action=1.0 对应的注入计数。"""
        return float(self.deg_per_action() * self.cfg.counts_per_360 / 360.0)

    def deg_per_count(self) -> float:
        """每计数转过的角度(度)。"""
        return 360.0 / float(self.cfg.counts_per_360)

    def cm360(self, dpi: float = 800.0) -> float:
        """等效 cm/360(按给定 DPI 换算,仅用于跟社区数据对拍)。

        cm/360 本身依赖 DPI,所以必须报出用的是哪个 DPI;默认 800 是社区惯例。
        """
        if dpi <= 0:
            raise ValueError(f"dpi 必须为正: {dpi}")
        return float(self.cfg.counts_per_360) * 2.54 / float(dpi)

    def deg_per_action(self) -> float:
        """action=1.0 对应的视线转角(度)。

        有 FOV 时 = speed_fraction × linFOV(把"扫过画面宽度的比例"换算成角度);
        无 FOV 时 = speed_fraction × 360(旧语义,过转见 GainConfig.fov_h_deg 的说明)。
        """
        lin = self.cfg.lin_fov_deg
        if lin is None:
            return float(self.cfg.speed_fraction_per_action) * 360.0
        return float(self.cfg.speed_fraction_per_action) * lin

    def describe(self) -> dict:
        d = asdict(self.cfg)
        d["counts_per_action"] = round(self.counts_per_action, 2)
        d["deg_per_count"] = round(self.deg_per_count(), 6)
        d["deg_per_action"] = round(self.deg_per_action(), 4)
        d["lin_fov_deg"] = (None if self.cfg.lin_fov_deg is None
                            else round(self.cfg.lin_fov_deg, 3))
        d["cm360"] = self.cm360()
        if self.cfg.lin_fov_deg is None:
            d["semantics"] = (
                "⚠️ 未指定 fov_h_deg:按旧语义 action=1 扫过整圈的 "
                "speed_fraction(deg_per_action = speed_fraction×360)。"
                "若 FOV≈103°,这比「扫过画面宽度的同一比例」过转 2.50 倍,"
                "一拍就把靶推过中心 35px —— 上线前务必显式给 --fov。"
            )
        else:
            d["semantics"] = (
                "action=1.0 使靶在画面中心移动 speed_fraction×W 像素"
                "(与离线靶场 14px/640px 同语义);角度量 = speed_fraction×linFOV,"
                "linFOV=2tan(FOV/2) 是中心处 px/度 的线性化。"
                "透视边缘偏差见 DECISIONS D26 的 px/度 实测。"
            )
        return d

    # -- 主接口 -------------------------------------------------------------

    def to_counts(self, action: np.ndarray) -> tuple[int, int]:
        """归一化 action -> 注入计数(已钳位、已翻转、已死区)。"""
        a = np.asarray(action, dtype=np.float64).reshape(-1)
        if a.size != 2:
            raise ValueError(f"action 必须是 (2,),收到 shape={np.shape(action)}")
        a = np.clip(a, -1.0, 1.0)
        if self.cfg.deadzone > 0:
            a = np.where(np.abs(a) < self.cfg.deadzone, 0.0, a)
        raw = a * self.counts_per_action
        if self.cfg.invert_x:
            raw[0] = -raw[0]
        if self.cfg.invert_y:
            raw[1] = -raw[1]
        m = float(self.cfg.max_counts_per_tick)
        dx = int(np.clip(np.round(raw[0]), -m, m))
        dy = int(np.clip(np.round(raw[1]), -m, m))
        return dx, dy

    # -- 持久化 -------------------------------------------------------------

    def save(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(
            json.dumps(self.describe(), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return p

    @classmethod
    def load(cls, path: str | Path) -> "GainModel":
        """从 JSON 载入。文件里的派生量字段(counts_per_action 等)被忽略。"""
        p = Path(path)
        raw = json.loads(p.read_text(encoding="utf-8"))
        keys = set(GainConfig.__dataclass_fields__)  # type: ignore[attr-defined]
        cfg = GainConfig(**{k: v for k, v in raw.items() if k in keys})
        return cls(cfg)
