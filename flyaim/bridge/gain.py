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

参数获取(tools/aimlab_gain.py):
    counts_per_360 = dpi * cm360 / 2.54     # 由 cm/360 与鼠标 DPI 推导
    例:800 DPI × 40 cm/360 -> 12598 counts/360°
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

# 离线靶场的默认语义(ArenaConfig 默认值),不要在这里悄悄改
_ARENA_SPEED_PX = 14.0
_ARENA_WIDTH_PX = 640.0


@dataclass
class GainConfig:
    """增益参数(全部可 JSON 持久化,保证可复现)。

    Attributes
    ----------
    counts_per_360:
        鼠标转一整圈(360°)需要的计数。由 cm/360 与 DPI 推导或实测。
    speed_fraction_per_action:
        action=1.0 时每拍扫过画面宽度的比例。默认 = 靶场 14px/640px。
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
    max_counts_per_tick: float = 600.0
    invert_x: bool = False
    invert_y: bool = False
    deadzone: float = 0.0

    def __post_init__(self) -> None:
        if not (0.0 < self.counts_per_360 < 10_000_000):
            raise ValueError(f"counts_per_360 不合理: {self.counts_per_360}")
        if self.speed_fraction_per_action <= 0 or self.speed_fraction_per_action > 1:
            raise ValueError(f"speed_fraction_per_action 必须在 (0,1]: {self.speed_fraction_per_action}")
        if self.max_counts_per_tick < 0:
            raise ValueError("max_counts_per_tick 不能为负")
        if self.deadzone < 0 or self.deadzone >= 1:
            raise ValueError(f"deadzone 必须在 [0,1): {self.deadzone}")

    @classmethod
    def from_cm360(
        cls, dpi: float, cm360: float, **kwargs
    ) -> "GainConfig":
        """由鼠标 DPI 与 cm/360 推导 counts_per_360。

        cm/360 实测法:游戏里让准星原地转一整圈,量鼠标在桌面上滑过的厘米数。
        """
        if dpi <= 0 or cm360 <= 0:
            raise ValueError(f"dpi/cm360 必须为正: {dpi}/{cm360}")
        cfg = cls(**kwargs)
        cfg.counts_per_360 = float(dpi) * float(cm360) / 2.54
        return cfg


class GainModel:
    """action [-1,1] -> (dx_counts, dy_counts)。无状态,可随便复用。"""

    def __init__(self, cfg: GainConfig) -> None:
        self.cfg = cfg

    # -- 派生量(供遥测与文档) --------------------------------------------

    @property
    def counts_per_action(self) -> float:
        """action=1.0 对应的注入计数。"""
        return float(self.cfg.speed_fraction_per_action * self.cfg.counts_per_360)

    def deg_per_count(self) -> float:
        """每计数转过的角度(度)。"""
        return 360.0 / float(self.cfg.counts_per_360)

    def deg_per_action(self) -> float:
        """action=1.0 对应的视线转角(度)。"""
        return self.counts_per_action * self.deg_per_count()

    def describe(self) -> dict:
        d = asdict(self.cfg)
        d["counts_per_action"] = round(self.counts_per_action, 2)
        d["deg_per_count"] = round(self.deg_per_count(), 6)
        d["deg_per_action"] = round(self.deg_per_action(), 4)
        d["semantics"] = (
            "action=1.0 每拍扫过画面宽度的 speed_fraction_per_action 比例"
            "(与离线靶场 14px/640px 同语义);角度量经由 counts_per_360 折算。"
            "屏幕像素≈角度的线性近似与 tick 频率依赖见 gain.py docstring。"
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
