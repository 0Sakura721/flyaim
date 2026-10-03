"""FlyAim: 用 MaleCNS v1.0 果蝇连接组驱动闭环瞄准控制。

模块划分(详见 CONTRACT.md):
    flyaim.neuron_index  索引空间与细胞角色选择      [Lead]
    flyaim.io            产物读写与 schema           [Lead]
    flyaim.config        配置数据类                  [Lead]
    flyaim.data          连接组下载/解析/导出        [A 线]
    flyaim.retina        人工复眼编码器              [B 线]
    flyaim.brain         LIF 稀疏仿真引擎 + 读出      [B 线]
    flyaim.arena         自建瞄准靶场                [C 线]
    flyaim.baselines     pid / shuffle / random      [C 线]
"""

__version__ = "0.1.0"

from flyaim.config import ArenaConfig, BrainConfig, ReadoutConfig, RetinaConfig
from flyaim.neuron_index import NeuronIndex, RoleSelection

__all__ = [
    "ArenaConfig",
    "BrainConfig",
    "ReadoutConfig",
    "RetinaConfig",
    "NeuronIndex",
    "RoleSelection",
]
