"""Aim Lab 桥接层:把 FlyAim 闭环从「自建 2D 靶场」搬到「真实游戏画面」。

===============================================================================
定位
===============================================================================
本项目原有的闭环是::

    Arena.render() -> Retina -> Connectome -> Readout -> (dx,dy) -> Arena.step()

桥接层把首尾两端换成真实世界的 I/O,**中间的神经链路一行不改**::

    屏幕捕获(FrameSource) -> Retina -> Connectome -> Readout -> (dx,dy)
        -> 增益层(GainModel) -> 鼠标注入(ActionSink,SendInput)

因此各模块的职责边界是:

    capture.py     FrameSource:屏幕区域 -> (H,W,3) uint8 RGB 帧(替代 Arena.render)
    inject.py      ActionSink:鼠标计数注入(替代 Arena.step 的准星位移)
    gain.py        GainModel:归一化 action [-1,1] -> 鼠标计数(替代 14px/action)
    detect.py      纯像素的靶标检测(遥测 + 管道校验用;**不是**果蝇臂的输入)
    controllers.py 控制器:fly(FlySystem)/ seek(纯视觉 PD,管道校验)/ random / zero
    loop.py        BridgeLoop:捕获线程 + 网络拍 + 注入 + 遥测 + 延迟记账

===============================================================================
诚实声明(与 README「已知限制」一致,接真实游戏前必须重读)
===============================================================================
1. **现有科学结论是负面的**(DECISIONS.md D10/D11):LIF 简化模型 + 静态连接组
   在自建 2D 靶场上不承载可用的视觉伺服信号。接入 Aim Lab 不会自动变好;
   本桥接层的目的是让「同一链路在 3D 光流统计下是否仍旧失效」可以被检验。
2. **读出语义是人为定义的**:trained 读出在 2D 靶场轨迹上训练,其 (dx,dy)
   语义是「屏幕像素速度」。3D 场景里鼠标增量是**角度增量**,两者只通过
   GainModel 的线性假设相连 —— 这个假设本身就是一个待检验的近似。
3. **闭环频率是硬约束**:全量网络 GPU 上 ~10 Hz(D17 工作点),人玩 Aim Lab
   是 60+ Hz。10 Hz 控制在机械上就瞄不快,任何命中率结果都必须连同
   tick_hz 一起解读。
4. **合规边界**:SendInput 是标准自动化输入(无驱动级注入、无反作弊规避);
   使用者应遵守 Aim Lab 服务条款,建议只用本地/自定义任务、不提交排行榜成绩
   (见 DECISIONS.md D18)。
"""

from flyaim.bridge.capture import ArraySource, ArenaSource, ScreenCapture, centered_region
from flyaim.bridge.controllers import (
    EyeController,
    FlyController,
    RandomController,
    SeekController,
    ZeroController,
)
from flyaim.bridge.gain import GainConfig, GainModel
from flyaim.bridge.inject import NullSink, SendInputSink
from flyaim.bridge.loop import BridgeLoop

__all__ = [
    "ArraySource",
    "ArenaSource",
    "ScreenCapture",
    "centered_region",
    "EyeController",
    "FlyController",
    "RandomController",
    "SeekController",
    "ZeroController",
    "GainConfig",
    "GainModel",
    "NullSink",
    "SendInputSink",
    "BridgeLoop",
]
