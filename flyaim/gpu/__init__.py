"""GPU rollout 组件(批量视网膜 + 批量靶场 + 特权教师标签)。

目的:把训练循环从"CPU 喂数据 + GPU 空等"改为"数据生成与训练都在 GPU"。
见 DECISIONS D24 与 CONTRACT_ANN 附录 A2。
"""

from flyaim.gpu.retina_gpu import RetinaGPU

__all__ = ["RetinaGPU"]
