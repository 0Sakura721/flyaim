"""端到端集成:把 数据(A线) → 复眼+仿真(B线) → 靶场(C线) 拼成可运行闭环。

三条并行工作线在此汇合。本模块是唯一的跨线粘合点,负责:
    1. 加载连接组产物并组装 `Retina` / `Connectome` / `Readout`
    2. 把三者包装成 `runner.Arm` 协议
    3. 强制「果蝇只能看像素」的约束(架构级,见 `FlyArm`)

**关键约束(不可绕过):**
    `FlyArm.act()` 接受 `state` 参数但**主动丢弃并记录**,
    保证 CONTRACT 禁止事项 1 不被违反。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from flyaim.io import (
    ARTIFACT_INDEX,
    ARTIFACT_MANIFEST,
    ARTIFACT_ROLES,
    ARTIFACT_WEIGHTS,
    ConnectomeArtifacts,
    Manifest,
    load_roles,
)

# ---------------------------------------------------------------- 装配


class FlySystem:
    """一条完整的果蝇控制链路:复眼 → 连接组 → 读出。

    分离出这个类是为了让「全量」与「shuffle 零模型」能共用同一前端与读出,
    只有连接组权重不同 —— 这正是零模型对照所需要控制的变量。
    """

    def __init__(self, data_dir: str | Path, retina_cfg, brain_cfg, readout_cfg,
                 weights_override: str | Path | None = None, device: str = "cpu"):
        """组装链路。

        device:
          "cpu"  → `flyaim.brain.lif.Connectome`(scipy 稀疏,单线程)
          "cuda" → `flyaim.brain.lif_gpu.ConnectomeGPU`(cuSPARSE)
          "auto" → 有可用 GPU 就用 GPU,否则 CPU

        实测(GTX 1660 Ti Max-Q,全量 166,700,dt=4ms/8 步):
            CPU 1148 ms/帧  vs  GPU 165 ms/帧  = **7.0x**
        数值等价性已证明:起始 6 步发放数逐位一致,脉冲一致率 99.28%,
        平均放电率相对差 0.009%(见 tools/bench_gpu.py)。
        """
        from flyaim.brain.readout import Readout
        from flyaim.retina.encoder import Retina

        data_dir = Path(data_dir)
        self.data_dir = data_dir

        self.roles = load_roles(data_dir / ARTIFACT_ROLES)
        self.manifest = (
            Manifest.load(data_dir / ARTIFACT_MANIFEST)
            if (data_dir / ARTIFACT_MANIFEST).exists()
            else None
        )

        weights_path = weights_override or (data_dir / ARTIFACT_WEIGHTS)

        # --- 设备选择 ---
        dev = str(device).lower()
        if dev == "auto":
            try:
                from flyaim.brain.lif_gpu import gpu_available

                dev = "cuda" if gpu_available() else "cpu"
            except Exception:
                dev = "cpu"
        self.device = "cuda" if dev in ("cuda", "gpu") else "cpu"

        if self.device == "cuda":
            from flyaim.brain.lif_gpu import ConnectomeGPU

            self.brain = ConnectomeGPU(weights_path, brain_cfg,
                                       input_neuron_ids=self.roles.visual_input)
        else:
            from flyaim.brain.lif import Connectome

            self.brain = Connectome(str(weights_path), brain_cfg,
                                    input_neuron_ids=self.roles.visual_input)

        # 复眼编码器:输入靶点 = roles.visual_input
        self.retina = Retina(retina_cfg, input_neuron_ids=self.roles.visual_input)
        self.readout = Readout(readout_cfg, roles=self.roles)

    def act(self, frame: np.ndarray) -> np.ndarray:
        """一帧:像素 → 复眼 → 仿真 steps_per_frame 步 → 读出 → (dx, dy)。"""
        drive = self.retina.frame_to_spikes(frame)
        in_exc = drive[:, 0]
        in_inh = drive[:, 1]
        n = self.brain.cfg.steps_per_frame

        # GPU 引擎提供 step_many:外部驱动只上传**一次**,避免每步 H2D 往返
        if hasattr(self.brain, "step_many"):
            self.brain.step_many(n, in_exc, in_inh, self.brain.cfg.dt_ms)
        else:
            for _ in range(n):
                self.brain.step(in_exc, in_inh, self.brain.cfg.dt_ms)
        return self.readout.act(self.brain)

    def reset(self) -> None:
        self.brain.reset()


# ---------------------------------------------------------------- Arm 包装


class FlyArm:
    """果蝇臂:符合 `runner.Arm` 协议。

    **state 被显式丢弃**。若有人试图让果蝇偷看靶位坐标,`state_access_log`
    会记录该尝试,`assert_state_blind()` 会失败。
    """

    def __init__(self, system_factory, name: str = "fly"):
        self._factory = system_factory
        self.name = name
        self.system: FlySystem | None = None
        self.state_access_log: list = []

    def reset(self, seed: int) -> None:
        self.system = self._factory(seed)
        self.system.reset()
        self.state_access_log.clear()

    def act(self, frame: np.ndarray, state: dict | None = None) -> np.ndarray:
        # ---- 架构级约束:记录并丢弃 state ----
        if state:
            self.state_access_log.append(sorted(state.keys()))
        assert self.system is not None, "FlyArm.act 前必须先 reset(seed)"
        return self.system.act(frame)

    def assert_state_blind(self) -> None:
        """CONTRACT 禁止事项 1 的自检钩子。"""
        if self.state_access_log:
            raise AssertionError(
                f"果蝇臂被传入了环境状态(共 {len(self.state_access_log)} 次):"
                f"{self.state_access_log[0]}。这违反 CONTRACT 禁止事项 1。"
            )


# ---------------------------------------------------------------- 便捷装配


def build_fly_system_factory(
    data_dir: str | Path,
    retina_cfg,
    brain_cfg,
    readout_cfg,
    weights_override: str | Path | None = None,
):
    """返回 `factory(seed) -> FlySystem`。

    每次调用重新构造,保证 episode 之间状态完全隔离(除训练好的读出权重)。
    """

    def factory(seed: int) -> FlySystem:
        return FlySystem(data_dir, retina_cfg, brain_cfg, readout_cfg,
                         weights_override=weights_override)

    return factory


def build_shuffle_override(data_dir: str | Path, seed: int, out_dir: str | Path) -> Path:
    """为指定 seed 生成打乱接线的连接组,返回 npz 路径。

    使用 C 线的 `flyaim.baselines.shuffle.shuffle_connectome`。
    """
    from flyaim.baselines.shuffle import shuffle_connectome

    data_dir = Path(data_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"shuffle_seed{seed}.npz"

    art = ConnectomeArtifacts.load(data_dir / ARTIFACT_WEIGHTS)
    shuffled = shuffle_connectome(art, seed)
    shuffled.save(out_path)
    return out_path


def check_data_ready(data_dir: str | Path) -> tuple[bool, list[str]]:
    """检查 A 线产物是否齐全。返回 (ready, 缺失项列表)。"""
    data_dir = Path(data_dir)
    missing = [
        n
        for n in (ARTIFACT_WEIGHTS, ARTIFACT_INDEX, ARTIFACT_ROLES, ARTIFACT_MANIFEST)
        if not (data_dir / n).exists()
    ]
    return (not missing), missing
