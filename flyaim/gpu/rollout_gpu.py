"""GPU 端到端 rollout:B 个环境并行、数据生成与网络前向全在显存。

解决的问题(实测):
    旧循环每帧 = CPU 视网膜 15ms + CPU 教师检测 18ms + arena 2.5ms,GPU 只在
    两次稀疏矩阵乘上忙 ~5ms,其余时间在等 CPU 喂数据 -> GPU 利用率个位数。

本模块把这条链全部搬到 GPU:
    GPUArenaBatch(B 环境) -> RetinaGPU(批量) -> 拼 u(1536,B)
    -> net.forward_batch(SpMM) -> teacher_labels(特权信息,零检测开销)
    -> 回流 arena.step

**全程无 cp.asnumpy**,一帧只在最后(可选,仪表盘)取一次样本;
输出 u/y 以 GPU 张量累积,训练侧直接消费,不回主机。
"""

from __future__ import annotations

import numpy as np

from flyaim.ann_train import EYE
from flyaim.config import ArenaConfig, RetinaConfig
from flyaim.gpu.arena_gpu import GPUArenaBatch
from flyaim.gpu.retina_gpu import RetinaGPU


class GPURollout:
    """B 环境锁步的 GPU rollout 采集器。"""

    def __init__(self, net, cpu_retina, roles, n_env: int = 64, seed: int = 0,
                 arena_cfg: ArenaConfig | None = None):
        import cupy as cp

        self.cp = cp
        self.net = net
        self.B = int(n_env)
        self.arena = GPUArenaBatch(arena_cfg or ArenaConfig(max_frames=900),
                                   n_env=self.B, seed=seed)
        self.retina = RetinaGPU(cpu_retina, n_env=self.B)
        self.roles = roles
        # retina_drive_to_u 的拓扑(在 GPU 上重推一遍,避免 import CPU 版)
        cp_retina = cpu_retina
        self._rows = [cp.asarray(r, dtype=cp.int64) for r in
                      (cp_retina._rows_lum, cp_retina._rows_r7, cp_retina._rows_r8)]
        self._cells = [cp.asarray(c, dtype=cp.int64) for c in
                       (cp_retina._cell_lum, cp_retina._cell_r7, cp_retina._cell_r8)]
        self._h = cp.zeros((net.n, self.B), dtype=cp.float32)
        self._frames = None

    # ------------------------------------------------------------------

    def _drive_to_u(self, drive):
        """(B, n_cells, 2) -> (1536, B):ON/OFF 各 768,组内均值,/200。"""
        cp = self.cp
        b = drive.shape[0]
        on = cp.zeros((b, EYE), dtype=cp.float32)
        off = cp.zeros((b, EYE), dtype=cp.float32)
        cnt = cp.zeros(EYE, dtype=cp.float32)
        for rows, cells in zip(self._rows, self._cells):
            if rows.size == 0:
                continue
            cp.add.at(on, (slice(None), cells), drive[:, rows, 0])
            cp.add.at(off, (slice(None), cells), drive[:, rows, 1])
            cnt[cells] += 1.0
        cnt = cp.maximum(cnt, 1.0)
        on = on / cnt[None, :]
        off = off / cnt[None, :]
        u = cp.concatenate([on, off], axis=1) / cp.float32(200.0)
        return u.T.astype(cp.float32)            # (1536, B)

    def reset(self):
        self.retina.reset()
        self._frames = self.arena.reset()

    def step(self, policy_driven: bool, hybrid: bool = False):
        """推进一帧。返回 (U (1536,B), Y (B,2), A (B,2), dist(B), hit(B))。"""
        cp = self.cp
        drive = self.retina.frame_to_spikes(self._frames)      # (B,n,2)
        U = self._drive_to_u(drive)                            # (1536,B)
        a_net, h_new, PU_t, z, cand = self.net.forward_batch(U, self._h)  # (2,B)
        Y = self.arena.teacher_labels()                        # (B,2)
        if policy_driven:
            A = a_net.T
            self._h = h_new
        else:
            A = Y
            # 教师驱动时网络仍前向传播(保持 h 有信号,供可视化/一致性)
            self._h = h_new
        self._frames, dist, hit = self.arena.step(A)
        return U, Y, A, dist, hit

    # ------------------------------------------------------------------

    def collect(self, n_frames: int, policy_driven: bool, publish=None,
                publish_every: int = 10, seed_off: int = 0):
        """采集 n_frames 帧(全局计数),返回 GPU 张量 U (1536,N) 与 Y (N,2)。

        只在 publish 时取一次样本(可选),其余全程不落主机。
        """
        cp = self.cp
        self.arena._rng = cp.random.default_rng(1000 + seed_off)
        self.reset()
        Us, Ys = [], []
        last = {}
        n = 0
        while n < n_frames:
            U, Y, A, dist, hit = self.step(policy_driven)
            Us.append(U)
            Ys.append(Y)
            n += self.B
            if publish is not None and (n % (publish_every * self.B) < self.B):
                act = cp.asnumpy(cp.abs(self._h[getattr(publish, "sample_idx", slice(None))]).mean(axis=1)) \
                    if getattr(publish, "sample_idx", None) is not None else None
                publish("rollout", policy=policy_driven, frames=n,
                        target_dist=float(cp.mean(dist)), hit=bool(cp.any(hit)),
                        activity=act)
        U = cp.concatenate(Us, axis=1)[:, :n_frames]
        Y = cp.concatenate(Ys, axis=0)[:n_frames]
        return U, Y
