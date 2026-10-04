"""GPU 版视网膜编码器(批量化,见 CONTRACT_ANN 附录 A2 的 GPU rollout 计划)。

与 CPU ``flyaim.retina.encoder.Retina`` 逐算子对应,但:
    - 在 GPU 上以 **批量** 张量 (B, H, W, 3) 运行;
    - 拓扑(小眼映射/细胞索引/感受器分组)**直接复用** CPU 实例,
      因此不会出现两套 mapping 漂移;
    - 时间差分状态 ``_prev_opp`` 也是批的 (B, r, c)。

数值等价性由 ``tools/verify_gpu_retina.py`` 与 CPU 版逐帧比对(容忍 float32 eps,
1e-6 量级)。**这里不做任何"优化近似"**,只是把 numpy 换 cupy、加一个 batch 维。

只接收像素(契约禁止事项 5.1):不 import arena,不接收目标坐标/标签。
"""

from __future__ import annotations

from typing import Any

import numpy as np

from flyaim.retina.encoder import Retina, _GAMMA, _NEIGHBOR_KERNEL


class RetinaGPU:
    """批量 GPU 视网膜。``n <= 1`` 时等价于单帧 CPU 版。"""

    def __init__(self, cpu_retina: Retina, n_env: int = 1):
        import cupy as cp

        self.cp = cp
        r = cpu_retina
        if not (r.encoding == "rate"):
            raise NotImplementedError(
                f"GPU 版仅实现 encoding='rate'(spike 通道是随机门,不属于确定性 rollout);"
                f"收到 {r.encoding!r}")
        if abs(r.temporal_diff) > 0 and r._gain_mode == "frame_rms":
            pass  # 支持,但下面 _gain_factor 需注意批量 RMS
        self.cpu = r
        self.n_env = int(n_env)
        self.eye_rows = int(r.eye_rows)
        self.eye_cols = int(r.eye_cols)
        self.n_ommatidia = int(r.n_ommatidia)
        self.max_drive_hz = float(r.max_drive_hz)
        self.temporal_diff = float(r.temporal_diff)
        self.lateral_inhibition = float(r.lateral_inhibition)
        self.use_on_off = bool(r.use_on_off)
        self._ref_lum = float(r._ref_lum)
        self._ref_col = float(r._ref_col)
        self._gain_mode = str(r._gain_mode)
        self.input_neuron_ids = np.asarray(r.input_neuron_ids)

        # 拓扑 -> GPU(一次性)
        self._lum_w = cp.asarray(r._lum_w.reshape(-1), dtype=cp.float32)
        self._kernel = cp.asarray(_NEIGHBOR_KERNEL[None, :, :], dtype=cp.float32)
        self._rows_lum = cp.asarray(r._rows_lum, dtype=cp.int64)
        self._rows_r7 = cp.asarray(r._rows_r7, dtype=cp.int64)
        self._rows_r8 = cp.asarray(r._rows_r8, dtype=cp.int64)
        self._cell_lum = cp.asarray(r._cell_lum, dtype=cp.int64)
        self._cell_r7 = cp.asarray(r._cell_r7, dtype=cp.int64)
        self._cell_r8 = cp.asarray(r._cell_r8, dtype=cp.int64)
        self.n_input = int(self.input_neuron_ids.size)
        self._gamma = float(_GAMMA)
        self._has_prev = False
        self._prev_lum = None
        self._prev_col = None

    # ------------------------------------------------------------------ 状态

    def reset(self) -> None:
        self._has_prev = False
        self._prev_lum = None
        self._prev_col = None

    def _zero_prev(self):
        cp = self.cp
        shape = (self.n_env, self.eye_rows, self.eye_cols)
        self._prev_lum = cp.zeros(shape, dtype=cp.float32)
        self._prev_col = cp.zeros(shape, dtype=cp.float32)

    # ------------------------------------------------------------------ 主接口

    def frame_to_spikes(self, frames) -> "Any":
        """(B,H,W,3) uint8 -> (B, n_input, 2) float32(GPU 数组)。"""
        cp = self.cp
        a = frames if isinstance(frames, cp.ndarray) else cp.asarray(frames)
        if a.ndim == 3:
            a = a[None, ...]
        b, h, w = int(a.shape[0]), int(a.shape[1]), int(a.shape[2])
        if self._prev_lum is None or self._prev_lum.shape[0] != b:
            self.n_env = b
            self._zero_prev()

        if a.shape[3] < 3:
            gray = a[..., 0].astype(cp.float32) / cp.float32(255.0)
            chroma = None
        else:
            rgb = a[..., :3].astype(cp.float32) / cp.float32(255.0)
            # 无 cuBLAS:显式加权和替代矩阵乘(逐元素,等价且更省)
            gray = (rgb[..., 0] * self._lum_w[0] + rgb[..., 1] * self._lum_w[1]
                    + rgb[..., 2] * self._lum_w[2])
            chroma = (rgb[..., 0], rgb[..., 1])

        center_l = cp.power(cp.maximum(self._ommatidial_mean(gray, h, w), 0.0), self._gamma)
        if chroma is None:
            signed_c = cp.zeros((b, self.eye_rows, self.eye_cols), dtype=cp.float32)
        else:
            cr = cp.power(cp.maximum(self._ommatidial_mean(chroma[0], h, w), 0.0),
                          self._gamma)
            cg = cp.power(cp.maximum(self._ommatidial_mean(chroma[1], h, w), 0.0),
                          self._gamma)
            signed_c = (cr - cg).astype(cp.float32)

        s_lum = self._channel_response(center_l, "lum")
        exc_l, inh_l = self._rectify(s_lum, self._ref_lum)
        s_col = self._channel_response(signed_c, "col")
        exc_c, inh_c = self._rectify(s_col, self._ref_col)

        out = cp.zeros((b, self.n_input, 2), dtype=cp.float32)
        ef_l, if_l = exc_l.reshape(b, -1), inh_l.reshape(b, -1)
        ef_c, if_c = exc_c.reshape(b, -1), inh_c.reshape(b, -1)
        if self._rows_lum.size:
            out[:, self._rows_lum, 0] = ef_l[:, self._cell_lum]
            out[:, self._rows_lum, 1] = if_l[:, self._cell_lum]
        if self._rows_r7.size:
            out[:, self._rows_r7, 0] = ef_c[:, self._cell_r7]
            out[:, self._rows_r7, 1] = if_c[:, self._cell_r7]
        if self._rows_r8.size:
            out[:, self._rows_r8, 0] = if_c[:, self._cell_r8]
            out[:, self._rows_r8, 1] = ef_c[:, self._cell_r8]
        return out

    # ------------------------------------------------------------------ 算子

    def _ommatidial_mean(self, gray, h: int, w: int):
        cp = self.cp
        r, c = self.eye_rows, self.eye_cols
        if h % r == 0 and w % c == 0:
            bh, bw = h // r, w // c
            return gray.reshape(-1, r, bh, c, bw).mean(axis=(2, 4), dtype=cp.float32)
        raise NotImplementedError(
            f"GPU 视网膜仅实现整除快路径(靶场 480x640 / 24x32);收到 {h}x{w}")

    def _channel_response(self, center, key: str):
        cp = self.cp
        x = center.astype(cp.float32)
        if self.lateral_inhibition > 0.0:
            surround = self._convolve_nearest(x)
            opp = x - cp.float32(self.lateral_inhibition) * surround
        else:
            opp = x
        opp = (opp - opp.mean(axis=(1, 2), dtype=cp.float32, keepdims=True)).astype(cp.float32)
        prev = self._prev_lum if key == "lum" else self._prev_col
        if self._has_prev and self.temporal_diff > 0.0:
            d = opp - prev
        else:
            d = cp.zeros_like(opp)
        if key == "lum":
            self._prev_lum = opp
        else:
            self._prev_col = opp
        # 注意:CPU 版 Retina 的 self._has_prev 除 init/reset 外**从不置 True**
        # (flyaim/retina/encoder.py:274/484,只在 603 读取)。因此 CPU 的时间差分
        # 分支实际永不生效,输出恒为 (1-temporal_diff)*opp。GPU 版忠实复刻这一
        # 行为以保证逐位等价(该 CPU 侧缺陷已记入 D24,不在本次改动中修复)。
        return ((cp.float32(1.0 - self.temporal_diff) * opp
                 + cp.float32(self.temporal_diff) * d).astype(cp.float32))

    def _convolve_nearest(self, x):
        """3x3 邻域均值(mode='nearest'),批量上等价于逐帧 scipy.ndimage.convolve。"""
        import cupyx.scipy.ndimage as ndi

        return ndi.convolve(x, self._kernel, mode="nearest")

    def _rectify(self, s, ref: float):
        cp = self.cp
        g = self._gain_factor(s, ref)
        on = cp.clip(cp.maximum(s, 0.0) * g, 0.0, 1.0) * cp.float32(self.max_drive_hz)
        if self.use_on_off:
            off = cp.clip(cp.maximum(-s, 0.0) * g, 0.0, 1.0) * cp.float32(self.max_drive_hz)
        else:
            off = cp.zeros_like(on)
        return on.astype(cp.float32), off.astype(cp.float32)

    def _gain_factor(self, s, ref: float):
        cp = self.cp
        if self._gain_mode == "unit":
            return 1.0
        if self._gain_mode == "frame_rms":
            rms = cp.sqrt(cp.mean(cp.square(s.astype(cp.float32)), axis=(1, 2),
                                  keepdims=True))
            return (cp.float32(0.5) / cp.maximum(rms, cp.float32(0.02)))
        return float(1.0 / max(ref, 1e-6))
