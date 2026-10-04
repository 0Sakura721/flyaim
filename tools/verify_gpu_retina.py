"""CPU vs GPU 视网膜编码器等价性验证。

    & $py tools/verify_gpu_retina.py

做法:构造一批随机 uint8 帧(含靶场式色块),同一序列喂给 CPU ``Retina`` 与
GPU ``RetinaGPU``(B=1 与 B=N 两种),逐帧比对 (n_input,2) 驱动。

通过标准:相对误差 max|Δ| / max|ref| ≤ 1e-5(允许 float32 归约顺序差异;
卷积在不同后端的求和次序不同,无法逐位相同)。任何超差 -> 非零退出,
GPU rollout 不得启用。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from flyaim.config import RetinaConfig  # noqa: E402
from flyaim.retina.encoder import Retina  # noqa: E402


def make_frames(n: int, seed: int = 0) -> np.ndarray:
    """混合帧:纯随机噪声 + 靶场风格(暗底 + 亮块 + 白十字)。"""
    rng = np.random.default_rng(seed)
    out = np.empty((n, 480, 640, 3), dtype=np.uint8)
    for i in range(n):
        if i % 3 == 0:
            out[i] = rng.integers(0, 256, size=(480, 640, 3), dtype=np.uint8)
        else:
            f = np.full((480, 640, 3), 18, dtype=np.uint8)
            f[..., 2] = 22
            cx, cy = rng.integers(60, 580), rng.integers(60, 420)
            yy, xx = np.ogrid[:480, :640]
            d2 = (xx - cx) ** 2 + (yy - cy) ** 2
            f[d2 <= 22 * 22] = (235, 70, 70)
            xh, yh = rng.integers(40, 600), rng.integers(40, 440)
            f[max(0, yh - 1):yh + 1, max(0, xh - 24):xh + 25] = (240, 240, 240)
            f[max(0, yh - 24):yh + 25, max(0, xh - 1):xh + 1] = (240, 240, 240)
            out[i] = f
    return out


def main() -> int:
    import cupy as cp
    from flyaim.gpu.retina_gpu import RetinaGPU
    from flyaim.io import load_roles

    frames = make_frames(24)
    roles = load_roles(ROOT / "flyaim" / "data" / "build" / "roles.json")

    RTOL = 1e-5
    cpu = Retina(RetinaConfig(), input_neuron_ids=roles.visual_input)
    ok = True

    def rel(ref, got) -> float:
        denom = max(float(np.abs(ref).max()), 1e-9)
        return float(np.abs(ref - got).max()) / denom

    # ---- B=1:逐帧 ------------------------------------------------
    gpu1 = RetinaGPU(cpu, n_env=1)
    worst = 0.0
    for i in range(len(frames)):
        ref = cpu.frame_to_spikes(frames[i])
        got = cp.asnumpy(gpu1.frame_to_spikes(frames[i:i + 1]))[0]
        r = rel(ref, got)
        worst = max(worst, r)
        if r > RTOL:
            ok = False
            print(f"  [B=1] frame {i}: rel={r:.3e} 超差")
    print(f"[B=1] 24 帧 max 相对误差 = {worst:.3e}  {'OK' if worst <= RTOL else 'FAIL'}")

    # ---- B=N:批量(逐 env 独立流,应与 B=1 一致) -----------------
    n = len(frames) // 2
    cpu2 = Retina(RetinaConfig(), input_neuron_ids=roles.visual_input)
    gpuN = RetinaGPU(cpu2, n_env=n)
    batch = frames[:n]
    ref_all = np.stack([cpu2.frame_to_spikes(batch[i]) for i in range(n)])
    got_all = cp.asnumpy(gpuN.frame_to_spikes(batch))
    rN = rel(ref_all, got_all)
    if rN > RTOL:
        ok = False
    print(f"[B={n}] {n} 帧同批 max 相对误差 = {rN:.3e}  {'OK' if rN <= RTOL else 'FAIL'}")
    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
