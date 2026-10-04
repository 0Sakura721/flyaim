"""批量反传 vs 单环境反传 等价性验证。

    & $py tools/verify_gpu_backward.py

做法:B 个环境喂**完全相同**的轨迹 -> 批量梯度的均值应等于单环境梯度。
在同一初始权重上分别调 ``backward_batch``(批量)与 ``backward_steps``(单帧),
比较更新后的 W/Wz/g/gz/R(Adam 确定,故应逐参数接近)。

通过标准:相对误差 ≤ 1e-4(允许 float32 归约与 kernel 求和次序差异)。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from flyaim.ann_gated import GatedConnectomeRNN  # noqa: E402


def main() -> int:
    import cupy as cp

    B, T = 3, 5
    rng = np.random.default_rng(7)
    # 所有环境用同一条 U 序列(广播到 B 列)与同一 Y
    net = GatedConnectomeRNN("real")
    U1 = cp.asarray(rng.standard_normal((1536,)).astype(np.float32))
    Y1 = cp.asarray(rng.standard_normal((2,)).astype(np.float32))
    # 打破 Wz=0.5W 关系,确保两条路径真正独立
    jitter = cp.asarray((1 + 0.1 * rng.standard_normal(net.Wz.data.shape)).astype(np.float32))
    net.Wz.data[...] = cp.asarray(net.Wz.data * jitter, dtype=cp.float32)
    net.Wz.sum_duplicates()

    # ---- 批量:构造 B 条相同轨迹 ---------------------------------
    st0 = net.save_state()
    hb = cp.zeros((net.n, B), dtype=cp.float32)
    tr_b = []
    U_b = cp.tile(U1[:, None], (1, B))
    for _ in range(T):
        a, hn, PU, z, cand = net.forward_batch(U_b, hb)
        tr_b.append({"a": a, "h": hn, "h_prev": hb, "PU": PU, "z": z, "cand": cand})
        hb = hn
    Y_b = [cp.tile(Y1[None, :], (B, 1)) for _ in range(T)]
    net.backward_batch(tr_b, Y_b)
    gpu_state = net.save_state()

    # ---- 单环境:同一轨迹,逐帧 ---------------------------------
    net.load_state(st0)
    h = cp.zeros(net.n, dtype=cp.float32)
    tr_s = []
    for _ in range(T):
        a, hn, PU, z, cand = net.forward_step(U1, h)
        tr_s.append({"a": a, "h": hn, "h_prev": h, "PU": PU, "z": z, "cand": cand})
        h = hn
    net.backward_steps(tr_s, [np.asarray(Y1.get(), dtype=np.float32) for _ in range(T)])
    cpu_state = net.save_state()

    ok = True
    for key in ("W_data", "Wz_data", "g", "gz", "R"):
        a = np.asarray(gpu_state[key])
        b = np.asarray(cpu_state[key])
        absd = float(np.abs(a - b).max())
        denom = max(float(np.abs(b).max()), 1e-9)
        rel = absd / denom
        # 判据:相对 ≤1e-4 或绝对 ≤1e-5。R 的更新是 Adam 对 ~±lr(3e-4) 的
        # 读出取符号,梯度里的 float32 归约噪声会让个别分量绝对值差 ~4e-7,
        # 此时相对误差被放大了三个数量级,故用绝对容差。
        good = (rel <= 1e-4) or (absd <= 1e-5)
        if not good:
            ok = False
        print(f"  {key:9s} abs {absd:.3e}  rel {rel:.3e}  {'OK' if good else 'FAIL'}")
    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
