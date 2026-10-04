"""A2 GPU 训练入口:GPU rollout + GPU TBPTT(替代 CPU rollout 版)。

    & $py tools/train_ann2_gpu.py --envs 64 --frames 12000

与 tools/train_ann2.py 的差别只在**数据生成与反传都在 GPU 上**:
    - GPUArenaBatch(B 环境) + RetinaGPU(批量) + 自定义 CSR SpMM;
    - 教师标签用特权坐标(零检测开销);
    - 反传用 backward_batch(批量 TBPTT,EdgeGrad 聚 gW)。

正确性由 tools/verify_gpu_retina.py、verify_gpu_backward.py 保证(与 CPU 版
等价,已通过)。本入口在 CPU 版上游最贵的两段(视网膜 15ms/帧 + 教师 18ms/帧)
上获得 6-7 倍吞吐提升,使 A2 战役从 ~2h 缩到 ~20min。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from flyaim.ann_train import EYE  # noqa: E402
from flyaim.config import ArenaConfig, RetinaConfig  # noqa: E402
from flyaim.gpu.arena_gpu import GPUArenaBatch  # noqa: E402
from flyaim.gpu.retina_gpu import RetinaGPU  # noqa: E402


def _drive_to_u(cp, drive, rows, cells):
    """(B,n,2) 驱动 -> (1536,B):ON/OFF 各 768,组内均值,/200。"""
    b = drive.shape[0]
    on = cp.zeros((b, EYE), dtype=cp.float32)
    off = cp.zeros((b, EYE), dtype=cp.float32)
    cnt = cp.zeros(EYE, dtype=cp.float32)
    for r, c in zip(rows, cells):
        if r.size == 0:
            continue
        cp.add.at(on, (slice(None), c), drive[:, r, 0])
        cp.add.at(off, (slice(None), c), drive[:, r, 1])
        cnt[c] += 1.0
    cnt = cp.maximum(cnt, 1.0)
    return (cp.concatenate([on / cnt, off / cnt], axis=1) / cp.float32(200.0)).T.astype(
        cp.float32)


def rollout_gpu(net, retina, arena, rows, cells, n_frames, policy_driven,
                publish=None, sample_idx=None, chunk=32):
    """采集 n_frames 帧(全局),返回 chunk 列表。

    **每个 chunk 只存输入**(U (1536,B) float32 + Y (B,2) float32,合计 ~49KB/B),
    不存激活 —— 激活有 166700 行,全存会 12GB+ OOM(实测)。训练时在 chunk 内
    重新前向重建 traces(CPU 版同哲学:只存 u,y,训练再 forward)。
    chunk 间截断,chunk 内完整 BPTT(与 CPU 版 CHUNK=32 一致)。
    """
    cp = net.cp
    retina.reset()
    frames = arena.reset()
    h = cp.zeros((net.n, arena.B), dtype=cp.float32)
    batches = []          # 每个元素: (list[U], list[Y])
    Us, Ys = [], []
    n = 0
    last_pub = 0
    while n < n_frames:
        drive = retina.frame_to_spikes(frames)
        U = _drive_to_u(cp, drive, rows, cells)
        a, h_new, PU, z, cand = net.forward_batch(U, h)
        Y = arena.teacher_labels()
        Us.append(U)
        Ys.append(Y)
        act = a.T if policy_driven else Y
        frames, dist, hit = arena.step(act)
        h = h_new
        n += arena.B
        if len(Us) >= chunk:
            batches.append((Us, Ys))
            Us, Ys = [], []
            # chunk 边界重置隐状态:训练时在 chunk 内从零前向重建,必须与
            # 采集一致(否则采集 h 与训练 h 错位)。CPU 版按 episode 重置,
            # 这里按 chunk 重置,粒度更细但自洽。
            h = cp.zeros((net.n, arena.B), dtype=cp.float32)
        if publish is not None and n - last_pub >= arena.B * 10:
            last_pub = n
            hh = None
            if sample_idx is not None:
                hh = cp.asnumpy(cp.abs(h)[cp.asarray(sample_idx)]).mean(axis=1)
            publish("rollout", policy=policy_driven, frames=n,
                    target_dist=float(cp.mean(dist)), hit=bool(cp.any(hit)),
                    activity=hh, frame=cp.asnumpy(frames[0]))
    if Us:
        batches.append((Us, Ys))
    return batches


def train_pass(net, batches, epochs=1):
    """对每个 chunk:chunk 内重放前向重建 traces,再做批量 TBPTT。"""
    cp = net.cp
    losses = []
    for _ in range(epochs):
        order = np.random.default_rng(1234 + net.t_adam).permutation(len(batches))
        for bi in order:
            Us, Ys = batches[bi]
            h = cp.zeros((net.n, arena_B(Us[0])), dtype=cp.float32)
            traces = []
            for U in Us:
                a, h_new, PU, z, cand = net.forward_batch(U, h)
                traces.append({"a": a, "h": h_new, "h_prev": h, "PU": PU,
                               "z": z, "cand": cand})
                h = h_new
            losses.append(net.backward_batch(traces, Ys))
            net.renorm_spectral(0.9)
    return float(np.mean(losses[-200:])) if losses else float("nan")


def arena_B(U) -> int:
    return int(U.shape[1])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--envs", type=int, default=64)
    ap.add_argument("--frames", type=int, default=12000)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--arms", default="real,shuffle")
    # TBPTT 长度 = 采集 chunk 长度。每帧 traces 存 4 个 (n,B) 张量:
    #   T=8/16/32 @B=64 约 1.4/2.7/5.5 GB。6GB 卡上 T=16 留足余量。
    ap.add_argument("--tbptt", type=int, default=16)
    args = ap.parse_args()

    import cupy as cp
    from flyaim.ann_gated import GatedConnectomeRNN, eval_arm
    from flyaim.io import load_roles, save_json
    from flyaim.retina.encoder import Retina

    ADIR = ROOT / "flyaim" / "runs" / "ann2_gpu"
    ADIR.mkdir(parents=True, exist_ok=True)
    roles = load_roles(ROOT / "flyaim" / "data" / "build" / "roles.json")
    cpu_retina = Retina(RetinaConfig(), input_neuron_ids=roles.visual_input)

    sys.path.insert(0, str(ROOT / "tools"))
    sample_idx = None
    layout = ROOT / ".cache/ann_layout.json"
    if layout.exists():
        sample_idx = np.array(json.loads(layout.read_text(encoding="utf-8"))["sample_idx"])
    from ann_dashboard import StatePublisher

    pub = StatePublisher(ROOT / ".cache/ann2_state.json", sample_idx=sample_idx,
                         total_frames=args.frames)

    t0 = time.perf_counter()
    results = {}
    for arm in args.arms.split(","):
        net = GatedConnectomeRNN(arm)
        retina = RetinaGPU(cpu_retina, n_env=args.envs)
        arena = GPUArenaBatch(ArenaConfig(max_frames=900), n_env=args.envs, seed=0)
        rows = [cp.asarray(getattr(cpu_retina, a), dtype=cp.int64)
                for a in ("_rows_lum", "_rows_r7", "_rows_r8")]
        cells = [cp.asarray(getattr(cpu_retina, a), dtype=cp.int64)
                 for a in ("_cell_lum", "_cell_r7", "_cell_r8")]

        rng = cp.random.default_rng(0)
        arena._rng = cp.random.default_rng(0)
        print(f"[{arm}] GPU 轮 0:BC {args.frames} 帧 (envs={args.envs})", flush=True)
        b1 = rollout_gpu(net, retina, arena, rows, cells, args.frames, False, pub,
                         sample_idx, chunk=args.tbptt)
        l0 = train_pass(net, b1, epochs=1)
        print(f"[{arm}] 轮 0 loss={l0:.4f} ({time.perf_counter()-t0:.0f}s)", flush=True)
        pub("round_done", round=0, loss=l0, policy=False)
        results[f"{arm}-bc"] = eval_arm(net, seeds=args.seeds, tag=f"{arm}-bc")
        pub("eval", arm=f"{arm}-bc",
            values=[r["mean_dist_px"] for r in results[f"{arm}-bc"]])

        for r in (1, 2):
            print(f"[{arm}] GPU 轮 {r}:DAgger 8000 帧 × 2 epochs", flush=True)
            b = rollout_gpu(net, retina, arena, rows, cells, 8000, True, pub, sample_idx,
                            chunk=args.tbptt)
            ll = train_pass(net, b, epochs=2)
            print(f"[{arm}] 轮 {r} loss={ll:.4f} ({time.perf_counter()-t0:.0f}s)",
                  flush=True)
            pub("round_done", round=r, loss=ll, policy=True)
        results[f"{arm}-ann2"] = eval_arm(net, seeds=args.seeds, tag=f"{arm}-ann2")
        pub("eval", arm=f"{arm}-ann2",
            values=[x["mean_dist_px"] for x in results[f"{arm}-ann2"]])
        print(f"[{arm}] 完成 {time.perf_counter()-t0:.0f}s", flush=True)

    out = {"arm_results": {k: [x["mean_dist_px"] for x in v] for k, v in results.items()},
           "wall_s": round(time.perf_counter() - t0, 1)}
    save_json(str(ADIR / "train_gpu.json"), out)
    print(f"\n完成,{out['wall_s']}s -> {ADIR/'train_gpu.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
