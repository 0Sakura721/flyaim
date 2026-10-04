"""连接组约束率网络(CONTRACT_ANN.md 附录 A):架构 = 真实接线图,训练 = Adam。

    h_t = tanh(W·h_{t-1} + g ⊙ (P·u_t) + b)     W: 25.6M 边稀疏(可训练)
    a_t = clip(R·h_t, −1, 1)                     P: 冻结随机输入投影(每行16非零)
                                                 g: 逐神经元输入增益(可训练)
                                                 R: 读出(可训练)

训练:行为克隆(导师 = seek,在线给标签),截断 BPTT(chunk=32),
手写 Adam(cupy 稀疏前向/反向 + 元素级 gather)。零新依赖。

诚实边界:这是"连接组作为网络架构"的研究纲领(Lappalainen et al. 2024
先例;网传 Apex 项目 YOLO+训练全部神经元的正规化),不是生物学习。
主张边界见预注册 A0/A4。
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np

from flyaim.config import ArenaConfig, RetinaConfig
from flyaim.io import ConnectomeArtifacts, load_roles

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "flyaim" / "data" / "build"
NORM = ROOT / "flyaim" / "runs" / "norm"

SEED = 20261004
CHUNK = 32
EYE = 768          # 24×32 小眼
U_DIM = EYE * 2    # u = 1536 维(ON/OFF 拼接)
LIF_SCALE = 3.2    # indeg 预归一化 × w_scale(16)×dt/tau(0.2)


def arm_weights(arm: str) -> Path:
    if arm == "real":
        return NORM / "connectome_indeg.npz"
    if arm == "shuffle":
        return NORM / "shuffle" / "connectome_shuffle_seed0.npz"
    raise ValueError(arm)


def retina_drive_to_u(retina, drive: np.ndarray) -> np.ndarray:
    """(n_cells,2) 驱动 → 1536 维小眼向量(ON/OFF 各 768,组内均值,/200)。"""
    on = np.zeros(EYE, dtype=np.float32)
    off = np.zeros(EYE, dtype=np.float32)
    cnt = np.zeros(EYE, dtype=np.float32)
    for rows, cells in ((retina._rows_lum, retina._cell_lum),
                        (retina._rows_r7, retina._cell_r7),
                        (retina._rows_r8, retina._cell_r8)):
        if rows.size == 0:
            continue
        np.add.at(on, cells, drive[rows, 0])
        np.add.at(off, cells, drive[rows, 1])
        np.add.at(cnt, cells, 1.0)
    np.maximum(cnt, 1.0, out=cnt)
    on /= cnt
    off /= cnt
    return (np.concatenate([on, off]) / 200.0).astype(np.float32)


def _scale_spectral_radius(W: "sp.csr_matrix", target: float = 0.9,
                           iters: int = 30) -> "sp.csr_matrix":
    """幂迭代估计谱半径并缩放到 target(A1 v3;CPU scipy,一次性)。"""
    import scipy.sparse as sp

    v = np.random.default_rng(SEED).standard_normal(W.shape[0])
    v /= np.linalg.norm(v)
    rho = 1.0
    for _ in range(iters):
        v = W @ v
        nrm = np.linalg.norm(v)
        if nrm < 1e-12:
            return W
        v /= nrm
        rho = nrm
    scale = target / max(rho, 1e-12)
    return (W * scale).tocsr()


class ConnectomeRNN:
    """稀疏率网络 + 截断 BPTT + 手写 Adam。参数驻留显存。"""

    def __init__(self, arm: str):
        import cupy as cp
        import cupyx.scipy.sparse as cpsp

        self.cp = cp
        art = ConnectomeArtifacts.load(arm_weights(arm))
        W = (art.W_exc * np.float32(LIF_SCALE)
             - art.W_inh * np.float32(LIF_SCALE)).tocsr().astype(np.float32)
        W = W.astype(np.float64)
        W = _scale_spectral_radius(W, target=0.9)  # A1 v3:tanh RNN 稳定初始化
        W = W.astype(np.float32).tocsr()
        W.sort_indices()
        if arm == "shuffle":  # 行内重排:行和精确不变,具体接线被打乱(P2 同哲学)
            rng = np.random.default_rng(SEED)
            data = W.data.copy()
            for i in range(W.shape[0]):
                seg = slice(W.indptr[i], W.indptr[i + 1])
                rng.shuffle(data[seg])
            W = type(W)((data, W.indices, W.indptr), shape=W.shape)

        self.n = W.shape[0]
        self.m = W.nnz
        self.W = cpsp.csr_matrix(
            (cp.asarray(W.data), cp.asarray(W.indices.astype(np.int32)),
             cp.asarray(W.indptr.astype(np.int32))), shape=W.shape)
        self.WT = cpsp.csr_matrix(self.W.T.tocsr())
        self.rows = cp.repeat(cp.arange(self.n, dtype=cp.int32),
                              cp.diff(self.W.indptr).astype(cp.int32))
        self.cols = self.W.indices

        rng = cp.random.default_rng(SEED)
        k = 16
        pnz = self.n * k
        P_rows = cp.repeat(cp.arange(self.n, dtype=cp.int32), k)
        P_cols = (rng.random(pnz) * U_DIM).astype(cp.int32)
        P_vals = (rng.standard_normal(pnz) / np.sqrt(k)).astype(cp.float32)
        indptr = cp.zeros(self.n + 1, dtype=cp.int32)
        indptr[1:] = cp.cumsum(cp.full(self.n, k, dtype=cp.int32))
        self.P = cpsp.csr_matrix((P_vals, P_cols, indptr), shape=(self.n, U_DIM))

        self.g = (rng.standard_normal(self.n) * 0.3).astype(cp.float32)  # A1 v2:破对称
        self.R = cp.zeros((2, self.n), dtype=cp.float32)
        self.b = (rng.standard_normal(self.n) * 0.3).astype(cp.float32)
        self.lr = 3e-4  # A1 v4
        self.t_adam = 0
        self._adam_state = {}

    # ---------------------------------------------------------------- 前向(逐步)

    def _readout_vec(self, h):
        """R·h,逐元素实现(避免 cuBLAS;R 只有 2 行)。"""
        cp = self.cp
        return cp.stack([cp.sum(self.R[0] * h), cp.sum(self.R[1] * h)])

    def forward_step(self, u: np.ndarray, h):
        cp = self.cp
        PU_t = self.P.dot(cp.asarray(u))                  # (n,)
        pre = self.W.dot(h) + self.g * PU_t + self.b
        h_new = cp.tanh(pre)
        a = np.clip(cp.asnumpy(self._readout_vec(h_new)), -1.0, 1.0).astype(np.float32)
        return a, h_new, PU_t

    # ---------------------------------------------------------------- 反传(分块)

    CLIP_NORM = 1.0  # A1 v3

    def renormalize_spectral_radius(self, target: float = 0.9, iters: int = 20) -> float:
        """训练后 W 重投影回目标谱半径(A1 v4,每 16 chunk)。GPU 幂迭代。"""
        cp = self.cp
        import numpy as _np
        v = cp.asarray(_np.random.default_rng(0).standard_normal(
            self.n).astype(cp.float32))   # numpy RNG:本机无 curand
        v /= cp.sqrt(cp.sum(v * v))
        rho = 1.0
        for _ in range(iters):
            v = self.W.dot(v)
            nrm = cp.sqrt(cp.sum(v * v))
            if float(nrm) < 1e-12:
                return 0.0
            v = v / nrm
            rho = float(nrm)
        self.W.data[...] = self.W.data * cp.float32(target / max(rho, 1e-12))
        return rho

    def backward_steps(self, traces: list[dict], ys: list[np.ndarray]) -> float:
        """traces[t] = {h_prev, h, PU};ys[t] = (2,) 导师动作。TBPTT + Adam。"""
        cp = self.cp
        T = len(traces)
        dR = cp.zeros((2, self.n), dtype=cp.float32)
        dg = cp.zeros(self.n, dtype=cp.float32)
        gW = cp.zeros(self.m, dtype=cp.float32)
        dh_next = cp.zeros(self.n, dtype=cp.float32)
        loss = 0.0
        # 不用稠密 GEMM/GEMV(本机无 cuBLAS):dR 逐帧元素级累加
        for t in range(T):
            e_t = cp.asarray(traces[t]["a"] - ys[t], dtype=cp.float32)
            dR[0] += e_t[0] * traces[t]["h"]
            dR[1] += e_t[1] * traces[t]["h"]
        for t in range(T - 1, -1, -1):
            tr = traces[t]
            a_err = cp.asarray(tr["a"] - ys[t], dtype=cp.float32)  # (2,)
            loss += float(cp.sum(a_err * a_err)) / (2.0 * T)
            dh = self.RT_dot(a_err) + self.WT.dot(dh_next)
            dpre = dh * (1.0 - tr["h"] * tr["h"])
            gW += dpre[self.rows] * tr["h_prev"][self.cols]
            dg += dpre * tr["PU"]
            dh_next = dpre
        dR *= 2.0 / T
        dg *= 2.0 / T
        gW *= 2.0 / T
        # 全局范数裁剪(A1 v3):三者合并范数超 1.0 时等比缩放
        total = float(cp.sqrt(cp.sum(dR * dR) + cp.sum(dg * dg)
                              + cp.sum(gW * gW)))
        if total > self.CLIP_NORM:
            f = self.CLIP_NORM / total
            dR *= f; dg *= f; gW *= f
        self._adam("W", self.W.data, gW)
        self._adam("g", self.g, dg)
        self._adam("R", self.R, dR)
        return loss

    def RT_dot(self, v):
        """(2,) 向量回传:逐元素组合,避免 cuBLAS(本机未装)。"""
        cp = self.cp
        v = cp.asarray(v)
        return v[0] * self.R[0] + v[1] * self.R[1]

    def _adam(self, name, param, grad):
        cp = self.cp
        st = self._adam_state.setdefault(
            name, {"m": cp.zeros_like(param), "v": cp.zeros_like(param)})
        st["m"] = 0.9 * st["m"] + 0.1 * grad
        st["v"] = 0.999 * st["v"] + 0.001 * grad * grad
        self.t_adam += 1
        mhat = st["m"] / (1 - 0.9 ** self.t_adam)
        vhat = st["v"] / (1 - 0.999 ** self.t_adam)
        param -= self.lr * mhat / (cp.sqrt(vhat) + 1e-8)

    # ---------------------------------------------------------------- 闭环/存取

    def act_closed_loop(self, u: np.ndarray, h) -> np.ndarray:
        cp = self.cp
        pre = self.W.dot(h) + self.g * (self.P.dot(cp.asarray(u))) + self.b
        h_new = cp.tanh(pre)
        return np.clip(cp.asnumpy(self._readout_vec(h_new)), -1.0, 1.0).astype(np.float32), h_new

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        cp = self.cp
        np.savez_compressed(path,
                            W_data=cp.asnumpy(self.W.data),
                            W_indices=cp.asnumpy(self.W.indices),
                            W_indptr=cp.asnumpy(self.W.indptr),
                            g=cp.asnumpy(self.g), R=cp.asnumpy(self.R),
                            b=cp.asnumpy(self.b))
        logger.info("ANN 权重已保存 %s", path)


def train_arm(arm: str, frames: int = 12000, out_dir: Path | None = None) -> dict:
    out_dir = Path(out_dir) if out_dir else (ROOT / "flyaim/runs/ann" / arm)
    out_dir.mkdir(parents=True, exist_ok=True)
    net = ConnectomeRNN(arm)
    roles = load_roles(D / "roles.json")
    from flyaim.retina.encoder import Retina
    from flyaim.bridge.controllers import SeekController

    retina = Retina(RetinaConfig(), input_neuron_ids=roles.visual_input)
    teacher = SeekController(ref_color=(235, 70, 70), tolerance=80.0,
                             use_aim_detect=False)
    from flyaim.arena.arena import Arena

    arena = Arena(ArenaConfig(max_frames=900), seed=0)
    cp = net.cp
    h = cp.zeros(net.n, dtype=cp.float32)

    t0 = time.perf_counter()
    frame_img = arena.reset()
    losses: list[float] = []
    n = 0
    best = {"loss": float("inf"), "state": None, "frame": -1}  # A1 v5
    print(f"[{arm}] ANN 训练开始: {frames} 帧, chunk={CHUNK}", flush=True)
    while n < frames:
        traces: list[dict] = []
        ys: list[np.ndarray] = []
        for _ in range(min(CHUNK, frames - n)):
            drive = retina.frame_to_spikes(frame_img)
            u = retina_drive_to_u(retina, drive)
            y = np.asarray(teacher.act(frame_img), dtype=np.float32)
            a, h_new, PU_t = net.forward_step(u, h)
            traces.append({"a": a, "h": h_new, "h_prev": h, "PU": PU_t})
            ys.append(y)
            h = h_new
            res = arena.step(a)
            frame_img = res.frame
            n += 1
        loss = net.backward_steps(traces, ys)
        net.renormalize_spectral_radius(0.9)  # A1 v4b:逐 chunk 重投影
        losses.append(loss)
        # A1 v5:最优检查点(最近 100 chunk 均值改进时,状态快照到 CPU)
        recent = losses[-100:]
        cur = float(np.mean(recent))
        if cur < best["loss"]:
            best["loss"] = cur
            best["frame"] = n
            best["state"] = {"W_data": net.cp.asnumpy(net.W.data).copy(),
                             "g": net.cp.asnumpy(net.g).copy(),
                             "R": net.cp.asnumpy(net.R).copy(),
                             "b": net.cp.asnumpy(net.b).copy()}
        if n % 500 < CHUNK:
            fps = n / max(time.perf_counter() - t0, 1e-9)
            print(f"  [{arm}] f={n} bc_loss={loss:.4f} ({fps:.1f} f/s)", flush=True)
    wall = time.perf_counter() - t0
    net.save(out_dir / f"{arm}_ann_final.npz")
    if best["state"] is not None:
        cp = net.cp
        best_np = {k: v for k, v in best["state"].items()}
        np.savez_compressed(out_dir / f"{arm}_ann_best.npz", **best_np)
        print(f"[{arm}] 最优检查点 @帧{best['frame']} loss={best['loss']:.4f} 已保存",
              flush=True)
    out = {"arm": arm, "frames": n, "wall_s": round(wall, 1),
           "loss_first": losses[0], "loss_last100": float(np.mean(losses[-100:])),
           "best_loss": best["loss"], "best_frame": best["frame"]}
    (out_dir / f"{arm}_train.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[{arm}] 训练完成: {n} 帧 / {wall:.0f}s,loss {losses[0]:.4f} → "
          f"{out['loss_last100']:.4f}", flush=True)
    return out


def eval_arm(net: ConnectomeRNN, seeds: int = 10, frames: int = 900,
             tag: str = "") -> list[dict]:
    roles = load_roles(D / "roles.json")
    from flyaim.retina.encoder import Retina
    from flyaim.arena.arena import Arena
    from flyaim.config import ArenaConfig

    retina = Retina(RetinaConfig(), input_neuron_ids=roles.visual_input)
    results = []
    for seed in range(seeds):
        arena = Arena(ArenaConfig(max_frames=frames), seed=seed)
        retina.reset()
        frame_img = arena.reset()
        h = net.cp.zeros(net.n, dtype=net.cp.float32)
        dists: list[float] = []
        hits = 0
        for _ in range(frames):
            drive = retina.frame_to_spikes(frame_img)
            u = retina_drive_to_u(retina, drive)
            a, h = net.act_closed_loop(u, h)
            res = arena.step(a)
            frame_img = res.frame
            dists.append(float(res.target_dist))
            hits += int(res.hit)
            if res.done:
                break
        results.append({"seed": seed, "mean_dist_px": float(np.mean(dists)),
                        "hits": hits})
        print(f"  eval[{tag}] seed={seed} mean_dist={results[-1]['mean_dist_px']:.0f}px "
              f"hits={hits}", flush=True)
    return results
