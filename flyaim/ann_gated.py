"""门控连接组率网络(CONTRACT_ANN.md 附录 A2)。

    z_t   = σ(Wz·h_{t-1} + gz⊙(P·u_t) + bz)     W/Wz: 25.6M 边稀疏(可训练)
    cand  = tanh(W·h_{t-1} + g⊙(P·u_t) + b)      P: 冻结随机输入投影
    h_t   = (1−z_t)⊙h_{t-1} + z_t⊙cand           g/gz/b/bz: 可训练向量
    a_t   = clip(R·h_t, −1, 1)                   R: 读出(可训练)

门控(GRU 思想,套在连接组拓扑上)让网络自行把混沌递归调到稳定区
—— 对策 D22/D23 的梯度爆炸。TBPTT + Adam + 谱半径保底重投影。
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np

from flyaim.config import ArenaConfig, RetinaConfig
from flyaim.io import ConnectomeArtifacts, load_roles
from flyaim.ann_train import arm_weights, retina_drive_to_u

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "flyaim" / "data" / "build"
SEED = 20261004
CHUNK = 32
PUBLISH_EVERY = 10  # 仪表盘发布粒度(帧)。1 = 逐帧;10 = 每 10 帧(2026-10-04 用户要求)


def _spectral_scale(W, target=0.9, iters=30):
    """CPU scipy:幂迭代谱半径缩放(A1 v3 程序)。"""
    v = np.random.default_rng(SEED).standard_normal(W.shape[0])
    v /= np.linalg.norm(v)
    rho = 1.0
    for _ in range(iters):
        v = W @ v
        nrm = np.linalg.norm(v)
        if nrm < 1e-12:
            return W, 1.0
        v = v / nrm
        rho = nrm
    return (W * (target / max(rho, 1e-12))).tocsr(), target / max(rho, 1e-12)


class GatedConnectomeRNN:
    def __init__(self, arm: str, lr: float = 3e-4):
        import cupy as cp
        import cupyx.scipy.sparse as cpsp

        self.cp = cp
        self.lr = lr
        art = ConnectomeArtifacts.load(arm_weights(arm))
        W = (art.W_exc * np.float32(3.2)
             - art.W_inh * np.float32(3.2)).tocsr().astype(np.float64)
        W, _ = _spectral_scale(W, 0.9)
        W = W.astype(np.float32).tocsr()
        W.sort_indices()
        if arm == "shuffle":  # 行内重排(行和不变,接线打乱;A1 v2 同哲学)
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
        self.Wz = cpsp.csr_matrix(
            (self.W.data * 0.5, self.W.indices, self.W.indptr), shape=W.shape)
        self.WT = cpsp.csr_matrix(self.W.T.tocsr())
        self.WzT = cpsp.csr_matrix(self.Wz.T.tocsr())
        self.rows = cp.repeat(cp.arange(self.n, dtype=cp.int32),
                              cp.diff(self.W.indptr).astype(cp.int32))
        self.cols = self.W.indices

        rng = cp.random.default_rng(SEED)
        k, pnz = 16, self.n * 16
        P_rows = cp.repeat(cp.arange(self.n, dtype=cp.int32), k)
        P_cols = (rng.random(pnz) * 1536).astype(cp.int32)
        P_vals = (rng.standard_normal(pnz) / np.sqrt(k)).astype(cp.float32)
        indptr = cp.zeros(self.n + 1, dtype=cp.int32)
        indptr[1:] = cp.cumsum(cp.full(self.n, k, dtype=cp.int32))
        self.P = cpsp.csr_matrix((P_vals, P_cols, indptr), shape=(self.n, 1536))

        self.g = (rng.standard_normal(self.n) * 0.3).astype(cp.float32)
        self.gz = (rng.standard_normal(self.n) * 0.3).astype(cp.float32)
        self.b = (rng.standard_normal(self.n) * 0.3).astype(cp.float32)
        self.bz = (rng.standard_normal(self.n) * 0.3).astype(cp.float32)
        self.R = cp.zeros((2, self.n), dtype=cp.float32)
        self.t_adam = 0
        self._adam_state = {}
        self._clip = 1.0

    # ---------------------------------------------------------------- 前向

    def forward_step(self, u: np.ndarray, h):
        cp = self.cp
        PU_t = self.P.dot(cp.asarray(u))                  # (n,)
        pre_z = self.Wz.dot(h) + self.gz * PU_t + self.bz
        z = 1.0 / (1.0 + cp.exp(-pre_z))
        pre_h = self.W.dot(h) + self.g * PU_t + self.b
        cand = cp.tanh(pre_h)
        h_new = (1.0 - z) * h + z * cand
        a = np.clip(cp.asnumpy(self._readout_vec(h_new)), -1.0, 1.0).astype(np.float32)
        return a, h_new, PU_t, z, cand

    def _readout_vec(self, h):
        cp = self.cp
        return cp.stack([cp.sum(self.R[0] * h), cp.sum(self.R[1] * h)])

    # ---------------------------------------------------------------- 反传

    def backward_steps(self, traces: list[dict], ys: list[np.ndarray]) -> float:
        """正确序的 BPTT:h_{t-1} 经直连、pre_h、pre_z 三条路径回传。"""
        cp = self.cp
        T = len(traces)
        dR = cp.zeros((2, self.n), dtype=cp.float32)
        dg = cp.zeros(self.n, dtype=cp.float32)
        dgz = cp.zeros(self.n, dtype=cp.float32)
        gW = cp.zeros(self.m, dtype=cp.float32)
        gWz = cp.zeros(self.m, dtype=cp.float32)
        loss = 0.0
        for t in range(T):  # dR 逐帧累加(无 cuBLAS)
            e_t = cp.asarray(traces[t]["a"] - ys[t], dtype=cp.float32)
            dR[0] += e_t[0] * traces[t]["h"]
            dR[1] += e_t[1] * traces[t]["h"]
        dh_carry = cp.zeros(self.n, dtype=cp.float32)  # ∂L/∂h_t 来自未来步
        for t in range(T - 1, -1, -1):
            tr = traces[t]
            a_err = cp.asarray(tr["a"] - ys[t], dtype=cp.float32)
            loss += float(cp.sum(a_err * a_err)) / (2.0 * T)
            dh_t = self.RT_dot(a_err) + dh_carry
            # 直连项:h_{t-1} += dh_t ⊙ (1−z)
            # pre_h 项:dpre_h = dh_t ⊙ z ⊙ (1−cand²),经 W^T 回传
            # pre_z 项:dpre_z = dh_t ⊙ (cand − h_prev) ⊙ z(1−z),经 Wz^T 回传
            dpre_h = dh_t * tr["z"] * (1.0 - tr["cand"] ** 2)
            dpre_z = dh_t * (tr["cand"] - tr["h_prev"]) * tr["z"] * (1.0 - tr["z"])
            dh_carry = dh_t * (1.0 - tr["z"]) + self.WT.dot(dpre_h)                 + self.WzT.dot(dpre_z)
            gW += dpre_h[self.rows] * tr["h_prev"][self.cols]
            gWz += dpre_z[self.rows] * tr["h_prev"][self.cols]
            dg += dpre_h * tr["PU"]
            dgz += dpre_z * tr["PU"]
        s = 2.0 / T
        dR *= s; dg *= s; dgz *= s; gW *= s; gWz *= s
        # 逐参数组 L2 裁剪(全局范数会被 25.6M 条边的 gW 支配,把 dR 压成 0)
        for grad in (dR, dg, dgz, gW, gWz):
            nrm = cp.sqrt(cp.sum(grad * grad))
            f = cp.minimum(1.0, self._clip / (nrm + 1e-9))
            grad *= f
        self.t_adam += 1
        self._adam("R", self.R, dR)      # 读出(重写时曾漏掉 → R 恒定 0)
        self._adam("W", self.W.data, gW)
        self._adam("Wz", self.Wz.data, gWz)
        self._adam("g", self.g, dg)
        self._adam("gz", self.gz, dgz)
        return loss

    def RT_dot(self, v):
        cp = self.cp
        v = cp.asarray(v)
        return v[0] * self.R[0] + v[1] * self.R[1]

    def _adam(self, name, param, grad):
        cp = self.cp
        st = self._adam_state.setdefault(
            name, {"m": cp.zeros_like(param), "v": cp.zeros_like(param)})
        st["m"] = 0.9 * st["m"] + 0.1 * grad
        st["v"] = 0.999 * st["v"] + 0.001 * grad * grad
        mhat = st["m"] / (1 - 0.9 ** self.t_adam)
        vhat = st["v"] / (1 - 0.999 ** self.t_adam)
        param -= self.lr * mhat / (cp.sqrt(vhat) + 1e-8)

    def renorm_spectral(self, target=0.9, iters=15) -> None:
        """保底:每 chunk 把 W/Wz 幂迭代重投影(防训练爬出稳定区)。"""
        cp = self.cp
        for M in (self.W, self.Wz):
            v = cp.asarray(np.random.default_rng(0).standard_normal(
                self.n).astype(cp.float32))
            v /= cp.sqrt(cp.sum(v * v))
            rho = 1.0
            for _ in range(iters):
                v = M.dot(v)
                nrm = cp.sqrt(cp.sum(v * v))
                if float(nrm) < 1e-12:
                    break
                v = v / nrm
                rho = float(nrm)
            M.data[...] = M.data * cp.float32(target / max(rho, 1e-12))

    def act_closed_loop(self, u: np.ndarray, h):
        cp = self.cp
        PU_t = self.P.dot(cp.asarray(u))
        pre_z = self.Wz.dot(h) + self.gz * PU_t + self.bz
        z = 1.0 / (1.0 + cp.exp(-pre_z))
        pre_h = self.W.dot(h) + self.g * PU_t + self.b
        cand = cp.tanh(pre_h)
        h_new = (1.0 - z) * h + z * cand
        return (np.clip(cp.asnumpy(self._readout_vec(h_new)), -1.0, 1.0)
                .astype(np.float32)), h_new

    def save_state(self) -> dict:
        cp = self.cp
        return {"W_data": cp.asnumpy(self.W.data).copy(),
                "Wz_data": cp.asnumpy(self.Wz.data).copy(),
                "g": cp.asnumpy(self.g).copy(), "gz": cp.asnumpy(self.gz).copy(),
                "b": cp.asnumpy(self.b).copy(), "bz": cp.asnumpy(self.bz).copy(),
                "R": cp.asnumpy(self.R).copy()}

    def load_state(self, st: dict) -> None:
        cp = self.cp
        self.W.data[...] = cp.asarray(st["W_data"])
        self.Wz.data[...] = cp.asarray(st["Wz_data"])
        self.g[...] = cp.asarray(st["g"])
        self.gz[...] = cp.asarray(st["gz"])
        self.b[...] = cp.asarray(st["b"])
        self.bz[...] = cp.asarray(st["bz"])
        self.R[...] = cp.asarray(st["R"])

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        cp = self.cp
        np.savez_compressed(
            path, W_data=cp.asnumpy(self.W.data), Wz_data=cp.asnumpy(self.Wz.data),
            g=cp.asnumpy(self.g), gz=cp.asnumpy(self.gz),
            b=cp.asnumpy(self.b), bz=cp.asnumpy(self.bz), R=cp.asnumpy(self.R))


def train_arm(arm: str, frames: int, out_dir: Path, publish=None) -> dict:
    """A2 程序:轮 0 = BC 12k(1 epoch);轮 1/2 = DAgger 8k(聚合 2 epochs)。

    publish(stage, **kv):可选状态发布钩子(仪表盘用)。每 100 rollout 帧与
    每轮训练后调用一次,传入 loss/fps/活动度等。
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    net = GatedConnectomeRNN(arm)
    roles = load_roles(D / "roles.json")
    from flyaim.retina.encoder import Retina
    from flyaim.bridge.controllers import SeekController
    from flyaim.ann_train import dagger, eval_arm

    retina = Retina(RetinaConfig(), input_neuron_ids=roles.visual_input)
    teacher = SeekController(ref_color=(235, 70, 70), tolerance=80.0,
                             use_aim_detect=False)
    cp = net.cp
    h = cp.zeros(net.n, dtype=cp.float32)

    def rollout_collect(policy_driven: bool, n_frames: int, seed_off: int) -> list:
        from flyaim.arena.arena import Arena

        buffer = []
        collected = 0
        seed = seed_off
        # 每个 arena 跑一个 episode(≤900 帧),累计到恰好 n_frames 帧。
        # 修正:旧实现内层 range(n_frames) 叠加 n_frames//900+1 个 arena,
        # 实际采集 ~14× 预算帧数,违反 C2 预注册预算 —— 2026-10-04 发现。
        while collected < n_frames:
            a = Arena(ArenaConfig(max_frames=900), seed=seed)
            seed += 1
            r2 = Retina(RetinaConfig(), input_neuron_ids=roles.visual_input)
            hh = cp.zeros(net.n, dtype=cp.float32)
            f = a.reset()
            ep = []
            for _ in range(min(900, n_frames - collected)):
                drive = r2.frame_to_spikes(f)
                u = retina_drive_to_u(r2, drive)
                y = np.asarray(teacher.act(f), dtype=np.float32)
                ep.append((u, y))
                if policy_driven:
                    a_act, hh = net.act_closed_loop(u, hh)
                    PU_view = None
                else:
                    # 教师驱动动作,但网络仍前向传播 —— 否则 h 恒为零,
                    # 仪表盘活动全灰(2026-10-04 实测 bug)
                    _, hh, PU_view, _, _ = net.forward_step(u, hh)
                    a_act = y.astype(np.float32)
                res = a.step(a_act)
                f = res.frame
                collected += 1
                if publish is not None and collected % PUBLISH_EVERY == 0:
                    # 只把采样点搬回 CPU(GPU 侧先切片):全脑 166,700 个
                    # 每帧 667 KB + JSON 编码会明显拖慢;采样后仅 ~16 KB。
                    act_idx = getattr(publish, "sample_idx", None)
                    idx = None if act_idx is None else cp.asarray(act_idx)
                    hi = cp.abs(hh)
                    if PU_view is not None:
                        # 教师阶段用 |输入驱动| 反映"信号到达强度"(非零)
                        hi = hi + cp.abs(PU_view) * 10.0
                    h_h = cp.asnumpy(hi if idx is None else hi[idx])
                    publish("rollout", policy=policy_driven, frames=collected,
                            target_dist=float(res.target_dist),
                            hit=bool(res.hit), activity=h_h, frame=f)
                if res.done or a.done:
                    break
            if ep:
                buffer.append(ep)
        return buffer

    def train_pass(buffer: list, epochs: int = 1) -> float:
        losses = []
        for _ in range(epochs):
            order = np.random.default_rng(SEED + net.t_adam).permutation(len(buffer))
            for ei in order:
                ep = buffer[ei]
                h = cp.zeros(net.n, dtype=cp.float32)
                traces, ys = [], []
                for u, y in ep:
                    a, h_new, PU_t, z, cand = net.forward_step(u, h)
                    traces.append({"a": a, "h": h_new, "h_prev": h, "PU": PU_t,
                                   "z": z, "cand": cand})
                    ys.append(y)
                    h = h_new
                losses.append(net.backward_steps(traces, ys))
                net.renorm_spectral(0.9)
        return float(np.mean(losses[-200:])) if losses else float("nan")

    t0 = time.perf_counter()
    log: list[dict] = []
    print(f"[{arm}] A2 轮 0:BC {frames} 帧 × 1 epoch", flush=True)
    buf = rollout_collect(False, frames, 0)
    l0 = train_pass(buf, epochs=1)
    log.append({"round": 0, "loss": l0})
    best = {"loss": l0, "state": net.save_state()}
    net.save(out_dir / f"{arm}_ann2_bc.npz")
    print(f"[{arm}] 轮 0 完成 loss={l0:.4f}({time.perf_counter()-t0:.0f}s)", flush=True)
    if publish is not None:
        publish("round_done", round=0, loss=l0, policy=False)

    res_bc = eval_arm(net, seeds=10, tag=f"{arm}-bc")
    if publish is not None:
        publish("eval", arm=f"{arm}-bc", values=[r["mean_dist_px"] for r in res_bc],
                hits=[r["hits"] for r in res_bc])

    for r, (n_dg, epochs) in ((1, (8000, 2)), (2, (8000, 2))):
        print(f"[{arm}] A2 轮 {r}:DAgger {n_dg} 帧 × {epochs} epochs", flush=True)
        buf += rollout_collect(True, n_dg, 100 * r)
        lr_loss = train_pass(buf, epochs=epochs)
        log.append({"round": r, "loss": lr_loss})
        cur = float(np.mean([e["loss"] for e in log[-3:]]))
        if cur < best["loss"]:
            best = {"loss": cur, "state": net.save_state()}
        print(f"[{arm}] 轮 {r} 完成 loss={lr_loss:.4f}({time.perf_counter()-t0:.0f}s)",
              flush=True)
        if publish is not None:
            publish("round_done", round=r, loss=lr_loss, policy=policy_driven)

    net.load_state(best["state"])
    net.save(out_dir / f"{arm}_ann2_best.npz")
    wall = time.perf_counter() - t0
    if publish is not None:
        res_fin = eval_arm(net, seeds=10, tag=f"{arm}-ann2")
        publish("eval", arm=f"{arm}-ann2",
                values=[r["mean_dist_px"] for r in res_fin],
                hits=[r["hits"] for r in res_fin])
    out = {"arm": arm, "rounds": log, "best_loss": best["loss"],
           "wall_s": round(wall, 1), "eval_bc": res_bc}
    (out_dir / f"{arm}_train.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[{arm}] 训练完成 {wall:.0f}s,最优 loss={best['loss']:.4f}", flush=True)
    return out


def eval_arm(net, seeds: int = 10, frames: int = 900, tag: str = "") -> list[dict]:
    """与 ann_train.eval_arm 同口径;h 逐 episode 归零。"""
    roles = load_roles(D / "roles.json")
    from flyaim.retina.encoder import Retina
    from flyaim.arena.arena import Arena
    from flyaim.config import ArenaConfig

    results = []
    for seed in range(seeds):
        arena = Arena(ArenaConfig(max_frames=frames), seed=seed)
        retina = Retina(RetinaConfig(), input_neuron_ids=roles.visual_input)
        frame_img = arena.reset()
        h = net.cp.zeros(net.n, dtype=net.cp.float32)
        dists, hits = [], 0
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
