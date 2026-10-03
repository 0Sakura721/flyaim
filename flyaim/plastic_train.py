"""可塑性训练与评估(CONTRACT_PLASTIC.md 附录 P 的执行器)。

流程(train_arm):
    1. 装配引擎(真实或 shuffle 权重)+ 复眼 + 冻结随机投影读出
    2. 探针期 200 帧:η 标定(m=1,P2 v2 程序)+ 读出 g 标定(m=0,P4)
    3. 训练期:闭环 36,000 帧,每帧一次可塑性更新,周期性落盘权重快照
    4. 全程记录:靶距曲线、DN 放电率、|ΔW|、g

评估(eval_arm):载入指定权重与冻结的 g,10 seeds × 900 帧闭环,
输出逐 seed 平均靶距(主指标,见 D12)与命中数(参考)。

诚实边界:
    - 训练期调制因子 m 使用靶场状态(oracle)——与离线读出训练的 PID
      导师同合法性:训练信号,不是策略输入;
    - 评估期无任何 oracle,策略 = 视网膜 → 网络 → 冻结随机读出;
    - 网络状态跨 episode 连续(episode 只是靶场重置,可塑性不断开)。
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np

from flyaim.config import ArenaConfig, BrainConfig, PlasticityConfig, RetinaConfig
from flyaim.io import load_roles

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "flyaim" / "data" / "build"
NORM = ROOT / "flyaim" / "runs" / "norm"

# 与 tools/run_experiment.py 相同的 Phase 2 工作点
CAL_W_SCALE = 16.0
CAL_INPUT_GAIN = 1.5
CAL_NORM = "indeg"


class RandomProjectionReadout:
    """冻结的种子随机投影读出(P4):任何行为改善只能来自网络可塑性。"""

    def __init__(self, dn_ids: np.ndarray, seed: int = 20261004,
                 rate_norm: float = 50.0) -> None:
        self.dn = np.asarray(dn_ids, dtype=np.int64)
        rng = np.random.default_rng(seed)
        self.W = (rng.standard_normal((self.dn.size, 2)) /
                  np.sqrt(self.dn.size)).astype(np.float32)
        self.rate_norm = float(rate_norm)
        self.g = 1.0  # 探针期末按 P4 冻结

    def raw(self, rates: np.ndarray) -> np.ndarray:
        """未缩放未裁剪输出(g 标定用)。"""
        x = np.asarray(rates, dtype=np.float32)[self.dn]
        return (x @ self.W) / self.rate_norm

    def act(self, rates: np.ndarray) -> np.ndarray:
        return np.clip(self.g * self.raw(rates), -1.0, 1.0).astype(np.float32)


def arm_weights(arm: str) -> Path:
    if arm == "real":
        return NORM / f"connectome_{CAL_NORM}.npz"
    if arm == "shuffle":
        return NORM / "shuffle" / "connectome_shuffle_seed0.npz"
    raise ValueError(f"未知 arm: {arm}")


def _build_brain(weights_npz: Path):
    """⚠️ 与 run_experiment 严格一致:norm npz 已预归一化,引擎不再二次归一化
    (weight_norm 默认 "none")。曾经传 "indeg" 导致二次归一化 → 网络静默。"""
    from flyaim.brain.lif_gpu import ConnectomeGPU

    bc = BrainConfig(weight_scale_exc=CAL_W_SCALE, weight_scale_inh=CAL_W_SCALE,
                     input_gain=CAL_INPUT_GAIN)
    roles = load_roles(D / "roles.json")
    brain = ConnectomeGPU(weights_npz, bc, input_neuron_ids=roles.visual_input)
    return brain, roles


def train_arm(arm: str = "real", frames: int = 36000, device: str = "cuda",
              out_dir: str | Path | None = None,
              pcfg: PlasticityConfig | None = None,
              log_every: int = 200) -> dict:
    out_dir = Path(out_dir) if out_dir else (ROOT / "flyaim/runs/plastic" / f"{arm}")
    out_dir.mkdir(parents=True, exist_ok=True)
    pcfg = pcfg or PlasticityConfig()
    wp = arm_weights(arm)

    brain, roles = _build_brain(wp)
    from flyaim.retina.encoder import Retina

    retina = Retina(RetinaConfig(), input_neuron_ids=roles.visual_input)
    readout = RandomProjectionReadout(roles.descending, seed=pcfg.readout_seed,
                                      rate_norm=pcfg.readout_rate_norm)
    from flyaim.brain.plasticity import PlasticOverlayGPU

    overlay = PlasticOverlayGPU(brain, pcfg, artifact_path=wp)
    from flyaim.arena.arena import Arena

    arena = Arena(ArenaConfig(max_frames=900), seed=0)
    steps = int(brain.cfg.steps_per_frame)

    t0 = time.perf_counter()
    frame_img = arena.reset()
    metrics: list[dict] = []
    snapshots: list[dict] = []
    err_hist: list[float] = []
    raw_hist: list[float] = []
    eta_frozen = False
    g_frozen = False
    n_frames = 0

    print(f"[{arm}] 训练开始: {frames} 帧(探针 {pcfg.probe_frames}),{device}", flush=True)

    while n_frames < frames:
        drive = retina.frame_to_spikes(frame_img)
        brain.step_many(steps, drive[:, 0], drive[:, 1])
        rates = brain.rates  # 每帧唯一一次同步

        if n_frames < pcfg.probe_frames:
            # ---- 探针期:m=1 标定 η;动作原始分布标定 g;无可塑性生效
            overlay.set_rates(rates)
            raw_hist.extend(np.asarray(readout.raw(rates), dtype=np.float64).ravel()[:512])
            overlay.probe_frame(1.0)
            action = readout.act(rates)  # g=1
            res = arena.step(action)
            frame_img = res.frame
            err_hist.append(float(res.target_dist))
            n_frames += 1
            if n_frames == pcfg.probe_frames:
                overlay.freeze_eta()
                eta_frozen = True
                sd = float(np.std(raw_hist)) or 1e-6
                readout.g = 0.3 / sd
                g_frozen = True
                print(f"  [{arm}] 探针完成: η={overlay.eta:.4g} g={readout.g:.3f} "
                      f"(std_raw={sd:.3f})", flush=True)
        else:
            action = readout.act(rates)
            res = arena.step(action)
            frame_img = res.frame
            err_hist.append(float(res.target_dist))
            if n_frames % int(pcfg.plastic_every) == 0:
                m = float(np.exp(-err_hist[-1] / pcfg.m_sigma_px))
                overlay.set_rates(rates)
                overlay.update(m)
                if n_frames - pcfg.probe_frames == 1000:
                    thr = 1e-9 * overlay._w_typ  # P2 v3.1:死可塑性底线
                    if overlay.last_abs_dW_active < thr:
                        raise RuntimeError(
                            f"[{arm}] 快速失败(P2 v3 止损):第 1000 训练帧 "
                            f"活跃边 mean|ΔW|={overlay.last_abs_dW_active:.2e} "
                            f"< 1e-9×w_typ={thr:.2e}")
                    print(f"  [{arm}] 止损检查通过: 活跃边 |ΔW|="
                          f"{overlay.last_abs_dW_active:.2e} "
                          f"({overlay.last_active_edges} 活跃边)", flush=True)
            n_frames += 1

        if res.done or arena.done:
            frame_img = arena.reset()
            retina.reset()

        if n_frames % log_every == 0:
            m_err = float(np.mean(err_hist[-log_every:]))
            dn_rate = float(np.mean(rates[roles.descending]))
            metrics.append({"frame": n_frames, "mean_err_px": m_err,
                            "dn_rate_hz": dn_rate,
                            "abs_dW_active": overlay.last_abs_dW_active,
                            "active_edges": overlay.last_active_edges,
                            "readout_g": readout.g})
            fps = n_frames / max(time.perf_counter() - t0, 1e-9)
            print(f"  [{arm}] f={n_frames} err={m_err:.0f}px dn={dn_rate:.1f}Hz "
                  f"|dW|act={overlay.last_abs_dW_active:.2e} ({fps:.1f} f/s)", flush=True)

        for k in (9000, 18000, 27000):
            if n_frames == k:
                p = overlay.save_artifacts(out_dir / f"{arm}_w{k}.npz")
                snapshots.append({"frame": k, "path": str(p)})

    wall = time.perf_counter() - t0
    final = overlay.save_artifacts(out_dir / f"{arm}_wfinal.npz")
    snapshots.append({"frame": n_frames, "path": str(final)})
    out = {"arm": arm, "frames": n_frames, "wall_s": round(wall, 1),
           "eta": overlay.eta, "w_typ": overlay._w_typ,
           "readout_g": readout.g, "metrics": metrics, "snapshots": snapshots,
           "err_last1k": float(np.mean(err_hist[-1000:])),
           "err_curve_downsampled": [float(x) for x in err_hist[::100]]}
    (out_dir / f"{arm}_train.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[{arm}] 训练完成: {n_frames} 帧 / {wall:.0f}s,末段靶距 "
          f"{out['err_last1k']:.0f}px,权重 {final.name}", flush=True)
    return out


def eval_arm(weights_npz: Path, readout_g: float, seeds: int = 10, frames: int = 900,
             tag: str = "") -> list[dict]:
    brain, roles = _build_brain(weights_npz)
    from flyaim.retina.encoder import Retina

    retina = Retina(RetinaConfig(), input_neuron_ids=roles.visual_input)
    readout = RandomProjectionReadout(roles.descending)
    readout.g = float(readout_g)
    from flyaim.arena.arena import Arena
    from flyaim.config import ArenaConfig

    steps = int(brain.cfg.steps_per_frame)
    results = []
    for seed in range(seeds):
        arena = Arena(ArenaConfig(max_frames=frames), seed=seed)
        brain.reset()
        retina.reset()
        frame_img = arena.reset()
        dists: list[float] = []
        hits = 0
        for _ in range(frames):
            drive = retina.frame_to_spikes(frame_img)
            brain.step_many(steps, drive[:, 0], drive[:, 1])
            res = arena.step(readout.act(brain.rates))
            frame_img = res.frame
            dists.append(float(res.target_dist))
            hits += int(res.hit)
            if res.done:
                break
        results.append({"seed": seed, "mean_dist_px": float(np.mean(dists)),
                        "hits": hits, "hit_rate": hits / max(len(dists), 1)})
        print(f"  eval[{tag or weights_npz.name}] seed={seed} "
              f"mean_dist={results[-1]['mean_dist_px']:.0f}px hits={hits}", flush=True)
    return results
