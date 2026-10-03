"""真实数据联调冒烟测试(B 线):加载 A 线产物 -> Retina -> Connectome -> Readout。

A 线完成后跑这个,用来回答三个**只有真实数据能回答**的问题:
  1. 真实连接组的稀疏度/规模下,单步耗时是多少?能否支撑 30 FPS(33 步/帧)?
  2. 真实网络被视驱动后到底会不会放电(非零比例、均值、最大率)?
  3. `input_gain`(外部驱动标定)取多少才合适?

用法(必须用捆绑解释器):
    C:\\Users\\Admin\\.dsh\\dsh-runtimes\\dsh-primary-runtime\\dependencies\\python\\python.exe \\
        flyaim/brain/selftest_real.py [--data-dir flyaim/data/build]

数据未就绪时**不报错**,打印 SKIP 并以 0 退出(A 线可能还没跑完)。
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import numpy as np

from flyaim.brain.lif import Connectome
from flyaim.brain.readout import Readout
from flyaim.config import BrainConfig, ReadoutConfig, RetinaConfig
from flyaim.io import ARTIFACT_INDEX, ARTIFACT_MANIFEST, ARTIFACT_ROLES, ARTIFACT_WEIGHTS, Manifest
from flyaim.neuron_index import NeuronIndex
from flyaim.retina.encoder import Retina

DEFAULT_DATA_DIR = Path("flyaim") / "data" / "build"
FRAME_HW = (480, 640)
EYE = (24, 32)

# 标定工作点(ws_exc, ws_inh, input_gain, 说明)。依据见 selftest_real 第 6/7 节:
#   解析律 weight_scale >= v_thresh / (f * S_i),以及实测的"全静默 <-> 全饱和"双稳态。
CALIB_POINTS = (
    (0.02, 0.02, 1.5, "raw 原设定:感光层饱和(1.5*drive 全过阈)但递归传不动 -> DN=0"),
    (0.10, 0.10, 0.30, "raw 临界下侧:全网络静默,DN=0(速度最快 ~1 ms/步)"),
    (0.12, 0.12, 0.20, "raw 刚点燃:DN 88% 有放电,但每步 9% 神经元发放 -> 很慢"),
    (0.12, 0.24, 0.20, "raw 推荐:DN 60% + 靶位可分辨,速度/可分辨性折中"),
    (0.12, 0.96, 0.20, "raw 强抑制:DN 32%,相对可分辨性最高,~7 FPS"),
    (0.12, 0.12, 1.50, "raw 感光层二值饱和:全局 140 Hz(癫痫式,不承载信息)"),
)

# 行归一化(weight_norm="indeg")工作点:把"全静默<->全饱和"的二元开关变成可调梯度。
# ws 从 8 提到 14,DN 均值从 0.03 Hz 连续升到 15.7 Hz(实测),这是唯一的可标定方向。
CALIB_POINTS_NORM = (
    (8.0, 16.0, 1.5, "indeg:DN 0.03 Hz(几乎静默)"),
    (10.0, 20.0, 1.5, "indeg:DN 1.1 Hz"),
    (12.0, 24.0, 1.5, "indeg:DN 4.3 Hz(生理区间下沿)"),
    (14.0, 28.0, 1.5, "indeg:DN 15.7 Hz(推荐:落在 2-20 Hz 目标区)"),
)

# 逐层统计用的 superclass 分组(Lead 要求的分层发放率)
LAYER_NAMES = (
    "ol_sensory",
    "ol_intrinsic",
    "visual_projection",
    "descending_neuron",
    "vnc_motor",
)


def section(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def make_frame(h: int, w: int, blob_xy: tuple[float, float] | None, radius: int = 22,
               rng: np.random.Generator | None = None) -> np.ndarray:
    """构造"靶场风格"帧:暗背景 + 红色圆靶(纯几何,与 C 线 Arena 无关)。

    注意:这是**测试夹具**;Retina 只接收像素数组,拿不到 blob 坐标。
    """
    img = np.empty((h, w, 3), dtype=np.uint8)
    img[:] = np.array([18, 18, 22], dtype=np.uint8)
    if blob_xy is not None:
        yy, xx = np.mgrid[0:h, 0:w]
        cx, cy = blob_xy
        img[(xx - cx) ** 2 + (yy - cy) ** 2 <= radius**2] = np.array([235, 70, 70], dtype=np.uint8)
    if rng is not None:
        n = rng.integers(-6, 7, size=img.shape, dtype=np.int16)
        img = np.clip(img.astype(np.int16) + n, 0, 255).astype(np.uint8)
    return img


def run_episode(
    retina: Retina, brain: Connectome, arena_rng: np.random.Generator, steps: int = 100,
    steps_per_frame: int = 33, gain_frames: bool = True
) -> dict:
    """按集成层的用法推进 steps 步(每 steps_per_frame 步换一帧)。"""
    h, w = FRAME_HW
    drive = None
    t0 = time.perf_counter()
    for i in range(steps):
        if drive is None or (gain_frames and i % steps_per_frame == 0):
            blob = (
                float(arena_rng.uniform(40, w - 40)),
                float(arena_rng.uniform(40, h - 40)),
            )
            drive = retina.frame_to_spikes(make_frame(h, w, blob, rng=arena_rng))
        brain.step(drive[:, 0], drive[:, 1], i * brain.cfg.dt_ms)
    wall = time.perf_counter() - t0
    rates = brain.rates
    return {
        "wall_s": wall,
        "rate_nonzero_frac": float(np.mean(rates > 0)),
        "rate_mean_hz": float(rates.mean()),
        "rate_max_hz": float(rates.max()),
        "finite": bool(np.all(np.isfinite(rates))),
        "spk_shape": tuple(brain.spikes.shape),
        "n_fired_last": int(brain._n_fired),
        "timing": brain.timing_stats(),
        "drive_stats": retina.drive_stats(drive),
    }


def main() -> int:
    data_dir = DEFAULT_DATA_DIR
    if "--data-dir" in sys.argv:
        data_dir = Path(sys.argv[sys.argv.index("--data-dir") + 1])
    p_w = data_dir / ARTIFACT_WEIGHTS
    p_i = data_dir / ARTIFACT_INDEX
    p_r = data_dir / ARTIFACT_ROLES
    p_m = data_dir / ARTIFACT_MANIFEST

    section("0. 产物检查")
    print(f"数据目录: {data_dir.resolve()}")
    missing = [p.name for p in (p_w, p_i, p_r) if not p.exists()]
    if missing:
        print(f"SKIP: A 线产物尚未就绪,缺少 {missing};先跑 flyaim/brain/selftest_synth.py")
        return 0
    for p in (p_w, p_i, p_r, p_m):
        print(f"  {'OK ' if p.exists() else '-- '} {p.name:24s} "
              f"{p.stat().st_size/1e6:9.2f} MB" if p.exists() else f"  --  {p.name} (缺失)")

    section("1. 加载真实产物")
    t0 = time.perf_counter()
    if p_m.exists():
        man = Manifest.load(p_m)
        print(f"manifest: N={man.n_neurons:,} edges={man.n_edges:,} "
              f"visual_input_fallback={man.visual_input_fallback} "
              f"strategy={man.visual_input_strategy}")
        print(f"  roles={man.roles}")
    idx = NeuronIndex.load(p_i)
    from flyaim.io import load_roles  # noqa: E402

    roles = load_roles(p_r)
    print(f"neuron_index: {idx.n:,} 行; roles: " + ", ".join(
        f"{k}={v}" for k, v in roles.summary().items() if k.startswith("n_")))
    sc = idx.df["superclass"].astype(str).to_numpy()
    layers = {n: np.flatnonzero(sc == n) for n in LAYER_NAMES}
    layers = {k: v for k, v in layers.items() if v.size}
    print("逐层 superclass 分组: " + ", ".join(f"{k}={v.size:,}" for k, v in layers.items()))
    if roles.visual_input.size == 0:
        print("  !! roles.visual_input 为空 -> 将走回退路径")

    t0 = time.perf_counter()
    brain = Connectome(str(p_w), BrainConfig(subnet_size=None),
                       input_neuron_ids=roles.visual_input, roles=roles)
    st = brain.stats()
    print(f"Connectome(全量): N={st['n_neurons']:,} m={st['n_sim']:,} "
          f"exc_nnz={st['n_edges_exc']:,} inh_nnz={st['n_edges_inh']:,}")
    print(f"  密度={st['density_sim']*100:.4f}% 平均入度={(st['n_edges_total']/st['n_neurons']):.1f} "
          f"矩阵内存={st['matrix_bytes']/1e6:.0f} MB 状态={st['state_bytes']/1e6:.1f} MB "
          f"加载+构建={time.perf_counter()-t0:.1f}s")
    print(f"  输入群: n={st['input_group']['n']:,} 落在子网内={st['input_group']['n_mapped_in_subnet']:,} "
          f"fallback={st['input_group']['fallback']} input_gain={st['input_group']['input_gain']}")

    section("2. Retina 在真实感光细胞群上的分组")
    r_cfg = RetinaConfig(eye_rows=EYE[0], eye_cols=EYE[1])
    r_cfg.data_dir = str(data_dir)
    t0 = time.perf_counter()
    retina = Retina(r_cfg, input_neuron_ids=roles.visual_input, neuron_index=idx)
    rd = retina.describe()
    print(f"{retina!r}  (构建 {time.perf_counter()-t0:.1f}s)")
    print(f"  通道分组: {rd['channel_map']}  (每小眼 {rd['photoreceptors_per_ommatidium']} 个感光细胞)")
    print(f"  type 来源={rd['types_source']}  视拓扑={rd['retinotopy']}")
    print(f"  gain_mode={rd['gain_mode']}  ref_lum={rd['reference_contrast']} "
          f"ref_color={rd['reference_contrast_color']}  max_drive_hz={rd['max_drive_hz']}")
    print(f"  小眼阵列 {rd['eye_rows']}x{rd['eye_cols']}={rd['n_ommatidia']} "
          f"(真实复眼约 750-800 个小眼)")
    if rd["input_neuron_ids_fallback"]:
        print("  !! 触发了输入群回退(manifest 必须标 visual_input_fallback=true)")

    # 真实 type 分布(从 neuron_index 直接统计,用于核对分组)
    types = idx.df["type"].astype(str).to_numpy()[roles.visual_input]
    vals, cnts = np.unique(types, return_counts=True)
    top = sorted(zip(cnts, vals), reverse=True)[:8]
    print("  真实 visual_input type 分布 top8: " + ", ".join(f"{t}={c}" for c, t in top))

    section("3. 帧 -> 驱动统计(红色靶在不同位置)")
    rng = np.random.default_rng(0)
    for name, blob in (("左上", (140.0, 90.0)), ("中心", (320.0, 240.0)), ("右下", (520.0, 400.0))):
        retina.reset()
        d = retina.frame_to_spikes(make_frame(*FRAME_HW, blob, rng=rng))
        s = retina.drive_stats(d)
        print(f"  {name}: exc 非零={s['exc_nonzero_frac']:.3f} 均值={s['exc_mean_hz']:6.1f} "
              f"最大={s['exc_max_hz']:6.1f} Hz | inh 非零={s['inh_nonzero_frac']:.3f} "
              f"均值={s['inh_mean_hz']:6.1f} 最大={s['inh_max_hz']:6.1f} Hz")
        assert d.shape == (roles.visual_input.size, 2) and d.dtype == np.float32
        assert np.all(np.isfinite(d)) and np.all(d >= 0), "驱动必须有限且非负"
        assert np.all(d[:, 0] * d[:, 1] == 0.0), "ON/OFF 通道必须互斥"

    section("4. 全量 100 步 + rates 统计(真实连接组)")
    try:
        res = run_episode(retina, brain, rng, steps=100)
        print(f"  rates: 非零比例={res['rate_nonzero_frac']:.4f} 均值={res['rate_mean_hz']:.3f} Hz "
              f"最大={res['rate_max_hz']:.1f} Hz 有限={res['finite']} 末步发放={res['n_fired_last']}")
        print(f"  spikes shape={res['spk_shape']} (必须 = (N,) 全局索引空间)")
        ts = res["timing"]
        print(f"  单步 p50={ts['p50_ms']:.3f} ms p95={ts['p95_ms']:.3f} ms 最大={ts['max_ms']:.3f} ms "
              f"-> 33 步/帧={ts['p50_ms']*33:.1f} ms ({1000.0/max(ts['p50_ms']*33,1e-9):.1f} FPS 上限)")
        assert res["finite"] and res["spk_shape"] == (brain.n_neurons,)
        dn_rates = brain.rates[roles.descending]
        print(f"  DN 群: 非零={np.mean(dn_rates>0):.3f} 均值={dn_rates.mean():.3f} Hz "
              f"最大={dn_rates.max():.1f} Hz")
    except ValueError as exc:
        print(f"  !! 全量模式失败: {exc}")

    section("5. 子网模式对比(真实数据,决策用)")
    rows = []
    for sub in (None, roles.visual_input.size + roles.descending.size + roles.motor.size,
                20000, 50000):
        label = "全量" if sub is None else f"子网 {sub:,}"
        try:
            cfg = BrainConfig(subnet_size=sub, subnet_seed=7)
            b = Connectome(str(p_w), cfg, input_neuron_ids=roles.visual_input, roles=roles)
            rr = Retina(r_cfg, input_neuron_ids=roles.visual_input, neuron_index=idx)
            r2 = run_episode(rr, b, np.random.default_rng(1), steps=100)
            t = r2["timing"]
            row = {
                "label": label, "m": b.n_sim, "nnz": b.stats()["n_edges_total"],
                "MB": b.stats()["matrix_bytes"] / 1e6, "p50": t["p50_ms"],
                "fired": r2["n_fired_last"] / max(b.n_sim, 1),
                "nz": r2["rate_nonzero_frac"], "mean": r2["rate_mean_hz"], "max": r2["rate_max_hz"],
            }
            rows.append(row)
            print(f"  {label:12s} m={row['m']:>7,} nnz={row['nnz']:>12,} {row['MB']:6.0f}MB | "
                  f"p50={row['p50']:7.3f} ms | 33 步/帧={row['p50']*33:7.1f} ms "
                  f"({1000.0/max(row['p50']*33,1e-9):5.1f} FPS) | rates 非零={row['nz']:.4f} "
                  f"均值={row['mean']:.3f} 最大={row['max']:.1f} Hz")
        except Exception as exc:  # 子网失败不应中断报告
            print(f"  {label:12s} !! 失败: {type(exc).__name__}: {exc}")

    section("7. 工作点标定(逐层发放率 / 靶位可分辨性 / 帧预算)")

    def workpoint(ws_e: float, ws_i: float, gain: float, tag: str, norm: str = "none") -> dict:
        cfg = BrainConfig(subnet_size=None, weight_scale_exc=ws_e, weight_scale_inh=ws_i)
        cfg.input_gain = gain
        cfg.weight_norm = norm
        b = Connectome(str(p_w), cfg, input_neuron_ids=roles.visual_input, roles=roles)
        rr = Retina(r_cfg, input_neuron_ids=roles.visual_input, neuron_index=idx)
        ro = Readout(ReadoutConfig(mode="trained"), roles=roles)
        feats, fired = [], []
        for blob in ((140.0, 90.0), (520.0, 400.0)):
            rr.reset()
            d = rr.frame_to_spikes(make_frame(*FRAME_HW, blob))
            for i in range(66):
                b.step(d[:, 0], d[:, 1], i)
                fired.append(b._n_fired / b.n_sim)
            feats.append(ro.features(b))
        r = b.rates
        dn = r[roles.descending]
        sep = float(np.abs(feats[0] - feats[1]).mean())
        per_layer = " ".join(
            f"{k.split('_')[0]}={r[g].mean():.2f}Hz({np.mean(r[g]>0)*100:.0f}%)"
            for k, g in layers.items()
        )
        p50 = b.timing_stats()["p50_ms"]
        print(f"  ws_e={ws_e:4.2f} ws_i={ws_i:4.2f} in_gain={gain:4.2f} norm={norm:5s} [{tag}]")
        print(f"      逐层均值率(非零比例): {per_layer}")
        print(f"      DN: 非零={np.mean(dn>0):.3f} 均值={dn.mean():7.2f} Hz | "
              f"靶位可分辨={sep:7.3f} Hz | 每步发放={np.mean(fired)*100:5.2f}% | "
              f"p50={p50:7.3f} ms -> 33 步/帧={p50*33:8.1f} ms ({1000.0/max(p50*33,1e-9):5.1f} FPS)")
        return {"ws_e": ws_e, "ws_i": ws_i, "gain": gain, "tag": tag,
                "dn_nz": float(np.mean(dn > 0)), "dn_mean": float(dn.mean()),
                "sep": sep, "p50": p50, "fired": float(np.mean(fired))}

    for ws_e, ws_i, gain, tag in CALIB_POINTS:
        workpoint(ws_e, ws_i, gain, tag, norm="none")
    for ws_e, ws_i, gain, tag in CALIB_POINTS_NORM:
        workpoint(ws_e, ws_i, gain, tag, norm="indeg")

    section("7d. 决定性检验:DN 输出到底有没有承载画面信息?")
    # 方法(两部分,缺一不可):
    #   (a) 同历史对照:完全相同的历史 + 只换画面 -> |ΔDN| 若 ~0,则视觉对 DN 完全无影响;
    #   (b) 重复性对照:同一批画面 + 不同随机历史 -> 得到"网络自身历史涨落"的噪声底。
    #   (a) 远大于 (b) 才说明刺激可分辨;比值 < ~3 意味着读出的 r2 会是 ~0。
    stims4 = ((120.0, 100.0, 40, 235), (520.0, 380.0, 40, 235),
              (320.0, 240.0, 40, 235), (120.0, 100.0, 22, 255))
    warm = ((200.0, 200.0, 30, 255), (400.0, 300.0, 30, 255), (300.0, 150.0, 30, 255))
    for ws_e, ws_i, gain, tag in CALIB_POINTS_NORM:
        cfg = BrainConfig(subnet_size=None, weight_scale_exc=ws_e, weight_scale_inh=ws_i)
        cfg.input_gain = gain
        cfg.weight_norm = "indeg"
        b = Connectome(str(p_w), cfg, input_neuron_ids=roles.visual_input, roles=roles)
        ro = Readout(ReadoutConfig(mode="trained"), roles=roles)

        def stim_frame(s):
            """(cx, cy, radius, level) -> 画面。"""
            cx, cy, rad, lv = s
            img = make_frame(*FRAME_HW, (cx, cy), radius=int(rad))
            if lv != 235:  # 需要不同的亮度
                yy, xx = np.mgrid[0:FRAME_HW[0], 0:FRAME_HW[1]]
                m = (xx - cx) ** 2 + (yy - cy) ** 2 <= rad**2
                img[m] = np.array([lv, max(lv // 3, 10), max(lv // 3, 10)], dtype=np.uint8)
            return img

        def settle_dn(fix_hist: bool, stim, rep: int, b=b, ro=ro):
            rr = Retina(r_cfg, input_neuron_ids=roles.visual_input, neuron_index=idx)
            b.reset()
            rr.reset()
            rng = np.random.default_rng(rep)
            for wi, w in enumerate(warm):
                ww = w if fix_hist else (
                    float(rng.uniform(40, 600)), float(rng.uniform(40, 440)), 22, 235
                )
                d = rr.frame_to_spikes(stim_frame(ww))
                for k in range(11):
                    b.step(d[:, 0], d[:, 1], wi * 11 + k)
            rr.reset()
            d = rr.frame_to_spikes(stim_frame(stim))
            for k in range(66):
                b.step(d[:, 0], d[:, 1], k)
            return ro.features(b).copy()

        same_hist = np.stack([settle_dn(True, s, 0) for s in stims4])
        re_dn = np.stack([settle_dn(False, s, r) for s in stims4 for r in range(3)])
        within = float(np.mean([np.abs(re_dn[i] - re_dn[j]).mean()
                                for i in range(len(re_dn)) for j in range(i + 1, len(re_dn))]))
        between = float(np.mean([np.abs(re_dn[i] - re_dn[j]).mean()
                                 for si in range(4) for sj in range(4) if si != sj
                                 for i in range(si * 3, si * 3 + 3) for j in range(sj * 3, sj * 3 + 3)]))
        dn_mean = float(same_hist.mean())
        stim_effect = float(np.mean([np.abs(same_hist[i] - same_hist[0]).mean() for i in (1, 2, 3)]))
        ratio = between / max(within, 1e-9)
        print(f"  indeg ws={ws_e}/{ws_i}: DN 均值={dn_mean:8.3f} Hz | 同历史换画面 |ΔDN|="
              f"{stim_effect:8.4f} Hz ({stim_effect/max(dn_mean,1e-9)*100:5.1f}% of mean)")
        print(f"      重复性对照:历史涨落噪声底={within:8.4f} 刺激间={between:8.4f} "
              f"**比值={ratio:5.2f}** ({'刺激可分辨' if ratio > 3 else '被历史涨落淹没 -> 读出 r2 会 ~0'})")

    section("6. 解析标定:发一个脉冲需要多少输入同时活跃")
    # LIF 稳态:v -> v_rest + I_syn,I_syn = weight_scale * sum_j W[i,j] * s_j。
    # 因此神经元 i 的"最大可达电流" = weight_scale_exc * S_i(S_i = 入权重总和)。
    # 若 S_i * scale < v_thresh,该神经元**无论如何都不可能发放**(输入全开也不行)
    # —— 这是"增益设置"而非"代码"问题,下面是量化诊断。
    S_exc = np.asarray(brain.art.W_exc.sum(axis=1)).reshape(-1).astype(np.float64)
    S_inh = np.asarray(brain.art.W_inh.sum(axis=1)).reshape(-1).astype(np.float64)
    scale = brain.w_exc
    v_th = brain.v_thresh
    never = S_exc * scale < v_th
    print(f"  W_exc 每神经元入权重总和 S_i: p10={np.percentile(S_exc,10):.0f} "
          f"p50={np.percentile(S_exc,50):.0f} p90={np.percentile(S_exc,90):.0f} "
          f"max={S_exc.max():.0f}; 抑制侧 S_i p50={np.percentile(S_inh,50):.0f}")
    print(f"  当前 weight_scale_exc={scale:g}, v_thresh={v_th:g}")
    print(f"  **永远无法发放的神经元(即使全部输入同时发放): {np.mean(never)*100:.1f}% "
          f"({int(never.sum()):,}/{brain.n_neurons:,})**")
    for q in (10, 25, 50, 75, 90):
        s_q = float(np.percentile(S_exc, q))
        print(f"    p{q:>2} 神经元(S={s_q:6.0f}): 需要 weight_scale_exc >= {v_th/s_q:.4f} "
              f"才能靠全部输入发放;若只有 25% 输入活跃则需 >= {v_th/(0.25*s_q):.4f}")
    # 定向验证:感光细胞 -> 一级靶(LMC)的入权重
    ph = roles.visual_input
    sub = brain.art.W_exc[ph]  # 感光细胞的行 = 它们的输入(不是我们要的)
    cols_targets = np.unique(brain.art.W_exc[:, ph].indices) if brain.art.W_exc.nnz else np.empty(0)
    toc = brain.art.W_exc[:, ph]  # 列切片:感光细胞 -> 下游
    wsum_to = np.asarray(toc.sum(axis=1)).reshape(-1)
    n_targets = np.asarray((toc != 0).sum(axis=1)).reshape(-1)
    has = wsum_to > 0
    print(f"  感光细胞直接投射到的下游神经元: {int(has.sum()):,} 个;"
          f"其 S_photo->target p50={np.percentile(wsum_to[has],50):.0f} "
          f"p90={np.percentile(wsum_to[has],90):.0f}, 突触前个数 p50={np.percentile(n_targets[has],50):.0f}")
    need_all = v_th / max(np.percentile(wsum_to[has], 50), 1e-9)
    print(f"  -> 一级靶要用**全部**感光细胞输入才发放,需要 weight_scale_exc >= {need_all:.4f}")
    del sub, toc

    section("7b. 最大视觉驱动测试(6098 个感光细胞全部 200 Hz 恒驱动)")
    # 目的:区分"增益不够"与"通路在过滤后的 A-A 子图里断了"。
    # 若感光细胞全速发放仍无法让 DN 放电,则问题在连接组结构(manifest 记录:
    # 151,856,684 行边表只保留了 25,582,938 条 A-A 边,丢弃的行里 body_post 不是
    # 任何有标注的 body),不是仿真参数。
    for ws in (0.3, 1.0, 3.0):
        cfg = BrainConfig(subnet_size=None, weight_scale_exc=ws)
        b = Connectome(str(p_w), cfg, input_neuron_ids=roles.visual_input, roles=roles)
        n_in = b.in_neuron_ids.size
        exc = np.full(n_in, 200.0, dtype=np.float32)
        inh = np.zeros(n_in, dtype=np.float32)
        for i in range(200):
            b.step(exc, inh, i * cfg.dt_ms)
        r = b.rates
        photo = r[roles.visual_input]
        dn = r[roles.descending]
        print(f"  ws={ws:5.3f}: 感光 非零={np.mean(photo>0):.3f} 均值={photo.mean():8.2f} | "
              f"DN 非零={np.mean(dn>0):.3f} 均值={dn.mean():8.4f} 最大={dn.max():7.2f} | "
              f"全群 非零={np.mean(r>0):.5f} 均值={r.mean():.4f} 最大={r.max():7.1f} Hz")
        # DN 的输入侧:看 DN 的突触前神经元此刻有多少在放
        dn_pre_w = b.art.W_exc[roles.descending]
        pre = np.unique(dn_pre_w.indices)
        print(f"           DN 群突触前神经元 {pre.size:,} 个,其中在发放 "
              f"{int(np.sum(r[pre] > 0)):,} 个({np.mean(r[pre]>0)*100:.4f}%)")

    section("7c. 结论:没有'稀疏活动'工作区(双稳态)")
    print("  实测证据(见 7 节表):weight_scale_exc <= 0.10 时全网络静默(DN=0);")
    print("  >= 0.12 时一步跨到 45-90% 神经元有放电(DN 也活),中间没有稳定稀疏区。")
    print("  机制:抑制侧入权重和(p50≈72)只有兴奋侧(p50≈267)的 27%,出-入权重失衡;")
    print("  确定性 LIF 无适应/噪声,活动一旦越过传播阈值就会自持到不应期上限。")
    print("  膜噪声测试(noise_sigma 0.05/0.2/0.5,见 lif.py)无法打开稀疏区。")
    print("  => 必须接受其中一个 regime,或改模型(适应/更强的抑制/噪声),见汇报。")

    section("8. Readout(真实 DN 群)")
    ro = Readout(ReadoutConfig(mode="fixed"), roles=roles, neuron_index=idx)
    d_ro = ro.describe()
    print(f"{ro!r}")
    print(f"  来源={d_ro['source_kind']} n_dn={d_ro['n_dn']:,} n_mn={d_ro['n_mn']:,} "
          f"分组方式={d_ro['grouping']} side 来源={d_ro['side_source']} 组={d_ro['groups']}")
    print(f"  注意: side 覆盖率不足时会退化为索引四分位(fixed 映射本身就是任意基线)")
    act = ro.act(brain)
    print(f"  fixed act={np.round(act, 5).tolist()} dtype={act.dtype} shape={act.shape}")
    assert act.shape == (2,) and act.dtype == np.float32 and np.all(np.abs(act) <= 1.0)

    # trained 模式的形状约定(不训练,只验证接口可用)
    ro_t = Readout(ReadoutConfig(mode="trained", ridge_lambda=1.0), roles=roles)
    X = np.stack([ro_t.features(brain) for _ in range(3)])
    Y = np.zeros((3, 2), dtype=np.float32)
    ro_t.fit(X, Y)
    w = ro_t.get_linear_weights()
    print(f"  trained 接口自检: W={w['W'].shape} mu={w['mu'].shape} sd={w['sd'].shape} "
          f"(n_features={ro_t.n_features})")
    assert w["W"].shape == (ro_t.n_features + 1, 2)
    assert w["mu"].shape == (ro_t.n_features,) and w["sd"].shape == (ro_t.n_features,)

    section("9. 冻结校验(仿真/读出不得改动连接组权重)")
    from flyaim.io import ConnectomeArtifacts  # noqa: E402

    art = ConnectomeArtifacts.load(p_w)
    print(f"  产物 W_exc.nnz={art.W_exc.nnz:,} W_inh.nnz={art.W_inh.nnz:,} "
          f"nnz 之和={art.W_exc.nnz+art.W_inh.nnz:,}")
    if p_m.exists():
        assert art.W_exc.nnz + art.W_inh.nnz == man.n_edges, "nnz 与 manifest 不一致"
        print(f"  与 manifest.n_edges={man.n_edges:,} 一致 OK")

    section("总结")
    print("真实数据冒烟测试通过")
    for r in rows:
        print(f"  {r['label']:12s} p50={r['p50']:7.3f} ms/步 -> 33 步/帧 {r['p50']*33:7.1f} ms "
              f"({1000.0/max(r['p50']*33,1e-9):5.1f} FPS 上限), 矩阵 {r['MB']:.0f} MB, "
              f"发放率 {r['fired']*100:.3f}%, 均值率 {r['mean']:.3f} Hz")
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    sys.exit(main())
