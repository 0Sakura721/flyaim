"""C 线(靶场/指标/基线)自检。零失败才算通过。

运行方式(二选一,都在仓库根目录下)::

    python -m flyaim.arena.selftest
    python flyaim/arena/selftest.py

检查项
------
1. 契约签名逐字一致(``inspect.signature``);
2. **确定性**:同 seed + 同 action 序列 -> 帧序列逐位相同(含运动靶/多靶配置),
   ``reset()`` 幂等;
3. 靶场基本合理性:PID(作弊上限)vs random(下界)的 900 帧命中率对比,并打印;
4. ``Metrics.summary()`` 必含键齐全、内部自洽、``add_extra`` 保护;
5. shuffle 零模型:逐行 nnz 不变、总 nnz 不变、权重逐元素不变、
   ``indices`` 多重集不变、拓扑确实改变、同 seed 可复现、存盘/读回一致;
6. 渲染:PNG 落盘且能被 Pillow 打开,亮度对比足够(复眼只看像素);
7. 曲线图:acq_curve / 命中率对比 PNG 落盘。

输出只用 ASCII,避免 Windows 控制台编码问题。
"""

from __future__ import annotations

import inspect
import sys
import time
from pathlib import Path

import numpy as np
import scipy
import scipy.sparse as sp

# 允许 `python flyaim/arena/selftest.py` 直接运行
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from flyaim.arena.arena import Arena, StepResult
from flyaim.arena.metrics import CORE_KEYS, Metrics
from flyaim.arena.rollout import as_act_fn, run_episode
from flyaim.baselines.pid import PIDBaseline
from flyaim.baselines.random import RandomBaseline
from flyaim.baselines.shuffle import shuffle_connectome, verify_shuffle
from flyaim.config import ArenaConfig
from flyaim.io import ConnectomeArtifacts

HERE = Path(__file__).resolve().parent
FAILURES: list[str] = []
T0 = time.time()


def check(cond: bool, msg: str) -> bool:
    """记录一条检查结果(不抛出,最后统一汇报,便于一次看到所有问题)。"""
    ok = bool(cond)
    print(f"  [{'PASS' if ok else 'FAIL'}] {msg}")
    if not ok:
        FAILURES.append(msg)
    return ok


def section(title: str) -> None:
    print()
    print("=" * 78)
    print(f"== {title}")
    print("=" * 78)


# --------------------------------------------------------------------- 1. 签名


def check_signatures() -> None:
    section("1. contract signatures (CONTRACT.md 2.4) - must match verbatim")

    def params(fn):
        return list(inspect.signature(fn).parameters)

    check(params(Arena.__init__) == ["self", "cfg", "seed"],
          f"Arena.__init__{inspect.signature(Arena.__init__)}")
    check(params(Arena.reset) == ["self"], f"Arena.reset{inspect.signature(Arena.reset)}")
    check(params(Arena.step) == ["self", "action"], f"Arena.step{inspect.signature(Arena.step)}")
    check(hasattr(Arena, "render"), "Arena.render exists (contract alias)")
    check(params(Arena.render) == ["self"], f"Arena.render{inspect.signature(Arena.render)}")
    check(params(Arena.get_state) == ["self"],
          f"Arena.get_state{inspect.signature(Arena.get_state)} (lead integration hook)")

    fields = list(StepResult.__dataclass_fields__)
    check(fields == ["frame", "hit", "target_dist", "done", "info"],
          f"StepResult fields = {fields}")

    check(params(Metrics.update) == ["self", "res"],
          f"Metrics.update{inspect.signature(Metrics.update)}")
    check(params(Metrics.summary) == ["self"],
          f"Metrics.summary{inspect.signature(Metrics.summary)}")
    try:
        Metrics()  # 集成层使用无参构造
        check(True, "Metrics() constructs with no arguments")
    except Exception as exc:  # pragma: no cover
        check(False, f"Metrics() must construct with no args, got {exc!r}")

    sig = inspect.signature(shuffle_connectome)
    check(list(sig.parameters)[:2] == ["art", "seed"],
          f"shuffle_connectome{sig} (2nd positional must be named seed)")

    state = Arena(ArenaConfig(), 0).get_state()
    for k in ("crosshair", "targets", "width", "height", "speed_px_per_action"):
        check(k in state, f"Arena.get_state() exposes '{k}' (PID reads it)")


# --------------------------------------------------------------------- 2. 确定性


def _rollout_frames(cfg: ArenaConfig, seed: int, actions: np.ndarray) -> list[np.ndarray]:
    arena = Arena(cfg, seed)
    frames = [arena.reset()]
    for a in actions:
        frames.append(arena.step(a).frame)
    return frames


def check_determinism() -> None:
    section("2. determinism: same seed + same actions -> bit-identical frames")

    rng = np.random.default_rng(20240607)
    actions = rng.uniform(-1.0, 1.0, size=(150, 2)).astype(np.float64)

    configs = {
        "default(single static target)": ArenaConfig(),
        "moving target": ArenaConfig(target_speed_px_per_frame=3.5),
        "3 targets + moving": ArenaConfig(n_targets=3, target_speed_px_per_frame=2.0),
        "fast crosshair + small target": ArenaConfig(
            speed_px_per_action=30.0, target_radius=10, max_frames=150
        ),
    }
    for name, cfg in configs.items():
        fa = _rollout_frames(cfg, 7, actions)
        fb = _rollout_frames(cfg, 7, actions)
        same = all(x.tobytes() == y.tobytes() for x, y in zip(fa, fb))
        check(same, f"bit-identical over {len(fa)} frames - {name}")

    # reset() 幂等
    a = Arena(ArenaConfig(), 3)
    f1 = a.reset()
    a.step(np.array([0.5, 0.5]))
    f2 = a.reset()
    check(f1.tobytes() == f2.tobytes(), "Arena.reset() is idempotent (same seed -> same frame)")

    # 不同 seed 必须给出不同初始场景(否则 seed 无效)
    fa = Arena(ArenaConfig(), 1).reset()
    fb = Arena(ArenaConfig(), 2).reset()
    check(fa.tobytes() != fb.tobytes(), "different seeds give different initial scenes")

    # 命中序列也必须可复现
    def hits_seq(seed: int) -> list[int]:
        arena = Arena(ArenaConfig(), seed)
        arena.reset()
        out = []
        for a_ in actions:
            out.append(int(arena.step(a_).hit))
        return out

    check(hits_seq(11) == hits_seq(11), "hit sequence reproducible for same seed")


# --------------------------------------------------------------------- 3. PID vs random


def check_baselines(seeds=(0, 1, 2), frames: int = 900) -> dict:
    section(f"3. PID (cheating upper bound) vs random (lower bound), {frames} frames x "
            f"{len(seeds)} seeds, identical arena seeds/budget")
    cfg = ArenaConfig()
    summaries: dict[str, list[dict]] = {"pid": [], "random": []}
    curves: dict[str, list[list[float]]] = {"pid": [], "random": []}

    print(f"  arena: {cfg.width}x{cfg.height}, radius={cfg.target_radius}px, "
          f"n_targets={cfg.n_targets}, speed={cfg.speed_px_per_action}px/action, "
          f"respawn_on_hit={cfg.respawn_on_hit}")

    for arm in ("pid", "random"):
        print(f"  --- arm '{arm}' ---")
        for s in seeds:
            controller = PIDBaseline(cfg) if arm == "pid" else RandomBaseline(seed=int(s))
            arena = Arena(cfg, int(s))
            ctl = as_act_fn(controller, "state")
            if hasattr(controller, "attach_arena"):  # 编排层钩子
                controller.attach_arena(arena)
                ctl = as_act_fn(controller, "state")
            m, _ = run_episode(arena, ctl, max_frames=frames, n_bins=10, fps=30.0)
            sm = m.summary()
            summaries[arm].append(sm)
            curves[arm].append(sm["acq_curve"])
            print(f"      seed={s:>2}  hits={sm['hits']:>3}/{sm['frames']}  "
                  f"hit_rate={sm['hit_rate']:.4f}  mean_dist={sm['mean_target_dist_px']:7.2f}px  "
                  f"ttff={sm['time_to_first_hit_ms']}")

    pid_rates = np.array([s["hit_rate"] for s in summaries["pid"]])
    rnd_rates = np.array([s["hit_rate"] for s in summaries["random"]])
    pid_hits = np.array([s["hits"] for s in summaries["pid"]])
    rnd_hits = np.array([s["hits"] for s in summaries["random"]])

    print("  --- comparison (mean over seeds) ---")
    print(f"      pid    : hit_rate={pid_rates.mean():.4f}  hits/episode={pid_hits.mean():.1f}"
          f"  (min {pid_hits.min()}, max {pid_hits.max()})")
    print(f"      random : hit_rate={rnd_rates.mean():.4f}  hits/episode={rnd_hits.mean():.1f}"
          f"  (min {rnd_hits.min()}, max {rnd_hits.max()})")
    ratio = (pid_rates.mean() / rnd_rates.mean()) if rnd_rates.mean() > 0 else float("inf")
    print(f"      ratio  : pid/random = {ratio:.1f}x")

    check(pid_rates.mean() > 0.01, f"PID hit_rate {pid_rates.mean():.4f} > 0.01 (aiming works)")
    check(pid_rates.mean() >= 5.0 * rnd_rates.mean() + 0.01,
          f"PID hit_rate >> random hit_rate ({pid_rates.mean():.4f} vs {rnd_rates.mean():.4f})")
    check(pid_rates.min() > rnd_rates.max(),
          f"every PID seed beats every random seed ({pid_rates.min():.4f} > {rnd_rates.max():.4f})")
    check(pid_rates.mean() > 0.02,
          "PID is a meaningful upper-bound ruler (hit_rate > 2%)")

    # PID 调参闭环(小网格,确定性)
    tuned = PIDBaseline(cfg).tune(seed=0, frames=240,
                                  grid=((1.0, 0.0), (1.0, 0.15)))
    check(tuned.kp in (1.0,) and tuned.kd in (0.0, 0.15),
          f"PIDBaseline.tune() returns valid gains (kp={tuned.kp}, kd={tuned.kd})")

    return {"summaries": summaries, "curves": curves,
            "pid_rates": pid_rates, "rnd_rates": rnd_rates}


# --------------------------------------------------------------------- 4. Metrics


def check_metrics() -> None:
    section("4. Metrics.summary() keys and internal consistency")
    cfg = ArenaConfig(max_frames=120)
    arena = Arena(cfg, 4)
    m = Metrics(n_bins=10, frames_budget=cfg.max_frames)
    m.set_fps(30.0)
    m.add_extra("arm", "unit-test")
    ctl = PIDBaseline(cfg).attach_arena(arena)
    run_episode(arena, as_act_fn(ctl, "state"), max_frames=120, metrics=m)

    sm = m.summary()
    missing = [k for k in CORE_KEYS if k not in sm]
    check(not missing, f"summary() contains all mandated keys {list(CORE_KEYS)}; missing={missing}")
    check(sm["shots"] == sm["frames"] == 120,
          f"shots == frames == 120 (per-frame auto fire), got {sm['shots']}/{sm['frames']}")
    check(abs(sm["hit_rate"] - sm["hits"] / sm["shots"]) < 1e-12, "hit_rate == hits / shots")
    check(len(sm["acq_curve"]) == 10, f"acq_curve has n_bins=10 entries, got {len(sm['acq_curve'])}")
    check(abs(sum(sm["acq_curve"]) / 10 - sm["hit_rate"]) < 1e-9,
          "mean(acq_curve) == hit_rate (bins are a partition of the episode)")
    check(sm["fps_loop"] == 30.0, f"fps_loop injected via set_fps -> {sm['fps_loop']}")
    check(sm["time_to_first_hit_ms"] is not None and sm["time_to_first_hit_ms"] > 0,
          f"time_to_first_hit_ms = {sm['time_to_first_hit_ms']}")
    check(sm["mean_target_dist_px"] > 0, f"mean_target_dist_px = {sm['mean_target_dist_px']:.2f}")
    check(sm.get("arm") == "unit-test", "add_extra() value is merged into summary()")
    try:
        m.add_extra("hit_rate", 1.0)
        check(False, "add_extra must reject keys colliding with contract keys")
    except ValueError:
        check(True, "add_extra() rejects keys colliding with contract keys")

    # 空 Metrics 不炸
    try:
        empty = Metrics().summary()
        check(empty["frames"] == 0 and empty["hit_rate"] == 0.0,
              "empty Metrics() summary is well-defined (frames=0, hit_rate=0)")
    except Exception as exc:  # pragma: no cover
        check(False, f"empty Metrics().summary() raised {exc!r}")

    # 只依赖 hit / target_dist
    class _Bare:
        hit = True
        target_dist = 1.0

    try:
        m2 = Metrics()
        m2.update(_Bare())
        ok = m2.summary()["hits"] == 1
    except Exception as exc:
        ok = False
        print(f"      update() raised {exc!r}")
    check(ok, "Metrics.update() only needs .hit/.target_dist (no info keys) - lead contract")

    # StepResult.done 存在(集成层 getattr 安全)
    res = Arena(ArenaConfig(max_frames=2), 0)
    res.reset()
    r0 = res.step(np.zeros(2))
    r1 = res.step(np.zeros(2))
    check(r0.done is False and r1.done is True, "StepResult.done flips at max_frames")


# --------------------------------------------------------------------- 5. shuffle


def _synthetic_connectome(n: int = 140, seed: int = 0) -> ConnectomeArtifacts:
    rng = np.random.default_rng(seed)
    n_exc, n_inh = 1800, 600
    W_exc = sp.csr_matrix(
        (
            rng.integers(1, 25, n_exc).astype(np.float32),
            (rng.integers(0, n, n_exc), rng.integers(0, n, n_exc)),
        ),
        shape=(n, n),
    )
    W_inh = sp.csr_matrix(
        (
            rng.integers(1, 12, n_inh).astype(np.float32),
            (rng.integers(0, n, n_inh), rng.integers(0, n, n_inh)),
        ),
        shape=(n, n),
    )
    return ConnectomeArtifacts(
        W_exc=W_exc, W_inh=W_inh,
        neuron_ids=np.arange(1000, 1000 + n, dtype=np.int64),
        soma_pos=rng.normal(size=(n, 3)).astype(np.float32),
    )


def check_shuffle() -> None:
    section("5. shuffle null model: keep statistics, destroy topology")
    art = _synthetic_connectome()
    print(f"  original: n={art.n}, edges={art.n_edges} "
          f"(exc={art.W_exc.nnz}, inh={art.W_inh.nnz}), density={art.stats()['density']:.4f}")

    for mode in ("row", "global"):
        sh = shuffle_connectome(art, seed=0, mode=mode)
        v = verify_shuffle(art, sh)
        print(f"  --- mode='{mode}' ---")
        print("      " + "  ".join(f"{k}={v[k]}" for k in
                                   ("row_nnz_equal", "nnz_equal", "data_equal", "indptr_equal",
                                    "rowsum_equal", "indices_multiset_equal",
                                    "indegree_distribution_equal", "colsum_multiset_equal",
                                    "topology_changed", "mode_guess")))

        # 逐行 nnz 严格相等(逐元素比较,不只看总数)
        row_ok = all(
            np.array_equal(np.diff(a.indptr), np.diff(b.indptr))
            for a, b in ((art.W_exc, sh.W_exc), (art.W_inh, sh.W_inh))
        )
        check(row_ok, f"[{mode}] per-row nnz identical row by row")
        check(sh.W_exc.nnz == art.W_exc.nnz and sh.W_inh.nnz == art.W_inh.nnz,
              f"[{mode}] total nnz identical (exc {art.W_exc.nnz}, inh {art.W_inh.nnz})")
        check(np.array_equal(sh.W_exc.data, art.W_exc.data)
              and np.array_equal(sh.W_inh.data, art.W_inh.data),
              f"[{mode}] weight arrays byte-identical (weight distribution preserved)")
        check(not np.array_equal(sh.W_exc.indices, art.W_exc.indices),
              f"[{mode}] topology actually changed (indices permuted)")
        check(not np.array_equal(sh.W_inh.indices, art.W_inh.indices),
              f"[{mode}] inh topology actually changed")

        # 行和不变 -> 下游每个神经元的"输出总权重"严格不变(两模式共有)
        d_exc = sh.W_exc.sum(axis=1) - art.W_exc.sum(axis=1)
        d_inh = sh.W_inh.sum(axis=1) - art.W_inh.sum(axis=1)
        check(float(np.max(np.abs(d_exc))) == 0.0 and float(np.max(np.abs(d_inh))) == 0.0,
              f"[{mode}] per-neuron out-weight sums exactly unchanged")

        if mode == "row":
            # 行内只是重排 -> 整体列索引多重集也不变(没有边被创造/消灭)
            check(sorted(sh.W_exc.indices.tolist()) == sorted(art.W_exc.indices.tolist()),
                  "[row] column-index multiset unchanged (no edge created/destroyed)")
        else:
            # 全局列置换会改变列索引多重集(这正是它的作用),但必须保持
            # 入度分布(多重集)与每个神经元输入总权重的多重集
            n = art.n
            same_indeg = np.array_equal(
                np.sort(np.bincount(art.W_exc.indices, minlength=n)),
                np.sort(np.bincount(sh.W_exc.indices, minlength=n)),
            )
            check(same_indeg, "[global] in-degree distribution (multiset) exactly preserved")
            same_col = np.allclose(
                np.sort(np.asarray(art.W_exc.sum(axis=0)).ravel()),
                np.sort(np.asarray(sh.W_exc.sum(axis=0)).ravel()),
            )
            check(same_col, "[global] in-weight sum multiset exactly preserved")
            changed_cols = int(np.count_nonzero(
                np.bincount(art.W_exc.indices, minlength=n)
                != np.bincount(sh.W_exc.indices, minlength=n)
            ))
            print(f"      global mode: {changed_cols}/{n} neurons got a different in-degree "
                  f"(distribution preserved, assignment destroyed)")

        # 元数据与浮点类型
        check(sh.neuron_ids.tolist() == art.neuron_ids.tolist()
              and np.array_equal(sh.soma_pos, art.soma_pos),
              f"[{mode}] neuron_ids / soma_pos carried over unchanged")
        check(sh.W_exc.dtype == np.float32 and sh.W_inh.dtype == np.float32,
              f"[{mode}] weights are float32 CSR (passes __post_init__ checks)")
        check(v["all_invariants_ok"], f"[{mode}] verify_shuffle() -> all_invariants_ok=True")

        # 确定性 + seed 敏感
        sh_again = shuffle_connectome(art, seed=0, mode=mode)
        check(np.array_equal(sh_again.W_exc.indices, sh.W_exc.indices)
              and np.array_equal(sh_again.W_inh.indices, sh.W_inh.indices),
              f"[{mode}] same seed -> identical permutation")
        sh_other = shuffle_connectome(art, seed=12345, mode=mode)
        check(not np.array_equal(sh_other.W_exc.indices, sh.W_exc.indices),
              f"[{mode}] different seed -> different permutation")

        # 下游可用性:matvec 有限、无自环爆炸;存盘/读回一致(集成层会 save)
        x = np.random.default_rng(1).random(art.n).astype(np.float32)
        y = sh.W_exc @ x
        check(np.all(np.isfinite(y)), f"[{mode}] CSR matvec finite (downstream LIF can use it)")
        check(bool(sh.W_exc.has_sorted_indices) == bool(_rows_sorted(sh.W_exc)),
              f"[{mode}] has_sorted_indices flag is truthful "
              f"({bool(sh.W_exc.has_sorted_indices)})")

    # 存盘 / 读回(集成层会 shuffled.save(...))
    import tempfile

    sh = shuffle_connectome(art, seed=0)
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "connectome.npz"
        sh.save(p)
        back = ConnectomeArtifacts.load(p)
        check(np.array_equal(back.W_exc.indices, sh.W_exc.indices)
              and np.array_equal(back.W_exc.data, sh.W_exc.data)
              and np.array_equal(back.W_inh.indptr, sh.W_inh.indptr)
              and np.array_equal(back.neuron_ids, sh.neuron_ids),
              "shuffled artifact survives save()/load() round-trip (npz)")

    # 原对象绝不被就地修改
    check(verify_shuffle(art, shuffle_connectome(art, 0))["data_equal"],
          "input ConnectomeArtifacts is not modified in place (returns a new object)")


def _rows_sorted(W: sp.csr_matrix) -> bool:
    rows = np.repeat(np.arange(W.shape[0], dtype=np.int64), np.diff(W.indptr))
    if W.indices.size == 0:
        return True
    same = rows[1:] == rows[:-1]
    return bool(np.all(W.indices[1:][same] > W.indices[:-1][same]))


# --------------------------------------------------------------------- 6. 渲染


def _downsample(fr: np.ndarray, rows: int, cols: int) -> np.ndarray:
    """把帧降采样成 (rows, cols) 的照度图(块均值),近似 B 线复眼的小眼阵列。

    640/32 == 480/24 == 20,默认配置下可以用 reshape 纯向量化;不能整除时退化到循环。
    """
    lum = 0.299 * fr[..., 0] + 0.587 * fr[..., 1] + 0.114 * fr[..., 2]
    h, w = lum.shape
    if h % rows == 0 and w % cols == 0:
        return lum.reshape(rows, h // rows, cols, w // cols).mean(axis=(1, 3))
    ys = np.linspace(0, h, rows + 1).astype(int)
    xs = np.linspace(0, w, cols + 1).astype(int)
    out = np.zeros((rows, cols), dtype=np.float64)
    for i in range(rows):
        for j in range(cols):
            out[i, j] = lum[ys[i]:max(ys[i + 1], ys[i] + 1), xs[j]:max(xs[j + 1], xs[j] + 1)].mean()
    return out


def check_retina_visibility() -> None:
    """靶在**复眼分辨率**下必须仍然可辨(否则 B 线再好的编码器也白搭)。

    B 线的 Retina 输入只有像素;这里把帧块均值降采样到 RetinaConfig 的
    小眼阵列 (eye_rows x eye_cols),检查:
      1. 含靶的小眼照度显著高于背景小眼(静态对比);
      2. 运动靶在 5 帧内造成显著的小眼照度变化(运动线索)。
    """
    from flyaim.config import RetinaConfig

    section("6b. retina visibility: is the target detectable at micro-eye resolution?")
    rc = RetinaConfig()
    cfg = ArenaConfig()
    er, ec = rc.eye_rows, rc.eye_cols
    print(f"      micro-eye grid: {er} x {ec}  (block {cfg.height // er} x {cfg.width // ec} px)")

    arena = Arena(cfg, 0)
    f0 = arena.reset()
    d0 = _downsample(f0, er, ec)
    bg_level = float(np.median(d0))
    peak = float(d0.max())
    print(f"      single static target: background={bg_level:.1f}, target cell peak={peak:.1f}, "
          f"delta={peak - bg_level:.1f} (0-255)")
    check(peak - bg_level > 40.0,
          f"target is clearly visible at micro-eye resolution (delta {peak - bg_level:.1f} > 40)")

    # 运动线索:运动靶 5 帧内的小眼照度变化
    cfg_m = ArenaConfig(target_speed_px_per_frame=3.0)
    am = Arena(cfg_m, 0)
    fa = am.reset()
    for _ in range(5):
        fb = am.step(np.zeros(2)).frame
    diff = np.abs(_downsample(fb, er, ec) - _downsample(fa, er, ec))
    print(f"      moving target (3 px/frame): max |delta| over 5 frames = {diff.max():.1f}, "
          f"cells changed > 2 levels: {int(np.count_nonzero(diff > 2))}")
    check(diff.max() > 3.0, f"moving target produces a temporal (motion) cue ({diff.max():.1f} > 3)")

    # 准星也必须可辨(否则控制器不知道自己在哪里)
    arena2 = Arena(cfg, 0)
    arena2.reset()
    fr_pre = arena2.render()
    for _ in range(12):
        fr_post = arena2.step(np.array([1.0, 0.0])).frame
    d_pre = _downsample(fr_pre, er, ec)
    d_post = _downsample(fr_post, er, ec)
    moved = np.abs(d_post - d_pre)
    print(f"      crosshair sweep (12 frames x 14 px): max |delta| = {moved.max():.1f}")
    check(moved.max() > 3.0, f"crosshair motion is visible ({moved.max():.1f} > 3)")


def check_render() -> Path:
    section("6. rendering: PNG on disk + luminance contrast for the retina")
    from PIL import Image, ImageDraw

    cfg = ArenaConfig()
    frames: list[tuple[str, np.ndarray]] = []

    # (a) 单靶首帧
    frames.append(("t=0 single target", Arena(cfg, 0).reset()))

    # (b) PID 闭环中的一帧 + (c) 一次命中的瞬间
    arena = Arena(cfg, 0)
    arena.reset()
    pid = PIDBaseline(cfg).attach_arena(arena)
    got_mid = got_hit = None
    for i in range(200):
        res = arena.step(pid.act())
        if i == 60:
            got_mid = res.frame
        if res.hit and got_hit is None:
            got_hit = res.frame
        if got_mid is not None and got_hit is not None:
            break
    frames.append(("pid mid-episode", got_mid if got_mid is not None else arena.render()))
    frames.append(("HIT moment (flash)", got_hit if got_hit is not None else arena.render()))

    # (d) 3 靶 + 运动
    cfg3 = ArenaConfig(n_targets=3, target_speed_px_per_frame=2.5)
    a3 = Arena(cfg3, 5)
    a3.reset()
    for _ in range(40):
        a3.step(np.array([0.2, -0.15]))
    frames.append(("3 moving targets", a3.render()))

    # 亮度对比检查(复眼只看像素,靶必须明显可辨)
    f = frames[0][1]
    check(f.dtype == np.uint8 and f.shape == (cfg.height, cfg.width, 3),
          f"frame is uint8 (H,W,3) = {f.shape} {f.dtype}")

    def lum(rgb):
        return 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]

    bg, tg, ch = cfg.bg_color, cfg.target_color, cfg.crosshair_color
    print(f"      luminance: bg={lum(bg):.1f}, target={lum(tg):.1f}, crosshair={lum(ch):.1f}")
    check(lum(tg) - lum(bg) > 60.0,
          f"target vs background luminance contrast {lum(tg) - lum(bg):.1f} > 60")
    check(lum(ch) - lum(bg) > 150.0,
          f"crosshair vs background luminance contrast {lum(ch) - lum(bg):.1f} > 150")

    unique_colors = np.unique(f.reshape(-1, 3), axis=0).shape[0]
    check(unique_colors >= 3, f"frame contains >= 3 distinct colors (found {unique_colors})")

    # 拼 2x2 蒙太奇
    tiles = []
    for label, fr in frames:
        img = Image.fromarray(fr, mode="RGB")
        tile = Image.new("RGB", (img.width, img.height + 18), (0, 0, 0))
        tile.paste(img, (0, 18))
        ImageDraw.Draw(tile).text((4, 4), label, fill=(255, 255, 0))
        tiles.append(tile)
    tw, th = tiles[0].size
    canvas = Image.new("RGB", (tw * 2 + 6, th * 2 + 6), (0, 0, 0))
    for i, tile in enumerate(tiles):
        canvas.paste(tile, ((i % 2) * (tw + 6), (i // 2) * (th + 6)))
    out = HERE / "arena_render.png"
    canvas.save(out)
    with Image.open(out) as chk:
        chk.verify()
    size_kb = out.stat().st_size / 1024
    print(f"      wrote {out}  ({canvas.width}x{canvas.height}, {size_kb:.0f} KB)")
    check(out.exists() and out.stat().st_size > 5_000, "arena_render.png exists and is not empty")
    return out


# --------------------------------------------------------------------- 7. 曲线图


def check_plots(bl: dict) -> list[Path]:
    section("7. curve PNGs (acq_curve + hit-rate comparison) for the Phase 3 report")
    from flyaim.arena import plot as pltmod

    summaries = {
        arm: {
            "hit_rate": float(np.mean([s["hit_rate"] for s in bl["summaries"][arm]])),
            "hit_rate_std": float(np.std([s["hit_rate"] for s in bl["summaries"][arm]], ddof=1)),
            "hits": int(np.sum([s["hits"] for s in bl["summaries"][arm]])),
            "frames": int(np.sum([s["frames"] for s in bl["summaries"][arm]])),
        }
        for arm in ("pid", "random")
    }
    for arm in summaries:
        n = len(bl["summaries"][arm])
        summaries[arm]["hit_rate_se"] = summaries[arm]["hit_rate_std"] / np.sqrt(n)

    curves = {
        arm: [float(v) for v in np.mean(np.asarray(bl["curves"][arm]), axis=0)]
        for arm in ("pid", "random")
    }

    outs = []
    outs.append(pltmod.plot_acq_curves(
        curves, HERE / "acq_curve.png",
        title="C-line selftest: mean hit rate per bin (pid vs random, 3 arena seeds)",
    ))
    outs.append(pltmod.plot_hit_rate_bars(
        summaries, HERE / "hit_rate_comparison.png",
        title="C-line selftest: hit rate by arm (900 frames x 3 seeds)",
    ))
    outs.append(pltmod.make_report_figure(curves, summaries, HERE / "report_panel.png"))

    from PIL import Image

    for p in outs:
        ok = p.exists() and p.stat().st_size > 5_000
        if ok:
            with Image.open(p) as im:  # 真能打开
                im.verify()
        check(ok, f"{p.name} written and decodable ({p.stat().st_size / 1024:.0f} KB)")
    return outs


# --------------------------------------------------------------------- main


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(
        description="FlyAim C-line selftest (arena / metrics / pid / random / shuffle)"
    )
    ap.add_argument("--seeds", type=int, default=3,
                    help="number of arena seeds for the pid-vs-random comparison (default 3)")
    ap.add_argument("--frames", type=int, default=900,
                    help="frames per episode (default 900 = ArenaConfig.max_frames)")
    args = ap.parse_args(argv)

    print("FlyAim C-line selftest (arena / metrics / pid / random / shuffle)")
    print(f"python: {sys.version.split()[0]}  cwd: {Path.cwd()}")
    print(f"numpy: {np.__version__}  scipy: {scipy.__version__}")

    check_signatures()
    check_determinism()
    bl = check_baselines(seeds=tuple(range(args.seeds)), frames=args.frames)
    check_metrics()
    check_shuffle()
    check_retina_visibility()
    png = check_render()
    curve_pngs = check_plots(bl)

    section("summary")
    if FAILURES:
        print(f"  {len(FAILURES)} CHECK(S) FAILED:")
        for f in FAILURES:
            print(f"    - {f}")
    else:
        print("  ALL CHECKS PASSED (0 failures)")
    print()
    print("  artifacts (absolute paths):")
    print(f"    render montage : {png.resolve()}")
    for p in curve_pngs:
        print(f"    curve figure   : {p.resolve()}")
    print(f"  elapsed: {time.time() - T0:.1f}s")
    print("  tip: python flyaim/arena/selftest.py --seeds 10   # tighter error bars")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
