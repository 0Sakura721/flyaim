"""B 线合成连接组自检(不依赖 A 线数据)。

跑三件事:
  1. 合成 N=2000 连接组 + 合成 neuron_index/roles -> 完整
     `Retina -> Connectome -> Readout` 链路 200 步,断言不变量;
  2. trained 读出:用合成靶场画面生成监督数据,跑通
     `fit / set_linear_weights / act / save / load`,并验证训练-评估种子分离;
  3. 真实规模压力测试:N=50000、密度 0.45% 的 CSR,实测单步耗时与内存。
     加 `--full-scale` 额外测 N=166700(真全局规模)作为参照。

运行(必须用捆绑解释器):
    C:\\Users\\Admin\\.dsh\\dsh-runtimes\\dsh-primary-runtime\\dependencies\\python\\python.exe \\他
        flyaim/brain/selftest_synth.py [--full-scale]

本文件是**测试夹具**,里面的"合成靶场"当然知道靶的坐标(用来生成监督标签),
但 `Retina.frame_to_spikes` 只接收像素帧——禁止事项 5.1 的隔离由接口保证。
"""

from __future__ import annotations

import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import scipy
import scipy.sparse as sp

from flyaim.brain.lif import Connectome
from flyaim.brain.readout import Readout
from flyaim.config import BrainConfig, ReadoutConfig, RetinaConfig
from flyaim.io import ConnectomeArtifacts, save_roles
from flyaim.neuron_index import NeuronIndex, RoleSelection
from flyaim.retina.encoder import Retina

# ------------------------------------------------------------------ 合成数据规模
N_SYNTH = 2000
N_R16, N_R7, N_R8 = 120, 40, 40          # 感光细胞(合成)
N_DN, N_MN = 40, 16                      # 下行 / 运动神经元
N_INH = 200                              # 抑制性(GABA)神经元
SYNTH_DENSITY = 0.03                     # 随机主体密度(入度 ≈ 60)
PHOTO_TO_DN = 200                        # 每个 DN 从感光细胞群直接接收的突触数
PHOTO_W = 60.0                           # 上述突触的计数权重
SYNTH_INPUT_GAIN = 0.06                  # 合成网络的输入增益标定(见下方说明)

# 合成网络是**人为标定**的:随机主体的入度(~60)远小于真实连接组(~911),
# 若用默认 input_gain=weight_scale_exc=0.02,整网几乎完全静默(实测 DN=0 Hz),
# 自检会退化成空转。这里显式提高输入增益并给 DN 一条强直连视觉通路,
# 目的是让合成网络落在"DN 真的在放电"的区间,从而**真正检验**链路与学习代码。
# 真实标定以 selftest_real.py 在真实数据上打印的 rates 统计为准。
EYE_SYNTH = (6, 8)                       # 合成链路的小眼阵列(200 感光细胞 / 48 小眼)
EYE_DEFAULT = (24, 32)                   # RetinaConfig 默认(= 回退规模 768)
FRAME = (48, 64)
BLOB_RADIUS = 6
STEPS_PER_FRAME = 20                     # 学习环节用(测试提速);主链路按 33 步/帧
DIR_SCALE = 24.0                         # 角误差 -> 期望方向 的归一化尺度(px)
DENSITY_REAL = 0.00546                   # A 线实测稀疏度 0.546%(151,856,684 条边)
RIDGE_LAMBDA_TEST = 10.0                 # 合成数据只有 ~144 样本,λ=1 会过拟合(实测)


def section(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def _cols_to_full(Wsub: sp.csr_matrix, cols_global: np.ndarray, n: int) -> sp.csr_matrix:
    """把 (n, k) 的列子集还原成 (n, n) 形状(列号映射回全局位置)。

    `ConnectomeArtifacts` 要求 W_exc/W_inh 都是 (N, N),这样兴奋/抑制两张矩阵
    才在同一索引空间里。
    """
    coo = Wsub.tocoo()
    col = np.asarray(cols_global, dtype=np.int64)[coo.col]
    return sp.csr_matrix((coo.data.astype(np.float32), (coo.row, col)), shape=(n, n))


def _save_index_parquet(idx: NeuronIndex, path: Path) -> None:
    """落 neuron_index.parquet 并立刻用真实读取路径回读校验。

    走 A 线实际使用的 `NeuronIndex.save()`(Lead 已修好 pandas 3.0 的
    `index_label` 兼容问题),再用 `NeuronIndex.load()` 回读 —— 这样自检覆盖
    的是**真实产物路径**,而不是测试自己发明的写法。
    """
    idx.save(path)
    if path.stat().st_size == 0:
        raise AssertionError("NeuronIndex.save() 写出了 0 字节文件")
    back = NeuronIndex.load(path)
    if back.n != idx.n:
        raise AssertionError(f"neuron_index 回读行数不符: {back.n} != {idx.n}")
    if not np.array_equal(back.body_ids, idx.body_ids):
        raise AssertionError("neuron_index 回读 body_id 不一致")


def build_synth_connectome(
    n: int = N_SYNTH, seed: int = 7
) -> tuple[ConnectomeArtifacts, RoleSelection, NeuronIndex]:
    """构造合成连接组 + 合成细胞元数据。

    组成:
      - `scipy.sparse.random` 随机主体(兴奋性,列 = 非抑制性神经元);
      - 显式"感光细胞 -> DN"通路(每个 DN 从感光细胞群取 PHOTO_TO_DN 个突触),
        让 DN 群对视觉刺激有**可学习的**响应 —— 这是对读出层学习代码的闭环检验;
      - 抑制性神经元的输出列单独放进 W_inh。
    """
    rng = np.random.default_rng(seed)
    n_photo = N_R16 + N_R7 + N_R8
    photo = np.arange(0, n_photo, dtype=np.int64)
    r16 = np.arange(0, N_R16, dtype=np.int64)
    r7 = np.arange(N_R16, N_R16 + N_R7, dtype=np.int64)
    r8 = np.arange(N_R16 + N_R7, n_photo, dtype=np.int64)
    dn = np.arange(n_photo, n_photo + N_DN, dtype=np.int64)
    mn = np.arange(n_photo + N_DN, n_photo + N_DN + N_MN, dtype=np.int64)
    inh = np.arange(n - N_INH, n, dtype=np.int64)
    exc_cols = np.setdiff1d(np.arange(n, dtype=np.int64), inh)

    # 随机主体:scipy.sparse.random(任务要求)
    W = sp.random(
        n, n, density=SYNTH_DENSITY, format="csr", random_state=int(seed), dtype=np.float32
    )
    W.data = np.maximum(np.ceil(W.data * 5.0), 1.0).astype(np.float32)  # 突触计数 1..5

    # 显式 感光细胞 -> DN 通路(每个 DN 固定 PHOTO_TO_DN 个突触)
    rows = np.repeat(dn, PHOTO_TO_DN)
    cols = photo[rng.integers(0, photo.size, size=rows.size)]
    extra = sp.csr_matrix(
        (np.full(rows.size, PHOTO_W, dtype=np.float32), (rows, cols)), shape=(n, n)
    )
    W = (W + extra).tocsr()

    # 按突触前神经元的抑制性拆分两张矩阵(保持 (N,N) 形状,列号回到全局位置)
    Wc = W.tocsc()
    inh_mask = np.zeros(n, dtype=bool)
    inh_mask[inh] = True
    W_inh = _cols_to_full(Wc[:, inh_mask].tocsr(), inh, n)
    W_exc = _cols_to_full(Wc[:, ~inh_mask].tocsr(), exc_cols, n)
    W_exc.sum_duplicates()
    W_inh.sum_duplicates()

    neuron_ids = np.arange(100000, 100000 + n, dtype=np.int64)
    art = ConnectomeArtifacts(
        W_exc=W_exc.astype(np.float32), W_inh=W_inh.astype(np.float32), neuron_ids=neuron_ids
    )

    types = np.empty(n, dtype=object)
    types[:] = "KCg"
    types[r16] = "R1-R6"
    types[r7] = "R7y"
    types[r8] = "R8p"
    types[dn] = "DNg01"
    types[mn] = "MN1"
    types[inh] = "GABA_intrinsic"
    sides = np.empty(n, dtype=object)
    sides[:] = "unknown"
    sides[r16[::2]] = "left"      # 合成感光细胞的 side 故意只标一半(真实数据也缺)
    sides[r16[1::2]] = "right"
    sides[dn[: N_DN // 2]] = "left"   # DN 的 side 标全,用于测试侧向分组
    sides[dn[N_DN // 2 :]] = "right"
    nts = np.empty(n, dtype=object)
    nts[:] = ""
    nts[inh] = "gaba"
    superclass = np.empty(n, dtype=object)
    superclass[:] = "intrinsic"
    superclass[photo] = "ol_sensory"
    superclass[dn] = "descending_neuron"
    superclass[mn] = "vnc_motor"

    idx = NeuronIndex.from_arrays(
        body_id=neuron_ids,
        name=types,
        cls_=superclass,
        type_=types,
        side=sides,
        nt=nts,
    )
    roles = RoleSelection(
        visual_input=photo,
        descending=dn,
        motor=mn,
        inhibitory=inh,
        visual_input_strategy="synthetic:ol_sensory",
        visual_input_fallback=False,
        notes=["合成连接组:非真实数据,仅用于链路自检"],
    )
    return art, roles, idx


# ------------------------------------------------------------------ 合成靶场
class SynthArena:
    """测试夹具:生成带红色靶团的画面,并给出监督方向。

    只用于生成**像素帧**和**训练标签**;Retina 永远只拿到 frame。
    """

    def __init__(self, h: int = FRAME[0], w: int = FRAME[1], radius: int = BLOB_RADIUS) -> None:
        self.h, self.w, self.radius = h, w, radius
        self.bg = np.array([18, 18, 22], dtype=np.uint8)
        self.fg = np.array([235, 70, 70], dtype=np.uint8)
        self.center = np.array([w / 2.0, h / 2.0])
        yy, xx = np.mgrid[0:h, 0:w]
        self._yy, self._xx = yy, xx

    def frame(self, blob_xy: tuple[float, float], noise_rng: np.random.Generator | None = None) -> np.ndarray:
        img = np.empty((self.h, self.w, 3), dtype=np.uint8)
        img[:] = self.bg
        cx, cy = blob_xy
        mask = (self._xx - cx) ** 2 + (self._yy - cy) ** 2 <= self.radius**2
        img[mask] = self.fg
        if noise_rng is not None:
            n = noise_rng.integers(-6, 7, size=img.shape, dtype=np.int16)
            img = np.clip(img.astype(np.int16) + n, 0, 255).astype(np.uint8)
        return img

    def desired_dir(self, blob_xy: tuple[float, float], scale: float = DIR_SCALE) -> np.ndarray:
        """靶相对准星(画面中心)的角误差 -> 期望鼠标方向(归一化到 ~[-1,1])。

        scale=24px(半个画面宽)让整个视野内的角误差都能落到 [-1,1],
        而不是早早饱和成 ±1 的阶跃标签(饱和标签无法检验回归的泛化)。
        """
        err = (np.asarray(blob_xy, dtype=np.float64) - self.center)
        return np.clip(err / scale, -1.0, 1.0).astype(np.float32)


def random_blob(rng: np.random.Generator, arena: SynthArena) -> tuple[float, float]:
    return (
        float(rng.uniform(arena.radius + 1, arena.w - arena.radius - 1)),
        float(rng.uniform(arena.radius + 1, arena.h - arena.radius - 1)),
    )


def grid_blobs(rng: np.random.Generator, arena: SynthArena, nx: int = 6, ny: int = 4):
    """均匀覆盖视野的靶位(带抖动),避免训练/评估位置分布不均。"""
    out = []
    for gy in np.linspace(arena.radius + 3, arena.h - arena.radius - 3, ny):
        for gx in np.linspace(arena.radius + 3, arena.w - arena.radius - 3, nx):
            out.append(
                (
                    float(np.clip(gx + rng.uniform(-3, 3), arena.radius + 1, arena.w - arena.radius - 1)),
                    float(np.clip(gy + rng.uniform(-3, 3), arena.radius + 1, arena.h - arena.radius - 1)),
                )
            )
    return out


def _retina_cfg(td: Path) -> RetinaConfig:
    """合成链路用的 Retina 配置(小眼阵列 6x8 -> 200 感光细胞约 4 个/小眼)。"""
    c = RetinaConfig(eye_rows=EYE_SYNTH[0], eye_cols=EYE_SYNTH[1])
    c.data_dir = str(td)  # 让 Retina 从 neuron_index.parquet 读 type
    return c


# ------------------------------------------------------------------ 主链路自检
def test_chain(tmp: Path) -> dict:
    section("1. 合成连接组链路自检(Retina -> Connectome -> Readout,200 步)")
    art, roles, idx = build_synth_connectome()
    n = art.n
    print(f"合成规模: N={n}, edges_exc={art.W_exc.nnz}, edges_inh={art.W_inh.nnz}, "
          f"density={art.stats()['density']:.4f}")
    print("角色: " + ", ".join(f"{k}={v}" for k, v in roles.summary().items() if k.startswith("n_")))

    weights_path = tmp / "connectome.npz"
    art.save(weights_path)
    save_roles(tmp / "roles.json", roles)
    _save_index_parquet(idx, tmp / "neuron_index.parquet")
    print(f"产物: {weights_path} ({weights_path.stat().st_size/1e6:.2f} MB), "
          f"roles.json, neuron_index.parquet")

    # ---- 冻结校验基线(与 lead 的 fit_readout.assert_connectome_frozen 同思路)
    frozen = {
        "exc_data": art.W_exc.data.tobytes(),
        "exc_idx": art.W_exc.indices.tobytes(),
        "exc_ptr": art.W_exc.indptr.tobytes(),
        "inh_data": art.W_inh.data.tobytes(),
        "inh_idx": art.W_inh.indices.tobytes(),
        "inh_ptr": art.W_inh.indptr.tobytes(),
    }

    # ---- 组装:注意 roles.json 在同目录,Connectome 自动读取
    r_cfg = _retina_cfg(tmp)
    b_cfg = BrainConfig(subnet_size=None, steps_per_frame=33)
    b_cfg.input_gain = SYNTH_INPUT_GAIN
    retina = Retina(r_cfg, input_neuron_ids=roles.visual_input)
    readout = Readout(ReadoutConfig(mode="fixed"), roles=roles, neuron_index=idx)
    brain = Connectome(str(weights_path), b_cfg, input_neuron_ids=roles.visual_input)
    print(f"\n{retina!r}\n{brain!r}\n{readout!r}")
    rd = retina.describe()
    print("Retina.describe(): "
          f"n_input={rd['n_input']}, channels={rd['channel_map']}, "
          f"types_source={rd['types_source']}, retinotopy={rd['retinotopy']}, "
          f"gain_mode={rd['gain_mode']}")
    print("Readout 分组: grouping=%s, groups=%s, dn=%d, mn=%d"
          % (readout.grouping, readout.describe()["groups"], readout.n_dn, readout.n_mn))

    assert rd["uses_target_coordinates"] is False
    assert rd["input_neuron_ids_fallback"] is False, "真实 roles 不该触发回退"
    assert rd["channel_map"]["luminance_R1R6"] == N_R16
    assert rd["channel_map"]["color_R7_red_excited"] == N_R7
    assert rd["channel_map"]["color_R8_green_excited"] == N_R8
    assert readout.grouping == "side_x_half", "DN 的 side 标注完整时应使用侧向分组"

    # ---- 链路:200 步,每 33 步换一帧
    arena = SynthArena()
    rng = np.random.default_rng(1234)
    drive_shapes, rate_shapes, nan_hits = set(), set(), 0
    max_rate = 0.0
    rate_nonzero = []
    t0 = time.perf_counter()
    blob = random_blob(rng, arena)
    drive = retina.frame_to_spikes(arena.frame(blob, rng))
    acts = []
    for i in range(200):
        if i % 33 == 0:
            blob = random_blob(rng, arena)
            drive = retina.frame_to_spikes(arena.frame(blob, rng))
            drive_shapes.add(drive.shape)
            assert drive.dtype == np.float32
            assert np.all(np.isfinite(drive))
            assert np.all(drive >= 0.0), "ON/OFF 驱动必须非负"
            assert np.all(drive[:, 0] * drive[:, 1] == 0.0), "ON/OFF 通道必须互斥"
        brain.step(drive[:, 0], drive[:, 1], i * b_cfg.dt_ms)
        rates = brain.rates
        spk = brain.spikes
        rate_shapes.add(rates.shape)
        assert rates.shape == (n,) and rates.dtype == np.float32
        assert spk.shape == (n,) and spk.dtype == bool
        if not np.all(np.isfinite(rates)):
            nan_hits += 1
        max_rate = max(max_rate, float(rates.max()))
        if i % 50 == 0:
            rate_nonzero.append(float(np.mean(rates > 0)))
            acts.append(readout.act(brain))
    wall = time.perf_counter() - t0
    print(f"\n链路 200 步: 帧驱动 shape={drive_shapes}, rates shape={rate_shapes}, "
          f"NaN 步数={nan_hits}, max_rate={max_rate:.1f} Hz, 墙钟={wall:.2f}s")
    print(f"rates 非零比例(每 50 步采样)={['%.3f' % v for v in rate_nonzero]}")
    print(f"fixed 读出输出样本={[list(np.round(a, 3)) for a in acts[:5]]}")
    assert nan_hits == 0, "rates 出现 NaN"
    assert drive_shapes == {(N_R16 + N_R7 + N_R8, 2)}
    assert rate_shapes == {(n,)}
    assert float(rates[roles.visual_input].max()) > 0, "感光细胞群完全静默"
    assert float(rates[roles.descending].max()) > 0, "DN 群完全静默,链路未真正被驱动"
    print(f"末步: 感光细胞群非零={np.mean(rates[roles.visual_input]>0):.3f}, "
          f"DN 群非零={np.mean(rates[roles.descending]>0):.3f}, "
          f"DN 均值={rates[roles.descending].mean():.2f} Hz")

    # ---- 缓冲区独立性:外部拿到的数组不能被后续 step 覆盖
    r1 = brain.rates
    s1 = brain.spikes
    r1[:] = -999.0
    s1[:] = True
    brain.step(drive[:, 0], drive[:, 1], 0.0)
    r2, s2 = brain.rates, brain.spikes
    assert not np.any(r2 < 0), "rates 复用了外部可见缓冲区(被外部写入污染)"
    assert r2 is not r1 and s2 is not s1
    print("缓冲区独立性: OK(rates/spikes 每次返回新数组)")

    # ---- 读出层断言
    act = readout.act(brain)
    assert act.shape == (2,) and act.dtype == np.float32
    assert np.all(np.isfinite(act)) and np.all(np.abs(act) <= 1.0)
    print(f"fixed 读出: act={np.round(act,4).tolist()} (shape/dtype/范围 OK)")

    # ---- 子网模式:形状必须回到全局索引空间
    sub_cfg = BrainConfig(subnet_size=1200, subnet_seed=3, steps_per_frame=33)
    sub_cfg.input_gain = SYNTH_INPUT_GAIN
    brain_sub = Connectome(str(weights_path), sub_cfg, input_neuron_ids=roles.visual_input)
    sim = brain_sub.sim_indices
    assert brain_sub.n_neurons == n
    assert brain_sub.n_sim == 1200, "子网规模应为 1200"
    assert np.all(brain_sub.compact_index_of(roles.descending) >= 0), "子网必须含全部 DN"
    assert np.all(brain_sub.compact_index_of(roles.motor) >= 0), "子网必须含全部 MN"
    assert np.all(brain_sub.compact_index_of(roles.visual_input) >= 0), "子网必须含全部输入群"
    outside = np.setdiff1d(np.arange(n), sim)
    assert outside.size > 0, "子网没有真的生效"
    for i in range(50):
        brain_sub.step(drive[:, 0], drive[:, 1], i * sub_cfg.dt_ms)
    r_sub, s_sub = brain_sub.rates, brain_sub.spikes
    assert r_sub.shape == (n,) and s_sub.shape == (n,) and r_sub.dtype == np.float32
    assert np.all(r_sub[outside] == 0.0), "子网外位置必须为 0"
    assert np.all(~s_sub[outside])
    assert np.any(r_sub[roles.descending] > 0), "子网内 DN 完全静默"
    print(f"子网模式: m={brain_sub.n_sim}/{n}, rates shape={r_sub.shape} "
          f"(全局索引空间 OK), 子网外全零 OK, DN 平均率={r_sub[roles.descending].mean():.2f} Hz")

    # ---- 输入群为空 -> 优雅回退(不得抛异常)
    # 真实 dataset 里 roles.visual_input 非空(6098),所以这里显式构造"视觉输入为空"
    # 的产物目录来覆盖降级路径:connectome.npz + roles.json(visual_input=空)。
    retina_fb = Retina(RetinaConfig())
    assert retina_fb.describe()["input_neuron_ids_fallback"] is True
    assert retina_fb.input_neuron_ids.size == EYE_DEFAULT[0] * EYE_DEFAULT[1]
    d_fb = retina_fb.frame_to_spikes(arena.frame(blob, rng))
    fb_dir = tmp / "no_visual"
    fb_dir.mkdir()
    shutil.copyfile(weights_path, fb_dir / "connectome.npz")
    save_roles(
        fb_dir / "roles.json",
        RoleSelection(
            descending=roles.descending,
            motor=roles.motor,
            inhibitory=roles.inhibitory,
            visual_input_strategy="synthetic:empty_visual",
            visual_input_fallback=True,
        ),
    )
    brain_fb = Connectome(
        str(fb_dir / "connectome.npz"), BrainConfig(subnet_size=1200, subnet_seed=1)
    )
    assert brain_fb.stats()["input_group"]["fallback"] is True, "空输入群必须走回退而不是报错"
    assert brain_fb.in_neuron_ids.size == retina_fb.input_neuron_ids.size, (
        "两端回退规模必须一致,否则驱动会错位"
    )
    for _ in range(20):
        brain_fb.step(d_fb[:, 0], d_fb[:, 1], 0.0)
    assert brain_fb.rates.shape == (n,)
    print(f"空输入群回退: Retina n_input={retina_fb.input_neuron_ids.size} (arange), "
          f"Connectome 回退={brain_fb.stats()['input_group']['fallback']} "
          f"规模一致={brain_fb.in_neuron_ids.size}, 20 步无异常 OK")

    # ---- 非整除帧尺寸(走 bincount 路径)
    odd = arena.frame(blob, rng)[:47, :61]
    d_odd = retina.frame_to_spikes(odd)
    assert d_odd.shape == (N_R16 + N_R7 + N_R8, 2) and np.all(np.isfinite(d_odd))
    print("非整除帧尺寸 (47,61): OK")

    # ---- 冻结校验:全程不得改动连接组权重
    art2 = ConnectomeArtifacts.load(weights_path)
    for name, W in (("exc", art2.W_exc), ("inh", art2.W_inh)):
        assert W.data.tobytes() == frozen[f"{name}_data"], f"W_{name}.data 被修改!"
        assert W.indices.tobytes() == frozen[f"{name}_idx"], f"W_{name}.indices 被修改!"
        assert W.indptr.tobytes() == frozen[f"{name}_ptr"], f"W_{name}.indptr 被修改!"
    print("连接组冻结校验: 磁盘产物 data/indices/indptr 逐位一致 OK")

    return {
        "brain": brain,
        "retina": retina,
        "readout": readout,
        "roles": roles,
        "arena": arena,
        "weights_path": weights_path,
        "sim_ok": True,
    }


# ------------------------------------------------------------------ trained 读出
def test_trained_readout(tmp: Path, ctx: dict) -> None:
    section("2. trained 读出:岭回归 / 权重注入 / save-load / 种子分离")
    roles = ctx["roles"]
    arena = ctx["arena"]
    weights_path = ctx["weights_path"]
    tmp_path = ctx["weights_path"].parent
    b_cfg = BrainConfig(subnet_size=None, steps_per_frame=STEPS_PER_FRAME)
    b_cfg.input_gain = SYNTH_INPUT_GAIN

    def collect(seed: int) -> tuple[np.ndarray, np.ndarray]:
        """一个种子 = 一轮独立 episode(靶位覆盖整个视野 + 抖动)。"""
        rng = np.random.default_rng(seed)
        retina = Retina(_retina_cfg(tmp_path), input_neuron_ids=roles.visual_input)
        brain = Connectome(str(weights_path), b_cfg, input_neuron_ids=roles.visual_input)
        ro = Readout(ReadoutConfig(mode="trained"), roles=roles)
        X, Y = [], []
        for blob in grid_blobs(rng, arena):
            retina.reset()
            drive = retina.frame_to_spikes(arena.frame(blob, rng))
            for _ in range(STEPS_PER_FRAME):
                brain.step(drive[:, 0], drive[:, 1], 0.0)
            X.append(ro.features(brain))
            Y.append(arena.desired_dir(blob))
        return np.asarray(X), np.asarray(Y)

    Xtr, Ytr = zip(*(collect(s) for s in (0, 1, 2)))
    Xtr, Ytr = np.vstack(Xtr), np.vstack(Ytr)
    Xte, Yte = zip(*(collect(s) for s in (100, 101, 102)))
    Xte, Yte = np.vstack(Xte), np.vstack(Yte)
    print(f"监督数据: train={Xtr.shape} (seeds={list(ReadoutConfig().train_seeds)}), "
          f"eval={Xte.shape} (seeds={list(ReadoutConfig().eval_seeds)})")
    print(f"DN 率统计: 非零比例={np.mean(Xtr>0):.3f}, 均值={Xtr.mean():.2f} Hz, 最大={Xtr.max():.1f} Hz")
    print(f"标签范围: dx[{Ytr[:,0].min():.2f},{Ytr[:,0].max():.2f}] "
          f"dy[{Ytr[:,1].min():.2f},{Ytr[:,1].max():.2f}] std={Ytr.std():.3f}")

    ro = Readout(ReadoutConfig(mode="trained", ridge_lambda=RIDGE_LAMBDA_TEST), roles=roles)
    ro.fit(Xtr, Ytr)
    m = ro.evaluate(Xte, Yte, prefix="eval_")
    print(f"拟合(ridge_lambda={RIDGE_LAMBDA_TEST}): train_r2={ro.train_info['train_r2_all']:.3f}, "
          f"eval_r2={m['eval_r2_all']:.3f} (dx={m['eval_r2_dx']:.3f}, dy={m['eval_r2_dy']:.3f}), "
          f"eval_rmse={m['eval_rmse']:.3f}")
    assert np.isfinite(ro.train_info["train_r2_all"])
    assert ro.train_info["train_r2_all"] > 0.5, "训练集都拟合不上,学习通路有问题"
    assert m["eval_r2_all"] > 0.10, "评估集无泛化:监督信号未真正进入 DN 群"

    # 前向一致性:手动按 docstring 约定算一遍,必须与 act() 的 raw 一致
    w = ro.get_linear_weights()
    z = (Xte - w["mu"]) / w["sd"]
    manual = (np.concatenate([z, np.ones((z.shape[0], 1))], axis=1) @ w["W"])[0]
    ro.reset()
    ro.set_linear_weights(w["W"], w["mu"], w["sd"])
    probe = ro._forward_trained(Xte[0])
    assert np.allclose(manual, probe, atol=1e-6), "act() 前向与 set_linear_weights 约定不一致"
    print(f"前向一致性: 手工 (x-mu)/sd + 偏置 @ W 与 _forward_trained 一致 OK "
          f"(样本输出 {np.round(manual,3).tolist()})")

    # W 形状约定
    assert w["W"].shape == (ro.n_features + 1, 2)
    assert w["mu"].shape == (ro.n_features,) and w["sd"].shape == (ro.n_features,)
    assert np.all(w["sd"] > 0)
    print(f"权重形状: W={w['W'].shape} (末行=偏置), mu={w['mu'].shape}, sd={w['sd'].shape} OK")

    # save / load 往返
    wp = tmp / "readout_weights.npz"
    ro.save(wp)
    ro2 = Readout(ReadoutConfig(mode="trained"), roles=roles)
    ro2.load(wp)
    a = ro.act(ctx["brain"])
    b = ro2.act(ctx["brain"])
    assert np.allclose(a, b, atol=1e-7), "save/load 往返结果不一致"
    print(f"save/load 往返: {wp.name} ({wp.stat().st_size/1024:.1f} KB), act 一致 OK")

    # 种子分离必须被强制
    try:
        Readout(ReadoutConfig(mode="trained", train_seeds=(0, 1), eval_seeds=(1, 2)), roles=roles)
    except ValueError as exc:
        print(f"种子分离校验: OK({exc})")
    else:
        raise AssertionError("训练/评估种子重叠时必须抛 ValueError")

    # 无权重时应明确报错,而不是返回垃圾
    ro3 = Readout(ReadoutConfig(mode="trained"), roles=roles)
    try:
        ro3.act(ctx["brain"])
    except RuntimeError:
        print("未训练就 act(): 正确抛出 RuntimeError OK")
    else:
        raise AssertionError("trained 模式无权重时必须报错")

    # DN 为空时回退 MN(契约补充)
    r_mn_only = RoleSelection(motor=roles.motor)
    ro_mn = Readout(ReadoutConfig(mode="fixed"), roles=r_mn_only)
    assert ro_mn.source_kind.startswith("mn") and ro_mn.readout_source_fallback
    assert ro_mn.act(ctx["brain"]).shape == (2,)
    print("DN 为空 -> 回退 MN: OK(readout_source_fallback=True)")

    # DN/MN 都为空 -> 返回 0,不冒充
    ro_none = Readout(ReadoutConfig(mode="fixed"), roles=RoleSelection())
    out_none = ro_none.act(ctx["brain"])
    assert np.all(out_none == 0.0)
    print("DN/MN 均为空 -> act() 恒 0(不冒充控制群)OK")


# ------------------------------------------------------------------ 压力测试
def _bench(
    art_path: Path, cfg: BrainConfig, n_input: int, steps: int, label: str, warm: int = 30
) -> dict:
    """单步耗时实测。

    `warm` 步预热是**必须的**:稀疏传播的 CSC 视图(`W.tocsc()`)是首用时惰性构建的
    (13.65M 非零约 0.3 s),而且网络从静默状态到产生脉冲也需要几步。若预热不足,
    一次性开销会落进计时窗口,把均值污染成几十毫秒(实测踩过这个坑)。
    因此同时报告 p50(稳健)与 mean。
    """
    t0 = time.perf_counter()
    brain = Connectome(str(art_path), cfg, input_neuron_ids=np.arange(n_input, dtype=np.int64))
    build_s = time.perf_counter() - t0
    n_in = brain.in_neuron_ids.size
    exc = np.random.default_rng(0).random(n_in).astype(np.float32) * 200.0
    inh = np.random.default_rng(1).random(n_in).astype(np.float32) * 50.0
    for i in range(warm):
        brain.step(exc, inh, i * cfg.dt_ms)
    t0 = time.perf_counter()
    for i in range(steps):
        brain.step(exc, inh, i * cfg.dt_ms)
    wall = time.perf_counter() - t0
    ts = brain.timing_stats()
    st = brain.stats()
    out = {
        "label": label,
        "n_global": st["n_neurons"],
        "m_sim": st["n_sim"],
        "subnet": st["subnet_applied"],
        "nnz": st["n_edges_total"],
        "matrix_MB": st["matrix_bytes"] / 1e6,
        "state_MB": st["state_bytes"] / 1e6,
        "build_s": build_s,
        "mean_ms": wall / steps * 1000.0,
        "p50_ms": ts["p50_ms"],
        "p95_ms": ts["p95_ms"],
        "fired_frac": float(brain._n_fired / max(brain.n_sim, 1)),
        "active_frac": float(np.mean(brain.rates_compact > 0)),
        "mean_rate_hz": float(brain.rates_compact.mean()),
        "path": brain.last_path,
    }
    print(
        f"  {label:26s} m={out['m_sim']:>7,} nnz={out['nnz']:>13,} {out['matrix_MB']:6.0f}MB | "
        f"p50={out['p50_ms']:7.3f} mean={out['mean_ms']:7.3f} p95={out['p95_ms']:8.3f} ms"
    )
    print(
        f"      发放率={out['fired_frac']*100:5.2f}% 均值率={out['mean_rate_hz']:6.2f}Hz "
        f"路径={out['path']} | **33 步/帧 = {out['p50_ms']*33:8.1f} ms "
        f"({1000.0/max(out['p50_ms']*33, 1e-9):5.1f} FPS 上限)** | 构造={out['build_s']:.1f}s "
        f"状态={out['state_MB']:.1f}MB"
    )
    del brain
    return out


def build_large_csr(n: int, in_degree: int, seed: int, inh_frac: float = 0.10):
    """快速构造固定入度的合成 CSR(比 sparse.random 省内存/时间)。

    用 int32 存列号与生成临时量,避免 1.5 亿非零时把内存打爆
    (全规模 166.7k/入度 911 的两张矩阵合计约 1.2 GB 数据)。
    """
    rng = np.random.default_rng(seed)
    n_inh = int(n * inh_frac)
    inh_ids = np.arange(n - n_inh, n, dtype=np.int32)
    exc_pool = np.arange(0, n - n_inh, dtype=np.int32)

    def block(k: int, pool: np.ndarray) -> sp.csr_matrix:
        kk = max(int(k), 1)
        indptr = (np.arange(n + 1, dtype=np.int64) * kk).astype(np.int32)
        sel = rng.integers(0, pool.size, size=n * kk, dtype=np.int32)
        cols = pool[sel]
        del sel
        data = rng.integers(1, 6, size=n * kk, dtype=np.int32).astype(np.float32)
        return sp.csr_matrix((data, cols, indptr), shape=(n, n))

    W_exc = block(round(in_degree * (1.0 - inh_frac)), exc_pool)
    W_inh = block(round(in_degree * inh_frac), inh_ids)
    return W_exc, W_inh


def test_stress(steps: int = 40, full_scale: bool = True) -> list[dict]:
    """真实规模压力测试 + 子网规模扫描(决策用)。

    full_scale=True 时构造 N=166,700 / 密度 0.546%(入度 911,1.519 亿条边)的合成
    连接组,直接给出"全量 vs 各规模子网"的单步耗时曲线 —— 这是 Lead 决定最终用
    子网还是全量的依据。用 `--quick` 跳过(只跑 N=50,000)。
    """
    section("3. 真实规模压力测试(密度 0.546% = A 线实测 151,856,684 条边)")
    results: list[dict] = []
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        if full_scale:
            n, k = 166700, 911  # 911/166700 = 0.546% -> 1.519e8 条边
            print(f"构造 N={n}, 密度 {k/n*100:.3f}%(入度 {k}),预计边数 {n*k:,} ...")
            t0 = time.perf_counter()
            W_e, W_i = build_large_csr(n, k, seed=13)
            art = ConnectomeArtifacts(
                W_exc=W_e, W_inh=W_i, neuron_ids=np.arange(n, dtype=np.int64) + 900000
            )
            p = td / "stress166k.npz"
            art.save(p)
            print(f"  nnz: exc={W_e.nnz:,} inh={W_i.nnz:,}(合计 {W_e.nnz+W_i.nnz:,}), "
                  f"文件={p.stat().st_size/1e6:.0f} MB, 构造+落盘={time.perf_counter()-t0:.1f}s")
            del W_e, W_i, art
            results.append(_bench(p, BrainConfig(subnet_size=None), 6098, 15, "全量 166.7k"))
            for sub in (8166, 12000, 20000, 30000, 50000):
                results.append(
                    _bench(p, BrainConfig(subnet_size=sub, subnet_seed=7), 6098, steps,
                           f"子网 {sub:,}")
                )
        else:
            n, k = 50000, 273
            print(f"构造 N={n}, 密度 {k/n*100:.3f}%(固定入度 {k})的合成 CSR ...")
            t0 = time.perf_counter()
            W_e, W_i = build_large_csr(n, k, seed=11)
            art = ConnectomeArtifacts(
                W_exc=W_e, W_inh=W_i, neuron_ids=np.arange(n, dtype=np.int64) + 500000
            )
            p = td / "stress50k.npz"
            art.save(p)
            print(f"  nnz: exc={W_e.nnz:,} inh={W_i.nnz:,}, "
                  f"文件={p.stat().st_size/1e6:.0f} MB, 构造+落盘={time.perf_counter()-t0:.1f}s")
            del W_e, W_i, art
            results.append(_bench(p, BrainConfig(subnet_size=None), 1024, steps, "全量 50k"))
            results.append(
                _bench(p, BrainConfig(subnet_size=20000, subnet_seed=5), 1024, steps, "子网 20k")
            )
    return results


def main() -> int:
    quick = "--quick" in sys.argv
    print("FlyAim B 线 合成连接组自检")
    print(f"numpy={np.__version__}, scipy={scipy.__version__}")
    t_start = time.perf_counter()
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        ctx = test_chain(tmp)
        test_trained_readout(tmp, ctx)
        stress = test_stress(full_scale=not quick)
    section("总结")
    print("全部断言通过(0 失败)")
    header = (
        f"  {'配置':<14} {'m':>8} {'p50 ms/步':>10} {'33步/帧 ms':>11} {'FPS 上限':>9} "
        f"{'矩阵 MB':>8} {'发放率':>7}"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))
    for r in stress:
        print(
            f"  {r['label']:<14} {r['m_sim']:>8,} {r['p50_ms']:>10.3f} {r['p50_ms']*33:>11.1f} "
            f"{1000.0/max(r['p50_ms']*33,1e-9):>9.1f} {r['matrix_MB']:>8.0f} "
            f"{r['fired_frac']*100:>6.2f}%"
        )
    print(f"总耗时 {time.perf_counter()-t_start:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
