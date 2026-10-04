"""冻结地基的回归测试:config / io / neuron_index / runner / stats / report。

运行方式::

    <捆绑 python> -m pytest flyaim/tests/test_core.py -q
    # 或(无 pytest 时)
    <捆绑 python> flyaim/tests/test_core.py

存在的理由:Phase 1 中 `NeuronIndex.save()` 因 pandas 3.0 的行为差异崩溃,
并且**第一版修复把连续性校验静默关掉了**。这两类问题都属于
「地基静默失效」,必须用测试钉死。

对应缺陷:
    - pandas 3.0: `to_parquet(index_label=...)` → TypeError + 0 字节残留文件
    - pandas 读 parquet 时具名索引还原到 `df.index` 而非列 → 校验被跳过
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from flyaim.io import ConnectomeArtifacts, Manifest, load_roles, save_roles  # noqa: E402
from flyaim.neuron_index import NeuronIndex, RoleSelection  # noqa: E402
from flyaim.stats import compare_all, paired_test  # noqa: E402

FAILS: list[str] = []


def check(cond: bool, msg: str) -> None:
    if cond:
        print(f"  PASS  {msg}")
    else:
        print(f"  FAIL  {msg}")
        FAILS.append(msg)


# ---------------------------------------------------------------- neuron_index


def test_index_roundtrip(tmp: Path) -> None:
    print("[test_index_roundtrip]")
    idx = NeuronIndex.from_arrays(
        body_id=np.arange(8) + 1000,
        type_=["R1-R6", "R1-R6", "R7y", "R8y", "DNa01", "DNa02", "MN1", "MN2"],
        superclass=["ol_sensory"] * 4 + ["descending_neuron"] * 2 + ["vnc_motor"] * 2,
    )
    p = tmp / "idx.parquet"
    idx.save(p)
    check(p.stat().st_size > 0, "save() 产出非空文件(pandas 3.0 回归)")
    back = NeuronIndex.load(p)
    check(np.array_equal(back.body_ids, idx.body_ids), "body_id round-trip")
    check(list(back.df["type"]) == list(idx.df["type"]), "type 列 round-trip")
    check(back.n == 8, "N 一致")

    roles = back.select_roles()
    check(roles.visual_input.size == 4, f"visual_input 选择正确 (={roles.visual_input.size})")
    check(roles.descending.size >= 2, f"descending 选择正确 (={roles.descending.size})")
    check(roles.motor.size == 2, f"motor 选择正确 (={roles.motor.size})")
    check(roles.visual_input_fallback is False,
          "能选到视觉输入即不标降级(min_visual_input 默认 0,避免小 fixture 假警报)")
    # 显式要求严格阈值时必须标记降级
    strict = back.select_roles(min_visual_input=100)
    check(strict.visual_input_fallback is True, "显式 min_visual_input 生效时正确标记降级")


def test_index_continuity_enforced(tmp: Path) -> None:
    """连续性校验必须在**两种 parquet 布局**下都生效。

    这是第一版修复的真实缺陷:具名索引被 pandas 还原到 df.index,
    `"index" in df.columns` 为假 → 校验被整段跳过。
    """
    print("[test_index_continuity_enforced]")
    idx = NeuronIndex.from_arrays(body_id=np.arange(5) + 100)

    # 布局一:索引写在 index 上(rename_axis + index=True)
    p1 = tmp / "bad_index.parquet"
    idx.df.iloc[[0, 1, 2, 4]].rename_axis("index").to_parquet(p1, index=True)
    check(_raises_valueerror(lambda: NeuronIndex.load(p1)),
          "非连续索引(index 布局)被拒绝")

    # 布局二:索引作为普通列
    p2 = tmp / "bad_column.parquet"
    d = idx.df.iloc[[0, 1, 2, 4]].copy()
    d.insert(0, "index", [0, 1, 2, 4])
    d.to_parquet(p2, index=False)
    check(_raises_valueerror(lambda: NeuronIndex.load(p2)),
          "非连续索引(列布局)被拒绝")

    # 起始非 0
    p3 = tmp / "bad_start.parquet"
    d2 = idx.df.iloc[[1, 2, 3]].copy()
    d2.insert(0, "index", [1, 2, 3])
    d2.to_parquet(p3, index=False)
    check(_raises_valueerror(lambda: NeuronIndex.load(p3)), "起始非 0 被拒绝")

    # 无索引语义的文件不应误报
    p4 = tmp / "plain.parquet"
    idx.df.to_parquet(p4, index=False)
    check(not _raises_valueerror(lambda: NeuronIndex.load(p4)),
          "无索引语义文件不误报")


def test_index_zero_byte(tmp: Path) -> None:
    print("[test_index_zero_byte]")
    p = tmp / "empty.parquet"
    p.write_bytes(b"")
    check(_raises_ioerror(lambda: NeuronIndex.load(p)), "0 字节文件报 IOError")
    # save() 应能覆盖残留的 0 字节文件
    idx = NeuronIndex.from_arrays(body_id=np.arange(3))
    idx.save(p)
    check(p.stat().st_size > 0, "save() 能覆盖残留 0 字节文件")


# ---------------------------------------------------------------- io


def test_artifacts_roundtrip(tmp: Path) -> None:
    print("[test_artifacts_roundtrip]")
    import scipy.sparse as sp

    n = 10
    rng = np.random.default_rng(0)
    W = sp.random(n, n, density=0.3, format="csr", dtype=np.float32, random_state=1)
    W_inh = sp.random(n, n, density=0.1, format="csr", dtype=np.float32, random_state=2)
    art = ConnectomeArtifacts(W_exc=W, W_inh=W_inh, neuron_ids=np.arange(n) + 7)
    p = tmp / "c.npz"
    art.save(p)
    back = ConnectomeArtifacts.load(p)
    check(back.n == n, "N round-trip")
    check(back.n_edges == W.nnz + W_inh.nnz, "边数 round-trip")
    check(np.allclose(back.W_exc.toarray(), W.toarray()), "W_exc 逐元素一致")
    check(np.allclose(back.W_inh.toarray(), W_inh.toarray()), "W_inh 逐元素一致")
    check(np.array_equal(back.neuron_ids, art.neuron_ids), "neuron_ids round-trip")
    check(back.soma_pos is None, "soma_pos 缺省为 None(合法)")

    # 形状不匹配必须被拒
    check(_raises(lambda: ConnectomeArtifacts(W_exc=W, W_inh=W_inh,
                                             neuron_ids=np.arange(n - 1))),
          "形状与 N 不符时抛异常")

    roles = RoleSelection(visual_input=np.array([1, 2]), descending=np.array([5]),
                          visual_input_strategy="unit_test", visual_input_fallback=False)
    rp = tmp / "roles.json"
    save_roles(rp, roles)
    r2 = load_roles(rp)
    check(np.array_equal(r2.visual_input, roles.visual_input), "roles round-trip")
    check(r2.visual_input_fallback is False, "roles 诊断字段 round-trip")

    m = Manifest(n_neurons=n, n_edges=art.n_edges, roles=roles.summary(),
                 visual_input_fallback=False, notes=["unit test"])
    mp = tmp / "manifest.json"
    m.save(mp)
    m2 = Manifest.load(mp)
    check(m2.n_neurons == n and m2.visual_input_fallback is False, "manifest round-trip")


# ---------------------------------------------------------------- stats


def test_stats() -> None:
    print("[test_stats]")
    good = [0.80, 0.82, 0.79, 0.85, 0.81, 0.83, 0.80, 0.84, 0.82, 0.81]
    bad = [0.50, 0.52, 0.49, 0.51, 0.50, 0.53, 0.48, 0.51, 0.52, 0.50]
    t = paired_test(np.array(good), np.array(bad), "fly", "shuffle", "hit_rate")
    check(t.significant and t.mean_diff > 0, "真实差异被判为显著")
    check(t.ci_low < t.mean_diff < t.ci_high, "bootstrap CI 包含点估计")
    check(t.n_pairs == 10, "配对样本数正确")

    # 零假设:两臂本质相同 → 必须**不显著**(否则会谎报有贡献)
    rng = np.random.default_rng(7)
    a = rng.normal(0.5, 0.02, 40)
    b = a + rng.normal(0, 0.001, 40)
    t0 = paired_test(a, b, "fly", "shuffle", "hit_rate")
    check(not t0.significant, f"零假设未被误判为显著 (p={t0.p_value:.3f})")

    # 判定文案:用**语义字段**断言,不匹配中文子串(否则文案微调就会误报失败)
    by_arm = {
        "fly": [{"summary": {"hit_rate": v}} for v in good],
        "shuffle": [{"summary": {"hit_rate": v}} for v in bad],
    }
    cmp = compare_all(by_arm)
    check(cmp["primary"] is not None, "compare_all 产出主判定")
    check(cmp["primary"]["significant"] is True, "fly 优于零模型时判为显著")
    check(cmp["primary"]["mean_diff"] > 0, "差值方向为正")
    check(isinstance(cmp["primary"]["verdict"], str) and len(cmp["primary"]["verdict"]) > 0,
          "主判定含非空结论文本")

    # 反向:fly 更差时必须如实反映方向,不能套用"有贡献"
    cmp2 = compare_all({
        "fly": [{"summary": {"hit_rate": v}} for v in bad],
        "shuffle": [{"summary": {"hit_rate": v}} for v in good],
    })
    check(cmp2["primary"]["mean_diff"] < 0, "fly 更差时差值方向为负")

    # 关键:无显著差异时必须判为"无因果贡献"(CONTRACT 预注册规则)
    rng2 = np.random.default_rng(11)
    base = rng2.normal(0.5, 0.02, 20)
    tie = base + rng2.normal(0, 0.0005, 20)
    cmp3 = compare_all({
        "fly": [{"summary": {"hit_rate": float(v)}} for v in base],
        "shuffle": [{"summary": {"hit_rate": float(v)}} for v in tie],
    })
    check(cmp3["primary"]["significant"] is False,
          "无差异时判为不显著(预注册规则:接线图无因果贡献)")
    check("无因果贡献" in cmp3["primary"]["verdict"],
          "不显著时的结论文案为「无因果贡献」")


def test_paired_length_mismatch() -> None:
    print("[test_paired_length_mismatch]")
    check(_raises(lambda: paired_test(np.zeros(3), np.zeros(4), "a", "b", "m")),
          "配对长度不一致时报错")


# ---------------------------------------------------------------- helpers


def _raises(fn) -> bool:
    try:
        fn()
        return False
    except Exception:
        return True


def _raises_valueerror(fn) -> bool:
    try:
        fn()
        return False
    except ValueError:
        return True


def _raises_ioerror(fn) -> bool:
    try:
        fn()
        return False
    except OSError:
        return True


# ---------------------------------------------------------------- retina 空间读出


def test_receptor_maps_roundtrip() -> None:
    """`Retina.receptor_maps()` 必须**保有视网膜的光栅空间结构**(D25 就靠它成立)。

    为什么必须测:eye-servo 的整条链路建立在"感光细胞驱动可以铺回 (24,32) 小眼空间图"
    这一条上。若块映射悄悄错位,空间图不会报错,只会让瞄准读出一个偏移的靶位
    —— 与 D7 的静默失效同一类风险。

    三条判据:
      (1) `m[cell[k]] == drive[rows[k]]` 对每一组、每个通道逐位成立
          —— 这一条**本身**就蕴含"同一 cell 的多个感光细胞驱动值相同"
          (否则后写覆盖会打破等式),也就是逆映射良定义;
      (2) 帧上一个**落在单格内部**的亮点,必须在图上落到**对的格子**(空间语义正确,
          而不只是"自洽");
      (3) 加权质心等于解析质心。
"""
    from flyaim.config import RetinaConfig
    from flyaim.retina.encoder import Retina

    n_lum, n_r7, n_r8 = 3377, 1385, 1329          # MaleCNS 实测规模
    n = n_lum + n_r7 + n_r8
    types = np.array(["R1-R6"] * n_lum + ["R7y"] * n_r7 + ["R8p"] * n_r8, dtype=object)
    ret = Retina(RetinaConfig(), input_neuron_ids=np.arange(n, dtype=np.int64),
                 types=types)
    check(ret._rows_lum.size == n_lum, f"R1-R6 分组数 = {ret._rows_lum.size}")
    check(ret._rows_r7.size == n_r7, f"R7 分组数 = {ret._rows_r7.size}")
    check(ret._rows_r8.size == n_r8, f"R8 分组数 = {ret._rows_r8.size}")
    check(not ret.input_neuron_ids_fallback, "合成输入不应触发 fallback")

    # ---- (1)(2) 1080x640? 不需要:直接喂编码器真实输出 ----
    frame = np.full((480, 640, 3), 18, dtype=np.uint8)
    frame[170, 310] = (235, 70, 70)               # 落在 cell(row=8, col=15) 内部
    rr, cc = 8, 15
    yy = np.arange(rr * 20, rr * 20 + 20)
    xx = np.arange(cc * 20, cc * 20 + 20)
    frame[np.ix_(yy, xx)] = (235, 70, 70)
    drive = ret.frame_to_spikes(frame)

    for ch in (0, 1):
        maps = ret.receptor_maps(drive, channel=ch)
        for key, rows, cells in (("lum", ret._rows_lum, ret._cell_lum),
                                 ("r7", ret._rows_r7, ret._cell_r7),
                                 ("r8", ret._rows_r8, ret._cell_r8)):
            m = maps[key].reshape(-1)
            # (1) 逐位逆映射(等式成立 ⇒ 同 cell 内驱动必然一致 ⇒ 逆映射良定义)
            check(np.array_equal(m[cells], drive[rows, ch]),
                  f"receptor_maps({key}, ch={ch}) 不是无损逆映射")
        check(maps["lum"].shape == (ret.eye_rows, ret.eye_cols),
              f"空间图形状 = {maps['lum'].shape}")

    # ---- (3) 空间语义:亮点必须落到对的格子 ----
    lum = ret.receptor_maps(drive, channel=0)["lum"]
    got = np.unravel_index(int(np.argmax(lum)), lum.shape)
    check((int(got[0]), int(got[1])) == (rr, cc),
          f"亮点落到格 {got},期望 {(rr, cc)} —— 空间结构没保住")
    chroma = np.clip(ret.receptor_maps(drive, 0)["r7"]
                     - ret.receptor_maps(drive, 0)["r8"], 0.0, None)
    gotc = np.unravel_index(int(np.argmax(chroma)), chroma.shape)
    check((int(gotc[0]), int(gotc[1])) == (rr, cc),
          f"色觉图上亮点落到格 {gotc},期望 {(rr, cc)}")

    # ---- 质心必须是权重意义上的精确质心 ----
    rng = np.random.default_rng(0)
    w = rng.random((ret.eye_rows, ret.eye_cols), dtype=np.float32)
    c = ret.centroid(w)
    check(c is not None, "全正权重的质心不应为 None")
    if c is not None:
        exp_c = float((w * np.arange(ret.eye_cols)[None, :]).sum() / w.sum())
        exp_r = float((w * np.arange(ret.eye_rows)[:, None]).sum() / w.sum())
        check(abs(c[0] - exp_c) < 1e-4 and abs(c[1] - exp_r) < 1e-4,
              f"centroid 偏差过大: {c} vs ({exp_c:.4f}, {exp_r:.4f})")


# ---------------------------------------------------------------- runner


def main() -> int:
    print("=" * 72)
    print("FlyAim 冻结地基回归测试")
    print("=" * 72)
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        test_index_roundtrip(tmp)
        test_index_continuity_enforced(tmp)
        test_index_zero_byte(tmp)
        test_artifacts_roundtrip(tmp)
        test_stats()
        test_paired_length_mismatch()
    test_receptor_maps_roundtrip()
    print("=" * 72)
    if FAILS:
        print(f"❌ {len(FAILS)} 项失败:")
        for f in FAILS:
            print(f"   - {f}")
        return 1
    print("✅ 全部通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


# ---------------------------------------------------------------- pytest 兼容


def test_all_pytest(tmp_path) -> None:
    """pytest 入口:复用同一组检查。"""
    FAILS.clear()
    test_index_roundtrip(tmp_path)
    test_index_continuity_enforced(tmp_path)
    test_index_zero_byte(tmp_path)
    test_artifacts_roundtrip(tmp_path)
    test_stats()
    test_paired_length_mismatch()
    test_receptor_maps_roundtrip()
    assert not FAILS, FAILS
