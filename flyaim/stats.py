"""统计检验:配对比较 + 效应量 + 置信区间。

CONTRACT.md 第 3 节要求:
    - 采样数 >= 10 个种子
    - 报告必须附效应量与置信区间,不能只报均值
    - 判定规则**预先注册**,不得事后修改

本模块只依赖 numpy,不引入 scipy.stats 之外的重型依赖(scipy 已可用)。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

BOOTSTRAP_N = 10000
BOOTSTRAP_SEED = 20260101  # 固定,保证报告可复现


@dataclass
class PairedTest:
    """配对检验结果。

    fly 与对照臂在同一批 seed 上跑 → 天然配对,用配对检验消除靶场难度差异。
    """

    arm_a: str
    arm_b: str
    metric: str
    n_pairs: int
    mean_a: float
    mean_b: float
    mean_diff: float          # a - b,配对均值
    ci_low: float
    ci_high: float
    cohens_dz: float
    t_stat: float
    p_value: float
    significant: bool         # p < alpha
    alpha: float

    def to_dict(self) -> dict:
        return {
            "arm_a": self.arm_a,
            "arm_b": self.arm_b,
            "metric": self.metric,
            "n_pairs": self.n_pairs,
            "mean_a": self.mean_a,
            "mean_b": self.mean_b,
            "mean_diff": self.mean_diff,
            "ci95": [self.ci_low, self.ci_high],
            "cohens_dz": self.cohens_dz,
            "t_stat": self.t_stat,
            "p_value": self.p_value,
            "significant": self.significant,
            "alpha": self.alpha,
            "verdict": self.verdict(),
        }

    def verdict(self) -> str:
        if not self.significant:
            return (
                f"{self.arm_a} 与 {self.arm_b} 无显著差异 "
                f"(p={self.p_value:.4f}, 配对差值均值={self.mean_diff:.4f}, "
                f"95%CI=[{self.ci_low:.4f},{self.ci_high:.4f}])"
            )
        direction = "优于" if self.mean_diff > 0 else "劣于"
        return (
            f"{self.arm_a} {direction} {self.arm_b} "
            f"(p={self.p_value:.4f}, 差值={self.mean_diff:.4f}, "
            f"95%CI=[{self.ci_low:.4f},{self.ci_high:.4f}], dz={self.cohens_dz:.3f})"
        )


def paired_test(
    a: np.ndarray,
    b: np.ndarray,
    arm_a: str,
    arm_b: str,
    metric: str,
    alpha: float = 0.05,
) -> PairedTest:
    """配对 t 检验 + bootstrap 置信区间 + Cohen's dz。

    要求 a、b 等长且按同一 seed 顺序配对。
    """
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    if a.size != b.size:
        raise ValueError(f"配对样本长度不一致: {a.size} vs {b.size}")
    if a.size < 2:
        raise ValueError("配对样本至少需要 2 个")

    d = a - b
    n = d.size
    mean_diff = float(d.mean())

    # 配对 t 检验(手算,避免对不同 scipy 版本的依赖)
    sd = float(d.std(ddof=1))
    se = sd / np.sqrt(n) if sd > 0 else 0.0
    t_stat = float(mean_diff / se) if se > 0 else 0.0
    p_value = float(_t_sf_two_sided(abs(t_stat), n - 1)) if se > 0 else 1.0

    # bootstrap 置信区间(百分位法)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    idx = rng.integers(0, n, size=(BOOTSTRAP_N, n))
    boot = d[idx].mean(axis=1)
    ci_low, ci_high = (float(x) for x in np.percentile(boot, [2.5, 97.5]))

    dz = float(mean_diff / sd) if sd > 0 else 0.0

    return PairedTest(
        arm_a=arm_a,
        arm_b=arm_b,
        metric=metric,
        n_pairs=n,
        mean_a=float(a.mean()),
        mean_b=float(b.mean()),
        mean_diff=mean_diff,
        ci_low=ci_low,
        ci_high=ci_high,
        cohens_dz=dz,
        t_stat=t_stat,
        p_value=p_value,
        significant=bool(p_value < alpha),
        alpha=alpha,
    )


def _t_sf_two_sided(t: float, df: int) -> float:
    """Student-t 双尾生存函数。优先用 scipy,缺失则回退到正态近似。"""
    try:
        from scipy import stats  # type: ignore

        return float(2.0 * stats.t.sf(t, df))
    except Exception:
        # 正态近似(大样本下足够;小样本会偏乐观,报告中已标注)
        from math import erfc, sqrt

        return float(erfc(t / sqrt(2.0)))


def summarize_arm(results: list[dict], metric: str = "hit_rate") -> dict:
    """把一个臂的多次 episode 汇总成 mean/std/median/min/max。"""
    vals = np.asarray([r["summary"].get(metric, np.nan) for r in results], dtype=np.float64)
    vals = vals[~np.isnan(vals)]
    if vals.size == 0:
        return {"metric": metric, "n": 0}
    return {
        "metric": metric,
        "n": int(vals.size),
        "mean": float(vals.mean()),
        "std": float(vals.std(ddof=1)) if vals.size > 1 else 0.0,
        "median": float(np.median(vals)),
        "min": float(vals.min()),
        "max": float(vals.max()),
        "values": vals.tolist(),
    }


def compare_all(
    by_arm: dict[str, list[dict]],
    reference: str = "shuffle",
    test_arm: str = "fly",
    metric: str = "hit_rate",
    alpha: float = 0.05,
) -> dict:
    """按 CONTRACT 第 3 节的预注册判定规则做全部比较。

    主判定: fly vs shuffle(零模型)
    参照:   fly vs pid / random(仅刻度,不参与"果蝇是否有效"的判定)
    """
    out: dict = {"metric": metric, "alpha": alpha, "tests": {}, "primary": None}

    def _vals(arm: str) -> np.ndarray | None:
        if arm not in by_arm or not by_arm[arm]:
            return None
        return np.asarray(
            [r["summary"].get(metric, np.nan) for r in by_arm[arm]], dtype=np.float64
        )

    arms = list(by_arm.keys())
    for other in arms:
        if other == test_arm:
            continue
        va, vb = _vals(test_arm), _vals(other)
        if va is None or vb is None or va.size != vb.size or va.size < 2:
            continue
        t = paired_test(va, vb, test_arm, other, metric, alpha=alpha)
        out["tests"][f"{test_arm}_vs_{other}"] = t.to_dict()

    # 预注册的主判定
    key = f"{test_arm}_vs_{reference}"
    if key in out["tests"]:
        t = out["tests"][key]
        if not t["significant"]:
            verdict = (
                f"判定:【接线图无因果贡献】{test_arm} 相对零模型 {reference} 无显著优势 "
                f"(p={t['p_value']:.4f})。"
            )
        elif t["mean_diff"] > 0:
            verdict = (
                f"判定:【接线图有可测贡献】{test_arm} 显著优于零模型 {reference} "
                f"(p={t['p_value']:.4f}, dz={t['cohens_dz']:.3f})。"
            )
        else:
            verdict = (
                f"判定:【接线图表现反而更差】{test_arm} 显著劣于零模型 {reference} "
                f"(p={t['p_value']:.4f})。"
            )
        # 注意:`{**t}` 展开会把 PairedTest.verdict(通用描述)盖到预注册结论上,
        # 那是本实验最关键的输出。因此显式把预注册文案放在最后,确保它生效;
        # 通用描述另存为 `generic_verdict`,便于报告同时展示两者。
        primary = dict(t)
        primary["generic_verdict"] = t.get("verdict")
        primary["verdict"] = verdict
        primary["comparison"] = key
        primary["preregistered"] = True
        out["primary"] = primary

    return out
