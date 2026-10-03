"""全局神经元索引空间与细胞角色选择。

索引空间是三方共享的地基(见 CONTRACT.md 第 1 节):
    - 全局索引 i ∈ [0, N),稳定不变
    - 索引 <-> body ID 双向映射
    - 细胞类型标注(class / type / side / neurotransmitter)

本模块只做**索引与元数据**,不做任何仿真,也不下载数据。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Literal, Sequence

import numpy as np
import pandas as pd

# 角色名 -> 选择策略说明
ROLE_VISUAL_INPUT = "visual_input"
ROLE_DESCENDING = "descending"
ROLE_MOTOR = "motor"
ROLE_INHIBITORY = "inhibitory"

# 抑制性神经递质判定集合
INHIBITORY_NTS = frozenset({"gaba", "glycine", "acetylcholine_gaba"})


@dataclass
class RoleSelection:
    """各功能角色的神经元索引集合。

    全部使用**全局索引空间**。空数组表示该角色在数据集中不存在,
    调用方必须显式处理(而不是静默降级)。
    """

    visual_input: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int64))
    descending: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int64))
    motor: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int64))
    inhibitory: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int64))

    # 诊断信息:视觉输入是否发生了降级
    visual_input_strategy: str = "none"
    visual_input_fallback: bool = False
    notes: list[str] = field(default_factory=list)

    def get(self, role: str) -> np.ndarray:
        if role not in ("visual_input", "descending", "motor", "inhibitory"):
            raise KeyError(f"未知角色: {role}")
        return getattr(self, role)

    def to_json(self) -> str:
        return json.dumps(
            {
                "visual_input": self.visual_input.tolist(),
                "descending": self.descending.tolist(),
                "motor": self.motor.tolist(),
                "inhibitory": self.inhibitory.tolist(),
                "visual_input_strategy": self.visual_input_strategy,
                "visual_input_fallback": self.visual_input_fallback,
                "notes": self.notes,
            },
            ensure_ascii=False,
            indent=2,
        )

    @classmethod
    def from_json(cls, text: str) -> "RoleSelection":
        d = json.loads(text)
        return cls(
            visual_input=np.asarray(d["visual_input"], dtype=np.int64),
            descending=np.asarray(d["descending"], dtype=np.int64),
            motor=np.asarray(d["motor"], dtype=np.int64),
            inhibitory=np.asarray(d["inhibitory"], dtype=np.int64),
            visual_input_strategy=d.get("visual_input_strategy", "none"),
            visual_input_fallback=bool(d.get("visual_input_fallback", False)),
            notes=list(d.get("notes", [])),
        )

    def summary(self) -> dict:
        return {
            "n_visual_input": int(self.visual_input.size),
            "n_descending": int(self.descending.size),
            "n_motor": int(self.motor.size),
            "n_inhibitory": int(self.inhibitory.size),
            "visual_input_strategy": self.visual_input_strategy,
            "visual_input_fallback": self.visual_input_fallback,
        }


def _is_trivial_rangeindex(index) -> bool:
    """判断索引是否为「无信息的默认 RangeIndex」。

    无信息的默认索引不代表任何语义,不应参与连续性校验
    (否则每个用默认索引落盘的文件都会被判为合法或非法,没有意义)。
    """
    if not isinstance(index, pd.RangeIndex):
        return False
    return index.start == 0 and index.step == 1 and index.name is None


def _check_continuity(index, source=None) -> None:
    """强制校验索引为 [0, N)。

    契约 CONTRACT.md 第 1 节要求全局索引空间严格为 `[0, N)`,
    任何缺口都会让「索引 i ↔ 神经元」的对应关系错位,进而毁掉整个仿真。
    """
    n = len(index)
    arr = index.to_numpy()
    if not np.array_equal(arr, np.arange(n)):
        where = f" (来源: {source})" if source is not None else ""
        head = arr[:6].tolist() if arr.size else []
        raise ValueError(
            f"neuron_index 索引不连续,违反全局索引空间契约{where}: "
            f"n={n}, 期望 [0,{n}), 实际前 6 个={head}, "
            f"min={arr.min() if arr.size else 'NA'}, max={arr.max() if arr.size else 'NA'}"
        )


class NeuronIndex:
    """全局索引空间 + 细胞元数据查询。

    典型用法::

        idx = NeuronIndex.load("flyaim/data/build/neuron_index.parquet")
        roles = idx.select_roles(visual_input_types=[...])
    """

    def __init__(self, df: pd.DataFrame):
        # df 行序即全局索引序
        required = {"body_id"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"neuron_index 缺少必需列: {sorted(missing)}")
        self.df = df.reset_index(drop=True)
        self._n = len(self.df)

        # 标准化文本列,缺失填 "unknown"
        # 注意:MaleCNS 的功能分类主字段是 `superclass`(见 select_roles docstring)
        for col in ("class", "superclass", "type", "side", "nt"):
            if col not in self.df.columns:
                self.df[col] = "unknown"
            self.df[col] = (
                self.df[col].astype("string").fillna("unknown").astype(str).str.strip()
            )

    # ------------------------------------------------------------ 基本属性

    @property
    def n(self) -> int:
        """神经元总数 N。"""
        return self._n

    @property
    def body_ids(self) -> np.ndarray:
        return self.df["body_id"].to_numpy(dtype=np.int64)

    @property
    def names(self) -> np.ndarray:
        col = "name" if "name" in self.df.columns else "type"
        return self.df[col].astype(str).to_numpy()

    def __len__(self) -> int:
        return self._n

    # ------------------------------------------------------------ 查询

    def indices_where(self, col: str, values: Iterable[str]) -> np.ndarray:
        """按某列取值筛选全局索引;大小写不敏感。"""
        if col not in self.df.columns:
            return np.empty(0, dtype=np.int64)
        want = {str(v).strip().lower() for v in values}
        got = self.df[col].astype(str).str.strip().str.lower()
        return np.flatnonzero(got.isin(want).to_numpy())

    def indices_type_prefix(self, prefixes: Sequence[str]) -> np.ndarray:
        """按 type 名称前缀筛选(如 'DN' 开头),大小写不敏感。"""
        if "type" not in self.df.columns:
            return np.empty(0, dtype=np.int64)
        t = self.df["type"].astype(str).str.strip().str.upper()
        mask = np.zeros(self._n, dtype=bool)
        for p in prefixes:
            mask |= t.str.startswith(str(p).strip().upper()).to_numpy()
        return np.flatnonzero(mask)

    def unique_values(self, col: str, top: int | None = None) -> list[tuple[str, int]]:
        """列出某列取值及计数,便于探查数据集标注体系。"""
        if col not in self.df.columns:
            return []
        vc = self.df[col].astype(str).value_counts()
        if top is not None:
            vc = vc.head(top)
        return [(str(k), int(v)) for k, v in vc.items()]

    # ------------------------------------------------------------ 角色选择

    def select_roles(
        self,
        visual_input_types: Sequence[str] | None = None,
        visual_input_superclasses: Sequence[str] = ("ol_sensory",),
        descending_superclasses: Sequence[str] = ("descending_neuron",),
        descending_prefixes: Sequence[str] = ("DN",),
        motor_superclasses: Sequence[str] = ("vnc_motor",),
        inhibitory_nts: Iterable[str] = INHIBITORY_NTS,
        min_visual_input: int = 0,
    ) -> RoleSelection:
        """按策略选出各角色索引。

        **重要(MaleCNS v1.0 实测得出):**
        该数据集的 `class` 列有约 87% 为 NaN(140,187/166,700),**不适合做主选择器**。
        真正的功能分类字段是 **`superclass`**(28 个取值,仅神经元行非空)。
        因此本方法以 superclass 为主、type 前缀为辅。

        实测(166,700 神经元):
            visual_input      : ol_sensory 6,091 个 -> R1-R6 3377 / R7* 1385 / R8* 1329
            descending        : descending_neuron 1,314 ∪ DN* 1,342 = 1,360
            motor             : vnc_motor 708
            (对照)ol_intrinsic 89,403 含 T4 6865 / T5 6720 运动检测神经元
                  visual_projection 9,201

        视觉输入策略优先级(逐级降级,每次降级都记录到 notes):
            1. 显式给定的 type 列表
            2. 显式给定的 superclass 列表  ← MaleCNS 实际走这条
            3. class 含 'visual' / 'optic' 的细胞
            4. type 前缀含 'R'(感光细胞命名法)且 superclass 为 ol_sensory
        """
        notes: list[str] = []
        strategy = "none"
        fallback = False

        vis = np.empty(0, dtype=np.int64)

        if visual_input_types:
            vis = self.indices_where("type", visual_input_types)
            if vis.size:
                strategy = f"explicit_type:{','.join(map(str, visual_input_types))}"

        if vis.size == 0 and visual_input_superclasses:
            vis = self.indices_where("superclass", visual_input_superclasses)
            if vis.size:
                strategy = f"superclass:{','.join(map(str, visual_input_superclasses))}"

        if vis.size == 0:
            vis = self.indices_where("class", ["visual", "optic", "visual_projection"])
            if vis.size:
                strategy = "class_contains_visual_optic"

        if vis.size == 0:
            # 最后手段:所有 ol_sensory 里 type 以 R 开头的
            r_all = self.indices_type_prefix(["R"])
            if r_all.size and "superclass" in self.df.columns:
                is_ol = (
                    self.df["superclass"].astype(str).str.strip().str.lower().eq("ol_sensory").to_numpy()
                )
                vis = r_all[is_ol[r_all]]
            if vis.size:
                strategy = "type_prefix_R_within_ol_sensory"

        # 阈值检查是**可选**的:默认 0 表示"只要能选到就不算降级"。
        # 为什么不默认 100:显式传入 selection 调用方(以及单元测试里的小规模 fixture)
        # 本来就只有少量细胞,若因数量少就标记降级会产生假警报,
        # 而 visual_input_fallback 这个字段是给报告用的关键诊断,不能被噪声污染。
        # 需要严格判定的调用方请显式传 min_visual_input。
        if vis.size and vis.size < min_visual_input:
            notes.append(
                f"视觉输入群仅 {vis.size} 个 (< min_visual_input={min_visual_input}),"
                f"策略={strategy};已标记为降级"
            )
            fallback = True
        if vis.size == 0:
            notes.append("未找到任何视觉输入候选")
            strategy = "none"
            fallback = True
        # 下行神经元:superclass 与 type 前缀取并集
        dn_sup = self.indices_where("superclass", descending_superclasses)
        dn_cls = self.indices_where("class", ["descending"])
        dn_prefix = self.indices_type_prefix(descending_prefixes)
        dn = np.unique(np.concatenate([dn_sup, dn_cls, dn_prefix]))
        if dn.size == 0:
            notes.append("未找到下行神经元——控制读出将无合法来源")
        else:
            notes.append(
                f"下行神经元: superclass={dn_sup.size}, class='descending'={dn_cls.size}, "
                f"type前缀DN={dn_prefix.size}, 取并集={dn.size}"
            )

        mn = self.indices_where("superclass", motor_superclasses)
        if mn.size == 0:
            mn = self.indices_where("class", ["motor"])
        if mn.size == 0:
            notes.append("未找到运动神经元")

        # 抑制性:按神经递质预测
        inh = np.empty(0, dtype=np.int64)
        if "nt" in self.df.columns:
            inh = self.indices_where("nt", inhibitory_nts)
        if inh.size == 0:
            notes.append("未找到抑制性递质预测;将回退到全兴奋网络(会削弱侧抑制动力学)")

        return RoleSelection(
            visual_input=np.asarray(vis, dtype=np.int64),
            descending=np.asarray(dn, dtype=np.int64),
            motor=np.asarray(mn, dtype=np.int64),
            inhibitory=np.asarray(inh, dtype=np.int64),
            visual_input_strategy=strategy,
            visual_input_fallback=fallback,
            notes=notes,
        )

    # ------------------------------------------------------------ 持久化

    def save(self, path: str | Path) -> None:
        """落盘为 parquet,行序即全局索引序。

        实现说明(踩过的坑):
            pandas 3.0 的 `DataFrame.to_parquet` **没有** `index_label` 形参,
            该 kwarg 会被透传给 `pyarrow.parquet.write_table`,抛
            `TypeError: __cinit__() got an unexpected keyword argument 'index_label'`,
            并且**会在磁盘上留下 0 字节文件**。

            正确做法是给索引命名(`rename_axis`)后 `index=True`,
            这样索引名会被写入 parquet schema,`load()` 才能读到并做连续性校验。
        """
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        # 先清理残留的 0 字节文件,避免上次失败留下的坏产物被误认为成功
        if p.exists() and p.stat().st_size == 0:
            p.unlink()
        self.df.rename_axis("index").to_parquet(p, index=True)
        if not p.exists() or p.stat().st_size == 0:
            raise IOError(f"parquet 写入失败(产物为空): {p}")

    @classmethod
    def load(cls, path: str | Path) -> "NeuronIndex":
        """从 parquet 读回,并**强制校验全局索引连续性**。

        实现说明(踩过的第二个坑):
            pandas 读 parquet 时,**具名索引会被还原成 DataFrame 的 index,而不是列**。
            因此只检查 `"index" in df.columns` 会让校验被整段跳过 ——
            非连续的索引会静默通过,破坏 `[0, N)` 契约。

            正确做法见下方 `_check_continuity`,它对「索引已在 index 上」
            与「索引作为列存在」两种情形都做校验。
        """
        p = Path(path)
        if p.exists() and p.stat().st_size == 0:
            raise IOError(
                f"{p} 是 0 字节文件。通常由 pandas 3.0 下误用 "
                f"`to_parquet(index_label=...)` 导致,请重新生成。"
            )
        df = pd.read_parquet(p)

        if "index" in df.columns:
            # 索引被当成普通列写入的情形
            df = df.set_index("index").sort_index()
            idx_is_meaningful = True
        else:
            # 索引已在 DataFrame 的 index 上(pandas 的常规还原路径)
            idx_is_meaningful = df.index.name is not None or not _is_trivial_rangeindex(df.index)

        _check_continuity(df.index, p if idx_is_meaningful else None)
        return cls(df.reset_index(drop=True))

    @classmethod
    def from_arrays(
        cls,
        body_id: np.ndarray,
        name: np.ndarray | None = None,
        cls_: np.ndarray | None = None,
        type_: np.ndarray | None = None,
        side: np.ndarray | None = None,
        nt: np.ndarray | None = None,
        superclass: np.ndarray | None = None,
        extra: dict[str, np.ndarray] | None = None,
    ) -> "NeuronIndex":
        """由原始列数组构建;行序即全局索引序。"""
        data: dict[str, object] = {"body_id": np.asarray(body_id, dtype=np.int64)}
        if name is not None:
            data["name"] = np.asarray(name)
        if cls_ is not None:
            data["class"] = np.asarray(cls_)
        if superclass is not None:
            data["superclass"] = np.asarray(superclass)
        if type_ is not None:
            data["type"] = np.asarray(type_)
        if side is not None:
            data["side"] = np.asarray(side)
        if nt is not None:
            data["nt"] = np.asarray(nt)
        for k, v in (extra or {}).items():
            data[k] = np.asarray(v)
        return cls(pd.DataFrame(data))
