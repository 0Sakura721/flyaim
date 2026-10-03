"""产物读写与 schema 定义。

契约见 CONTRACT.md 第 4 节。所有跨模块传递的落盘格式在此集中定义,
避免 A/B/C 三线各自发明格式。
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import scipy.sparse as sp

# ---------------------------------------------------------------- 文件名常量

ARTIFACT_WEIGHTS = "connectome.npz"
ARTIFACT_INDEX = "neuron_index.parquet"
ARTIFACT_ROLES = "roles.json"
ARTIFACT_MANIFEST = "manifest.json"


# ---------------------------------------------------------------- 校验和


def sha256_file(path: str | Path, chunk_mb: int = 8) -> str:
    """流式计算文件 sha256(用于 manifest 记录数据来源完整性)。"""
    h = hashlib.sha256()
    chunk = chunk_mb * 1024 * 1024
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


# ---------------------------------------------------------------- 连接组产物


@dataclass
class ConnectomeArtifacts:
    """连接组的落盘表示。

    W_exc / W_inh 均为 CSR,形状 (N, N),语义为 **突触计数缩放后的电导**
    (W[i, j] = j -> i 的投射强度)。
    突触计数本身是非负整数;兴奋/抑制的符号由递质预测决定,分行存储。
    """

    W_exc: sp.csr_matrix
    W_inh: sp.csr_matrix
    neuron_ids: np.ndarray  # (N,) int64, body IDs,行序即全局索引序
    soma_pos: np.ndarray | None = None  # (N, 3) float32,可缺省

    def __post_init__(self) -> None:
        n = self.neuron_ids.size
        if self.W_exc.shape != (n, n) or self.W_inh.shape != (n, n):
            raise ValueError(
                f"权重矩阵形状 {self.W_exc.shape}/{self.W_inh.shape} 与 N={n} 不符"
            )
        if self.W_exc.dtype != np.float32:
            self.W_exc = self.W_exc.astype(np.float32)
        if self.W_inh.dtype != np.float32:
            self.W_inh = self.W_inh.astype(np.float32)

    @property
    def n(self) -> int:
        return int(self.neuron_ids.size)

    @property
    def n_edges(self) -> int:
        return int(self.W_exc.nnz + self.W_inh.nnz)

    def stats(self) -> dict:
        return {
            "n_neurons": self.n,
            "n_edges_exc": int(self.W_exc.nnz),
            "n_edges_inh": int(self.W_inh.nnz),
            "n_edges": self.n_edges,
            "density": self.n_edges / max(self.n * self.n, 1),
            "has_soma_pos": self.soma_pos is not None,
        }

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = {
            "neuron_ids": self.neuron_ids.astype(np.int64),
            "W_exc_data": self.W_exc.data,
            "W_exc_indices": self.W_exc.indices,
            "W_exc_indptr": self.W_exc.indptr,
            "W_inh_data": self.W_inh.data,
            "W_inh_indices": self.W_inh.indices,
            "W_inh_indptr": self.W_inh.indptr,
            "n": np.asarray([self.n], dtype=np.int64),
        }
        if self.soma_pos is not None:
            payload["soma_pos"] = self.soma_pos.astype(np.float32)
        np.savez(p, **payload)

    @classmethod
    def load(cls, path: str | Path) -> "ConnectomeArtifacts":
        z = np.load(path, allow_pickle=False)
        n = int(z["n"][0])
        W_exc = sp.csr_matrix(
            (z["W_exc_data"], z["W_exc_indices"], z["W_exc_indptr"]), shape=(n, n)
        )
        W_inh = sp.csr_matrix(
            (z["W_inh_data"], z["W_inh_indices"], z["W_inh_indptr"]), shape=(n, n)
        )
        soma = z["soma_pos"] if "soma_pos" in z.files else None
        return cls(
            W_exc=W_exc.astype(np.float32),
            W_inh=W_inh.astype(np.float32),
            neuron_ids=z["neuron_ids"].astype(np.int64),
            soma_pos=None if soma is None else soma.astype(np.float32),
        )


# ---------------------------------------------------------------- Manifest


@dataclass
class Manifest:
    """数据来源与角色选择的完整记录。

    visual_input_fallback 是**必填的诊断字段**:若为 True,说明
    "果蝇的眼睛"不是真实生物接线,报告里必须显式声明。
    """

    dataset: str = "MaleCNS v1.0"
    license: str = "CC-BY-4.0"
    created_utc: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))

    n_neurons: int = 0
    n_edges: int = 0
    roles: dict = field(default_factory=dict)

    visual_input_fallback: bool = True
    visual_input_strategy: str = "none"
    notes: list[str] = field(default_factory=list)

    source_files: dict[str, dict] = field(default_factory=dict)  # name -> {url, bytes, sha256}
    extra: dict = field(default_factory=dict)

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(asdict(self), ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "Manifest":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in known})


# ---------------------------------------------------------------- run 产出


def save_json(path: str | Path, obj: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")


def load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _json_default(o: Any) -> Any:
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    if hasattr(o, "to_dict"):
        return o.to_dict()
    raise TypeError(f"无法序列化: {type(o)}")


def new_run_dir(base: str | Path, tag: str = "") -> Path:
    p = Path(base) / (time.strftime("%Y%m%d-%H%M%S") + (f"-{tag}" if tag else ""))
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_roles(path: str | Path, roles) -> None:
    """保存 RoleSelection(避免与 neuron_index 循环导入,故用鸭子类型)。"""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(roles.to_json(), encoding="utf-8")


def load_roles(path: str | Path):
    from flyaim.neuron_index import RoleSelection

    return RoleSelection.from_json(Path(path).read_text(encoding="utf-8"))
