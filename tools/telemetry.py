"""训练/评估过程的实时遥测落盘(供 `tools/live_view.py` 可视化)。

设计:训练进程只写文件(开销极低),可视化在**独立进程**里读文件刷新。
这样即使可视化关掉/卡住,训练也不受影响。

每个 run 的目录结构::

    <out_dir>/telemetry.jsonl     逐条 JSON(首条 kind="header",其余 kind="frame"/"diag")
    <out_dir>/preview/*.png       周期性靶场预览图(可选)

`act` 数组是拼接的:
    act = [ full_rate_sample(按 sample_idx 下采样) , dn_rates ]
其中 `sample_idx` 是全局神经元索引,排序方式见 `header.sample_order`。
"""

from __future__ import annotations

import base64
import json
import shutil
import time
from pathlib import Path

import numpy as np


def _pack(arr: np.ndarray) -> str:
    """把浮点数组压成 base64(float16)。比 JSON 数字数组小 ~6 倍。"""
    return base64.b64encode(np.asarray(arr, np.float16).tobytes()).decode("ascii")


def _unpack(s: str) -> np.ndarray:
    return np.frombuffer(base64.b64decode(s), np.float16).astype(np.float32)

_LAYER_PRIORITY = ("ol_sensory", "ol_intrinsic", "visual_projection", "visual_centrifugal",
                   "descending_neuron", "vnc_motor")


class TelemetryWriter:
    """把每帧的脑状态 + 靶场状态写成 JSONL,可选保存预览帧。

    开销:JSON 序列化 ~1 ms/帧,预览编码 ~3 ms/帧(每 preview_every 帧才发生一次)。
    """

    def __init__(self, out_dir: str | Path, *, sample_shape=(96, 96),
                 sample_idx=None, dn_idx=None, layer_labels=None,
                 preview_every: int = 5, run_tag: str = ""):
        self.out = Path(out_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        self.jsonl = self.out / "telemetry.jsonl"
        self.preview_every = int(preview_every)
        self.run_tag = str(run_tag)

        self.sample_idx = None if sample_idx is None else np.asarray(sample_idx, np.int64)
        self.dn_idx = None if dn_idx is None else np.asarray(dn_idx, np.int64)
        self.sample_shape = tuple(int(x) for x in sample_shape)

        self.sample_order, self.layer_bounds, self.layer_names = \
            self._build_order(self.sample_idx, layer_labels)

        self.n_frames = 0
        self.t0 = time.perf_counter()
        self._fh = open(self.jsonl, "a", encoding="utf-8")
        self._preview_dir: Path | None = None
        self._write_header()

    # ---------------------------------------------------------------- 构造辅助

    @staticmethod
    def _build_order(sample_idx, layer_labels):
        """按「层 → 层内索引」排序,使热图按解剖层次分块显示。"""
        if sample_idx is None:
            return None, [], []
        idx = np.asarray(sample_idx, np.int64)
        if layer_labels is None:
            return idx, [], []
        lab = np.asarray([str(x) for x in layer_labels])
        present = [c for c in _LAYER_PRIORITY if np.any(lab == c)]
        present += sorted({*set(lab)} - set(present))
        order_parts, bounds, cur = [], [], 0
        for c in present:
            sel = np.flatnonzero(lab == c)
            if sel.size == 0:
                continue
            order_parts.append(sel[np.argsort(idx[sel], kind="stable")])
            cur += sel.size
            bounds.append(cur)
        return np.concatenate(order_parts) if order_parts else np.arange(idx.size), bounds, present

    def _write_header(self) -> None:
        if self.sample_idx is not None and len(self.sample_idx) != int(np.prod(self.sample_shape)):
            raise ValueError(f"sample_idx 长度 {len(self.sample_idx)} != "
                             f"sample_shape {self.sample_shape}")
        if self.dn_idx is not None:
            self.dn_idx = np.asarray(self.dn_idx, np.int64)
        hdr = {
            "kind": "header",
            "run_tag": self.run_tag,
            "sample_shape": list(self.sample_shape),
            "sample_order": (None if self.sample_order is None
                             else np.asarray(self.sample_order, np.int64).tolist()),
            "sample_idx": (None if self.sample_idx is None
                           else np.asarray(self.sample_idx, np.int64).tolist()),
            "n_dn": 0 if self.dn_idx is None else int(self.dn_idx.size),
            "layer_bounds": [int(x) for x in self.layer_bounds],
            "layer_names": [str(x) for x in self.layer_names],
            "preview_every": self.preview_every,
        }
        self._fh.write(json.dumps(hdr, ensure_ascii=False) + "\n")
        self._fh.flush()

    # ---------------------------------------------------------------- 每帧

    def step(self, *, t_ms: float, frame: np.ndarray | None,
             spikes: np.ndarray, rates: np.ndarray,
             meta: dict | None = None) -> None:
        """记录一帧。spikes/rates 是**全局索引空间**的数组(长度 = N)。

        若 `meta["no_brain"]` 为真(或 spikes/rates 长度为 1),
        则记录靶场状态但**不记录神经活动**(供 pid/random 等无脑臂使用)。
        """
        sp = np.asarray(spikes).ravel()
        rt = np.asarray(rates).ravel()
        if sp.shape != rt.shape:
            raise ValueError(f"spikes {sp.shape} 与 rates {rt.shape} 形状不一致")

        no_brain = bool((meta or {}).get("no_brain", False)) or sp.size <= 1

        if no_brain:
            act = np.empty(0, np.float32)
            layer_mean = {}
        else:
            full = (rt[self.sample_order] if self.sample_order is not None
                    else self._subsample(rt))
            dn = rt[self.dn_idx] if self.dn_idx is not None else np.empty(0, np.float32)
            act = np.concatenate([full.astype(np.float32), dn.astype(np.float32)])
            layer_mean = {}
            for name, lo, hi in self._iter_layers():
                layer_mean[name] = float(sp[lo:hi].mean()) if hi > lo else 0.0

        rec = {
            "kind": "frame",
            "i": int(self.n_frames),
            "t_ms": float(t_ms),
            "wall_s": round(time.perf_counter() - self.t0, 4),
            "act": _pack(act),
            "act_n": int(act.size),
            "spike_rate": (float(sp.mean()) if not no_brain else 0.0),
            "active": (int((rt > 0).sum()) if not no_brain else 0),
            "layer_mean_hz": layer_mean,
        }
        if meta:
            rec["meta"] = {k: (v if isinstance(v, (int, float, str, bool))
                               else _pack(np.asarray(v).ravel()))
                           for k, v in meta.items()}

        if frame is not None and self.preview_every > 0 and \
                self.n_frames % self.preview_every == 0:
            rec["frame_png"] = self._save_preview(frame)

        self._fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        self._fh.flush()
        self.n_frames += 1

    def _iter_layers(self):
        if not self.layer_bounds:
            yield ("all", 0, int(self.sample_idx.size) if self.sample_idx is not None else 0)
            return
        lo = 0
        for name, hi in zip(self.layer_names, self.layer_bounds):
            yield (name, int(lo), int(hi))
            lo = int(hi)

    def _subsample(self, rt: np.ndarray) -> np.ndarray:
        n = rt.size
        k = int(np.prod(self.sample_shape))
        if n <= k:
            out = np.zeros(k, np.float32)
            out[:n] = rt
            return out
        sel = (np.arange(k) * (n // k)).astype(np.int64)
        return rt[sel].astype(np.float32)

    def _save_preview(self, frame: np.ndarray) -> str | None:
        try:
            from PIL import Image
        except ImportError:
            return None
        if self._preview_dir is None:
            self._preview_dir = self.out / "preview"
            self._preview_dir.mkdir(parents=True, exist_ok=True)
        p = self._preview_dir / f"f{self.n_frames:06d}.png"
        Image.fromarray(np.asarray(frame).astype(np.uint8)).save(p, optimize=False)
        return str(p.relative_to(self.out))

    def diag(self, **kv) -> None:
        """写一条诊断记录(如读出层 R²),会显示在可视化的标题里。"""
        rec = {"kind": "diag", "i": int(self.n_frames),
               "wall_s": round(time.perf_counter() - self.t0, 4)}
        rec.update(kv)
        self._fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        self._fh.flush()

    # ---------------------------------------------------------------- 收尾

    def close(self) -> dict:
        if self._fh:
            self._fh.flush()
            self._fh.close()
            self._fh = None
        n_prev = 0
        if self._preview_dir and self._preview_dir.exists():
            n_prev = sum(1 for _ in self._preview_dir.glob("*.png"))
        return {"frames": int(self.n_frames),
                "jsonl": str(self.jsonl), "bytes": self.jsonl.stat().st_size,
                "previews": n_prev}

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


def make_sample_plan(n_neurons: int, roles, neuron_index=None,
                     shape=(96, 96)):
    """为 TelemetryWriter 准备下采样方案(全网热图 + DN 列表 + 层标签)。"""
    k = int(np.prod(shape))
    rng = np.random.default_rng(20261003)
    if n_neurons <= k:
        sample_idx = np.arange(n_neurons, dtype=np.int64)
    else:
        sample_idx = np.sort(rng.choice(n_neurons, size=k, replace=False).astype(np.int64))

    labels = None
    if neuron_index is not None:
        try:
            sup = neuron_index.df["superclass"].astype(str).to_numpy()
            labels = sup[sample_idx]
        except Exception:
            labels = None

    return {
        "sample_idx": sample_idx,
        "sample_shape": tuple(shape),
        "dn_idx": np.asarray(roles.descending, np.int64),
        "layer_labels": labels,
    }
