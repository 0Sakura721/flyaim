"""打乱接线零模型(shuffle null model)。契约见 CONTRACT.md 第 3 节。

``shuffle`` 臂要回答的问题是:

    "果蝇的成绩,是因为这张**特定的接线图**,还是随便一张同样稠密的随机图也能做到?"

为此需要构造一个 **保持整体统计量、摧毁拓扑信息** 的连接组副本。

为什么"重排列索引"是有效的零模型
--------------------------------
CSR 表示里 ``W[i, j] = j -> i`` 的强度。我们**只置换 ``indices``(列索引),
``data``(权重)与 ``indptr``(行指针)原样不动**:

* ``indptr`` 不变 -> 每个神经元的**出度逐个精确保持**(第 i 行非零个数逐行相等,
  总 nnz 相等);
* ``data`` 不变 -> 权重的多重集(以及逐行权重多重集)完全不变,连数据数组都能
  逐字节比对;
* 只有"第 i 个神经元到底连到哪些 j"被打散 -> **拓扑信息(哪些神经元连哪些)
  被摧毁**,而这正是"接线图是否有因果贡献"要检验的那个变量。

于是 ``fly`` vs ``shuffle`` 的差异只能归因于拓扑,而不是"边变少了/权重变小了/
网络变稀疏了"这些平凡解释。这是标准的 wiring-shuffle 零模型。

两种模式
--------
``mode="row"``(默认)
    在每个 **行内** 对列索引做确定性置换(Fisher-Yates 风格,用同一个 PRNG 流按
    行序消费)。逐行 nnz、逐行权重多重集严格保持;每个神经元的**入度**会随机变化
    (总入度守恒)。
``mode="global"``
    对整个索引空间取一个置换 ``pi``,令 ``indices -> pi[indices]``。出度逐行精确
    保持、**入度分布(多重集)也精确保持**(只是重新分配给随机神经元),同时
    ``W_exc`` 与 ``W_inh`` 共用同一个 ``pi``,因而兴奋/抑制的**共同靶向结构也保持**,
    是更保守的零模型。内存开销只有一个 ``(N,)`` 置换表,适合 1.25e8 边的全量连接组。

 reproducibility
----------------
置换完全由 ``np.random.default_rng(seed)`` 驱动;``mode="row"`` 为了内存友好采用
固定大小的分块处理(``_BLOCK_NNZ``),因此"结果只依赖 seed"这一性质绑定在
当前实现与 ``_BLOCK_NNZ`` 常量上 —— 同一个 seed、同一份代码 -> 同一条置换,
跨版本不保证逐位相同(零模型的统计性质不受影响)。
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from flyaim.io import ConnectomeArtifacts

__all__ = ["shuffle_connectome", "verify_shuffle"]

# 行模式分块处理的最大 nnz。实测(166,700 神经元 / 25.6M 边真实连接组,exc 20.6M 边):
# 4M -> 14-17s,1M -> 10s,250k -> 9.7s。块太大反而慢(lexsort 超出缓存),
# 取 1M 兼顾速度与峰值内存(约 1M * 20B = 20MB/块)。
_BLOCK_NNZ = 1_000_000


def shuffle_connectome(
    art: ConnectomeArtifacts, seed: int, mode: str = "row"
) -> ConnectomeArtifacts:
    """返回一个**新的** :class:`ConnectomeArtifacts`,拓扑被打乱、统计量被保持。

    Parameters
    ----------
    art:
        原始连接组(不会被修改)。
    seed:
        置换种子(第二个位置参数,名为 ``seed``,集成层按位置调用)。
    mode:
        ``"row"``(默认)行内列索引置换;``"global"`` 全局列标签置换。

    Returns
    -------
    ConnectomeArtifacts
        新对象,**原样带上** ``neuron_ids`` 与 ``soma_pos``;``W_exc``/``W_inh``
        为 float32 CSR(可通过 ``ConnectomeArtifacts.__post_init__`` 的形状校验),
        ``indptr`` 与 ``data`` 与原矩阵逐元素相同。
    """
    if mode not in ("row", "global"):
        raise ValueError(f"mode 必须是 'row' 或 'global',收到 {mode!r}")
    n = art.n
    rng = np.random.default_rng(int(seed))

    if mode == "global":
        perm = rng.permutation(n).astype(np.int64)
        W_exc = _apply_column_perm(art.W_exc, perm)
        W_inh = _apply_column_perm(art.W_inh, perm)
    else:
        W_exc = _shuffle_rows(art.W_exc, rng)
        W_inh = _shuffle_rows(art.W_inh, rng)

    return ConnectomeArtifacts(
        W_exc=W_exc,
        W_inh=W_inh,
        neuron_ids=np.array(art.neuron_ids, dtype=np.int64, copy=True),
        soma_pos=None if art.soma_pos is None else np.array(art.soma_pos, copy=True),
    )


def verify_shuffle(original: ConnectomeArtifacts, shuffled: ConnectomeArtifacts) -> dict:
    """零模型不变量自检(供 selftest / 报告引用)。

    两种模式共有的不变量(``universal_ok``):

    * ``row_nnz_equal``          每行非零个数逐行相等(出度结构严格保持)
    * ``nnz_equal``              总 nnz 相等
    * ``data_equal``             ``data``(权重)逐元素相同 -> 权重分布严格保持
    * ``indptr_equal``           行指针逐元素相同
    * ``rowsum_equal``           每个神经元的输出总权重严格不变
    * ``neuron_ids_equal`` / ``soma_pos_equal``  元数据原样保留

    模式特有的不变量:

    * ``mode="row"``:``indices_multiset_equal=True``(行内只是重排,整体列索引
      多重集也不变,没有边被创造或消灭)
    * ``mode="global"``:列索引整体多重集**会变**(这正是置换的作用),但
      ``indegree_distribution_equal=True``(入度多重集严格保持)且
      ``colsum_multiset_equal=True``(每个神经元的输入总权重多重集严格保持)。

    ``all_invariants_ok`` = 共有不变量 + (行模式不变量 或 全局模式不变量) +
    拓扑确实改变(``topology_changed``)。``mode_guess`` 是对模式的启发式判断。
    """
    out: dict = {}
    row_ok = True
    nnz_ok = True
    data_ok = True
    indptr_ok = True
    multiset_ok = True
    rowsum_ok = True
    indeg_ok = True
    colsum_ok = True
    changed = False
    for a, b in ((original.W_exc, shuffled.W_exc), (original.W_inh, shuffled.W_inh)):
        a = a.tocsr()
        b = b.tocsr()
        row_ok = row_ok and bool(np.array_equal(np.diff(a.indptr), np.diff(b.indptr)))
        nnz_ok = nnz_ok and (a.nnz == b.nnz)
        data_ok = data_ok and bool(np.array_equal(a.data, b.data))
        indptr_ok = indptr_ok and bool(np.array_equal(a.indptr, b.indptr))
        multiset_ok = multiset_ok and bool(
            np.array_equal(np.sort(a.indices), np.sort(b.indices))
        )
        rowsum_ok = rowsum_ok and bool(
            np.array_equal(
                np.asarray(a.sum(axis=1)).ravel(), np.asarray(b.sum(axis=1)).ravel()
            )
        )
        n = a.shape[1]
        indeg_ok = indeg_ok and bool(
            np.array_equal(
                np.sort(np.bincount(a.indices, minlength=n)),
                np.sort(np.bincount(b.indices, minlength=n)),
            )
        )
        colsum_ok = colsum_ok and bool(
            np.allclose(
                np.sort(np.asarray(a.sum(axis=0)).ravel()),
                np.sort(np.asarray(b.sum(axis=0)).ravel()),
            )
        )
        changed = changed or (not np.array_equal(a.indices, b.indices))

    n = original.n
    diag_a = _self_loops(original.W_exc) + _self_loops(original.W_inh)
    diag_b = _self_loops(shuffled.W_exc) + _self_loops(shuffled.W_inh)

    out["row_nnz_equal"] = bool(row_ok)
    out["nnz_equal"] = bool(nnz_ok)
    out["data_equal"] = bool(data_ok)
    out["indptr_equal"] = bool(indptr_ok)
    out["rowsum_equal"] = bool(rowsum_ok)
    out["indices_multiset_equal"] = bool(multiset_ok)
    out["indegree_distribution_equal"] = bool(indeg_ok)
    out["colsum_multiset_equal"] = bool(colsum_ok)
    out["topology_changed"] = bool(changed)
    out["neuron_ids_equal"] = bool(np.array_equal(original.neuron_ids, shuffled.neuron_ids))
    out["soma_pos_equal"] = (
        True
        if original.soma_pos is None and shuffled.soma_pos is None
        else bool(
            original.soma_pos is not None
            and shuffled.soma_pos is not None
            and np.array_equal(original.soma_pos, shuffled.soma_pos)
        )
    )
    out["n"] = int(n)
    out["self_loops_before"] = int(diag_a)
    out["self_loops_after"] = int(diag_b)

    universal = bool(
        row_ok and nnz_ok and data_ok and indptr_ok and rowsum_ok and changed
        and out["neuron_ids_equal"] and out["soma_pos_equal"]
    )
    row_mode_ok = bool(multiset_ok)
    global_mode_ok = bool(indeg_ok and colsum_ok)
    out["universal_ok"] = universal
    out["row_mode_invariants_ok"] = row_mode_ok
    out["global_mode_invariants_ok"] = global_mode_ok
    out["mode_guess"] = "row" if multiset_ok else ("global" if global_mode_ok else "unknown")
    out["all_invariants_ok"] = bool(universal and (row_mode_ok or global_mode_ok))
    return out


# ---------------------------------------------------------------- 内部实现


def _self_loops(W: sp.csr_matrix) -> int:
    rows = np.repeat(np.arange(W.shape[0], dtype=np.int64), np.diff(W.indptr))
    return int(np.count_nonzero(rows == W.indices))


def _apply_column_perm(W: sp.csr_matrix, perm: np.ndarray) -> sp.csr_matrix:
    """全局列标签置换:``indices -> perm[indices]``,data/indptr 不动。"""
    W = W.tocsr()
    new_indices = perm[W.indices]
    out = sp.csr_matrix((W.data, new_indices, W.indptr), shape=W.shape, copy=False)
    _set_sorted_flag(out, new_indices, W.indptr)
    return out.astype(np.float32, copy=False)


def _shuffle_rows(W: sp.csr_matrix, rng: np.random.Generator) -> sp.csr_matrix:
    """行内列索引置换(分块 lexsort,内存 O(block))。"""
    W = W.tocsr()
    indptr = W.indptr
    indices = W.indices
    nnz = int(indices.size)
    if nnz == 0:
        out = sp.csr_matrix(W.shape, dtype=W.dtype)
        return out.astype(np.float32, copy=False)

    new_indices = np.empty_like(indices)
    n_rows = indptr.size - 1
    for start in range(0, nnz, _BLOCK_NNZ):
        stop = min(start + _BLOCK_NNZ, nnz)
        # 该 nnz 区间覆盖的行范围(不物化整条 row_of,省内存)
        r0 = int(np.searchsorted(indptr, start, side="right")) - 1
        r1 = int(np.searchsorted(indptr, stop - 1, side="right")) - 1
        rows = np.repeat(
            np.arange(r0, r1 + 1, dtype=np.int64), np.diff(indptr[r0 : r1 + 2])
        )[start - int(indptr[r0]) : stop - int(indptr[r0])]
        keys = rng.random(stop - start)
        order = np.lexsort((keys, rows))  # 主键 = 行,次键 = 随机键
        new_indices[start:stop] = indices[start:stop][order]

    out = sp.csr_matrix((W.data, new_indices, indptr), shape=W.shape, copy=False)
    _set_sorted_flag(out, new_indices, indptr)
    return out.astype(np.float32, copy=False)


def _set_sorted_flag(W: sp.csr_matrix, indices: np.ndarray, indptr: np.ndarray) -> None:
    """显式告诉 scipy 行内是否已排序。

    置换之后行内索引**通常不再有序**,而某些 scipy 版本在手工构造 CSR 时会把
    ``has_sorted_indices`` 默认成 ``True``,这会让下游 ``sorted_indices()`` /
    ``sum_duplicates()`` 出错。这里自己算准(注意:绝不能调用 ``sort_indices()``,
    排序会把行内置换**整个抵消**掉,拓扑就白打乱了)。
    """
    if indices.size == 0:
        sorted_flag = True
    else:
        rows = np.repeat(np.arange(indptr.size - 1, dtype=np.int64), np.diff(indptr))
        adjacent_same_row = rows[1:] == rows[:-1]
        sorted_flag = bool(np.all(indices[1:][adjacent_same_row] > indices[:-1][adjacent_same_row]))
    try:
        W._has_sorted_indices = sorted_flag  # type: ignore[attr-defined]
    except Exception:  # pragma: no cover - scipy 内部结构变化时退化
        pass
