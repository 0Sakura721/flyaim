"""自定义 CSR SpMM(CuPy RawKernel)。

为什么不用 cuSPARSE:
    本机(GTX 1660 Ti / CUDA 13.4 环境)cuSPARSE **没有优化 SpMV/SpMM**:
    check_availability('csrmv')=False、'csrmm2'=False、'spmm'=True 但实测
    按列线性退化(spmm B=64 耗时 1843ms,等于逐列跑慢速 SpMV)。cupy 的
    ``M.dot(H)`` 走 csrmm->asfortranarray 直接 ImportError(cublasLt 缺失)。
    结果:稀疏矩阵乘成为唯一瓶颈,批量并行完全失效。

本 kernel:
    block-per-row,threadIdx = 列 b(批量索引)。
    - W 行的 data[j]/indices[j] 被同一 block 所有线程读 -> 广播,只读一次;
    - H[indices[j]*B + b] 跨线程连续 -> 合并访存;
    - 输出 out[row*B+b] 连续写。
    这样每个 nnz 的 W(8B) 只读一次,而 H 读 nnz*B*4。

已知约束:行 nnz 分布不均时块间负载不平衡(连接组平均 153,可接受)。
"""

from __future__ import annotations

_KERNEL_SRC = r"""
extern "C" __global__
void csrmm_rowblock(const int* __restrict__ indptr,
                    const int* __restrict__ indices,
                    const float* __restrict__ data,
                    const float* __restrict__ H,
                    float* __restrict__ out,
                    int n, int B) {
    int row = blockIdx.x;
    if (row >= n) return;
    int b = threadIdx.x;
    if (b >= B) return;
    int s = indptr[row], e = indptr[row + 1];
    float acc = 0.f;
    for (int j = s; j < e; ++j) {
        acc += data[j] * H[indices[j] * B + b];
    }
    out[row * B + b] = acc;
}
"""


_SCATTER_SRC = r"""
extern "C" __global__
void edge_grad(const int* __restrict__ rows,
               const int* __restrict__ cols,
               const float* __restrict__ A,   // (n,B)
               const float* __restrict__ Hp,  // (n,B)
               float* __restrict__ gW,        // (m,)
               int m, int B) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= m) return;
    int r = rows[j], c = cols[j];
    const float* ar = A + (long long)r * B;
    const float* hc = Hp + (long long)c * B;
    float acc = 0.f;
    for (int b = 0; b < B; ++b) acc += ar[b] * hc[b];
    gW[j] = acc;
}
"""


class EdgeGrad:
    """按边求梯度 gW[j] = Σ_b A[rows[j],b]·H[cols[j],b](无原子,线程独占边)。"""

    def __init__(self, cp):
        self.cp = cp
        self._k = cp.RawKernel(_SCATTER_SRC, "edge_grad")

    def __call__(self, rows, cols, A, Hp):
        cp = self.cp
        m = int(rows.size)
        gW = cp.empty(m, dtype=cp.float32)
        Ac = cp.ascontiguousarray(A, dtype=cp.float32)
        Hc = cp.ascontiguousarray(Hp, dtype=cp.float32)
        B = int(Ac.shape[1])
        blk = 256
        self._k(((m + blk - 1) // blk,), (blk,), (rows, cols, Ac, Hc, gW, m, B))
        return gW


class CSRSpMM:
    """把 CSR 权重与自定义 kernel 绑定(构建一次,复用)。"""

    def __init__(self, cp, W, block=None):
        self.cp = cp
        self.indptr = cp.ascontiguousarray(W.indptr, dtype=cp.int32)
        self.indices = cp.ascontiguousarray(W.indices, dtype=cp.int32)
        self.data = cp.ascontiguousarray(W.data, dtype=cp.float32)
        self.n = int(W.shape[0])
        self._kernel = cp.RawKernel(_KERNEL_SRC, "csrmm_rowblock")
        self._block = block

    def __call__(self, H):
        """H (K,B) -> (n,B);n 为本矩阵行数,K = H 行数(=矩阵列数)。"""
        cp = self.cp
        n = self.n
        B = H.shape[1]
        Hc = cp.ascontiguousarray(H, dtype=cp.float32)
        out = cp.empty((n, B), dtype=cp.float32)
        block = self._block or max(32, min(256, ((B + 31) // 32) * 32))
        self._kernel((n,), (block,),
                     (self.indptr, self.indices, self.data, Hc, out, n, B))
        return out
