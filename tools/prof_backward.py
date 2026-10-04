"""反传各步耗时剖析(临时工具)。"""
import sys, time
sys.path.insert(0, ".")
import numpy as np, cupy as cp
from flyaim.ann_gated import GatedConnectomeRNN

net = GatedConnectomeRNN("real")
t0 = time.perf_counter(); net._bt_ops(); print("build _bt_ops %.1fs" % (time.perf_counter()-t0), flush=True)
t0 = time.perf_counter(); net._spmm_ops(); print("build _spmm_ops %.1fs" % (time.perf_counter()-t0), flush=True)

B, T = 64, 8
h = cp.zeros((net.n, B), dtype=cp.float32)
U = cp.random.default_rng(0).standard_normal((1536, B)).astype(cp.float32)
trs = []
t0 = time.perf_counter()
for t in range(T):
    a, hn, PU, z, cand = net.forward_batch(U, h)
    trs.append({"a": a, "h": hn, "h_prev": h, "PU": PU, "z": z, "cand": cand}); h = hn
cp.cuda.Stream.null.synchronize()
print("forward_batch x%d = %.2fs (%.1f ms/步)" % (T, time.perf_counter()-t0, (time.perf_counter()-t0)/T*1000), flush=True)

ys = [cp.random.default_rng(1).standard_normal((B, 2)).astype(cp.float32) for _ in range(T)]
# time individual pieces
opWT, opWzT, opEdge = net._bt_ops()
dpre = cp.random.default_rng(2).standard_normal((net.n, B)).astype(cp.float32)
hp = cp.random.default_rng(3).standard_normal((net.n, B)).astype(cp.float32)
def bench(fn, n=3):
    fn(); cp.cuda.Stream.null.synchronize(); t0=time.perf_counter()
    for _ in range(n): fn()
    cp.cuda.Stream.null.synchronize(); return (time.perf_counter()-t0)/n*1000
print("opWT  (W.T SpMM) %.1f ms" % bench(lambda: opWT(dpre)), flush=True)
print("opEdge(gW)       %.1f ms" % bench(lambda: opEdge(net._rows_c, net._cols_c, dpre, hp)), flush=True)
cp.cuda.Stream.null.synchronize(); t0 = time.perf_counter()
L = net.backward_batch(trs, ys); cp.cuda.Stream.null.synchronize()
print("backward_batch T=%d B=%d = %.2fs (%.1f ms/步)" % (T, B, time.perf_counter()-t0, (time.perf_counter()-t0)/T*1000), flush=True)
