"""最小探查:weights feather 的 batch 结构 + 前 5 行。"""

import ctypes
import ctypes.wintypes as wt
import time

import pyarrow as pa

W = r"D:\dsh\autoaim\.cache\connectome-weights-male-cns-v1.0-minconf-0.5.feather"


class PMC(ctypes.Structure):
    _fields_ = [
        ("cb", wt.DWORD),
        ("PageFaultCount", wt.DWORD),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
    ]


def rss_mb() -> float:
    p = PMC()
    p.cb = ctypes.sizeof(PMC)
    ctypes.windll.psapi.GetProcessMemoryInfo(
        ctypes.windll.kernel32.GetCurrentProcess(), ctypes.byref(p), p.cb
    )
    return p.WorkingSetSize / 1048576


t0 = time.time()
with pa.memory_map(W, "r") as src:
    r = pa.ipc.open_file(src)
    print("num_record_batches", r.num_record_batches)
    print(r.schema)
    nrows = 0
    for i in range(r.num_record_batches):
        nrows += r.get_batch(i).num_rows
    print("total rows", nrows)
    b0 = r.get_batch(0)
    print("batch0 rows", b0.num_rows, "nbytes", b0.nbytes)
    print(b0.slice(0, 5).to_pandas().to_string())
    blast = r.get_batch(r.num_record_batches - 1)
    print("last batch rows", blast.num_rows)
    print(blast.slice(max(0, blast.num_rows - 5), 5).to_pandas().to_string())

print("elapsed %.1fs" % (time.time() - t0))
print("RSS MB %.0f" % rss_mb())
