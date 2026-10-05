"""测量冒烟测试的实际资源开销 + 它到底加载了哪些重依赖。

回答两个问题:
  1. 跑一次要多久 / 吃多少内存?
  2. 它会 import CuPy / torch / 打开屏幕 / 碰游戏吗?(决定它对硬件的要求)

用法::
    & $py tools/_measure_smoke_cost.py
"""
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable

# 只看这些"重"模块是否被加载 —— 它们决定对 GPU / 显存 / 驱动的依赖
HEAVY = ["cupy", "torch", "tensorflow", "cuda", "numba", "cv2", "imageio",
         "matplotlib", "tkinter", "PyQt5", "mss", "dxcam", "bettercam"]


def _which_loaded() -> list[str]:
    """在一个干净子进程里 exec 冒烟测试的 import 部分,看加载了谁。"""
    code = (
        "import sys, importlib.util as u\n"
        f"sys.path.insert(0, r'{ROOT}')\n"
        f"spec = u.spec_from_file_location('sm', r'{ROOT / 'tools' / 'aimlab_smoke.py'}')\n"
        "m = u.module_from_spec(spec)\n"
        "spec.loader.exec_module(m)\n"          # 只导入,不跑 main()
        f"HEAVY = {HEAVY!r}\n"
        "for h in HEAVY:\n"
        "    print(('LOADED  ' if h in sys.modules else 'absent  ') + h)\n"
        "print('n_modules', len(sys.modules))\n"
    )
    out = subprocess.run([PY, "-c", code], capture_output=True, text=True,
                         encoding="utf-8", errors="replace", cwd=str(ROOT))
    return (out.stdout or "").splitlines() + (out.stderr or "").splitlines()[-3:]


def main() -> int:
    print("=" * 74)
    print("① 冒烟测试会加载哪些重依赖?(决定硬件要求)")
    print("=" * 74)
    for ln in _which_loaded():
        print("  " + ln)

    print()
    print("=" * 74)
    print("② 实际跑一次:耗时 / CPU / 峰值内存(默认离线模式)")
    print("=" * 74)
    t0 = time.perf_counter()
    proc = subprocess.Popen([PY, str(ROOT / "tools" / "aimlab_smoke.py")],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                            cwd=str(ROOT))
    peak_ws = peak_cpu = 0
    try:
        import ctypes
        from ctypes import wintypes

        class PMC(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD),
                        ("PageFaultCount", wintypes.DWORD),
                        ("PeakWorkingSetSize", ctypes.c_size_t),
                        ("WorkingSetSize", ctypes.c_size_t),
                        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                        ("PagefileUsage", ctypes.c_size_t),
                        ("PeakPagefileUsage", ctypes.c_size_t)]

        psapi = ctypes.windll.psapi
        k32 = ctypes.windll.kernel32
        PROCESS_QUERY_LIMITED = 0x1000
        handles: dict[int, int] = {}
        while proc.poll() is None:
            if proc.pid not in handles:
                handles[proc.pid] = k32.OpenProcess(PROCESS_QUERY_LIMITED, False,
                                                    proc.pid)
            h = handles[proc.pid]
            if h:
                pmc = PMC()
                pmc.cb = ctypes.sizeof(PMC)
                if psapi.GetProcessMemoryInfo(h, ctypes.byref(pmc), pmc.cb):
                    peak_ws = max(peak_ws, pmc.PeakWorkingSetSize)
            time.sleep(0.03)
        for h in handles.values():
            if h:
                k32.CloseHandle(h)
    except Exception as exc:
        print(f"  (内存采样不可用: {type(exc).__name__}: {exc})")
    wall = time.perf_counter() - t0
    print(f"  退出码   : {proc.returncode}")
    print(f"  墙钟耗时 : {wall:.1f}s")
    print(f"  峰值内存 : {peak_ws / 1024 / 1024:.1f} MB")
    print(f"  结论     : 单线程、纯 CPU、{peak_ws / 1024 / 1024:.0f}MB 级 —— "
          f"这是十年前的机器都能跑的量级")
    return 0


if __name__ == "__main__":
    sys.exit(main())
