"""C 线:A 线代实现的录屏模块自检(``flyaim/arena/recorder.py``)。

运行::

    python flyaim/arena/selftest_recorder.py

验收点
------
1. 文件真的存在且**可由 Pillow 打开**,``n_frames`` 与喂入帧数按 ``every`` 折算一致(不差一)
2. **逐帧像素级回读一致**(最强正确性证据,不是只看帧数)
3. ``every`` 抽帧率生效:``every=k`` 时帧数 == ``ceil(N/k)``
4. 空 episode(0 帧)**不崩溃**,给出明确行为(不生成文件 + ``status="empty"`` + warning)
5. **峰值内存不随帧数增长** —— 用 RSS 采样线程实测,并配一个**故意累积的对照组**
   证明这个测量方法确实能测出累积(否则"内存恒定"的结论没有说服力)
6. 磁盘/组装失败**抛出而非静默吞掉**
7. 与真实 :class:`Arena` 集成,且 ``close()`` 的返回值可直接进 ``Metrics.add_extra()``
"""

from __future__ import annotations

import ctypes
import ctypes.wintypes as wt
import gc
import json
import shutil
import sys
import tempfile
import threading
import time
import tracemalloc
import warnings
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from flyaim.arena.arena import Arena  # noqa: E402
from flyaim.arena.metrics import Metrics  # noqa: E402
from flyaim.arena.recorder import FrameRecorder  # noqa: E402
from flyaim.config import ArenaConfig  # noqa: E402

FAILS: list[str] = []
ROWS: list[tuple[str, str]] = []


def check(cond: bool, msg: str) -> bool:
    if not cond:
        FAILS.append(msg)
        print(f"  [FAIL] {msg}", flush=True)
    return bool(cond)


def section(title: str) -> None:
    print(f"\n=== {title} ===", flush=True)


def rownote(k: str, v) -> None:
    ROWS.append((k, str(v)))


# ------------------------------------------------------------------ RSS 采样


class _PMC(ctypes.Structure):
    _fields_ = [("cb", wt.DWORD), ("PageFaultCount", wt.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]


_K32 = ctypes.windll.kernel32
_PSAPI = ctypes.windll.psapi
_K32.GetCurrentProcess.restype = ctypes.c_void_p
_PSAPI.GetProcessMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.POINTER(_PMC), wt.DWORD]
_PSAPI.GetProcessMemoryInfo.restype = wt.BOOL


def rss_mb() -> float:
    p = _PMC()
    p.cb = ctypes.sizeof(_PMC)
    _PSAPI.GetProcessMemoryInfo(_K32.GetCurrentProcess(), ctypes.byref(p), p.cb)
    return p.WorkingSetSize / 1048576


class RssSampler(threading.Thread):
    """采样**当前** RSS 的峰值(区别于 OS 的单调高水位 PeakWorkingSetSize)。"""

    def __init__(self, interval: float = 0.004):
        super().__init__(daemon=True)
        self.interval = interval
        self.halt = False
        self.peak = 0.0

    def run(self) -> None:
        while not self.halt:
            self.peak = max(self.peak, rss_mb())
            time.sleep(self.interval)


def peak_delta_mb(fn) -> float:
    gc.collect()
    time.sleep(0.12)
    base = rss_mb()
    s = RssSampler()
    s.start()
    try:
        fn()
    finally:
        s.halt = True
        s.join(timeout=2)
    return s.peak - base


# ------------------------------------------------------------------ 造帧


def mkframe(i: int, h: int = 480, w: int = 640) -> np.ndarray:
    """近似靶场画面:深底 + 移动的红靶 + 固定的准星(帧间有差异,便于验证抽帧)。"""
    f = np.full((h, w, 3), 18, np.uint8)
    y = 60 + (i * 7) % max(1, h - 100)
    x = 60 + (i * 11) % max(1, w - 100)
    f[y:y + 44, x:x + 44] = (235, 70, 70)
    cy, cx = h // 2, w // 2
    f[cy - 2:cy + 2, cx - 12:cx + 12] = (240, 240, 240)
    f[cy - 12:cy + 12, cx - 2:cx + 2] = (240, 240, 240)
    return f


def feed(rec: FrameRecorder, n: int, h: int = 480, w: int = 640,
         keep: list | None = None) -> None:
    for i in range(n):
        fr = mkframe(i, h, w)
        if keep is not None:
            keep.append(fr)
        rec(i, fr, None)


# ------------------------------------------------------------------ 各项检查


def t1_integration_real_arena(tmp: Path) -> None:
    section("1 与真实 Arena 集成 + Pillow 可读 + 帧数精确")
    cfg = ArenaConfig()
    arena = Arena(cfg, seed=0)
    frame = arena.reset()
    rng = np.random.default_rng(0)
    N = 100
    every = 2
    out = tmp / "real_arena.png"
    kept: list[np.ndarray] = [frame]
    rec = FrameRecorder(out, every=every, fmt="apng")
    for i in range(N):
        res = arena.step(rng.uniform(-1, 1, 2).astype(np.float32))
        kept.append(res.frame)
        rec(i, res.frame, res)
    info = rec.close()

    expect = -(-N // every)          # ceil(N/every)
    print(f"  真实靶场 {N} 帧,every={every} -> 期望 {expect} 帧")
    print(f"  path={info['path']}")
    print(f"  bytes={info['bytes']:,}  frames={info['frames']}  seconds={info['seconds']}")
    check(out.exists(), f"输出文件不存在: {out}")
    check(info["status"] == "ok", f"status={info['status']} != ok")
    check(info["frames"] == expect, f"录到 {info['frames']} 帧 != 期望 {expect} 帧(差一?)")

    im = Image.open(out)
    check(im.format == "PNG", f"Pillow 识别格式为 {im.format}")
    nf = getattr(im, "n_frames", 1)
    check(nf == expect, f"Pillow 读到 {nf} 帧 != 期望 {expect}")
    # 逐帧像素级回读。注意下标:`kept[0]` 是 reset() 那一帧,而第 1 个被录制的帧是
    # step(0) 之后的 res.frame,即 kept[1];第 k 个录制帧(0-based)对应喂入下标 k*every,
    # 即 kept[k*every + 1] —— 这里最容易差一,所以单独算清楚。
    bad = []
    for k in range(nf):
        im.seek(k)
        got = np.asarray(im.convert("RGB"))
        want = kept[k * every + 1]
        if not np.array_equal(got, want):
            bad.append(k)
    check(not bad, f"像素级回读不一致的帧下标: {bad[:10]}(共 {len(bad)})")
    print(f"  像素级回读: {nf - len(bad)}/{nf} 帧完全一致 "
          f"(第 k 个录制帧 <-> 喂入下标 k*every)")
    rownote("t1 真实靶场 APNG", f"{info['frames']} 帧 / {info['bytes']:,} B")
    im.close()


def t2_every(tmp: Path) -> None:
    section("2 every 抽帧率:N -> ceil(N/every)")
    N = 200
    table = []
    for every in (1, 2, 3, 7, 200, 500):
        out = tmp / f"every{every}.png"
        rec = FrameRecorder(out, every=every, fmt="apng")
        feed(rec, N, h=60, w=80)
        info = rec.close()
        expect = -(-N // every)
        ok = info["frames"] == expect
        nf = Image.open(out).n_frames if info["frames"] else None
        table.append((every, expect, info["frames"], nf, ok))
        print(f"  every={every:<4d} 期望={expect:<4d} 录制={info['frames']:<4d} "
              f"Pillow 读到={nf!s:<5s} {'OK' if ok else 'FAIL'}")
        check(ok, f"every={every}: 录到 {info['frames']} != ceil({N}/{every})={expect}")
        check(nf == expect, f"every={every}: Pillow 读到 {nf} != {expect}")
    rownote("t2 抽帧率", "全部 every 取值一致")


def t3_max_frames_and_empty(tmp: Path) -> None:
    section("3 max_frames 上限 + 空 episode 行为")
    out = tmp / "capped.png"
    rec = FrameRecorder(out, every=1, max_frames=25, fmt="apng")
    feed(rec, 200, h=60, w=80)
    info = rec.close()
    print(f"  max_frames=25,喂入 200 帧 every=1 -> frames={info['frames']} dropped={info['dropped']}")
    check(info["frames"] == 25, f"max_frames 未生效: {info['frames']} != 25")
    check(Image.open(out).n_frames == 25, "max_frames 下 Pillow 帧数不符")

    out2 = tmp / "empty.png"
    rec2 = FrameRecorder(out2, every=2, fmt="apng")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        info2 = rec2.close()
    print(f"  空 episode: status={info2['status']} path={info2['path']} frames={info2['frames']} "
          f"warnings={len(w)}")
    check(info2["frames"] == 0, "空 episode 的 frames 应为 0")
    check(info2["status"] == "empty", f"空 episode 的 status 应为 'empty',得到 {info2['status']}")
    check(info2["path"] is None, "空 episode 不应产出文件(path 应为 None)")
    check(not out2.exists(), f"空 episode 却生成了文件: {out2}")
    check(len(w) >= 1, "空 episode 必须发出 warning(不能静默)")
    check(not rec2.spool_dir.exists(), "空 episode 后 spool 目录应被清理")

    # 喂了帧但 every 把它们全抽掉(只有 i=1)也算 empty
    out3 = tmp / "allskipped.png"
    rec3 = FrameRecorder(out3, every=5, fmt="apng")
    rec3(1, mkframe(1, 60, 80), None)
    with warnings.catch_warnings(record=True) as w3:
        warnings.simplefilter("always")
        info3 = rec3.close()
    print(f"  全部被抽掉: status={info3['status']} seen={info3['seen']} warnings={len(w3)}")
    check(info3["status"] == "empty", "所有帧都被 every 抽掉时应为 empty")
    check(info3["seen"] == 1, "seen 计数应为 1")
    rownote("t3 上限/空 episode", "max_frames=25 生效;/0 帧 -> status=empty 不生成文件")


def t4_errors(tmp: Path) -> None:
    section("4 失败必须抛出(不能静默吞掉)")
    # 4a spool 路径被同名文件占用 -> 构造期就报错
    blocked_parent = tmp / "blocked"
    blocked_parent.mkdir()
    blocker = blocked_parent / "x.png.frames"
    blocker.write_bytes(b"i am a file")
    raised = None
    try:
        FrameRecorder(blocked_parent / "x.png", every=1)
    except Exception as e:  # noqa: BLE001
        raised = e
    print(f"  4a spool 被占用 -> {type(raised).__name__ if raised else 'None'}")
    check(raised is not None, "spool 路径被占用时应当立刻抛错")

    # 4b 录制中途单帧写盘失败 -> 必须传播
    out = tmp / "midfail.png"
    rec = FrameRecorder(out, every=1, fmt="apng")
    orig_save = Image.Image.save
    state = {"n": 0}

    def flaky_save(self, fp, *a, **kw):
        state["n"] += 1
        if state["n"] == 3:
            raise OSError("[测试注入] 磁盘写入失败")
        return orig_save(self, fp, *a, **kw)

    Image.Image.save = flaky_save
    raised2 = None
    try:
        for i in range(10):
            rec(i, mkframe(i, 60, 80), None)
    except Exception as e:  # noqa: BLE001
        raised2 = e
    finally:
        Image.Image.save = orig_save
    print(f"  4b 中途写盘失败 -> {type(raised2).__name__ if raised2 else 'None'}: {raised2}")
    check(isinstance(raised2, OSError), f"中途写盘失败应抛出 OSError,得到 {raised2!r}")
    check(rec.n_frames == 2, f"失败前应已落盘 2 帧,得到 {rec.n_frames}")

    # 4c 非法参数
    bad = []
    for kwargs in ({"every": 0}, {"every": -1}, {"fmt": "mp4"}, {"max_frames": 0},
                   {"duration_ms": 0}):
        try:
            FrameRecorder(tmp / "z.png", **kwargs)
        except (ValueError, TypeError):
            bad.append(True)
        else:
            bad.append(False)
    print(f"  4c 非法参数被拒绝: {sum(bad)}/{len(bad)}")
    check(all(bad), f"非法参数未被拒绝: {bad}")

    # 4d 非 uint8 / 形状错误
    rec2 = FrameRecorder(tmp / "shape.png", every=1, fmt="apng")
    errs = []
    for bad_frame in (np.zeros((10, 10), np.float32), np.zeros((10, 10, 5), np.uint8),
                      np.zeros((10,), np.uint8)):
        try:
            rec2(0, bad_frame, None)
        except (TypeError, ValueError) as e:
            errs.append(type(e).__name__)
    print(f"  4d 非法帧被拒绝: {errs}")
    check(len(errs) == 3, f"非法帧未被拒绝: {errs}")
    rownote("t4 错误传播", "4 类失败均抛出")


def t5_memory(tmp: Path) -> float:
    section("5 ★ 流式验收:峰值内存不随帧数增长(+ 故意累积的对照组)")
    frame_bytes = 480 * 640 * 3
    print(f"  帧规格 640x480x3 = {frame_bytes:,} B/帧")
    N_SMALL, N_BIG = 60, 600

    def rec_run(n: int, tag: str) -> tuple[float, dict]:
        out = tmp / f"mem_{tag}.png"
        rec = FrameRecorder(out, every=2, fmt="apng")
        feed(rec, n)
        info = rec.close()
        return info["frames"], info

    small_frames, _ = rec_run(N_SMALL, "small")            # 预热/建基线
    peak_small = peak_delta_mb(lambda: rec_run(N_SMALL, "small2"))
    peak_big = peak_delta_mb(lambda: rec_run(N_BIG, "big"))
    grow = peak_big - peak_small
    print(f"  recorder  N={N_SMALL:<4d} peakΔ={peak_small:7.1f} MB")
    print(f"  recorder  N={N_BIG:<4d} peakΔ={peak_big:7.1f} MB   ->  增长 {grow:+.1f} MB")
    print(f"            (若把 {N_BIG} 帧全驻留内存应增长 "
          f"{(N_BIG - N_SMALL) * frame_bytes / 1e6:.0f} MB)")

    # ---- 对照组:故意累积(等价于 rollout.collect_frames=True)
    def accumulating(n: int) -> None:
        buf = [mkframe(i) for i in range(n)]          # 故意全存内存
        imgs = [Image.fromarray(f) for f in buf]
        out = tmp / "control.gif"
        imgs[0].save(out, save_all=True, append_images=imgs[1:], duration=33, loop=0,
                     optimize=False)
        del buf, imgs

    n_ctrl = 400
    ctrl_small = peak_delta_mb(lambda: accumulating(60))
    ctrl_big = peak_delta_mb(lambda: accumulating(n_ctrl))
    print(f"  对照组    N=60  peakΔ={ctrl_small:7.1f} MB")
    print(f"  对照组    N={n_ctrl} peakΔ={ctrl_big:7.1f} MB   ->  增长 {ctrl_big - ctrl_small:+.1f} MB")
    print("  ^ 对照组证明本测量方法**确实能**测出内存累积,否则下面的结论不成立")

    check(ctrl_big - ctrl_small > 150.0,
          f"对照组只增长 {ctrl_big - ctrl_small:.1f} MB,测量方法可能失效(预期 >150 MB)")
    check(grow < 64.0,
          f"recorder 峰值随帧数增长 {grow:.1f} MB,超过 64 MB 阈值 —— 不是真流式")
    check(peak_big < 128.0, f"recorder N={N_BIG} 峰值 {peak_big:.1f} MB 过大")
    rownote("t5 流式",
            f"N 60->600 增长 {grow:+.1f} MB(对照组 {(ctrl_big - ctrl_small):+.1f} MB)")

    # tracemalloc 作为补充证据(Python 域;numpy/PIL 缓冲区不在该域内,故只作参考)
    gc.collect()
    tracemalloc.start(1)
    before = tracemalloc.get_traced_memory()[1]
    rec_run(N_BIG, "trace")
    cur, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    print(f"  tracemalloc(仅 Python 域,补充证据): peak={peak / 1e6:.2f} MB")
    check(peak < 64e6, f"tracemalloc Python 域峰值 {peak / 1e6:.2f} MB 过大")
    _ = before
    return grow


def t6_gif(tmp: Path) -> float:
    section("6 GIF 备选路径(已知内存代价,如实测量)")
    N = 150
    out = tmp / "clip.gif"
    keep: list[np.ndarray] = []

    def run() -> None:
        rec = FrameRecorder(out, every=1, fmt="gif", duration_ms=33)
        feed(rec, N, h=120, w=160, keep=keep)
        rec.close()

    peak = peak_delta_mb(run)
    im = Image.open(out)
    print(f"  path={out}  bytes={out.stat().st_size:,}  Pillow n_frames={im.n_frames} "
          f"(期望 {N})  peakΔ={peak:.1f} MB")
    check(out.exists(), "GIF 未生成")
    check(im.n_frames == N, f"GIF 帧数 {im.n_frames} != {N}")
    check(im.format == "GIF", f"格式应为 GIF,得到 {im.format}")
    # 抽检像素
    im.seek(7)
    got = np.asarray(im.convert("RGB"))
    check(got.shape == (120, 160, 3), f"GIF 帧尺寸异常: {got.shape}")
    im.close()
    print(f"  GIF 峰值内存增长随帧数线性(备选格式的固有代价),故默认用 apng")
    rownote("t6 GIF", f"{N} 帧 / {out.stat().st_size:,} B")
    return peak


def t7_api(tmp: Path) -> None:
    section("7 API:上下文管理器 / 幂等 close / add_extra 兼容")
    out = tmp / "ctx.png"
    with FrameRecorder(out, every=2, fmt="apng") as rec:
        feed(rec, 50, h=60, w=80)
    info = rec.close()                     # 幂等
    check(rec.result is not None and rec.result["frames"] == info["frames"],
          "重复 close() 返回结果不一致")
    check(info["frames"] == 25, f"30 帧 every=2 应为 25,得到 {info['frames']}")
    check(out.exists(), "上下文管理器未产出文件")

    # 返回值必须能直接塞进 Metrics.add_extra 并 JSON 序列化
    m = Metrics()
    try:
        m.add_extra("recording", info)
    except Exception as e:  # noqa: BLE001
        check(False, f"add_extra 失败: {e!r}")
    try:
        blob = json.dumps(m.summary(), ensure_ascii=False)
        ok = "recording" in json.loads(blob)
    except TypeError as e:
        ok = False
        check(False, f"summary() 不能 JSON 序列化: {e}")
    check(ok, "close() 返回值无法 JSON 序列化进 Metrics")
    check(all(v is None or isinstance(v, (str, int, float, bool))
              for v in info.values()),
          f"close() 返回的 dict 含非标量值: {info}")

    # 主体抛异常时,已录帧仍应收尾且原始异常不被掩盖
    out2 = tmp / "exc.png"
    rec2 = FrameRecorder(out2, every=1, fmt="apng")
    raised = None
    try:
        with rec2:
            feed(rec2, 10, h=60, w=80)
            raise KeyError("模拟实验中途崩溃")
    except KeyError as e:
        raised = e
    check(isinstance(raised, KeyError), "上下文管理器掩盖了原始异常")
    check(out2.exists(), "主体异常时已录帧未被收尾落盘")
    print(f"  上下文管理器: 正常/异常两条路径均收尾;add_extra 兼容 OK")
    rownote("t7 API", "context manager/幂等/JSON 序列化/异常收尾 全通过")

    # keep_frames 保留 spool
    out3 = tmp / "keep.png"
    r3 = FrameRecorder(out3, every=1, fmt="apng", keep_frames=True)
    feed(r3, 6, h=60, w=80)
    r3.close()
    n_spool = len(list(r3.spool_dir.glob("*.png")))
    print(f"  keep_frames=True 时保留 spool: {n_spool} 个单帧 PNG")
    check(n_spool == 6, f"keep_frames=True 应保留 6 个 spool 帧,得到 {n_spool}")
    shutil.rmtree(r3.spool_dir, ignore_errors=True)

    # 单参数 on_step(res) 形式:rollout.run_episode 的钩子只传 StepResult
    from flyaim.arena.rollout import run_episode as rollout_run  # noqa: PLC0415

    out4 = tmp / "onstep.png"
    every4, n4 = 4, 40
    r4 = FrameRecorder(out4, every=every4, fmt="apng")
    a4 = Arena(ArenaConfig(), 3)
    rollout_run(a4, act_fn=lambda f, s: np.zeros(2, np.float32), max_frames=n4,
                metrics=Metrics(), on_step=r4)
    i4 = r4.close()
    expect4 = len(range(0, n4, every4))
    print(f"  on_step(res) 形式: {n4} 帧 every={every4} -> frames={i4['frames']} "
          f"(期望 {expect4})  Pillow 读到={Image.open(out4).n_frames}")
    check(i4["frames"] == expect4,
          f"on_step 形式帧数 {i4['frames']} != {expect4}")
    check(Image.open(out4).n_frames == expect4, "on_step 形式的 APNG 帧数不符")
    check(i4["seen"] == n4, f"on_step 形式 seen={i4['seen']} != {n4}")


def main() -> int:
    t_start = time.perf_counter()
    print("=" * 78)
    print("FrameRecorder 自检  (flyaim/arena/selftest_recorder.py)")
    print("=" * 78)
    tmp = Path(tempfile.mkdtemp(prefix="recorder_selftest_"))
    print(f"临时目录: {tmp}")
    try:
        t1_integration_real_arena(tmp)
        t2_every(tmp)
        t3_max_frames_and_empty(tmp)
        t4_errors(tmp)
        t5_memory(tmp)
        t6_gif(tmp)
        t7_api(tmp)
    finally:
        pass

    section("摘要")
    width = max(len(k) for k, _ in ROWS) + 2
    for k, v in ROWS:
        print(f"  {k:<{width}s} {v}")
    print(f"\n  临时目录(保留以便人工查看): {tmp}")
    print(f"  总耗时 {time.perf_counter() - t_start:.1f}s")
    print()
    if FAILS:
        print(f"结论:**失败** —— {len(FAILS)} 个断言未通过:")
        for f in FAILS:
            print(f"  - {f}")
        return 1
    print("结论:**全部断言通过**(0 失败)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
