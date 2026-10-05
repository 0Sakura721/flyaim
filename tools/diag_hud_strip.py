"""HudStrip(D42.7)取帧通路诊断 —— 不需要游戏,抓桌面即可。

验证四件事:
  1. `hud_region_for` 解出的绝对框能真的抓到帧(不是"区域非法"被拒);
  2. 抓到的帧尺寸 == bbox 尺寸(OCR 依赖逐像素形状,缩放会改变笔画宽度);
  3. 连续多次抓帧不返回 None(HUD 每 score_every 拍读一次,必须稳定);
  4. 🔴 **跨线程并发**:HudStrip 由主线程调用,而 DualWindowSource 在捕获线程里
     用**同一个 dxcam 单例相机** —— D42.7 之前不存在这种并发。本项验证
     `_DXCAM_LOCK` 真的把它串住了(无异常、尺寸恒正确、不静默丢帧)。

用法::

    & $py tools/diag_hud_strip.py
"""
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from flyaim.bridge.capture import ScreenCapture  # noqa: E402
from flyaim.bridge.score_ocr import HudStrip, ScoreHUD, hud_region_for  # noqa: E402

REGION = (0, 0, 1920, 1080)
CENTER = (510, 90, 900, 900)


def _latency_and_shape() -> None:
    print(f"hud_region_for {REGION} -> {hud_region_for(REGION)}")
    for backend in ("dxcam", "mss", "pil"):
        try:
            strip = HudStrip(REGION, backend=backend)
        except Exception as exc:
            print(f"[{backend:6s}] 构造失败: {type(exc).__name__}: {exc}")
            continue
        print(f"[{backend:6s}] backend={strip.backend} bbox={strip.bbox}")
        ok, shapes, lat = 0, [], []
        for i in range(12):
            try:
                t0 = time.perf_counter()
                img = strip.read()
                lat.append((time.perf_counter() - t0) * 1000.0)
                shapes.append(img.shape)
                ok += 1
            except Exception as exc:
                print(f"          第 {i} 次抓帧失败: {type(exc).__name__}: {exc}")
            time.sleep(0.02)      # 模拟"每 10 拍读一次";睡眠不计入延迟
        a = np.asarray(lat)
        exp = (strip.bbox[3], strip.bbox[2], 3)
        same = all(s == exp for s in shapes)
        print(f"           成功 {ok}/12  尺寸 {set(shapes)}  期望 {exp}  一致={same}")
        print(f"           抓帧延迟 p50={np.percentile(a, 50):.2f}ms "
              f"p90={np.percentile(a, 90):.2f}ms max={a.max():.2f}ms "
              f"(首次 {lat[0]:.2f}ms = 相机冷启动)")
        hud = ScoreHUD(grab_fn=strip.read)
        print(f"           OCR 通路(桌面内容读不出分数是正常的): {hud.read()}")
        strip.close()
        break


def _baseline_capture(seconds: float = 3.0) -> float:
    """**对照组**:只跑捕获线程,不跑 HudStrip,测它的 stale%。

    没有这个基线就没法回答"我引入 HudStrip 之后,视觉管线是不是开始吃旧帧了"。
    先验预期:捕获线程以 ~256Hz 轮询 60Hz 屏幕,dxcam 只在有新帧时返回图像,
    所以 stale% 本来就该在 ~75% 量级 —— 但这是**推理**,必须用实测确认,
    否则"68% 是本来就有的"就成了一句没有依据的辩解。
    """
    cap = ScreenCapture(region=CENTER, out_size=None, backend="dxcam")
    stop = threading.Event()
    stats = {"ok": 0, "err": 0}
    measuring = threading.Event()

    def loop() -> None:
        first = True
        while not stop.is_set():
            try:
                cap.read()
                if measuring.is_set() or not first:
                    if measuring.is_set():
                        stats["ok"] += 1
            except Exception:
                if measuring.is_set():
                    stats["err"] += 1
            if first:
                first = False
                measuring.set()          # 拿到首帧即开始计
    th = threading.Thread(target=loop, daemon=True)
    th.start()
    time.sleep(0.5)                       # 预热(与并发测试同一口径)
    s0, q0 = cap.n_stale, cap.seq + 1
    time.sleep(seconds)
    s1, q1 = cap.n_stale, cap.seq + 1
    stop.set()
    th.join(timeout=2.0)
    n, st = q1 - q0, s1 - s0
    pct = 100.0 * st / n if n else 0.0
    print(f"           对照组(仅捕获线程,{seconds:.0f}s):{n} 次读 / stale {st}"
          f" ({pct:.0f}%) / 异常 {stats['err']}")
    cap.close()
    return pct


def _concurrent(baseline_stale_pct: float = 0.0) -> None:
    """主线程 HudStrip + 捕获线程 ScreenCapture 并发抢同一个 dxcam 单例。

    **必须分预热期与测量期**:两线程同时冷启会互相抢首帧,那批异常是启动竞争,
    不是稳态缺陷 —— 混在一起统计会把两个不同的问题当成一个。

    真正要看的稳态指标是 **stale%(复用旧帧占比)**:
      - HUD 侧 stale 高 ⇒ 分数读的是旧画面(但仍比 GDI 冻结帧好得多);
      - 捕获侧 stale 高 ⇒ **视觉管线开始吃旧帧 = 闭环延迟变大**,这才是我
        引入 HudStrip 后最该担心的事。
    """
    strip = HudStrip(REGION, backend="dxcam")
    cap = ScreenCapture(region=CENTER, out_size=None, backend="dxcam")
    stop = threading.Event()
    stats = {"cap_ok": 0, "cap_err": 0, "hud_ok": 0, "hud_err": 0,
             "cap_bad": 0, "hud_bad": 0}
    warm = {"cap_err": 0, "hud_err": 0}
    errs: list[str] = []
    measuring = threading.Event()

    def capture_thread() -> None:
        while not stop.is_set():
            try:
                f, _m = cap.read()
                if measuring.is_set():
                    stats["cap_ok"] += 1
                    if f.shape != (900, 900, 3):
                        stats["cap_bad"] += 1
            except Exception as exc:
                if measuring.is_set():
                    stats["cap_err"] += 1
                else:
                    warm["cap_err"] += 1
                if len(errs) < 4:
                    errs.append(f"capture: {type(exc).__name__}: {exc}")

    th = threading.Thread(target=capture_thread, daemon=True)
    th.start()
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < 5.0:
        phase = "warm" if time.perf_counter() - t0 < 2.0 else "measure"
        if phase == "measure":
            measuring.set()
            s0, c0 = strip.n_stale, cap.n_stale
        try:
            img = strip.read()
            if measuring.is_set():
                stats["hud_ok"] += 1
                if img.shape != (30, 735, 3):
                    stats["hud_bad"] += 1
        except Exception as exc:
            if measuring.is_set():
                stats["hud_err"] += 1
            else:
                warm["hud_err"] += 1
            if len(errs) < 4:
                errs.append(f"hud: {type(exc).__name__}: {exc}")
        time.sleep(0.05)                              # ~20Hz,比实盘 6Hz 更激进
    stop.set()
    th.join(timeout=2.0)
    print(f"           预热期(前 2s,不计入):捕获异常 {warm['cap_err']} / "
          f"HUD 异常 {warm['hud_err']}   ← 冷启抢首帧,不是稳态缺陷")
    print(f"           测量期 捕获线程 {stats['cap_ok']} 成功 / {stats['cap_err']} 异常"
          f" / {stats['cap_bad']} 尺寸错 / **stale {cap.n_stale}/{cap.seq + 1}**")
    print(f"           测量期 主线程HUD {stats['hud_ok']} 成功 / {stats['hud_err']} 异常"
          f" / {stats['hud_bad']} 尺寸错 / **stale {strip.n_stale}/{strip.n_reads}**"
          f" ({strip.stale_pct:.0f}%)")
    if errs:
        print("           错误样本: " + " | ".join(errs))
    ok = (stats["cap_err"] == 0 and stats["hud_err"] == 0
          and stats["cap_bad"] == 0 and stats["hud_bad"] == 0
          and stats["cap_ok"] > 50 and stats["hud_ok"] > 40)
    print(f"           {'✅ 稳态并发安全' if ok else '❌ 稳态并发有问题'}")
    cap_pct = 100.0 * cap.n_stale / max(cap.seq + 1, 1)
    print(f"           📊 捕获侧 stale:{cap_pct:.0f}% vs 对照组 "
          f"{baseline_stale_pct:.0f}%  →  "
          + ("**HudStrip 没有让视觉管线更容易吃旧帧**"
             if cap_pct <= baseline_stale_pct + 8 else
             f"⚠️ **捕获侧变差 {cap_pct - baseline_stale_pct:+.0f} 个百分点**,"
             f"HudStrip 在抢占新帧,需加大 --score-every 或降 grab_retry"))
    strip.close()
    cap.close()


if __name__ == "__main__":
    _latency_and_shape()
    print("[对照] 先测**没有 HudStrip** 时捕获线程的陈旧帧占比")
    base = _baseline_capture(3.0)
    print("[并发] 主线程 HudStrip + 捕获线程 900x900 共用 dxcam 单例")
    _concurrent(base)
