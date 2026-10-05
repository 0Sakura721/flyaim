#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""参考视频量化测量(手机 Aim Lab Gridshot 实录):HUD OCR + 命中计分 + 摄像机节奏。

只读视频,只写 `.cache/video_ref/`。不修改任何其它项目文件。
OCR 复用 `flyaim/bridge/score_ocr.py` 的 `ScoreHUD`(布局常量在此处运行时 monkeypatch)。

阶段:
    probe   视频元数据 + 逐帧 PTS + 朝向判定
    layout  实测 HUD 布局(三框 x、数字行 y),存审计 PNG
    all     全流程:一次解码 → 白字掩码 + 相位相关位移 → 离线 OCR → CSV/JSON

用法:
  <py> tools/analyze_ref_video.py --stage layout
  <py> tools/analyze_ref_video.py --stage all

✅ **本脚本的 `--stage all` 产出已独立复核通过(2026-10-06 最终版),可以使用。**
   它的终局读数 **130,709 / 96% / 00:00** 被 Lead 用一条**完全独立**的路径验中:
   0.15s 步长密集抽帧目检,分数在 `00:00` 之后仍继续上涨
   (t=61.00→129,221 → t=61.45→130,335 → t=61.60→**130,709** 并保持到
   t=62.05,HUD 于 t≈62.2 消失)。详见 DECISIONS D42.11。

   **历史(必须留着,否则会重蹈)**:
     ① 首版只解码了 1908 帧里的 500 帧(前 ~16.6s),`final_points` 写成 1207,
        该批产物已隔离到 `.cache/video_ref/_INVALID_subagent_16s_window/`;
     ② Lead 用 `diag_ref_hud_grid.py` 的 **`fps=1` 抽帧**读到 130,335/95%,
        并据此"纠正"了用户原始的 130,709/96% —— **那个纠正是错的**。
        `fps=1` 滤镜有自己的相位(实测 ≈ +0.45s),真正的终局值出现在 t≈61.6,
        而 HUD 在 t≈62.2 消失,**恰好落在 cell 61 与 cell 62 之间被采样漏掉**。
        教训:**用固定相位抽样去读一个"末值",必须自己验相位,而不能假设它对齐。**

   **与本脚本结论冲突、且尚未解决的先验(不要当成已定论)**:
     - 每球得分:**本脚本实测 ≈ 固定 373/球,未支持"按时距加权"**(r=-0.23,
       间隔 2 倍变化只带来 7% 得分变化)。这与 D42.1 记录的"间隔加权"相反。
     - 瞄准节奏:本脚本实测 duty 44.8% / dwell p50 66.2ms,与 D41 用的
       5% / 629ms 基线严重不符。但**这一项本脚本自己承认不可靠** ——
       5.93 hits/s 的靶子生灭会造成大量伪位移,中值滤波把 dwell 从 66ms
       抬到 232ms 就是这个原因。**duty/dwell 在这段录像上尚无定论。**
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import subprocess
import sys
import time
import warnings

import numpy as np

warnings.filterwarnings("ignore", message="All-NaN slice encountered")
warnings.filterwarnings("ignore", message="Mean of empty slice")
np.seterr(all="ignore")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTDIR = os.path.join(ROOT, ".cache", "video_ref")
OCR_PATH = os.path.join(ROOT, "flyaim", "bridge", "score_ocr.py")
DEFAULT_VIDEO = r"C:\Users\Admin\Downloads\Screenrecording_20261005_230205.mp4"
_FF_CANDIDATES = [
    r"C:\Users\Admin\.dsh\dsh-runtimes\dsh-primary-runtime\dependencies\python"
    r"\Lib\site-packages\imageio_ffmpeg\binaries\ffmpeg-win-x86_64-v7.1.exe",
]

FOV_H_DEG = 106.26          # D28:CS2/16:9 水平渲染 FOV(项目既定值)
FPS_NOMINAL = 30.21


# --------------------------------------------------------------------- 工具

def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def find_ffmpeg() -> str:
    for c in _FF_CANDIDATES:
        if os.path.isfile(c):
            return c
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return "ffmpeg"


def load_score_ocr():
    """按文件路径加载 ScoreHUD 模块(不 import flyaim 包,避免副作用)。"""
    spec = importlib.util.spec_from_file_location("score_ocr_ref", OCR_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _read_exact(f, n: int) -> bytes:
    buf = f.read(n)
    while len(buf) < n:
        more = f.read(n - len(buf))
        if not more:
            break
        buf += more
    return buf


# ------------------------------------------------------------------ 阶段 0:probe

def probe_video(ff: str, video: str) -> dict:
    """元数据 + 逐帧 pts_time(showinfo)+ 朝向判定。"""
    log("probe: ffmpeg 元数据")
    p = subprocess.run([ff, "-hide_banner", "-nostdin", "-i", video],
                       capture_output=True, text=True, errors="replace")
    meta = {"ffmpeg_stderr_head": "\n".join(p.stderr.splitlines()[:14])}
    for line in p.stderr.splitlines():
        if line.strip().startswith("Duration:"):
            meta["duration_str"] = line.strip()
        if "fps," in line and "Video:" in line:
            meta["stream_line"] = line.strip()
        if "displaymatrix" in line:
            meta["displaymatrix"] = line.strip()
        if "com.android.version" in line:
            meta["android"] = line.strip()

    log("probe: 逐帧 PTS(showinfo)—用于处理 VFR")
    t0 = time.time()
    p = subprocess.run(
        [ff, "-hide_banner", "-nostdin", "-v", "info", "-i", video,
         "-fps_mode", "passthrough", "-vf", "showinfo", "-f", "null", "-"],
        capture_output=True, text=True, errors="replace")
    pat = re.compile(r"\bn:\s*(\d+).*?\bpts_time:([-\d.eE+]+)")
    idx, pts = [], []
    for line in p.stderr.splitlines():
        m = pat.search(line)
        if m:
            idx.append(int(m.group(1)))
            pts.append(float(m.group(2)))
    pts = np.asarray(pts, dtype=np.float64)
    log(f"probe: showinfo {len(pts)} 帧,耗时 {time.time()-t0:.1f}s")

    # 朝向判定:两种 reshape 的像素数相同,只能看内容(哪一对边是暗条)
    p = subprocess.run([ff, "-hide_banner", "-nostdin", "-v", "error", "-i", video,
                        "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                       capture_output=True)
    raw = p.stdout
    npix = len(raw) // 3
    cand = {}
    for name, (h, w) in (("H720_W1600", (720, 1600)), ("H1600_W720", (1600, 720))):
        if h * w != npix:
            continue
        a = np.frombuffer(raw, np.uint8).reshape(h, w, 3)
        cand[name] = {"top80_mean": float(a[:80].mean()), "bot80_mean": float(a[-80:].mean()),
                      "left80_mean": float(a[:, :80].mean()),
                      "right80_mean": float(a[:, -80:].mean())}
        cand[name]["lr_sum"] = cand[name]["left80_mean"] + cand[name]["right80_mean"]
        cand[name]["tb_sum"] = cand[name]["top80_mean"] + cand[name]["bot80_mean"]
    orient = None
    if len(cand) == 2:
        a, b = cand["H720_W1600"], cand["H1600_W720"]
        # 真横屏:左右是窄黑边 -> H720_W1600 的 lr_sum 更小
        orient = ("H720_W1600(landscape)" if a["lr_sum"] < b["lr_sum"]
                  else "H1600_W720(portrait)")
    meta["first_frame_orientation_probe"] = cand
    meta["decoded_orientation"] = orient
    log(f"probe: 朝向 = {orient}")
    log(f"  判据 {json.dumps(cand, ensure_ascii=False)}")
    log("  (注:首帧可能是亮屏启动画面,黑边判据以 layout 阶段的多帧中位为准)")

    dt = np.diff(pts) if pts.size > 1 else np.array([np.nan])
    meta["n_frames_showinfo"] = int(pts.size)
    meta["pts_first"] = float(pts[0]) if pts.size else None
    meta["pts_last"] = float(pts[-1]) if pts.size else None
    meta["duration_from_pts"] = float(pts[-1] - pts[0]) if pts.size else None
    meta["dt_median"] = float(np.median(dt))
    meta["dt_p05"] = float(np.percentile(dt, 5))
    meta["dt_p95"] = float(np.percentile(dt, 95))
    meta["dt_min"] = float(dt.min())
    meta["dt_max"] = float(dt.max())
    meta["vfr_note"] = ("恒定帧率" if meta["dt_max"] - meta["dt_min"] < 1e-3
                        else "疑似 VFR / 有丢帧间隔")
    log(f"probe: {meta['n_frames_showinfo']} 帧, dt median={meta['dt_median']*1000:.2f}ms "
        f"p05={meta['dt_p05']*1000:.2f} p95={meta['dt_p95']*1000:.2f} "
        f"min={meta['dt_min']*1000:.2f} max={meta['dt_max']*1000:.2f} -> {meta['vfr_note']}")
    return meta, pts


# ------------------------------------------------------------------ 帧迭代器

def iter_frames(ff: str, video: str, nframes: int, chunk: int = 16,
                t0: float | None = None, t1: float | None = None):
    """产出 (frame_index_in_stream, (H,W,3) uint8)。-fps_mode passthrough 保帧序。"""
    cmd = [ff, "-hide_banner", "-nostdin", "-v", "error"]
    if t0 is not None:
        cmd += ["-ss", f"{t0:.3f}"]
    cmd += ["-i", video]
    if t1 is not None:
        cmd += ["-t", f"{t1 - (t0 or 0.0):.3f}"]
    cmd += ["-fps_mode", "passthrough", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    # 注意:stderr 必须 DEVNULL。若用 PIPE 而无人读取,ffmpeg 写满 64KB 管道缓冲后会
    # 卡死,消费端读 stdout 也随之死锁(实测发生过)。
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                         bufsize=1 << 24)
    fs = 1600 * 720 * 3
    k = 0
    try:
        while True:
            buf = _read_exact(p.stdout, fs * chunk)
            if len(buf) < fs:
                break
            n = len(buf) // fs
            arr = np.frombuffer(buf[:n * fs], np.uint8).reshape(n, 720, 1600, 3)
            for i in range(n):
                yield k, arr[i]
                k += 1
            if len(buf) < fs * chunk:
                break
    finally:
        p.stdout.close()
        try:
            p.wait(timeout=20)
        except Exception:
            p.kill()


# ------------------------------------------------------------------ 相位相关

def _hann2d(h: int, w: int) -> np.ndarray:
    return np.outer(np.hanning(h), np.hanning(w)).astype(np.float32)


def phase_shift(a: np.ndarray, b: np.ndarray, win: np.ndarray | None = None,
                hp: bool = False) -> tuple[float, float, float]:
    """相位相关求 b 相对 a 的整/亚像素位移(px)。返回 (dy, dx, peak)。

    约定:若 b(x) = a(x - d)(即画面内容整体平移 +d),返回 d。
    推导:g(x)=f(x-d) -> G(k)=F(k)e^{-i2πk·d/N} -> conj(A)·B 的逆变换峰值在 +d。
    自检见主流程(对真实静帧做已知 np.roll 位移)。
    """
    a = a.astype(np.float32, copy=False)
    b = b.astype(np.float32, copy=False)
    if hp:
        a = a - _box_blur(a, 5)
        b = b - _box_blur(b, 5)
    a = a - a.mean()
    b = b - b.mean()
    if win is not None:
        a = a * win
        b = b * win
    A = np.fft.rfft2(a)
    B = np.fft.rfft2(b)
    R = np.conj(A) * B
    R /= (np.abs(R) + 1e-9)
    r = np.fft.irfft2(R, s=a.shape)
    h, w = r.shape
    pk = int(np.argmax(r))
    py, px = divmod(pk, w)
    peak = float(r[py, px]) / float(r.size) * 1e4   # 归一化峰高(可比)
    # 抛物线亚像素
    def _par(vm, v0, vp):
        den = (vm - 2.0 * v0 + vp)
        return 0.0 if abs(den) < 1e-12 else float(0.5 * (vm - vp) / den)
    sy = _par(r[(py - 1) % h, px], r[py, px], r[(py + 1) % h, px])
    sx = _par(r[py, (px - 1) % w], r[py, px], r[py, (px + 1) % w])
    dy = py + sy
    dx = px + sx
    if dy > h / 2:
        dy -= h
    if dx > w / 2:
        dx -= w
    return float(dy), float(dx), peak


def _box_blur(a: np.ndarray, k: int) -> np.ndarray:
    """cumsum 实现的分离盒滤波(k 奇数)。"""
    r = k // 2
    p = np.pad(a, ((r + 1, r), (0, 0)), mode="edge")
    c = np.cumsum(p, axis=0)
    out = (c[k:] - c[:-k]) / k
    p = np.pad(out, ((0, 0), (r + 1, r)), mode="edge")
    c = np.cumsum(p, axis=1)
    return ((c[:, k:] - c[:, :-k]) / k).astype(np.float32)


def to_gray_half(frame: np.ndarray, vp: tuple[int, int, int, int],
                 down: int = 2) -> np.ndarray:
    """视口 -> 亮度 -> down x down 均值池化(降低 FFT 成本)。"""
    x0, x1, y0, y1 = vp
    sub = frame[y0:y1, x0:x1].astype(np.float32)
    g = 0.299 * sub[:, :, 0] + 0.587 * sub[:, :, 1] + 0.114 * sub[:, :, 2]
    h, w = g.shape
    h2, w2 = (h // down) * down, (w // down) * down
    g = g[:h2, :w2].reshape(h2 // down, down, w2 // down, down).mean(axis=(1, 3))
    return g.astype(np.float32)


# ------------------------------------------------------------------ HUD 布局

def white_mask(frame: np.ndarray, thr: int = 190) -> np.ndarray:
    return ((frame[:, :, 0] > thr) & (frame[:, :, 1] > thr)
            & (frame[:, :, 2] > thr))


def runs_of(cols: np.ndarray, gap: int) -> list[tuple[int, int]]:
    xs = np.where(cols)[0]
    if xs.size == 0:
        return []
    segs, s = [], int(xs[0])
    for i in range(1, xs.size):
        if int(xs[i]) - int(xs[i - 1]) >= gap:
            segs.append((s, int(xs[i - 1]) + 1))
            s = int(xs[i])
    segs.append((s, int(xs[-1]) + 1))
    return segs


def save_png(arr: np.ndarray, path: str, scale: int = 1) -> None:
    from PIL import Image
    im = Image.fromarray(arr.astype(np.uint8))
    if scale != 1:
        im = im.resize((im.width * scale, im.height * scale), Image.NEAREST)
    im.save(path)
    log(f"  写出 {path} ({im.width}x{im.height}, {os.path.getsize(path)/1024:.1f} KB)")


def measure_layout(ff: str, video: str, pts: np.ndarray, sample_ts: list[float]) -> dict:
    log("layout: 抽样解码帧")
    frames = {}
    for t in sample_ts:
        for _, fr in iter_frames(ff, video, 0, chunk=1, t0=t, t1=t + 0.05):
            frames[t] = fr.copy()
            break
        if t not in frames:
            log(f"  !! t={t} 取帧失败")
    ts = sorted(frames)
    log(f"layout: 取得 {len(ts)} 帧 @ {ts}  shape={frames[ts[0]].shape}")

    # --- 1) 黑边(letterbox):列中位亮度 << 场景亮度
    colmed = np.median(np.stack([frames[t].mean(axis=2) for t in ts]), axis=(0, 1))
    rowmed = np.median(np.stack([frames[t].mean(axis=2) for t in ts]), axis=0).mean(axis=1)
    scene = float(np.median(colmed))
    log(f"layout: 列中位亮度总体中位={scene:.1f}; 左 0..200 采样="
        f"{[int(colmed[i]) for i in range(0, 200, 20)]}; 右 1400..1600 采样="
        f"{[int(colmed[i]) for i in range(1400, 1600, 20)]}")
    bar_thr = max(45.0, 0.45 * scene)
    x_left = 0
    while x_left < 1600 and colmed[x_left] < bar_thr:
        x_left += 1
    x_right = 1600
    while x_right > 0 and colmed[x_right - 1] < bar_thr:
        x_right -= 1
    log(f"layout: 黑边阈值 {bar_thr:.1f} -> 左黑边宽 {x_left}, 右黑边宽 {1600-x_right}; "
        f"视口 x[{x_left},{x_right}) 宽 {x_right-x_left} 高 720 纵横比 "
        f"{(x_right-x_left)/720:.4f}")
    log(f"layout: 行中位亮度 上 0..60 采样={[int(rowmed[i]) for i in range(0, 60, 5)]}")

    # --- 2) 时间中位数图(框底稳定,数字被抹掉)-> 用「数字下方那一行框底」切三个框
    med = np.median(np.stack([frames[t][:120].astype(np.float32) for t in ts]),
                    axis=0).astype(np.uint8)
    save_png(med, os.path.join(OUTDIR, "layout_top120_median.png"))
    # 框底行:数字带下方、框底上方的纯背景行(逐行试,取能切出 3 个宽游程者)
    box_runs = []
    for y_probe in (62, 66, 70, 58, 72):
        prof = med[y_probe].mean(axis=1)
        rr = [(a, b) for a, b in runs_of(prof < 95, 3) if b - a > 60]
        rr = [(a, b) for a, b in rr if a >= x_left - 5 and b <= x_right + 5]
        log(f"layout: 中位图 y={y_probe} 行亮度<95 的游程(宽>60)= {rr}")
        if len(rr) == 3:
            box_runs, box_row_used = rr, y_probe
            break
    if not box_runs:
        # 退化:用 y∈[55,72] 的暗列占比
        frac = (med[55:73].mean(axis=2) < 95).mean(axis=0)
        box_runs = [(a, b) for a, b in runs_of(frac > 0.6, 3) if b - a > 60]
        box_runs = [(a, b) for a, b in box_runs if a >= x_left - 5 and b <= x_right + 5]
        box_row_used = "y55-72暗列占比>0.6"
    log(f"layout: 三个 HUD 框 = {box_runs} (判据 {box_row_used})")

    # 框的 y 范围(取第一个框中间一列)
    if box_runs:
        xc = (box_runs[0][0] + box_runs[0][1]) // 2
        colp = med[:, xc].mean(axis=1)
        ys = np.where(colp < 92)[0]
        box_y = (int(ys.min()), int(ys.max()) + 1) if ys.size else (0, 0)
        log(f"layout: 框 y 范围(中位图 x={xc} 列)<92 = y[{box_y[0]},{box_y[1]})")
    else:
        box_y = (0, 0)

    # --- 3) 白字掩码
    per_frame_white = {t: white_mask(frames[t][:120]) for t in ts}
    orw = np.zeros_like(per_frame_white[ts[0]])
    for t in ts:
        orw |= per_frame_white[t]
    andw = np.ones_like(orw)
    for t in ts:
        andw &= per_frame_white[t]
    rows_or_all = orw.sum(axis=1)
    log("layout: 全宽白像素行分布(前 90 行,OR of %d 帧):" % len(ts))
    log("   " + " ".join(f"{y}:{int(rows_or_all[y])}" for y in range(0, 90)))
    log(f"layout: AND(全部帧皆白的像素)总数 = {int(andw.sum())}")

    # --- 4) 数字 x 紧包围盒(在每个框内)
    colw = orw.sum(axis=0)
    boxes_tight = []
    if box_runs:
        gx0, gx1 = box_runs[0][0], box_runs[-1][1]
        for a, b in box_runs:
            xs = np.where(colw[a:b] > 0)[0]
            boxes_tight.append((int(a + xs.min()), int(a + xs.max()) + 1) if xs.size
                               else (a, b))
        log(f"layout: 各框内数字 x 紧包围盒 = {boxes_tight}")
    else:
        gx0, gx1 = x_left, x_right

    # --- 5) 数字行 y 带(只在数字组 x 范围内统计)
    r = orw[:, gx0:gx1].sum(axis=1)
    ys = np.where(r > 4)[0]
    band = (int(ys.min()), int(ys.max()) + 1) if ys.size else (0, 0)
    log(f"layout: 数字行带(x∈[{gx0},{gx1}], 行白>4)= y[{band[0]},{band[1]}) 高 {band[1]-band[0]}")
    log("layout: 该范围内逐行白像素: " +
        " ".join(f"{y}:{int(orw[y, gx0:gx1].sum())}"
                 for y in range(max(0, band[0] - 8), min(120, band[1] + 8))))

    # --- 6) 逐帧字符段数(间隙阈值 3,与 ScoreHUD._GAP 一致)
    log("layout: 逐帧字符段数(间隙阈值 _GAP=3;x 用框范围,数字无跨框风险)")
    for t in ts:
        w = per_frame_white[t]
        segrow, widths = [], []
        for (a, b) in (box_runs if box_runs else boxes_tight):
            sub = w[band[0]:band[1], a:b]
            rr = runs_of(sub.sum(axis=0) > 0, 3)
            segrow.append(len(rr))
            widths.append([int(y - x) for x, y in rr])
        log(f"   t={t:5.1f}s  段数={segrow}  各段宽={widths}")

    # --- 7) 白阈值/亮度
    vals = []
    for t in ts:
        reg = frames[t][band[0]:band[1], gx0:gx1]
        vals.append(reg.reshape(-1, 3).max(axis=1))
    v = np.concatenate(vals)
    hist = {f">={thr}": int((v >= thr).sum()) for thr in (150, 170, 190, 210, 230)}
    log(f"layout: 数字带内像素最大通道值 p50={np.percentile(v,50):.0f} "
        f"p90={np.percentile(v,90):.0f} p99={np.percentile(v,99):.0f} p99.9={np.percentile(v,99.9):.0f}")
    log(f"layout: 白像素计数 {hist}")

    # 数字笔画的实际峰值(取白像素的众数)
    white_vals = v[v > 150]
    log(f"layout: >150 像素数 {white_vals.size},其中位 {np.median(white_vals) if white_vals.size else -1:.0f}")

    # --- 8) 审计 PNG
    save_png(frames[ts[len(ts) // 2]][:100, max(0, x_left - 20):x_right + 20],
             os.path.join(OUTDIR, "layout_strip_raw.png"))
    if boxes_tight:
        y0 = max(0, band[0] - 10)
        y1 = min(120, band[1] + 10)
        crop = frames[ts[len(ts) // 2]][y0:y1, boxes_tight[0][0] - 10:boxes_tight[-1][1] + 10]
        save_png(crop, os.path.join(OUTDIR, "layout_hud_zoom.png"), scale=4)
    save_png(med[20:95, max(0, x_left - 10):x_right + 10],
             os.path.join(OUTDIR, "layout_boxband_median.png"), scale=2)

    return {
        "frame_shape": list(frames[ts[0]].shape),
        "letterbox_left_x": int(x_left), "letterbox_right_x": int(x_right),
        "letterbox_bar_threshold": float(bar_thr),
        "viewport": [int(x_left), int(x_right), 0, 720],
        "viewport_w": int(x_right - x_left),
        "viewport_aspect": float((x_right - x_left) / 720.0),
        "digit_band_y": [int(band[0]), int(band[1])],
        "hud_boxes_xy": [[int(a), int(b)] for a, b in box_runs],
        "hud_boxes_y": [int(box_y[0]), int(box_y[1])],
        "boxes_tight_xy": [[int(a), int(b)] for a, b in boxes_tight],
        "white_pixel_hist": hist,
        "max_channel_percentiles": {
            "p50": float(np.percentile(v, 50)), "p90": float(np.percentile(v, 90)),
            "p99": float(np.percentile(v, 99)), "p999": float(np.percentile(v, 99.9))},
    }


# ------------------------------------------------------------------ 主流程

def build_hud_crop(score_ocr, layout, digit_pad=4):
    """把 ScoreHUD 的布局常量 patch 到手机视频实测布局。

    _BOX_X 直接用实测的三个 HUD 框 x 范围(框与框之间是场景,无串字风险);
    _DIGIT_Y 覆盖数字行带 + digit_pad 行空白(空白行不含 >190 像素,不改变字形)。
    """
    y0 = layout["digit_band_y"][0] - digit_pad
    y1 = layout["digit_band_y"][1] + digit_pad
    boxes_xy = layout["hud_boxes_xy"]
    assert len(boxes_xy) == 3, f"期望 3 个 HUD 框,实测 {boxes_xy}"
    x_left = boxes_xy[0][0]
    x_right = boxes_xy[-1][1]
    names = ("points", "time", "acc")
    boxes = {nm: (boxes_xy[i][0] - x_left, boxes_xy[i][1] - x_left)
             for i, nm in enumerate(names)}
    bb = (x_left, y0, x_right, y1)
    score_ocr.GRAB_BBOX = bb
    score_ocr._DIGIT_Y = (0, y1 - y0)
    score_ocr._BOX_X = boxes
    return {"GRAB_BBOX": list(bb), "_DIGIT_Y": [0, y1 - y0],
            "_BOX_X": {k: list(v) for k, v in boxes.items()},
            "band_y_abs": [y0, y1],
            "hud_boxes_xy": [list(b) for b in boxes_xy]}


def _json_default(o):
    """numpy 标量 -> Python(否则 json.dump 抛 TypeError)。"""
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def pack_mask(m: np.ndarray) -> bytes:
    return np.packbits(m.reshape(-1)).tobytes()


def unpack_mask(buf: bytes, shape: tuple[int, int]) -> np.ndarray:
    n = shape[0] * shape[1]
    return np.unpackbits(np.frombuffer(buf, np.uint8))[:n].reshape(shape).astype(bool)


def mask_to_img(m: np.ndarray) -> np.ndarray:
    """白字掩码 -> 三通道 0/255 图。与原始帧对 `>190` 的二值化结果逐位等价。"""
    return np.repeat((m * 255).astype(np.uint8)[:, :, None], 3, axis=2)


def cluster_glyphs(glyphs: list[np.ndarray], tol: float = 0.02):
    """贪心聚类,返回 (labels, centers)。"""
    centers: list[np.ndarray] = []
    labels: list[int] = []
    cnt: list[int] = []
    for g in glyphs:
        if centers:
            d = [float(np.mean((g - c) ** 2)) for c in centers]
            j = int(np.argmin(d))
            if d[j] <= tol:
                centers[j] = (centers[j] * cnt[j] + g) / (cnt[j] + 1)
                cnt[j] += 1
                labels.append(j)
                continue
        centers.append(g.copy())
        cnt.append(1)
        labels.append(len(centers) - 1)
    return labels, centers, cnt


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default=DEFAULT_VIDEO)
    ap.add_argument("--stage", default="all", choices=["probe", "layout", "scan", "analyze", "all"])
    ap.add_argument("--digit-pad", type=int, default=4)
    ap.add_argument("--down", type=int, default=2, help="相位相关降采样倍数")
    ap.add_argument("--stride", type=int, default=1, help="HUD 每 N 帧读一次")
    ap.add_argument("--limit", type=int, default=0, help="只处理前 N 帧(调试)")
    ap.add_argument("--skip-motion", action="store_true", help="跳过相位相关(快速调 OCR)")
    ap.add_argument("--no-hp", action="store_true", help="相位相关不做高通")
    args = ap.parse_args()

    os.makedirs(OUTDIR, exist_ok=True)
    ff = find_ffmpeg()
    log(f"ffmpeg = {ff}")
    log(f"video  = {args.video}")
    log(f"outdir = {OUTDIR}")

    meta, pts = probe_video(ff, args.video)
    n_total = int(pts.size) if pts.size else 1909
    if args.limit:
        n_total = min(n_total, args.limit)

    layout = measure_layout(ff, args.video, pts, [1.0, 5.0, 15.0, 30.0, 45.0, 55.0, 61.0])
    with open(os.path.join(OUTDIR, "layout_measured.json"), "w", encoding="utf-8") as f:
        json.dump(layout, f, ensure_ascii=False, indent=2)
    log(f"layout: {json.dumps(layout, ensure_ascii=False)}")

    if args.stage in ("probe", "layout"):
        return 0

    score_ocr = load_score_ocr()
    hud_patch = build_hud_crop(score_ocr, layout, args.digit_pad)
    log(f"OCR 布局 patch = {json.dumps(hud_patch, ensure_ascii=False)}")
    band = hud_patch["band_y_abs"]
    x_l, x_r = hud_patch["GRAB_BBOX"][0], hud_patch["GRAB_BBOX"][2]
    bh = band[1] - band[0]
    log(f"HUD 掩码条: x[{x_l},{x_r}) y[{band[0]},{band[1]}) = {x_r-x_l}x{bh}")

    # ---------------------------------------------------------- 单次解码全片
    vp = (layout["viewport"][0], layout["viewport"][1], 0, 720)
    win = _hann2d((vp[3] - vp[2]) // args.down, (vp[1] - vp[0]) // args.down)
    log(f"相位相关窗口 {win.shape}, 视口 {vp}, down={args.down}")

    masks: list[bytes] = []
    shifts = np.full(n_total, np.nan)
    peaks = np.full(n_total, np.nan)
    gtimes = pts[:n_total].copy() if pts.size >= n_total else np.arange(n_total) / FPS_NOMINAL

    scan_path = os.path.join(OUTDIR, "_scan.npz")
    if args.stage == "analyze" and os.path.isfile(scan_path):
        log(f"载入扫描缓存 {scan_path}")
        z = np.load(scan_path, allow_pickle=False)
        # 存成 (N, nb) uint8 而不是 bytes 对象:numpy 的 S dtype 标量转 Python bytes
        # 会吃掉结尾的 NUL,长度就不对了(踩过)。
        marr = z["masks"]
        if marr.dtype.kind == "S":      # 兼容早期 bytes 对象格式(尾 NUL 被吃掉,补零即还原)
            nb = (bh * (x_r - x_l) + 7) // 8
            masks = [marr[i].tobytes().ljust(nb, b"\x00") for i in range(marr.shape[0])]
        else:
            masks = [marr[i].tobytes() for i in range(marr.shape[0])]
        shifts = z["shifts"]
        gtimes = z["gtimes"]
        n_total = len(masks)
        log(f"  缓存 {n_total} 帧, {gtimes[0]:.2f}s -> {gtimes[-1]:.2f}s, "
            f"shift 非空 {int(np.isfinite(shifts).sum())}")
    else:
        prev_gray = None
        t_start = time.time()
        n_done = 0
        for k, fr in iter_frames(ff, args.video, n_total, chunk=16):
            if k >= n_total:
                break
            g = to_gray_half(fr, vp, args.down)
            if (not args.skip_motion) and prev_gray is not None and g.shape == prev_gray.shape:
                dy, dx, pk = phase_shift(prev_gray, g, win=win, hp=not args.no_hp)
                shifts[k] = float(np.hypot(dy, dx) * args.down)
                peaks[k] = pk
            prev_gray = g
            masks.append(pack_mask(white_mask(fr[band[0]:band[1], x_l:x_r])))
            n_done += 1
            if n_done % 200 == 0 or n_done == 1:
                el = time.time() - t_start
                eta = el / n_done * (n_total - n_done)
                sm = (np.nanmedian(shifts[:n_done]) if np.any(np.isfinite(shifts[:n_done]))
                      else float("nan"))
                log(f"scan {n_done}/{n_total} 帧 ({el:.0f}s 已用, ETA {eta:.0f}s) shift中位={sm:.2f}px")
        log(f"scan 完成:{n_done} 帧,{time.time()-t_start:.1f}s")
        if n_done < n_total:
            log(f"!! 实际解码 {n_done} 帧 < 预期 {n_total},按实际裁剪")
            n_total = n_done
            gtimes = gtimes[:n_total]
            shifts = shifts[:n_total]
            peaks = peaks[:n_total]
        np.savez_compressed(scan_path,
                            masks=np.frombuffer(b"".join(masks), np.uint8)
                                   .reshape(len(masks), -1),
                            shifts=shifts, gtimes=gtimes)
        log(f"写出扫描缓存 {scan_path} ({os.path.getsize(scan_path)/1e6:.2f} MB)")

    if args.stage == "scan":
        return 0

    mask_shape = (bh, x_r - x_l)

    # ------------------------------------------------- 相位相关自检(已知位移)
    log("自检:相位相关精度(对静态帧做已知位移)")
    f_test = None
    for _, fr in iter_frames(ff, args.video, 0, chunk=1, t0=61.5, t1=61.6):
        f_test = fr.copy()
        break
    if f_test is not None:
        base = to_gray_half(f_test, vp, args.down)
        errs = []
        for (ty, tx) in ((0, 0), (1, 0), (0, 1), (3, -5), (-7, 4), (10, 10)):
            sh = np.roll(np.roll(base, ty, axis=0), tx, axis=1)
            dy, dx, pk = phase_shift(base, sh, win=win, hp=not args.no_hp)
            errs.append((ty, tx, dy, dx, pk))
            log(f"   真值(dy={ty:3d},dx={tx:3d}) -> 测得(dy={dy:+.2f},dx={dx:+.2f}) peak={pk:.3f}")
        selftest = [{"true_dy": a, "true_dx": b, "got_dy": c, "got_dx": d, "peak": e}
                    for a, b, c, d, e in errs]
    else:
        selftest = []

    # ---------------------------------------------------------- OCR 自举
    log("OCR 自举:解包 TIME 框字形并聚类定标")
    hud = score_ocr.ScoreHUD()
    hud.templates = {}          # 清空 PC 字体种子,全部从手机字体学
    time_box = score_ocr._BOX_X["time"]
    acc_box = score_ocr._BOX_X["acc"]
    pt_box = score_ocr._BOX_X["points"]

    imgs: list[np.ndarray | None] = [None] * n_total
    time_glyphs: list[list[np.ndarray] | None] = [None] * n_total
    acc_glyphs: list[list[np.ndarray] | None] = [None] * n_total
    n_time_seg: dict[int, int] = {}
    n_acc_seg: dict[int, int] = {}
    for k in range(n_total):
        m = unpack_mask(masks[k], mask_shape)
        img = mask_to_img(m)
        imgs[k] = img
        tg = hud._split_glyphs(img, *time_box)
        ag = hud._split_glyphs(img, *acc_box)
        time_glyphs[k] = tg
        acc_glyphs[k] = ag
        n_time_seg[len(tg)] = n_time_seg.get(len(tg), 0) + 1
        n_acc_seg[len(ag)] = n_acc_seg.get(len(ag), 0) + 1
    log(f"TIME 段数分布 = {dict(sorted(n_time_seg.items()))}")
    log(f"ACC  段数分布 = {dict(sorted(n_acc_seg.items()))}")
    pt_seg = {}
    for k in range(0, n_total, 7):
        n = len(hud._split_glyphs(imgs[k], *pt_box))
        pt_seg[n] = pt_seg.get(n, 0) + 1
    log(f"POINTS 段数分布(每7帧采样)= {dict(sorted(pt_seg.items()))}")

    # ------------------------------------------------------------------
    # TIME 文本分组(自举定标的核心)
    # 为什么不用「字形聚类」:手机帧里数字只有 ~10x14 px,`_split_glyphs` 会把 22 行
    # 数字带整体双线性缩放到 20x28,±1 px 的上下抖动被放大成很大的 MSE,同一个数字
    # 会散成几十个簇(实测 76 簇),靠簇无法定标。
    # 改用**原始掩码逐位比较**:同一秒内 HUD 静止,掩码几乎逐位相同;跨秒必然变化。
    # 于是「帧间掩码差异 > 阈值」处切段,每段 = 一个显示值;倒计时每段约 1 秒、
    # 末段 = 00:00,故从末段往前数即得每个显示段的秒值。此法不依赖任何字形归一化。
    # ------------------------------------------------------------------
    tb = time_box
    tflat = np.stack([unpack_mask(masks[k], mask_shape)[:, tb[0]:tb[1]].reshape(-1)
                      for k in range(n_total)])
    t_nonempty = tflat.any(axis=1)
    tdiff = np.zeros(n_total)
    tdiff[1:] = (tflat[1:] != tflat[:-1]).mean(axis=1)
    log("TIME 掩码帧间差异 d 的分位: p50=%.6f p90=%.6f p99=%.6f max=%.4f"
        % (np.median(tdiff), np.percentile(tdiff, 90), np.percentile(tdiff, 99), tdiff.max()))
    # 单帧 d 噪声很大(手机录像 H.264 把 >190 的边缘像素翻来翻去),没有干净的
    # 双峰可分。改用**梳状拟合**:倒计时严格每秒跳一次,所以变化帧必定落在
    # {t0+1, t0+2, ..., t0+59}(PTS 为墙钟时间)。只有 t0 一个自由度,
    # 用 Σ_j d(最近帧) 扫描 t0 即可;59 项求和把单帧噪声平均掉。
    dsm = np.maximum(np.maximum(tdiff, np.roll(tdiff, 1)), np.roll(tdiff, -1))
    nz = np.where(t_nonempty)[0]
    runs = []
    s = nz[0]
    for i in range(1, nz.size):
        if nz[i] - nz[i - 1] > 2:
            runs.append((s, nz[i - 1]))
            s = nz[i]
    runs.append((s, nz[-1]))
    ka, kb = max(runs, key=lambda r: r[1] - r[0])
    log(f"TIME 框可见区间(最长连续段)= 帧[{ka},{kb}] {gtimes[ka]:.2f}-{gtimes[kb]:.2f}s "
        f"({kb-ka+1} 帧); 段列表 {[(int(a),int(b)) for a,b in runs]}")

    def _k_at(t: float) -> int:
        k = int(np.searchsorted(gtimes, t))
        return min(max(k, 1), n_total - 1)

    lo = float(gtimes[ka])
    grid = np.arange(lo, lo + 8.0, 1.0 / 240.0)
    scores = np.empty(grid.size)
    for gi, t0 in enumerate(grid):
        acc = 0.0
        for j in range(1, 60):
            acc += dsm[_k_at(t0 + j)]
        scores[gi] = acc
    bi = int(np.argmax(scores))
    t0_hat = float(grid[bi])
    prom = float(scores[bi] / max(1e-12, np.percentile(scores, 95)))
    log(f"梳状拟合 t0(回合计时起点)= {t0_hat:.3f}s(相对视频), 得分 {scores[bi]:.4f},"
        f" 峰/95分位 = {prom:.3f}")
    log(f"   得分曲线 top5: " + ", ".join(
        f"{grid[i]:.3f}s:{scores[i]:.3f}" for i in np.argsort(-scores)[:5]))

    bnd = [ka] + [_k_at(t0_hat + j) for j in range(1, 60)] + [kb + 1]
    bnd = sorted(set(bnd))
    grp = [(bnd[i], bnd[i + 1] - 1) for i in range(len(bnd) - 1) if bnd[i + 1] - 1 >= bnd[i]]
    log(f"倒计时链 = {len(grp)} 段,{gtimes[grp[0][0]]:.2f}s -> {gtimes[grp[-1][1]]:.2f}s,"
        f" 时长 {gtimes[grp[-1][1]]-gtimes[grp[0][0]]:.2f}s")
    seg_lens = np.array([gtimes[b] - gtimes[a] for a, b in grp])
    log("   段时长: min=%.3f p50=%.3f max=%.3f s" % (seg_lens.min(), np.median(seg_lens),
                                                     seg_lens.max()))
    log("   段时长直方(ms): " + str({f"{a}-{b}": int(((seg_lens*1000 >= a) & (seg_lens*1000 < b)).sum())
                                     for a, b in ((0, 900), (900, 1100), (1100, 1400), (1400, 100000))}))

    val_of_frame = np.full(n_total, -1, dtype=int)
    k_t0 = _k_at(t0_hat)
    val_of_frame[ka:k_t0] = 60          # 开局前静止显示的 01:00(不属于回合)
    for i, (a, b) in enumerate(grp):
        lo_i = max(a, k_t0)
        if lo_i <= b:
            val_of_frame[lo_i:b + 1] = len(grp) - 1 - i  # 末段 = 0
    cd = [k for k in range(k_t0, grp[-1][1] + 1)]
    b0, b1 = k_t0, grp[-1][1]
    cd_start_val = len(grp) - 1
    cd_end_val = 0
    cd_span = float(gtimes[b1] - gtimes[b0])

    # 用模块自身的 calibrate_from_time 学模板(每段取 2 帧,段内 30%/70% 处)
    hud = score_ocr.ScoreHUD()
    hud.templates = {}          # 清空 PC 字体种子,全部从手机字体学
    learned_at = []
    for i, (a, b) in enumerate(grp):
        val = len(grp) - 1 - i
        for frac in (0.3, 0.7):
            k = a + int((b - a) * frac)
            if hud.calibrate_from_time(imgs[k], val):
                learned_at.append((round(float(gtimes[k]), 2), val))
    log(f"calibrate_from_time 学习 {len(learned_at)} 帧;模板库 = " +
        "{" + ", ".join(f"{c}:{len(v)}" for c, v in sorted(hud.templates.items())) + "}")

    # 用「段标签」当 TIME 的独立地面真值,测 OCR 在 TIME 上的逐帧准确率
    time_ok = time_bad = time_none = 0
    time_conf = []
    for k in cd:
        r = hud.read(imgs[k])
        got = r["time_s"]
        want = val_of_frame[k]
        if got is None:
            time_none += 1
        elif got == want:
            time_ok += 1
        else:
            time_bad += 1
            if len(time_conf) < 12:
                time_conf.append((round(float(gtimes[k]), 2), want, got))
    log(f"TIME OCR 校验(对照段标签): 对 {time_ok}, 错 {time_bad}, 未读出 {time_none} "
        f"-> 准确率 {time_ok/max(1,time_ok+time_bad):.4f}")
    if time_conf:
        log(f"   错读样例(视频时间, 期望, 读到) = {time_conf}")

    seq = [(k, gtimes[k], f"{val_of_frame[k]//60:02d}:{val_of_frame[k]%60:02d}")
           for k in cd]

    # 学 '%':ACC 末段聚类,取与数字模板最小 MSE 明显偏大的簇
    acc_last = [acc_glyphs[k][-1] for k in range(n_total)
                if acc_glyphs[k] and len(acc_glyphs[k]) >= 2]
    pct_note = "无 ACC 字形"
    if acc_last:
        labels, centers, cnt = cluster_glyphs(acc_last, tol=0.02)
        best = int(np.argmax(cnt))
        members = [g for g, l in zip(acc_last, labels) if l == best]
        # 取「到簇均值最近的成员」(medoid)而不是均值本身:字形抖动会把均值抹糊
        g = min(members, key=lambda x: float(np.mean((x - centers[best]) ** 2)))
        dmin, cmin = 1e9, None
        for c, samples in hud.templates.items():
            for t_ in samples:
                d = float(np.mean((g - t_) ** 2))
                if d < dmin:
                    dmin, cmin = d, c
        pct_note = (f"ACC 末段 {len(centers)} 簇,主体簇 n={cnt[best]}(medoid),"
                    f"与已有模板最近 MSE={dmin:.4f}(char={cmin})")
        log("  " + pct_note)
        if dmin > 0.05:
            hud.templates["%"] = [g]
            log("  -> 认作 '%' 模板注入")
        else:
            log("  -> 该簇其实是数字,未注入 '%'(可能 ACC 段被切开)")

    # ---------------------------------------------------------- 全片 HUD 读取
    tmpl_digits = sorted(c for c in hud.templates if c.isdigit())
    log(f"模板库数字 = {''.join(tmpl_digits)} (缺 "
        f"{''.join(c for c in '0123456789' if c not in tmpl_digits) or '无'})")
    log(f"HUD 读取:每 {args.stride} 帧一次")
    rows = []
    n_ok_p = n_ok_a = 0
    for k in range(n_total):
        if k % args.stride:
            continue
        r = hud.read(imgs[k])
        ok_p = r["points"] is not None
        ok_a = r["acc_pct"] is not None
        n_ok_p += ok_p
        n_ok_a += ok_a
        rows.append({"video_t_s": round(float(gtimes[k]), 4),
                     "frame": k,
                     "points": r["points"] if ok_p else "",
                     "acc_pct": r["acc_pct"] if ok_a else "",
                     "time_s": r["time_s"] if r["time_s"] is not None else "",
                     "ok_points": int(ok_p), "ok_acc": int(ok_a)})
        if len(rows) % 200 == 0:
            log(f"  HUD {len(rows)} 行,points 可读 {n_ok_p},acc 可读 {n_ok_a}")
    log(f"HUD 完成:{len(rows)} 行,points 可读 {n_ok_p}({n_ok_p/max(1,len(rows)):.1%}),"
        f"acc 可读 {n_ok_a}({n_ok_a/max(1,len(rows)):.1%})")
    log(f"hud.n_unknown(字符未认出次数)= {hud.n_unknown},n_reads={hud.n_reads}")

    # ---------------------------------------------------------- 写 CSV
    csv_p = os.path.join(OUTDIR, "hud_timeline.csv")
    with open(csv_p, "w", encoding="utf-8") as f:
        f.write("video_t_s,points,acc_pct,time_s,ok_points,ok_acc\n")
        for r in rows:
            f.write(f"{r['video_t_s']},{r['points']},{r['acc_pct']},"
                    f"{r['time_s']},{r['ok_points']},{r['ok_acc']}\n")
    log(f"写出 {csv_p} ({len(rows)} 行)")

    # ---------------------------------------------------------- OCR 审计蒙太奇
    try:
        sel = list(range(0, n_total, max(1, n_total // 20)))[:21]
        tiles = []
        for k in sel:
            segs = [imgs[k][:, score_ocr._BOX_X[nm][0]:score_ocr._BOX_X[nm][1]]
                    for nm in ("points", "time", "acc")]
            w = sum(s.shape[1] + 4 for s in segs)
            row = np.zeros((imgs[k].shape[0], w, 3), np.uint8)
            x = 0
            for s in segs:
                row[:, x:x + s.shape[1]] = s
                x += s.shape[1] + 4
            tiles.append(row)
        pad = max(t.shape[1] for t in tiles)
        mono = []
        for t in tiles:
            row = np.zeros((t.shape[0] + 2, pad, 3), np.uint8)
            row[:t.shape[0], :t.shape[1]] = t
            mono.append(row)
        save_png(np.vstack(mono), os.path.join(OUTDIR, "hud_montage_masks.png"), scale=2)
        log("  蒙太奇行序(每行 = 该帧的 POINTS|TIME|ACC 掩码): " +
            ", ".join(f"{gtimes[k]:.1f}s" for k in sel))
    except Exception as e:      # 审计图失败不影响测量
        log(f"  !! 蒙太奇生成失败: {e}")

    # ---------------------------------------------------------- 回合窗口
    # 回合起点 = 梳状拟合出的 t0(倒计时 60 -> 0 的计时起点);回合长 60s。
    round_start = float(t0_hat)
    round_end = float(t0_hat + 60.0)
    log(f"回合窗口 = [{round_start:.3f}, {round_end:.3f}] s(视频时间),长 60.000s;"
        f" 起点不确定度约 ±{meta.get('dt_median', 0.033):.3f}s")

    # ---------------------------------------------------------- Task B
    # 分数是「阶梯函数」:一次命中 -> 显示值跳一档。但 HUD 会做约 1 帧的数字滚动动画,
    # 于是偶尔出现「半档」中间值。所以按**平台(run)**而不是按逐帧差分来切事件:
    # 相同值合并成平台,相邻平台的差 = 一次事件;小数片段自然落在 round(x/unit)=0。

    def hist_of(a, edges):
        h, _ = np.histogram(a, bins=edges)
        return {f"{edges[i]}-{edges[i+1]}": int(h[i]) for i in range(len(h))}

    log("Task B:命中/失手分段(平台法)")
    valid = [(r["video_t_s"], r["points"]) for r in rows if r["ok_points"]]
    valid = [(t, p) for t, p in valid if round_start - 0.3 <= t <= round_end + 0.6]
    pts_series = np.array([p for _, p in valid], dtype=np.int64)
    t_series = np.array([t for t, _ in valid], dtype=np.float64)
    log(f"  有效 points 读 {len(pts_series)} 个;首={pts_series[0] if len(pts_series) else None} "
        f"末={pts_series[-1] if len(pts_series) else None} "
        f"max={pts_series.max() if len(pts_series) else None}")

    # (a) 稳健去毛刺:分数序列是「阶梯 + 小额扣分」,任何一帧内跨好几档的跳变都是 OCR
    #     错读。用「合理性 + 持续性」逐帧判定:不合理就沿用上一个可信值。
    raw_steps = np.diff(pts_series) if len(pts_series) > 1 else np.array([], dtype=np.int64)
    plaus_ref = raw_steps[(raw_steps > 100) & (raw_steps < 1000)]
    unit0 = int(np.median(plaus_ref)) if plaus_ref.size else 373
    plaus_max = int(6 * unit0)
    good = np.zeros(len(pts_series), dtype=bool)
    clean = pts_series.copy()
    good[0] = True
    last = int(pts_series[0])
    pend: list[int] = []
    glitch_idx = []
    for i in range(1, len(pts_series)):
        d = int(pts_series[i]) - last
        if -100 <= d <= plaus_max:
            good[i] = True
            last = int(pts_series[i])
            pend = []
        else:
            clean[i] = last
            glitch_idx.append(i)
            pend.append(i)
            # 连续 3 帧都是同一个「不合理」值 -> 分数确实跳了(或模板系统性错),接受它
            if len(pend) >= 3 and len({int(pts_series[j]) for j in pend[-3:]}) == 1:
                good[i] = True
                last = int(pts_series[i])
                pend = []
    log(f"  步进合理性上限 plaus_max = {plaus_max}(=6x 初估单位分 {unit0});"
        f" 判为 OCR 毛刺并沿用前值的帧 = {len(glitch_idx)}")
    log("  毛刺帧(视频时间, 原读值, 沿用值): " + ", ".join(
        f"({t_series[i]:.2f},{int(pts_series[i])}->{int(clean[i])})" for i in glitch_idx[:20]))

    # (b) 平台
    runs = []
    s = 0
    for i in range(1, len(clean)):
        if clean[i] != clean[i - 1]:
            runs.append((int(clean[i - 1]), s, i - 1))
            s = i
    runs.append((int(clean[-1]), s, len(clean) - 1))
    run_len = np.array([r[2] - r[1] + 1 for r in runs])
    steps = np.array([runs[i][0] - runs[i - 1][0] for i in range(1, len(runs))], dtype=np.int64)
    step_t = np.array([t_series[runs[i][1]] for i in range(1, len(runs))])
    step_len = run_len[1:]
    log(f"  平台数 = {len(runs)};平台长度分位 p50={np.median(run_len):.0f} 帧;"
        f" 步进数 = {len(steps)}")
    log("  步进取值直方(全部平台间)=" + json.dumps(hist_of(steps, [-10 ** 9, -1000, -100, -50,
        -30, -10, 0, 10, 50, 100, 150, 200, 250, 300, 340, 360, 370, 380, 390, 400, 430,
        10 ** 9]), ensure_ascii=False))

    pos = steps[steps > 0]
    neg = steps[steps < 0]
    # 单位分 = 正步进的众数(最密 10 分箱的中点)
    if pos.size:
        h, e = np.histogram(pos, bins=np.arange(0, pos.max() + 11, 10))
        pk = int(np.argmax(h))
        unit = int((e[pk] + e[pk + 1]) / 2)
        in_unit = pos[(pos >= 0.75 * unit) & (pos <= 1.25 * unit)]
        unit = int(np.median(in_unit)) if in_unit.size else unit
    else:
        unit = None
    log(f"  单位分 unit(正步进众数)= {unit}")

    # (c) 每个正步进折算命中数:round(step/unit);片段(<0.5 unit)自然为 0
    hits_total = 0
    per_hit_awards = []
    hit_times = []
    if unit:
        for i in range(len(steps)):
            if steps[i] > 0:
                n = int(round(steps[i] / float(unit)))
                hits_total += n
                if n == 1:
                    per_hit_awards.append(int(steps[i]))
                    hit_times.append(float(step_t[i]))
                elif n > 1:
                    # 一步含多次命中:均分时间戳
                    for j in range(n):
                        hit_times.append(float(step_t[i]) + j * 1e-3)
    per_hit_awards = np.array(per_hit_awards, dtype=np.int64)
    log(f"  推断命中总数 = {hits_total}(其中 {len(per_hit_awards)} 步 = 单次命中)")

    # (d) miss:负步进。以 HUD 的 ACC 交叉验证:miss 应为固定扣分
    neg_vals = {}
    for v in neg:
        neg_vals[int(v)] = neg_vals.get(int(v), 0) + 1
    miss_deduct = None
    if neg.size:
        cand = [v for v in neg if -60 < v < 0]
        if cand:
            miss_deduct = int(np.median(cand))
    n_miss = int(sum(1 for v in neg if v == miss_deduct)) if miss_deduct else 0
    log(f"  负步进取值 = {neg_vals};推定单次 miss 扣分 = {miss_deduct},miss 事件数 = {n_miss}")
    miss_times = [float(step_t[i]) for i in range(len(steps))
                  if steps[i] == (miss_deduct or 0)]
    log(f"  miss 时间戳数 = {len(miss_times)}")

    # 直方图
    per_hit_stats = {}
    if per_hit_awards.size:
        per_hit_stats = {
            "n": int(per_hit_awards.size),
            "min": int(per_hit_awards.min()), "p25": float(np.percentile(per_hit_awards, 25)),
            "p50": float(np.percentile(per_hit_awards, 50)),
            "p75": float(np.percentile(per_hit_awards, 75)),
            "max": int(per_hit_awards.max()), "mean": float(per_hit_awards.mean()),
            "std": float(per_hit_awards.std()),
            "hist": hist_of(per_hit_awards, list(range(0, 501, 20))),
        }
    log(f"  单次命中得分分布 = {json.dumps(per_hit_stats, ensure_ascii=False)}")

    # 每球得分是否与「距上一球的间隔」相关(时间加权模型检验)
    # 用**平台步进**的间隔,而不是合成出来的命中时间戳(多命中步进会被塞进 1ms 假间隔)
    tw = None
    if unit:
        single_mask = (steps >= int(0.75 * unit)) & (steps <= int(1.25 * unit))
        if int(single_mask.sum()) > 20:
            idxs = np.where(single_mask)[0]
            iv = np.diff(step_t[idxs])
            aw = steps[idxs[1:]]
            keep = iv < 1.0
            if int(keep.sum()) > 15:
                cc = float(np.corrcoef(iv[keep], aw[keep])[0, 1])
                tw = {"n_pairs": int(keep.sum()), "pearson_r_interval_vs_award": cc,
                      "interval_p10_s": float(np.percentile(iv[keep], 10)),
                      "interval_p50_s": float(np.percentile(iv[keep], 50)),
                      "interval_p90_s": float(np.percentile(iv[keep], 90)),
                      "award_std": float(aw[keep].std()),
                      "award_range_p5_p95": [float(np.percentile(aw[keep], 5)),
                                             float(np.percentile(aw[keep], 95))]}
                log(f"  每球得分 vs 上一球间隔: r={cc:+.3f}(n={int(keep.sum())});"
                    f" 间隔 p10/p50/p90 = {tw['interval_p10_s']:.3f}/{tw['interval_p50_s']:.3f}/"
                    f"{tw['interval_p90_s']:.3f}s;得分 p5-p95 = "
                    f"{tw['award_range_p5_p95'][0]:.0f}-{tw['award_range_p5_p95'][1]:.0f}")

    # 命中率随时间(5s 分箱)
    buckets = []
    if hit_times:
        edges = np.arange(round_start, round_end + 5.0, 5.0)
        cnt, _ = np.histogram(np.array(hit_times), bins=edges)
        for i in range(len(cnt)):
            buckets.append({"t0": round(float(edges[i]), 2),
                            "t1": round(float(edges[i + 1]), 2),
                            "hits": int(cnt[i]),
                            "hits_per_s": round(float(cnt[i]) / (edges[i + 1] - edges[i]), 3)})
        log("  命中率(5s 箱): " + ", ".join(f"{b['t0']:.0f}-{b['t1']:.0f}s:{b['hits']}"
                                            for b in buckets))
        xs = np.array(sorted(hit_times))
        if len(xs) > 10:
            slope = float(np.polyfit(xs - round_start, np.arange(len(xs)), 1)[0])
            half = len(xs) // 2
            r1 = half / (xs[half] - round_start)
            r2 = (len(xs) - half) / (xs[-1] - xs[half])
            log(f"  累计命中斜率 = {slope:.2f} hits/s(全程);前半 {r1:.2f} vs 后半 {r2:.2f} hits/s")

    # 分数与精度终点
    final_points = int(pts_series[-1]) if len(pts_series) else None
    max_points = int(pts_series.max()) if len(pts_series) else None
    accs = [(r["video_t_s"], r["acc_pct"]) for r in rows if r["ok_acc"]]
    if round_start is not None:
        accs_r = [(t, a) for t, a in accs if round_start - 0.2 <= t <= round_end + 0.5]
    else:
        accs_r = accs
    final_acc = int(accs_r[-1][1]) if accs_r else None
    acc_vals = np.array([a for _, a in accs_r], dtype=np.int64)
    log(f"  终点 points(回合内最后一次有效读)= {final_points};回合内最大 = {max_points}")
    log(f"  ACC 读 {len(accs_r)} 个;取值分布 = "
        f"{np.unique(acc_vals, return_counts=True) if acc_vals.size else 'n/a'}")
    # 独立交叉验证:ACC = hits/(hits+misses) 应与 HUD 的 ACC 一致
    acc_from_events = None
    if hits_total and hits_total + n_miss > 0:
        acc_from_events = 100.0 * hits_total / (hits_total + n_miss)
        log(f"  交叉验证:hits={hits_total}, miss={n_miss} -> ACC={acc_from_events:.2f}%"
            f" vs HUD 显示 {final_acc}%")

    # ---------------------------------------------------------- Task C
    log("Task C:运动节奏")
    have_motion = bool(np.any(np.isfinite(shifts[:n_total])))
    if not have_motion:
        log("  !! 本次运行未计算位移(--skip-motion),Task C 跳过")
    # 静帧噪声地板:优先用「TIME 框已消失」的帧(结算画面 = 相机确定静止),
    # 不够再用回合结束后 1 秒。
    static_m = (~t_nonempty) & np.isfinite(shifts[:n_total])
    static_src = "TIME 框消失(结算画面)"
    if int(static_m.sum()) < 8:
        lo2 = min(float(gtimes[n_total - 1]), round_end + 1.0)
        static_m = (gtimes[:n_total] >= lo2) & np.isfinite(shifts[:n_total])
        static_src = f"t>{lo2:.2f}s(回合结束后)"
    noise_tail = shifts[:n_total][static_m]
    med_tail = float(np.median(noise_tail)) if noise_tail.size else float("nan")
    if have_motion:
        log(f"  静帧噪声地板({static_src}, {noise_tail.size} 帧):中位={med_tail:.3f}px "
            f"p90={(np.percentile(noise_tail,90) if noise_tail.size else float('nan')):.3f} "
            f"max={(noise_tail.max() if noise_tail.size else float('nan')):.3f}")

    def motion_metrics(thr_px, lo, hi, series=None):
        ser = shifts[:n_total] if series is None else series
        m = (gtimes[:n_total] >= lo) & (gtimes[:n_total] <= hi) & np.isfinite(ser)
        s = ser[m]
        if s.size == 0:
            return None
        mv = s > thr_px
        duty = float(mv.mean())
        dts = np.diff(gtimes[:n_total][m])
        dtm = float(np.median(dts)) if dts.size else 1.0 / FPS_NOMINAL
        runs = []
        i = 0
        while i < len(mv):
            if not mv[i]:
                j = i
                while j + 1 < len(mv) and not mv[j + 1]:
                    j += 1
                if i > 0 and j < len(mv) - 1:      # 只统计两端都被运动包围的完整停留
                    runs.append((j - i + 1) * dtm)
                i = j + 1
            else:
                i += 1
        runs = np.array(runs) if runs else np.array([])
        return {"thr_px": thr_px, "n_frames": int(s.size), "duty": duty,
                "n_stationary_runs": int(runs.size),
                "dwell_p50_ms": float(np.median(runs) * 1000) if runs.size else None,
                "dwell_p90_ms": float(np.percentile(runs, 90) * 1000) if runs.size else None,
                "dwell_mean_ms": float(runs.mean() * 1000) if runs.size else None,
                "dwell_max_ms": float(runs.max() * 1000) if runs.size else None,
                "dwell_hist_ms": {f"{a}-{b}": int(((runs * 1000 >= a) & (runs * 1000 < b)).sum())
                                  for a, b in ((0, 40), (40, 70), (70, 100), (100, 160),
                                               (160, 250), (250, 400), (400, 10000))}
                if runs.size else {},
                "shift_p50_px": float(np.median(s)), "shift_p90_px": float(np.percentile(s, 90)),
                "shift_mean_px": float(s.mean())}

    px_per_deg = (vp[1] - vp[0]) / FOV_H_DEG
    log(f"  角尺度:{vp[1]-vp[0]} px 视口 / {FOV_H_DEG}° = {px_per_deg:.4f} px/deg"
        f" (即 {1/px_per_deg:.4f} °/px)")
    # 阈值:噪声地板中位的若干倍,且对应一个物理角速率
    thr0 = max(3.0 * med_tail, 0.25 * px_per_deg) if np.isfinite(med_tail) else 0.25 * px_per_deg
    log(f"  选定阈值 thr0 = {thr0:.3f}px = {thr0/px_per_deg:.4f}°/帧 = "
        f"{thr0/px_per_deg*FPS_NOMINAL:.3f}°/s")
    sens = []
    for fac in (0.25, 0.5, 1.0, 2.0, 4.0):
        mm = motion_metrics(thr0 * fac, round_start, round_end)
        if mm:
            mm["factor"] = fac
            sens.append(mm)
            log(f"    x{fac:<4} thr={thr0*fac:6.3f}px -> duty={mm['duty']:.3f} "
                f"dwell_p50={None if mm['dwell_p50_ms'] is None else round(mm['dwell_p50_ms'],1)}ms "
                f"runs={mm['n_stationary_runs']}")
    main_m = next((m for m in sens if abs(m["factor"] - 1.0) < 1e-9), None)

    # 平滑稳健性检验:位移序列做 k 帧中值滤波后再判。若 duty 大幅下降,说明「运动」里
    # 有相当一部分是单帧噪声(靶子生灭 / 命中特效造成的伪位移),dwell 也就不可信。
    smooth_sens = []
    for w in (3, 5):
        pad = w // 2
        sp = np.pad(shifts[:n_total], (pad, pad), mode="edge")
        sm = np.array([np.nanmedian(sp[i:i + w]) for i in range(n_total)])
        mm = motion_metrics(thr0, round_start, round_end, series=sm)
        if mm:
            mm["median_window"] = w
            smooth_sens.append(mm)
            log(f"   中值滤波 w={w} @thr0 -> duty={mm['duty']:.3f} "
                f"dwell_p50={None if mm['dwell_p50_ms'] is None else round(mm['dwell_p50_ms'],1)}ms "
                f"runs={mm['n_stationary_runs']}")

    shift_all = shifts[:n_total]
    csv_m = os.path.join(OUTDIR, "motion_timeline.csv")
    with open(csv_m, "w", encoding="utf-8") as f:
        f.write("video_t_s,shift_px,shift_deg,is_moving,in_round\n")
        for k in range(n_total):
            s = shift_all[k]
            ss = "" if np.isnan(s) else f"{s:.4f}"
            sd = "" if np.isnan(s) else f"{s/px_per_deg:.5f}"
            mv = "" if np.isnan(s) else int(s > thr0)
            inr = int(round_start <= gtimes[k] <= round_end)
            f.write(f"{gtimes[k]:.4f},{ss},{sd},{mv},{inr}\n")
    log(f"写出 {csv_m} ({n_total} 行)")

    # 全片与回合内的位移分位
    def q(a, ps=(5, 10, 25, 50, 75, 90, 95, 99)):
        return {f"p{p}": float(np.nanpercentile(a, p)) for p in ps}

    sel_round = (gtimes[:n_total] >= round_start) & (gtimes[:n_total] <= round_end)
    shift_round = shift_all[sel_round]
    # 位移分布形状(判断阈值法是否可靠的关键证据)
    edges_h = [0, 0.25, 0.5, 1, 1.5, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 1e9]
    hh, _ = np.histogram(shift_round[np.isfinite(shift_round)], bins=edges_h)
    shift_hist = {f"{edges_h[i]:g}-{edges_h[i+1]:g}px": int(hh[i]) for i in range(len(hh))}
    log("  回合内位移直方(px): " + json.dumps(shift_hist, ensure_ascii=False))

    # ---------------------------------------------------------- 汇总
    summary = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "video": args.video,
        "video_meta": meta,
        "processing": {"stage": args.stage, "stride": args.stride,
                       "phase_corr_downsample": args.down,
                       "phase_corr_highpass": (not args.no_hp),
                       "frames_decoded": n_total,
                       "full_frame_rate": args.stride == 1},
        "measured_hud_layout": {
            "letterbox": {"left_x": layout["letterbox_left_x"],
                          "right_x": layout["letterbox_right_x"],
                          "viewport_w": layout["viewport"][1] - layout["viewport"][0]},
            "digit_band_y": layout["digit_band_y"],
            "boxes_tight_xy": layout["boxes_tight_xy"],
            "ocr_patch": hud_patch,
            "white_pixel_hist": layout["white_pixel_hist"],
            "max_channel_percentiles": layout["max_channel_percentiles"],
        },
        "ocr": {
            "segment_count_hist": {"time": {str(k): v for k, v in n_time_seg.items()},
                                   "acc": {str(k): v for k, v in n_acc_seg.items()},
                                   "points_sampled": {str(k): v for k, v in pt_seg.items()}},
            "countdown_block": {"frame0": int(b0), "frame1": int(b1),
                                "t0": round(float(gtimes[b0]), 3),
                                "t1": round(float(gtimes[b1]), 3),
                                "n_frames": len(cd)},
            "countdown_decode_frames": len(seq),
            "countdown_label_source": "TIME 框掩码帧间变化 + 每秒一次的梳状拟合(PTS 墙钟)",
            "countdown_round_start_t0_s": t0_hat,
            "countdown_comb_peak_ratio": prom,
            "countdown_n_groups": len(grp),
            "countdown_group_durations_s": [round(float(x), 3) for x in seg_lens],
            "countdown_first_seconds": cd_start_val,
            "countdown_last_seconds": cd_end_val,
            "countdown_span_s": cd_span,
            "time_ocr_check": {"correct": int(time_ok), "wrong": int(time_bad),
                               "unreadable": int(time_none),
                               "accuracy": float(time_ok / max(1, time_ok + time_bad)),
                               "wrong_examples": time_conf},
            "text_group_spans_s": [[round(float(gtimes[a]), 2), round(float(gtimes[b]), 2)]
                                   for a, b in grp],
            "templates_learned": {c: len(v) for c, v in hud.templates.items()},
            "calibrate_samples": learned_at,
            "percent_template_note": pct_note,
            "n_unknown_chars": int(hud.n_unknown),
        },
        "hud_read": {
            "rows": len(rows),
            "points_readable": int(n_ok_p), "acc_readable": int(n_ok_a),
            "points_rejected": int(len(rows) - n_ok_p),
            "acc_rejected": int(len(rows) - n_ok_a),
            "final_points": final_points,
            "max_points_in_round": max_points,
            "final_acc_pct": final_acc,
            "acc_value_distribution": (
                {int(v): int(c) for v, c in zip(*np.unique(acc_vals, return_counts=True))}
                if acc_vals.size else {}),
            "round_start_video_t_s": round_start,
            "round_end_video_t_s": round_end,
            "round_duration_s": round(round_end - round_start, 3),
            "round_start_uncertainty_s": 0.05,
            "round_start_source": "倒计时梳状拟合 t0(与 TIME OCR 观测到的 60->59 切换一致到 ~50ms)",
        },
        "scoring": {
            "n_valid_point_reads": int(len(pts_series)),
            "n_plateaus": int(len(runs)),
            "n_steps": int(len(steps)),
            "n_glitch_frames_removed": int(len(glitch_idx)),
            "step_value_hist": hist_of(steps, [-10 ** 9, -1000, -100, -50, -30, -10, 0, 10,
                                               50, 100, 150, 200, 250, 300, 340, 360, 370,
                                               380, 390, 400, 430, 10 ** 9]),
            "negative_step_values": neg_vals,
            "unit_points": unit,
            "inferred_hits": int(hits_total),
            "inferred_miss_events": int(n_miss),
            "miss_deduction_points": miss_deduct,
            "total_deducted_points": int(n_miss * miss_deduct) if miss_deduct else 0,
            "per_hit_points": per_hit_stats,
            "mean_points_per_hit": (float(final_points / hits_total)
                                    if (final_points and hits_total) else None),
            "total_points": final_points,
            "acc_cross_check_pct": acc_from_events,
            "acc_hud_pct": final_acc,
            "time_weighted_model_test": tw,
            "hit_rate_buckets_5s": buckets,
            "hit_rate_overall_hits_per_s": (
                float(hits_total / (round_end - round_start))
                if (round_start and round_end and round_end > round_start) else None),
        },
        "motion": {
            "px_per_deg_assumed": px_per_deg,
            "fov_h_deg_assumed": FOV_H_DEG,
            "render_width_px_assumed": vp[1] - vp[0],
            "noise_floor_static_tail": {
                "static_source": static_src,
                "n_frames": int(noise_tail.size),
                "median_px": med_tail,
                "p90_px": float(np.percentile(noise_tail, 90)) if noise_tail.size else None,
                "max_px": float(noise_tail.max()) if noise_tail.size else None,
                "median_deg": med_tail / px_per_deg if np.isfinite(med_tail) else None},
            "phase_corr_selftest": selftest,
            "shift_px_all_video": q(shift_all),
            "shift_px_round_only": q(shift_round),
            "shift_deg_round_only": {k: v / px_per_deg for k, v in q(shift_round).items()},
            "shift_hist_round_px": shift_hist,
            "threshold_px_chosen": thr0,
            "threshold_deg_chosen": thr0 / px_per_deg,
            "threshold_justification":
                f"3x 静帧噪声地板中位({3*med_tail:.3f}px) 与 0.25px/deg({0.25*px_per_deg:.3f}px) 取大",
            "primary": main_m,
            "sensitivity": sens,
            "smoothing_robustness": smooth_sens,
        },
    }
    sp = os.path.join(OUTDIR, "hud_summary.json")
    with open(sp, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=_json_default)
    log(f"写出 {sp}")
    mp = os.path.join(OUTDIR, "motion_summary.json")
    with open(mp, "w", encoding="utf-8") as f:
        json.dump(summary["motion"] | {"round_window": [round_start, round_end]},
                  f, ensure_ascii=False, indent=2, default=_json_default)
    log(f"写出 {mp}")

    log("=== 关键结果 ===")
    log(f"  终局分数 = {final_points}(回合内最大 {max_points}) / ACC = {final_acc}%")
    log(f"  命中 = {hits_total},每球均分 = "
        f"{summary['scoring']['mean_points_per_hit']},miss 事件 = {int(n_miss)}"
        f"(每次 {miss_deduct} 分)")
    if main_m:
        log(f"  duty = {main_m['duty']:.4f}, dwell p50 = {main_m['dwell_p50_ms']} ms,"
            f" p90 = {main_m['dwell_p90_ms']} ms")
    return 0


if __name__ == "__main__":
    sys.exit(main())
