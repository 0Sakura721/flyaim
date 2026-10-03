"""进程内实时渲染(训练/评估边跑边画窗口,不落盘、不开子进程)。

性能与窗口行为的关键设计
------------------------
1. **绝不调用 `plt.pause()`** —— 它内部会 `canvas.draw()`(完整重绘)+
   `start_event_loop()`(Tk `update()` 会把窗口 **raise 到最前**)+
   `time.sleep()`(Windows 计时器 15 ms 粒度,每帧白等)。
   一条语句同时造成"抢焦点"和"卡"。改用 `draw_idle()` + `update_idletasks()`:
   后者**只重绘、不 raise 窗口、不阻塞**。

2. **绝不调用 `fig.show()` / `window.lift()`** —— matplotlib 的
   `FigureManagerTk.show()` 会 `deiconify()` + `lift()` + 短暂 `topmost`,
   这就是"每次刷新都跑到最前面"的直接原因。改为只 `deiconify()`,
   并显式 `attributes('-topmost', False)`,**绝不 `lift()`**。

3. **降采样渲染面积** —— DN 时序矩阵若按 1360×N 画,每帧 50 万像素是大头。
   默认降到 256 行 × 256 列(6.5 万像素,快 8 倍)。

4. **刷新率可配** —— 脑仿真一帧 ~300 ms,渲染无需每帧都做。
   `refresh_every=N` 表示每 N 帧才重绘一次。

用法::

    from tools.live_render import make_renderer

    live = make_renderer(roles, idx, window_title="FlyAim",
                         figsize=(14, 9), refresh_every=2)
    for i in range(n_frames):
        ... 跑一步 ...
        live.update(frame=frame, spikes=brain.spikes,
                    rates=brain.rates, meta={"target_dist": d, "R2": r2})
    live.close()
"""

from __future__ import annotations

import sys
import time

import numpy as np

_LAYER_PRIORITY = ("ol_sensory", "ol_intrinsic", "visual_projection", "visual_centrifugal",
                   "descending_neuron", "vnc_motor")


def _say(msg: str) -> None:
    """诊断信息一律写 stderr 并 flush(stdout 重定向后是块缓冲,会卡住线索)。"""
    print(msg, file=sys.stderr, flush=True)


class _NullRenderer:
    """渲染不可用时的降级实现(接口一致,什么都不做)。"""

    is_live = False

    def update(self, **kw):
        return None

    def close(self):
        return None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class LiveRenderer:
    """把脑活动 + 靶场状态直接画成一个 2×2 的实时窗口。"""

    is_live = True

    def __init__(self, roles, neuron_index=None, *, sample_shape=(96, 96),
                 figsize=(13.6, 8.2), refresh_every: int = 2,
                 window_title: str = "FlyAim", history: int = 256,
                 dn_rows: int = 256, enabled: bool = True):
        self.enabled = bool(enabled)
        self.sample_shape = tuple(int(x) for x in sample_shape)
        self.refresh_every = max(1, int(refresh_every))
        self.history = max(16, int(history))
        self.dn_rows = max(16, int(dn_rows))
        self.figsize = (float(figsize[0]), float(figsize[1]))
        self.n_frames = 0
        self._last_draw = 0.0

        # --- 下采样方案(全网热图) ---
        df = (neuron_index.df if hasattr(neuron_index, "df") else neuron_index)
        n = int(df.shape[0]) if df is not None else 0
        k = int(np.prod(self.sample_shape))
        rng = np.random.default_rng(20261003)
        if n <= k:
            self.sample_idx = np.arange(n, dtype=np.int64)
        else:
            self.sample_idx = np.sort(rng.choice(n, size=k, replace=False).astype(np.int64))
        self.dn_idx = np.asarray(roles.descending, np.int64)

        sup = None
        if df is not None and "superclass" in getattr(df, "columns", []):
            sup = df["superclass"].astype(str).to_numpy()[self.sample_idx]
        self.order, self.bounds, self.names = self._build_order(self.sample_idx, sup)

        # DN 行降采样(1360 -> dn_rows),显著降低 DN 矩阵的渲染成本
        if self.dn_idx.size > self.dn_rows:
            sel = (np.arange(self.dn_rows) * (self.dn_idx.size // self.dn_rows)).astype(np.int64)
            self.dn_idx = self.dn_idx[np.sort(sel)]

        self._init_plot(window_title)

    # ---------------------------------------------------------------- 绘图初始化

    def _init_plot(self, title: str) -> None:
        import matplotlib
        try:
            matplotlib.use("TkAgg", force=False)
        except Exception:
            pass
        # Windows 上 DejaVu Sans 无 CJK 字形,中文标签会变方框。
        self._zh = True
        try:
            matplotlib.rcParams["font.sans-serif"] = [
                "Microsoft YaHei", "SimHei", "SimSun", "Noto Sans CJK SC",
                "Source Han Sans SC", "DejaVu Sans",
            ]
            matplotlib.rcParams["font.family"] = "sans-serif"
            matplotlib.rcParams["axes.unicode_minus"] = False
        except Exception:
            self._zh = False

        def T(zh: str, en: str) -> str:
            return zh if self._zh else en

        import matplotlib.pyplot as plt

        self._plt = plt
        self.T = T

        # constrained_layout:拖拽窗口时 axes 自适应
        self.fig = plt.figure(figsize=self.figsize, layout="constrained")
        self._busy = False

        gs = self.fig.add_gridspec(2, 2)

        self.ax_prev = self.fig.add_subplot(gs[0, 0])
        self.ax_rast = self.fig.add_subplot(gs[0, 1])
        self.ax_dn = self.fig.add_subplot(gs[1, 0])
        self.ax_met = self.fig.add_subplot(gs[1, 1])

        self.ax_prev.set_title(T("靶场画面", "Arena"), fontsize=10)
        self.ax_prev.set_xticks([])
        self.ax_prev.set_yticks([])
        self.prev_img = self.ax_prev.imshow(np.zeros((48, 64, 3), np.uint8),
                                           interpolation="nearest")

        self.ax_rast.set_title(T("全网活动热图(按解剖层次分块)",
                                 "Network activity (by layer)"), fontsize=10)
        self.rast_img = self.ax_rast.imshow(
            np.zeros(self.sample_shape, np.float32),
            aspect="auto", cmap="inferno", interpolation="nearest")
        self.ax_rast.set_xticks([])
        self.ax_rast.set_yticks([])
        self.ax_rast.set_ylabel(T("神经元(层序)", "neurons (layer order)"), fontsize=9)
        for bi, b in enumerate(self.bounds):
            self.ax_rast.axhline(b, color="w", lw=0.7, alpha=0.5)
            if bi < len(self.names) and b > 0:
                y = (0 if bi == 0 else self.bounds[bi - 1] + b) / 2.0
                self.ax_rast.text(1.5, y, self.names[bi], color="w", fontsize=6.5,
                                  va="center", ha="left", alpha=0.85)
        if self.bounds:
            self.ax_rast.set_ylim(self.bounds[-1], 0)

        self.ax_dn.set_title(T("下行神经元放电率(Hz) — 滚动时序",
                               "DN rate (Hz) — rolling"), fontsize=10)
        self.ax_dn.set_xlabel(T("帧", "frame"), fontsize=9)
        self.ax_dn.set_ylabel(T("DN 索引", "DN index"), fontsize=9)
        self.dn_img = self.ax_dn.imshow(np.zeros((1, 1), np.float32), aspect="auto",
                                        cmap="magma", interpolation="nearest")

        self.ax_met.set_title(T("指标曲线", "Metrics"), fontsize=10)
        self.ax_met.set_xlabel(T("帧", "frame"), fontsize=9)
        self.ax_met.set_ylabel(T("靶距(px) / spike_rate", "dist (px) / spike_rate"),
                               fontsize=9)
        (self.line_dist,) = self.ax_met.plot([], [], lw=1.8, color="#1f77b4")
        (self.line_sr,) = self.ax_met.plot([], [], lw=1.2, color="#7f7f7f", alpha=0.85)
        self.ax_r2 = self.ax_met.twinx()
        self.ax_r2.set_ylabel("R²", fontsize=9)
        (self.line_r2,) = self.ax_r2.plot([], [], lw=1.8, color="#d62728", ls="--")
        # 不用 legend(每帧重绘图例很贵),把含义写进小标题
        self.ax_met.text(0.01, 0.98,
                         T("蓝=靶距  灰=spike×1000  红虚=R²",
                           "blue=dist  grey=spike x1000  red dash=R2"),
                         transform=self.ax_met.transAxes, fontsize=7.5,
                         va="top", ha="left", alpha=0.75)

        self.fig.suptitle(title, fontsize=11)

        # ---- 显示窗口但**不 raise**(不调用 fig.show / lift) ----
        self._tkwin = None
        try:
            self._tkwin = self.fig.canvas.get_tk_widget().winfo_toplevel()
            self._tkwin.title(title)
            self._tkwin.attributes("-topmost", False)
            self._tkwin.deiconify()
        except Exception:
            self._tkwin = None

        self.fig.canvas.draw()
        self._flush()

    # ---------------------------------------------------------------- 排序

    @staticmethod
    def _build_order(sample_idx, labels):
        if labels is None:
            return np.arange(len(sample_idx)), [], []
        lab = np.asarray([str(x) for x in labels])
        present = [c for c in _LAYER_PRIORITY if np.any(lab == c)]
        present += sorted({*set(lab)} - set(present))
        parts, bounds, cur = [], [], 0
        for c in present:
            sel = np.flatnonzero(lab == c)
            if sel.size == 0:
                continue
            parts.append(sel)
            cur += sel.size
            bounds.append(cur)
        order = np.concatenate(parts) if parts else np.arange(lab.size)
        return order, bounds, present

    # ---------------------------------------------------------------- 刷新

    def _flush(self) -> None:
        """重绘但**不 raise 窗口、不阻塞**。

        关键:用 `update_idletasks()` 而不是 `plt.pause()` / `update()`。
        后者会进入 Tk 事件循环并 raise 窗口;前者只执行 idle 重绘回调。
        """
        self.fig.canvas.draw_idle()
        if self._tkwin is not None:
            try:
                self._tkwin.update_idletasks()
            except Exception:
                pass
        else:
            try:
                self.fig.canvas.flush_events()
            except Exception:
                pass

    # ---------------------------------------------------------------- 每帧

    def update(self, *, frame=None, spikes=None, rates=None, meta=None) -> None:
        """每帧调用一次。内部按 refresh_every 决定是否真的重绘。"""
        try:
            self._update(frame, spikes, rates, meta or {})
        except Exception:
            self.enabled = False

    def _update(self, frame, spikes, rates, meta) -> None:
        if not self.enabled:
            return
        self.n_frames += 1

        # ---- 窗口是否还在 ----
        if self._tkwin is not None:
            try:
                if not self._tkwin.winfo_exists():
                    self.enabled = False
                    return
            except Exception:
                pass

        # ---- 收集数据(每次都收,保证曲线连续) ----
        i = self.n_frames
        rt = None if rates is None else np.asarray(rates).ravel()
        sp = None if spikes is None else np.asarray(spikes).ravel()

        self._xs = np.append(getattr(self, "_xs", np.empty(0)), i)
        self._dist = np.append(getattr(self, "_dist", np.empty(0)),
                               float(meta.get("target_dist", np.nan)))
        self._sr = np.append(getattr(self, "_sr", np.empty(0)),
                             float(sp.mean()) if sp is not None else np.nan)
        r2 = meta.get("r2")
        if r2 is not None and np.isfinite(r2):
            self._rx = np.append(getattr(self, "_rx", np.empty(0)), i)
            self._ry = np.append(getattr(self, "_ry", np.empty(0)), float(r2))

        if rt is not None and self.dn_idx.size and rt.size > int(self.dn_idx.max()):
            row = rt[self.dn_idx][None, :]
            self._dn_buf = row if getattr(self, "_dn_buf", None) is None else \
                np.concatenate([self._dn_buf, row], axis=0)[-self.history:]

        # ---- 只在该刷新时才动 artist + 重绘 ----
        if (self.n_frames % self.refresh_every) != 0:
            return

        if frame is not None:
            self.prev_img.set_data(np.asarray(frame))

        if rt is not None and rt.size > self.sample_idx.size:
            full = rt[self.order].reshape(self.sample_shape)
            self.rast_img.set_data(full)
            self.rast_img.set_clim(0.0, max(float(full.max()), 1e-3))

        buf = getattr(self, "_dn_buf", None)
        if buf is not None and buf.size:
            self.dn_img.set_data(buf)
            self.dn_img.set_extent((max(0, i - buf.shape[0]), i, buf.shape[0], 0))
            self.dn_img.set_clim(0.0, max(float(buf.max()), 1e-3))

        self.line_dist.set_data(self._xs, self._dist)
        self.line_sr.set_data(self._xs, self._sr * 1000.0)
        ry = getattr(self, "_ry", np.empty(0))
        if ry.size:
            self.line_r2.set_data(self._rx, ry)
            self.ax_r2.set_ylim(min(0.0, float(ry.min()) - 0.05),
                                max(1.0, float(ry.max()) + 0.05))
            self.ax_r2.set_xlim(0, max(50, i))
        self.ax_met.set_xlim(0, max(50, i))
        ok = np.isfinite(self._dist)
        if ok.any():
            self.ax_met.set_ylim(0.0, max(float(self._dist[ok].max()) * 1.15, 40.0))

        # ---- 标题(便宜) ----
        parts = [f"#{i}"]
        for k in ("target_dist", "hit_rate", "hits"):
            v = meta.get(k)
            if isinstance(v, (int, float)) and np.isfinite(v):
                parts.append(f"{k}={v:.2f}" if isinstance(v, float) else f"{k}={v}")
        if sp is not None:
            parts.append(f"spike={sp.mean():.4f}")
        if rt is not None:
            parts.append(f"active={int((rt > 0).sum())}")
        if r2 is not None and np.isfinite(r2):
            parts.append(f"R2={r2:.3f}")
        if "n_samples" in meta:
            parts.append(f"n={meta['n_samples']}")
        try:
            self.fig.suptitle("   ".join(parts), fontsize=10)
        except Exception:
            pass

        self._flush()

    # ---------------------------------------------------------------- 收尾

    def close(self) -> None:
        try:
            self._plt.close(self.fig)
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


def make_renderer(roles, neuron_index=None, **kw):
    """构造 LiveRenderer;渲染不可用时返回 no-op(不抛异常)。

    诊断写 stderr 并 flush —— stdout 重定向后是块缓冲,
    会把失败原因卡在缓冲区里,让人误以为"窗口没反应"。
    """
    try:
        r = LiveRenderer(roles, neuron_index, **kw)
    except Exception:
        import traceback

        _say("[live] 渲染不可用,降级为无窗口模式")
        traceback.print_exc(file=sys.stderr)
        sys.stderr.flush()
        return _NullRenderer()

    import matplotlib

    title = ""
    try:
        title = (r._tkwin.title() if r._tkwin is not None
                 else r.fig.canvas.manager.get_window_title())
    except Exception:
        pass
    _say(f"[live] OK 渲染窗口已创建  backend={matplotlib.get_backend()}  "
         f"title={title!r}  figsize={r.figsize}  refresh_every={r.refresh_every}")
    _say("[live]    窗口不会自动置顶;若仍跑到最前,说明有别的进程在 raise 它")
    return r
