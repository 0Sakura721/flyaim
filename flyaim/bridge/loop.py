"""桥接闭环(BridgeLoop):捕获线程 + 网络拍 + 注入 + 遥测 + 延迟记账。

===============================================================================
线程模型(为什么捕获在别的线程)
===============================================================================
网络的拍频(~4-10 Hz)远低于屏幕刷新(60-240 Hz)。若在同一线程里
「截屏 -> 仿真 -> 注入」,每 tick 的画面年龄 = 截屏耗时 + 仿真耗时,
延迟直接翻倍。因此:

    捕获线程(高频):  source.read() -> 最新帧槽(单槽,旧的直接被覆盖)
    主线程(网络拍):  取最新帧 -> controller.act -> 增益 -> sink.send -> 遥测

画面年龄(age)单独记账进 summary —— 这是接入真实游戏后最要紧的指标:
网络「看到」的永远是过去的画面,age 越大,闭环相位滞后越严重。

与自建靶场的另一个语义差异(必须知道):
    Arena.step(action) 是「先动准星再渲染」,帧与动作严格因果相接;
    真实屏幕源做不到 —— 注入的移动要等游戏渲染完才会出现在下一帧里。
    即真实闭环自带 ≥1 帧的传输延迟,这会写进每篇相关报告的限定条件。

遥测与 tools/telemetry.py 兼容(BridgeLoop 只要求鸭子类型:
.step(t_ms=, frame=, spikes=, rates=, meta=) / .diag(**kv) / .close()),
所以 tools/live_view.py / live_web.py 可以直接看桥接运行。
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def _pct(xs: list[float], q: float) -> float:
    return round(float(np.percentile(np.asarray(xs, dtype=np.float64), q)), 3) if xs else -1.0


class BridgeLoop:
    """把 FrameSource / controller / GainModel / ActionSink 拧成一个实时闭环。

    用法::

        loop = BridgeLoop(source=cap, sink=NullSink(), controller=fly,
                          gain=GainModel(cfg), telemetry=writer,
                          max_frames=300)
        summary = loop.run()

    telemetry 为 None 时不落盘。writer 的 spikes/rates 由 controller.brain
    提供(仅 FlyController 有);其余控制器记 no_brain。
    """

    def __init__(
        self,
        source: Any,
        sink: Any,
        controller: Any,
        gain: Any,
        telemetry: Any | None = None,
        *,
        min_tick_interval_s: float = 0.0,
        max_frames: int | None = None,
        max_seconds: float | None = None,
        capture_stall_warn_s: float = 2.0,
        focus_watchdog_title: str | None = None,
    ) -> None:
        self.source = source
        self.sink = sink
        self.controller = controller
        self.gain = gain
        self.telemetry = telemetry
        self.min_tick_interval_s = float(min_tick_interval_s)
        self.max_frames = None if max_frames is None else int(max_frames)
        self.max_seconds = None if max_seconds is None else float(max_seconds)
        self.capture_stall_warn_s = float(capture_stall_warn_s)
        # 焦点看门狗:SendInput 只进焦点窗口,运行中焦点被切走 = 注入静默丢失
        # (2026-10-04 实测)。每 2s 检查并夺回,丢失次数写入 summary。
        self.focus_watchdog_title = focus_watchdog_title

        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._slot: tuple[np.ndarray, dict] | None = None
        self._cap_thread: threading.Thread | None = None
        self._last_capture_t = time.perf_counter()  # 防启动瞬间误报停滞

    # ---------------------------------------------------------------- 捕获线程

    def _capture_worker(self) -> None:
        while not self._stop.is_set():
            try:
                frame, meta = self.source.read()
            except Exception as exc:  # 捕获失败不静默:记账并重试
                logger.error("捕获失败(%s: %s),100ms 后重试", type(exc).__name__, exc)
                if self._stop.wait(0.1):
                    return
                continue
            with self._lock:
                self._slot = (frame, meta)
            self._last_capture_t = time.perf_counter()
            # 不加额外 sleep:read() 本身耗时(grab+resize)就是天然节流;
            # Windows 默认定时器精度 15.6ms,这里每多等一次就把拍频拉低一截。

    def _latest_frame(self) -> tuple[np.ndarray, dict] | None:
        with self._lock:
            return self._slot

    # ---------------------------------------------------------------- 主循环

    def run(self) -> dict:
        """跑到停止条件;Ctrl-C 也会优雅落账。返回 summary dict。"""
        # 提升 Windows 定时器精度到 1ms(默认 15.6ms 会把 Event.wait 的
        # 0.5-2ms 放大一个量级,直接压低拍频)。进程级设置,退出时恢复。
        self._timer_period_set = False
        try:
            import ctypes as _ct

            if hasattr(_ct, "windll") and _ct.windll.winmm.timeBeginPeriod(1) == 0:
                self._timer_period_set = True
        except Exception:
            pass

        has_brain = getattr(self.controller, "brain", None) is not None
        self._stop.clear()
        self._cap_thread = threading.Thread(
            target=self._capture_worker, name="bridge-capture", daemon=True
        )
        self._cap_thread.start()

        n_frames = 0
        focus_losses = 0
        last_focus_check = 0.0
        t_start = time.perf_counter()
        act_ms: list[float] = []
        inject_ms: list[float] = []
        tick_ms: list[float] = []
        age_ms: list[float] = []
        grab_ms: list[float] = []
        last_seq = -1
        stall_warned = 0.0
        hit_counts = 0

        logger.info(
            "BridgeLoop 启动: controller=%s sink=%s gain_counts/action=%.1f%s",
            getattr(self.controller, "name", "?"), getattr(self.sink, "name", "?"),
            self.gain.counts_per_action,
            f" (上限 {self.max_frames} 帧)" if self.max_frames else "",
        )

        try:
            while True:
                t_tick = time.perf_counter()

                # -- 停止条件
                if self.max_frames is not None and n_frames >= self.max_frames:
                    break
                if self.max_seconds is not None and (
                    time.perf_counter() - t_start >= self.max_seconds
                ):
                    break

                # -- 取最新帧(等待新帧,防同一帧被网络重复消费)
                # 注意:必须真正 sleep 等待 —— 忙等自旋会占住 GIL,
                # 把捕获线程饿到只剩几 FPS(2026-10-04 实测 5.4 FPS 的根因)。
                frame_meta = self._latest_frame()
                while frame_meta is None or frame_meta[1].get("seq", last_seq) == last_seq:
                    now = time.perf_counter()
                    if (now - self._last_capture_t > self.capture_stall_warn_s
                            and now - stall_warned > 5.0):
                        stall_warned = now
                        logger.warning(
                            "捕获停滞(%.1fs 无新帧,seq=%d)—— 检查游戏/窗口是否最小化",
                            now - self._last_capture_t, last_seq,
                        )
                    if self._stop.wait(0.002):
                        break
                    frame_meta = self._latest_frame()
                    if self.max_seconds is not None and (
                        time.perf_counter() - t_start >= self.max_seconds
                    ):
                        frame_meta = None
                        break
                if frame_meta is None:
                    break
                frame, meta = frame_meta
                last_seq = int(meta.get("seq", -1))
                age = time.perf_counter() - float(meta.get("t_grab", time.perf_counter()))
                age_ms.append(age * 1000.0)
                if "grab_ms" in meta:
                    grab_ms.append(float(meta["grab_ms"]))

                # -- 决策(网络拍)
                t0 = time.perf_counter()
                action = np.asarray(
                    self.controller.act(frame), dtype=np.float32
                ).reshape(2)
                act_ms.append((time.perf_counter() - t0) * 1000.0)

                # -- 环境回调(ArenaSource 消费;屏幕源忽略)
                push = getattr(self.source, "push", None)
                if callable(push):
                    push(action)

                # -- 注入
                t0 = time.perf_counter()
                dx, dy = self.gain.to_counts(action)
                ok = self.sink.send(dx, dy)
                inject_ms.append((time.perf_counter() - t0) * 1000.0)
                if not ok:
                    logger.warning("注入失败一次(delta=(%d,%d))", dx, dy)

                # -- 遥测
                if self.telemetry is not None:
                    meta_out = {
                        "act": action.astype(np.float32),
                        "counts": np.array([dx, dy], dtype=np.float32),
                        "frame_seq": last_seq,
                        "age_ms": round(age * 1000.0, 2),
                        "grab_ms": float(meta.get("grab_ms", -1.0)),
                        "read_ms": float(meta.get("read_ms", -1.0)),
                        "hit": bool(meta.get("hit", False)),
                        "target_dist": float(meta.get("target_dist", -1.0)),
                    }
                    det = getattr(self.controller, "last_detection", None)
                    if det is not None:
                        meta_out["det"] = np.array(
                            [det.cx, det.cy, det.radius_px, float(det.ok)],
                            dtype=np.float32,
                        )
                    if has_brain:
                        brain = self.controller.brain
                        self.telemetry.step(
                            t_ms=float(n_frames), frame=frame,
                            spikes=brain.spikes, rates=brain.rates, meta=meta_out,
                        )
                    else:
                        meta_out["no_brain"] = True
                        self.telemetry.step(
                            t_ms=float(n_frames), frame=frame,
                            spikes=np.zeros(1, np.float32),
                            rates=np.zeros(1, np.float32), meta=meta_out,
                        )
                if meta.get("hit"):
                    hit_counts += 1  # 仅 ArenaSource 之类会报 hit 的源有效

                # -- 焦点看门狗(每 ~2s)
                if self.focus_watchdog_title and t_tick - last_focus_check > 2.0:
                    last_focus_check = t_tick
                    try:
                        import ctypes as _c

                        from flyaim.bridge.capture import _fg_matches

                        if not _fg_matches(self.focus_watchdog_title):
                            focus_losses += 1
                            from flyaim.bridge.capture import focus_window as _fw

                            _fw(self.focus_watchdog_title)
                            logger.warning("焦点曾被切走(第 %d 次),已夺回", focus_losses)
                    except Exception:
                        pass

                n_frames += 1
                tick_ms.append((time.perf_counter() - t_tick) * 1000.0)

                # -- 拍频下限(避免快控制器空转烧 CPU)
                if self.min_tick_interval_s > 0:
                    remain = self.min_tick_interval_s - (time.perf_counter() - t_tick)
                    if remain > 0:
                        self._stop.wait(remain)

        except KeyboardInterrupt:
            logger.info("收到 Ctrl-C,优雅停止")
        finally:
            self._stop.set()
            if self._cap_thread is not None:
                self._cap_thread.join(timeout=2.0)
            try:
                self.controller.close()
            except Exception:
                pass
            try:
                self.source.close()
            except Exception:
                pass
            try:
                self.sink.close()
            except Exception:
                pass
            tel_info: dict = {}
            if self.telemetry is not None:
                try:
                    tel_info = self.telemetry.close()  # type: ignore[union-attr]
                except Exception:
                    tel_info = {}
            if getattr(self, "_timer_period_set", False):
                try:
                    import ctypes as _ct

                    _ct.windll.winmm.timeEndPeriod(1)
                except Exception:
                    pass

        wall = time.perf_counter() - t_start
        summary = {
            "controller": getattr(self.controller, "name", "?"),
            "sink": getattr(self.sink, "name", "?"),
            "source": str(getattr(self.source, "backend", "?")),
            "frames": n_frames,
            "wall_s": round(wall, 2),
            "tick_hz": round(n_frames / wall, 2) if wall > 0 else 0.0,
            "latency_ms": {
                "act_p50": _pct(act_ms, 50),
                "act_p95": _pct(act_ms, 95),
                "inject_p50": _pct(inject_ms, 50),
                "tick_p50": _pct(tick_ms, 50),
                "tick_p95": _pct(tick_ms, 95),
                "capture_age_p50": _pct(age_ms, 50),
                "capture_age_p95": _pct(age_ms, 95),
                "grab_p50": _pct(grab_ms, 50),
            },
            "counts_total": [int(v) for v in np.asarray(getattr(self.sink, "total_counts", (0, 0))).ravel()],
            "n_hits_env": int(hit_counts),
            "focus_losses": int(focus_losses),
            "telemetry": tel_info,
            "gain": self.gain.describe() if hasattr(self.gain, "describe") else {},
        }
        if getattr(self.controller, "name", "") == "random":
            pass
        logger.info(
            "BridgeLoop 结束: %d 帧 / %.1fs = %.1f Hz;act p95=%.0fms;画面年龄 p95=%.0fms",
            n_frames, wall, summary["tick_hz"],
            summary["latency_ms"]["act_p95"], summary["latency_ms"]["capture_age_p95"],
        )
        return summary
