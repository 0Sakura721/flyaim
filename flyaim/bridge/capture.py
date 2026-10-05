"""屏幕捕获(FrameSource):真实画面 -> (H, W, 3) uint8 RGB 帧。

===============================================================================
设计
===============================================================================
`Arena.render()` 的替代品。所有源都产出与靶场帧同构的数组:
    - dtype uint8,shape (H, W, 3),RGB 顺序(`Retina._to_channels` 的约定);
    - 每次调用返回新数组,不与内部缓冲共享(Retina 的时间差分会保存引用)。

后端选择(`ScreenCapture`,按可用性自动回退):
    dxcam  DXGI Desktop Duplication,最快(~240 FPS),需要 `pip install dxcam`;
    mss    纯 Python 截屏(~30-60 FPS 小区域),需要 `pip install mss`;
    pil    PIL.ImageGrab(零新依赖,~10-30 FPS 小区域)—— 本机默认。

实测注意:本项目所在网络 pip 限速严重,如果 dxcam/mss 装不上,PIL 后端
在 640x480 区域足够支撑 ~10-30 FPS 捕获,高于全量网络的 ~10 Hz 拍频,
不是当前瓶颈。

`ArenaSource` 不是屏幕源:它把自建 2D 靶场包成 FrameSource 并**闭环消费**
action(push -> arena.step),用于在没有游戏/没有屏幕的情况下彩排整条
桥接链路 —— 语义与正式实验完全一致,只是"屏幕"换成已知环境。
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

Region = tuple[int, int, int] | tuple[int, int, int, int]  # (l, t, w, h) 或 (l, t, r, b)


# ---------------------------------------------------------------- 纯数组源


class ArraySource:
    """回放一组现成的帧(测试 / 离线重放)。frame 序列循环使用。"""

    backend = "array"

    def __init__(self, frames: list[np.ndarray] | np.ndarray) -> None:
        fs = [np.asarray(f, dtype=np.uint8) for f in frames]
        if not fs:
            raise ValueError("ArraySource 需要至少一帧")
        h, w = fs[0].shape[:2]
        for f in fs:
            if f.shape[:2] != (h, w):
                raise ValueError("ArraySource: 所有帧必须同尺寸")
        self._frames = fs
        self._i = -1
        self.t_grab = 0.0
        self.seq = -1

    def read(self) -> tuple[np.ndarray, dict]:
        self._i = (self._i + 1) % len(self._frames)
        self.seq += 1
        self.t_grab = time.perf_counter()
        return self._frames[self._i].copy(), {"seq": self.seq, "backend": self.backend,
                                              "t_grab": self.t_grab}

    def close(self) -> None:
        pass


# ---------------------------------------------------------------- 自建靶场源


class ArenaSource:
    """把自建 2D 靶场包成 FrameSource(无头彩排用)。

    **步进语义(2026-10-04 修正)**:环境步进只发生在 `push(action)`(消费
    线程)里,捕获线程只做 `render()`。若把 arena.step 放进捕获线程,捕获
    提速后每个消费拍会 step 十几次同一动作,等效控制增益暴增(实测把 T5
    彩排的 seek 从收敛打到过冲)。修正后:每消费拍恰好步进一次,与离线
    run_episode 和真实 FPS 的"画面快照 + 指令在拍间生效"语义一致。
    """

    backend = "arena"

    def __init__(self, arena: Any, closed_loop: bool = True) -> None:
        self.arena = arena
        self.closed_loop = bool(closed_loop)
        self._started = False
        self._last_action = np.zeros(2, dtype=np.float32)
        self.t_grab = 0.0
        self.seq = -1
        self._last_info: dict = {"target_dist": -1.0, "hit": False}

    def read(self) -> tuple[np.ndarray, dict]:
        if not self._started:
            self.arena.reset()
            self._started = True
        frame = np.asarray(self.arena.render(), dtype=np.uint8)
        self.seq += 1
        self.t_grab = time.perf_counter()
        meta = {"seq": self.seq, "backend": self.backend, "t_grab": self.t_grab,
                "target_dist": float(self._last_info.get("target_dist", -1.0)),
                "hit": bool(self._last_info.get("hit", False))}
        return frame, meta

    def push(self, action: np.ndarray) -> None:
        """消费线程每拍调用一次:closed_loop 时环境恰好步进一次。"""
        a = np.clip(np.asarray(action, dtype=np.float32).reshape(2), -1.0, 1.0)
        if self.closed_loop:
            res = self.arena.step(a)
            self._last_info = {"target_dist": float(res.target_dist), "hit": bool(res.hit)}

    def close(self) -> None:
        pass


# ---------------------------------------------------------------- 屏幕捕获


def centered_region(monitor_wh: tuple[int, int], size_wh: tuple[int, int]) -> tuple[int, int, int, int]:
    """返回主屏中央的 (left, top, width, height) 区域。"""
    mw, mh = int(monitor_wh[0]), int(monitor_wh[1])
    w, h = int(size_wh[0]), int(size_wh[1])
    if w > mw or h > mh:
        raise ValueError(f"捕获区域 {w}x{h} 大于主屏 {mw}x{mh}")
    left = (mw - w) // 2
    top = (mh - h) // 2
    return (left, top, w, h)


def primary_monitor_size() -> tuple[int, int]:
    """主屏分辨率(零依赖,ctypes)。"""
    import ctypes

    user32 = ctypes.windll.user32
    try:
        user32.SetProcessDPIAware()
    except Exception:
        pass
    return int(user32.GetSystemMetrics(0)), int(user32.GetSystemMetrics(1))


# ---------------------------------------------------------------- 窗口定位


def list_windows(min_client_wh: int = 100) -> list[dict]:
    """列出所有可见且有标题的顶层窗口(客户区矩形,屏幕坐标,物理像素)。

    用客户区而不是窗口矩形:排除标题栏/边框,截到的就是纯画面。
    DPI 感知:进程级 SetProcessDPIAware 后坐标为物理像素。
    """
    import ctypes
    from ctypes import wintypes

    user32 = ctypes.windll.user32
    try:
        user32.SetProcessDPIAware()
    except Exception:
        pass

    out: list[dict] = []
    proto = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)

    def _cb(hwnd, _lparam):
        if not user32.IsWindowVisible(hwnd):
            return True
        n = user32.GetWindowTextLengthW(hwnd)
        if n <= 0:
            return True
        buf = ctypes.create_unicode_buffer(n + 1)
        user32.GetWindowTextW(hwnd, buf, n + 1)
        rect = wintypes.RECT()
        if not user32.GetClientRect(hwnd, ctypes.byref(rect)):
            return True
        w, h = int(rect.right) - int(rect.left), int(rect.bottom) - int(rect.top)
        if w < min_client_wh or h < min_client_wh:
            return True
        pt = wintypes.POINT(0, 0)
        user32.ClientToScreen(hwnd, ctypes.byref(pt))
        out.append({
            "hwnd": int(hwnd),
            "title": buf.value,
            "left": int(pt.x), "top": int(pt.y),
            "width": w, "height": h,
        })
        return True

    user32.EnumWindows(proto(_cb), 0)
    return out


def find_window_region(title_substr: str) -> tuple[int, int, int, int]:
    """按标题子串找窗口,返回客户区 (left, top, width, height)。

    多个匹配时取客户区面积最大者;找不到抛 RuntimeError 并列出候选。
    """
    subs = str(title_substr).strip().lower()
    cands = [w for w in list_windows() if subs in w["title"].lower()]
    if not cands:
        known = [f'"{w["title"]}" {w["width"]}x{w["height"]}' for w in list_windows()[:12]]
        raise RuntimeError(
            f'找不到标题含 "{title_substr}" 的窗口。当前可见窗口(前 12):\n  '
            + "\n  ".join(known)
        )
    cands.sort(key=lambda w: w["width"] * w["height"], reverse=True)
    b = cands[0]
    logger.info("find_window_region: %r -> (%d,%d,%d,%d)",
                b["title"], b["left"], b["top"], b["width"], b["height"])
    return (b["left"], b["top"], b["width"], b["height"])


def _fg_matches(title_substr: str) -> bool:
    """当前前台窗口标题是否含 title_substr(焦点看门狗用)。"""
    import ctypes

    user32 = ctypes.windll.user32
    hwnd = user32.GetForegroundWindow()
    if not hwnd:
        return False
    buf = ctypes.create_unicode_buffer(256)
    user32.GetWindowTextW(hwnd, buf, 256)
    return str(title_substr).lower() in buf.value.lower()


def focus_window(title_substr: str) -> bool:
    """把标题含 title_substr 的窗口调到前台。

    **注入的前提**:SendInput 的相对移动只进焦点窗口 —— 游戏不在前台时
    注入会静默丢失(2026-10-04 实测:注入 15000 计数后画面 0px 变化)。
    Windows 防抢焦点机制会拒绝裸 SetForegroundWindow,标准解法是先用
    AttachThreadInput 把本线程附加到当前前台线程再调。
    """
    import ctypes

    subs = str(title_substr).strip().lower()
    cands = sorted(
        (w for w in list_windows() if subs in w["title"].lower()),
        key=lambda w: w["width"] * w["height"], reverse=True,
    )
    if not cands:
        return False
    hwnd = cands[0]["hwnd"]
    user32 = ctypes.windll.user32
    kernel32 = ctypes.windll.kernel32
    try:
        user32.SetProcessDPIAware()
    except Exception:
        pass
    if user32.GetForegroundWindow() == hwnd:
        return True

    fg = user32.GetForegroundWindow()
    tid_me = kernel32.GetCurrentThreadId()
    tid_fg = user32.GetWindowThreadProcessId(fg, None) if fg else 0
    tid_tgt = user32.GetWindowThreadProcessId(hwnd, None)
    if tid_fg and tid_fg != tid_me:
        user32.AttachThreadInput(tid_me, tid_fg, True)
    if tid_tgt and tid_tgt != tid_me:
        user32.AttachThreadInput(tid_me, tid_tgt, True)
    try:
        user32.BringWindowToTop(hwnd)
        user32.SetForegroundWindow(hwnd)
    finally:
        if tid_fg and tid_fg != tid_me:
            user32.AttachThreadInput(tid_me, tid_fg, False)
        if tid_tgt and tid_tgt != tid_me:
            user32.AttachThreadInput(tid_me, tid_tgt, False)
    return bool(user32.GetForegroundWindow() == hwnd)


def _resize_rgb(rgb: np.ndarray, out_size: tuple[int, int] | None) -> np.ndarray:
    """resize 到 out_size(保持 uint8 RGB)。None 表示原样。"""
    if out_size is None or (rgb.shape[1], rgb.shape[0]) == (out_size[0], out_size[1]):
        return rgb
    from PIL import Image

    im = Image.fromarray(rgb)
    im = im.resize((int(out_size[0]), int(out_size[1])), Image.BILINEAR)
    return np.asarray(im, dtype=np.uint8)


# Desktop Duplication 相机是进程级单例:bettercam 重复 create 会返回旧实例,
# 而 release 过的旧实例会导致后续 grab 全部失败(2026-10-04 实测 0 帧)。
# 因此全局缓存一个实例;region 是 grab 时参数,一个实例可服务所有区域。
_DXCAM_CACHE: dict = {}

# 🔴 相机**不是线程安全的**,而 D42.7 起它真的会被两个线程同时用:
#     - 捕获线程:DualWindowSource 读全屏窗/中心窗;
#     - 主线程  :HudStrip 每 score_every 拍读 HUD 条(D42 分数闭环)。
# 在 D42.7 之前不存在这种并发(两个 ScreenCapture 是**同一个捕获线程**交替用的),
# 所以此前没暴露。这里用一把**进程级可重入锁**把 create 与 grab 都串起来 ——
# 粒度取"整个 grab 调用"而不是"取帧缓冲",因为 dxcam 内部会改自身状态。
# 代价:主线程读 HUD 最多等一次捕获抓帧(实测 8~13ms),6Hz 下可忽略。
_DXCAM_LOCK = threading.RLock()


def _get_dd_camera(output_color: str = "BGRA"):
    """取(或创建)进程级 bettercam/dxcam 相机单例(线程安全)。"""
    with _DXCAM_LOCK:
        if output_color in _DXCAM_CACHE:
            return _DXCAM_CACHE[output_color]
        import ctypes

        try:  # bettercam/dxcam 的 region 校验用物理像素,必须先声明 DPI 感知
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass
        cam, name = None, None
        try:
            import bettercam

            cam = bettercam.create(output_color=output_color)
            name = "bettercam"
        except ImportError:
            pass
        if cam is None:
            try:
                import dxcam

                cam = dxcam.create(output_color=output_color)
                name = "dxcam"
            except Exception:
                cam = None
        if cam is None:
            raise RuntimeError("bettercam/dxcam 均不可用")
        _DXCAM_CACHE[output_color] = (cam, name)
        return _DXCAM_CACHE[output_color]


class ScreenCapture:
    """屏幕区域捕获,dxcam -> mss -> PIL 自动回退。

    用法::

        cap = ScreenCapture()                       # 主屏中央 640x480
        frame, meta = cap.read()                    # (H,W,3) uint8 RGB
        cap.close()

    注意:
      - 坐标系是**虚拟桌面坐标**(多显示器时负值/超界合法,但本类不校验,
        传错区域会抓到黑屏/别的显示器 —— 用 `--region` 前先量好)。
      - 游戏建议用**无边框窗口**;独占全屏下 Desktop Duplication 仍可抓,
        但 PIL 后端可能抓不到或抓到黑帧(实测因驱动而异),dxcam/mss 更稳。
    """

    backend = "screen"

    def __init__(
        self,
        region: tuple[int, int, int, int] | None = None,
        out_size: tuple[int, int] | None = (640, 480),
        backend: str = "auto",
        grab_retry: int = 0,
        grab_retry_ms: float = 3.0,
    ) -> None:
        if region is None:
            region = centered_region(primary_monitor_size(), (640, 480))
        self.region = tuple(int(v) for v in region)
        if len(self.region) != 4:
            raise ValueError(f"region 必须是 (left, top, width, height),收到 {region}")
        self.out_size = None if out_size is None else (int(out_size[0]), int(out_size[1]))
        # 见 _mk_dxcam 的说明:Desktop Duplication 只在**有新帧**时返回图像。
        # 若同一相机被两个线程共用(捕获线程高频 + HudStrip 低频),低频方会被
        # 高频方"抢光"所有新帧,从而长期拿不到首帧。有界重试给低频方一次机会。
        # 默认 0 = 保持旧行为不变(热路径不引入额外延迟)。
        self._grab_retry = max(0, int(grab_retry))
        self._grab_retry_ms = max(0.0, float(grab_retry_ms))
        self.t_grab = 0.0
        self.seq = -1
        self.n_stale = 0          # 复用上一帧的次数(该帧不是本次新抓的)

        self._backend = self._make_backend(str(backend).lower())
        logger.info("ScreenCapture: region=%s out=%s backend=%s retry=%d",
                    self.region, self.out_size, self._backend[0], self._grab_retry)

    # -- 后端 ---------------------------------------------------------------

    def _make_backend(self, prefer: str):
        """返回 (名字, grab_fn);grab_fn() -> RGB uint8 (H,W,3)。"""
        order = {
            "auto": ("dxcam", "mss", "pil"),
            "dxcam": ("dxcam",),
            "mss": ("mss",),
            "pil": ("pil",),
        }.get(prefer)
        if order is None:
            raise ValueError(f"未知捕获后端: {prefer}(可用 auto/dxcam/mss/pil)")
        errs: list[str] = []
        for name in order:
            try:
                if name == "dxcam":
                    return ("dxcam", self._mk_dxcam())
                if name == "mss":
                    return ("mss", self._mk_mss())
                if name == "pil":
                    return ("pil", self._mk_pil())
            except Exception as exc:
                errs.append(f"{name}: {type(exc).__name__}: {exc}")
        raise RuntimeError(
            "没有可用的屏幕捕获后端。请 `pip install mss`(本网络请用 "
            "tools/fetch_fast.py 并发下载)或使用 PIL(应总是可用)。失败明细:\n  "
            + "\n  ".join(errs)
        )

    def _mk_dxcam(self):
        """Desktop Duplication 后端:优先 bettercam(dxcam 维护分支),退回 dxcam。

        两者只有在做颜色转换(output_color=RGB 等)时才 import cv2;
        用原生 **BGRA** 输出即可零 cv2 依赖 —— 通道重排已在 read() 的
        BGRA 快速路径里(与 resize 交换,小图上做)。
        相机为进程级单例(见 _get_dd_camera),close() 不释放。
        """
        cam, backend_name = _get_dd_camera("BGRA")
        self._fmt = "bgra"
        self._dd_name = backend_name
        l, t, w, h = self.region

        def grab() -> np.ndarray:
            # region 是 (l, t, r, b);无新帧时复用上一帧(与游戏帧率解耦)。
            #
            # ⚠️ Desktop Duplication 只在**有新帧**时返回图像(None = 本刷新周期
            # 已被取走)。所以当同一相机被两个线程共用时(捕获线程高频 +
            # HudStrip 低频),低频方会被高频方抢光所有新帧 —— 实测主线程
            # 59 次读里 12 次拿不到首帧(D42.7d)。`grab_retry` 让调用方声明
            # "我愿意多等几个刷新周期"。**睡眠必须在锁外** —— 否则会把
            # 捕获线程一起拖住。
            f = None
            for attempt in range(self._grab_retry + 1):
                with _DXCAM_LOCK:
                    f = cam.grab(region=(l, t, l + w, t + h))
                if f is not None:
                    break
                if attempt < self._grab_retry:
                    time.sleep(self._grab_retry_ms / 1000.0)
            if f is None:
                f = getattr(grab, "_last", None)
                if f is None:
                    raise RuntimeError("尚无首帧")
                grab._stale = True      # type: ignore[attr-defined]
                return f
            grab._last = f  # type: ignore[attr-defined]
            grab._stale = False         # type: ignore[attr-defined]
            return np.asarray(f, dtype=np.uint8)

        return grab

    def _mk_mss(self):
        import mss  # 延迟 import

        l, t, w, h = self.region
        monitor = {"left": l, "top": t, "width": w, "height": h}
        sct = mss.mss()
        self._fmt = "bgra"  # 返回原始 BGRA;重排推迟到 resize 之后(见 read)

        def grab() -> np.ndarray:
            shot = sct.grab(monitor)
            return np.asarray(shot, dtype=np.uint8)  # (H, W, 4) BGRA,连续

        return grab

    def _mk_pil(self):
        from PIL import ImageGrab  # Pillow 必备依赖,总是可用

        l, t, w, h = self.region

        def grab() -> np.ndarray:
            im = ImageGrab.grab(bbox=(l, t, l + w, t + h), all_screens=True)
            return np.asarray(im.convert("RGB"), dtype=np.uint8)

        return grab

    # -- 接口 ---------------------------------------------------------------

    @property
    def backend_name(self) -> str:
        return self._backend[0]

    def read(self) -> tuple[np.ndarray, dict]:
        from PIL import Image

        grab = self._backend[1]
        t0 = time.perf_counter()
        raw = grab()
        grab_ms = (time.perf_counter() - t0) * 1000.0
        # t_grab 取「像素内容定格」时刻(抓帧完成);resize 不改内容但计入 read_ms,
        # 这样 age_ms 包含了本帧自己的 resize 时间 —— 延迟记账不做美化。
        self.t_grab = time.perf_counter()
        fmt = getattr(self, "_fmt", "rgb")
        if fmt == "bgra" and self.out_size is not None:
            # 通道重排与 resize 都是逐通道线性操作,可交换:
            # 先在 4 通道上 resize(2MP),再在小图(0.3MP)上 BGRA->RGB。
            # 实测比「先重排非连续视图再 resize」快 2x(53.8 -> 26.7ms),
            # 且数值逐位一致 —— 旧路径 PIL 对负步长数组走慢路径。
            im = Image.fromarray(raw)  # RGBA 模式,字节序实为 BGRA
            # BOX = 面积平均,是降采样的正确滤波(与视网膜分块均值同语义),
            # 实测也比 BILINEAR 快 5ms(19.9 vs 25.0ms)
            small = np.asarray(im.resize((self.out_size[0], self.out_size[1]), Image.BOX))
            frame = np.ascontiguousarray(small[:, :, :3][:, :, ::-1])
        elif fmt == "bgra":
            frame = np.ascontiguousarray(raw[:, :, :3][:, :, ::-1])
        else:
            frame = _resize_rgb(np.asarray(raw, dtype=np.uint8), self.out_size)
        read_ms = (time.perf_counter() - t0) * 1000.0
        stale = bool(getattr(grab, "_stale", False))
        if stale:
            self.n_stale += 1
        self.seq += 1
        return frame, {
            "seq": self.seq,
            "backend": f"screen:{self._backend[0]}",
            "grab_ms": round(grab_ms, 3),
            "read_ms": round(read_ms, 3),
            "t_grab": self.t_grab,
            # True = 本次拿到的是**上一帧的复用**(本刷新周期已被别的调用取走)。
            # HUD 读数若长期 stale,说明它看到的是旧画面,不是"与渲染同步"。
            "grab_stale": stale,
        }

    def push(self, action: np.ndarray) -> None:  # 屏幕源不消费 action
        del action

    def close(self) -> None:
        # 相机是进程级单例,不 release(release 后 bettercam 仍返回旧实例,
        # 会导致后续 ScreenCapture 全部 0 帧);进程退出时由 OS 回收。
        self._backend = None


class DualWindowSource:
    """搜索窗(全屏) / 跟踪窗(中心小窗)自动切换的帧源(2026-10-05)。

    ===========================================================================
    为什么需要它 —— 这是「命中率低」层级的真因,不是调参能解决的
    ===========================================================================
    「中心窗」优化的前提是「靶已经在画面中心附近」。搜靶阶段这个前提**不成立**:

        实测(2026-10-05 14:2x,目标帧):靶在全屏 (144, 48),半径 25px。
          全屏检测          -> ok=True  (144,48)
          中心窗 700        -> ok=False  (窗范围 x[610,1310] y[190,890])
          中心窗 900        -> ok=False  (窗范围 x[510,1410] y[ 90,990])

    靶在窗口**左 366px、上 42px** —— 完全在窗外。此前跑出的 play-cov900 那局
    2152 帧 det-ok=0、|act| 全程 0,拍频 36Hz,相机对着空墙站 60 秒开火 0 次,
    就是这条。而 play-smooth 那局能开火 117 次,**纯粹是运气**:当时相机碰巧
    朝向靶,靶落在中心窗内。**同一份参数、同一份代码,结果 0 与 117 的差别
    全在「靶是否恰好在窗外」** —— 这正说明中心窗方案在几何上是错的。

    而全屏捕获贵(read 24.7ms vs 中心窗 3.6ms,且不缩放时像素 2.07M vs 0.49M)。
    所以正确解法是**分阶段用不同的窗**,而不是二选一:

        - 未锁定(搜靶):抓**全屏**,保证任意位置的靶都看得见;
        - 已锁定(跟踪):抓**中心窗**,保住 7x 速度与原始像素精度。

    锁定信号怎么来:看上一拍控制器写下的 `last_detection`,若 ok 且离中心
    在 `lock_switch_px` 以内,就认为「已经锁上、可以切小窗」;否则回大窗。

    诚实声明:本类只改变**送给控制器的画面范围**,不改变任何决策语义;
    两个窗的**坐标都是各自窗内的局部坐标**,而控制器用的正是局部坐标
    (误差 = 靶 - 本帧中心),所以切换不影响控制律的正确性。唯一需要
    对齐的是开火层的 sticky 状态 —— 跨窗切换时靶的局部坐标会突变,
    故切换时把 `meta["window_switched"]=True` 传给上层,由上层清 sticky。
    """

    backend = "dual-window"

    def __init__(
        self,
        full_region: tuple[int, int, int, int],
        center_size: int = 900,
        *,
        lock_switch_px: float = 260.0,
        full_downsample_size: int | None = 960,
        backend: str = "auto",
        controller: Any | None = None,
    ) -> None:
        """full_region: 全屏 region(左,上,宽,高)。
        center_size: 跟踪窗边长(裁在全屏 region 的几何中心)。
        lock_switch_px: 靶离窗中心多近算「已锁定」→ 切小窗。
        full_downsample_size: 搜靶全屏帧的降采样边长(None=不缩放,更慢但更准)。
        controller: 用来读 last_detection 判断是否已锁定;可后置 set_controller。
        """
        self.full_region = tuple(int(v) for v in full_region)
        self.center_size = int(center_size)
        self.lock_switch_px = float(lock_switch_px)
        self._down = None if full_downsample_size is None else int(full_downsample_size)
        self._controller = controller

        fl, ft, fw, fh = self.full_region
        self._center_region = (
            fl + (fw - self.center_size) // 2,
            ft + (fh - self.center_size) // 2,
            self.center_size,
            self.center_size,
        )
        # 搜靶窗:全屏,但可选降采样(降采样只影响像素量,坐标由 ScreenCapture
        # 缩放到 out_size 后再 ×ds 还原 —— 这里 down 由控制器侧负责,故只用 out)
        self._full_cap = ScreenCapture(
            region=self.full_region, out_size=None, backend=backend
        )
        self._center_cap = ScreenCapture(
            region=self._center_region, out_size=None, backend=backend
        )
        # 当前生效的窗:True=全屏(搜靶)
        self.searching = True
        self.n_switches = 0
        self.n_full_reads = 0
        self.n_center_reads = 0
        self._last_read_ms = 0.0

    def set_controller(self, controller: Any) -> None:
        self._controller = controller

    # -- 判定:该用哪个窗 ----------------------------------------------------

    def _target_in_full(self) -> bool:
        """上一拍检测若 ok 且落在中心窗内(按它自己的坐标判断),说明锁上了。

        `last_detection` 的坐标与**上一拍用的窗**同口径:上一拍用全屏时是
        全屏坐标,用中心窗时是窗内坐标。所以这里必须记住上一拍用的窗。
        """
        ctrl = self._controller
        if ctrl is None:
            return False
        det = getattr(ctrl, "last_detection", None)
        if det is None or not getattr(det, "ok", False):
            return False
        if self.searching:
            # 上一拍是全屏:靶必须落在中心窗矩形内才能切小窗
            cl, ct, cw, ch = self._center_region
            fl, ft, _, _ = self.full_region
            # det 坐标是「降采样后的帧坐标」;本类不降采样(交给控制器),
            # 故直接用全屏坐标比较
            return (cl - fl) <= det.cx <= (cl - fl + cw) and \
                   (ct - ft) <= det.cy <= (ct - ft + ch)
        # 上一拍已是中心窗:只要还在窗内(离中心不算太远)就继续用
        h = w = self.center_size
        return 0 <= det.cx <= w and 0 <= det.cy <= h

    def read(self) -> tuple[np.ndarray, dict]:
        want_full = not self._target_in_full()
        switched = want_full != self.searching
        if switched:
            self.searching = want_full
            self.n_switches += 1
        cap = self._full_cap if self.searching else self._center_cap
        frame, meta = cap.read()
        if self.searching:
            self.n_full_reads += 1
        else:
            self.n_center_reads += 1
        meta = dict(meta)
        meta["window"] = "full" if self.searching else "center"
        meta["window_switched"] = bool(switched)
        meta["center_region"] = self._center_region
        # 搜靶窗是全屏(较大),控制器应据 window 字段选更激进的 downsample;
        # 这里只负责把窗类型告诉下游,**不做缩放**(缩放交给 detect.downsample,
        # 它已经把坐标/半径/面积原样还原回本窗口径 —— 在 source 里再缩一次
        # 会让坐标口径与控制器约定不一致)。
        return frame, meta

    def push(self, action: np.ndarray) -> None:
        del action

    def close(self) -> None:
        try:
            self._full_cap.close()
        finally:
            self._center_cap.close()

    # 供桥接层查询
    @property
    def region(self):
        return self.full_region
