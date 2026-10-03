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


def _get_dd_camera(output_color: str = "BGRA"):
    """取(或创建)进程级 bettercam/dxcam 相机单例。"""
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
    ) -> None:
        if region is None:
            region = centered_region(primary_monitor_size(), (640, 480))
        self.region = tuple(int(v) for v in region)
        if len(self.region) != 4:
            raise ValueError(f"region 必须是 (left, top, width, height),收到 {region}")
        self.out_size = None if out_size is None else (int(out_size[0]), int(out_size[1]))
        self.t_grab = 0.0
        self.seq = -1

        self._backend = self._make_backend(str(backend).lower())
        logger.info("ScreenCapture: region=%s out=%s backend=%s",
                    self.region, self.out_size, self._backend[0])

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
            # region 是 (l, t, r, b);无新帧时复用上一帧(与游戏帧率解耦)
            f = cam.grab(region=(l, t, l + w, t + h))
            if f is None:
                f = getattr(grab, "_last", None)
                if f is None:
                    raise RuntimeError("尚无首帧")
                return f
            grab._last = f  # type: ignore[attr-defined]
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
        self.seq += 1
        return frame, {
            "seq": self.seq,
            "backend": f"screen:{self._backend[0]}",
            "grab_ms": round(grab_ms, 3),
            "read_ms": round(read_ms, 3),
            "t_grab": self.t_grab,
        }

    def push(self, action: np.ndarray) -> None:  # 屏幕源不消费 action
        del action

    def close(self) -> None:
        # 相机是进程级单例,不 release(release 后 bettercam 仍返回旧实例,
        # 会导致后续 ScreenCapture 全部 0 帧);进程退出时由 OS 回收。
        self._backend = None
