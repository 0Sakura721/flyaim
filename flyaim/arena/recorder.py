"""C 线:流式录屏(A 线代 Lead 实现,新增模块,不改动既有代码)。

契约依据
--------
``CONTRACT.md`` §4 要求 ``flyaim/runs/<ts>/`` 里有「录屏」。本模块提供把
``Arena`` 的逐帧画面落成动画文件的能力,接口刻意设计成可以直接当
``flyaim.runner.run_episode(..., on_frame=...)`` 的回调使用。

为什么不能用 Pillow 直接写动画(实测结论,决定了本模块的架构)
------------------------------------------------------------
1. ``Image.save(save_all=True, append_images=<generator>)`` **不可用**:
   generator 会被提前耗尽,Pillow 静默产出一个**单帧 PNG**
   (实测:20 帧只写出 182 B,文件里 ``acTL``/``fcTL`` 计数为 0)。
   必须传**完整的 list**,于是全部帧都驻留内存。
2. ``GIF`` 走 list 路径时实测峰值内存按 **~307 KB/帧** 增长
   (150 帧 +53 MB,600 帧 +196 MB → 450 帧多出 142 MB);APNG 走 list 路径
   则按 **~921 KB/帧**(原始 RGB)增长。900 帧就是 276 MB / 829 MB。
   ``rollout.run_episode(collect_frames=True)`` 的 docstring 同样警告过这一点。

因此本模块采用**两段式流式**方案::

    __call__   -> 立刻把这一帧编码成**单帧 PNG** 写进 spool 目录(内存 O(1))
    close()    -> **流式**读回 spool,按 chunk 组装成单个 APNG(内存 O(1 帧))

APNG 由模块内的 :func:`_assemble_apng` 按 PNG chunk 规范手写组装
(``IHDR`` + ``acTL`` + 首帧 ``fcTL``/``IDAT`` + 每帧 ``fcTL``/``fdAT`` + ``IEND``),
因为 Pillow 无法增量追加帧。写法与 Pillow 的 list 路径产物结构一致,并已用
Pillow **逐帧像素级回读**验证(见 ``selftest_recorder.py``)。

用法::

    from flyaim.arena.recorder import FrameRecorder

    with FrameRecorder(run_dir / "episode_fly_seed0.png", every=2) as rec:
        runner.run_episode(..., on_frame=rec)
    info = rec.result            # {'path', 'frames', 'bytes', 'seconds', ...}
    metrics.add_extra("recording", info)

:class:`FrameRecorder` 的 ``close()`` 返回值只含 str/int/float/None,
可直接交给 ``Metrics.add_extra(key, value)``。
"""

from __future__ import annotations

import shutil
import struct
import sys
import time
import warnings
import zlib
from pathlib import Path

import numpy as np
from PIL import Image

__all__ = ["FrameRecorder", "FORMATS"]

#: 支持的容器格式。``apng`` 为默认(无损、峰值内存恒定);``gif`` 为备选(见 close 的说明)。
FORMATS = ("apng", "gif")

_PNG_SIG = b"\x89PNG\r\n\x1a\n"

# APNG dispose_op / blend_op 常量
_DISPOSE_NONE = 0
_BLEND_SOURCE = 0


# ------------------------------------------------------------------ PNG chunk 工具


def _png_chunks(data: bytes):
    """产出单帧 PNG 的 ``(type, payload)``。数据只有几 KB,不构成内存问题。"""
    if data[:8] != _PNG_SIG:
        raise ValueError("spool 里的文件不是合法 PNG(缺少 PNG 签名)")
    off = 8
    total = len(data)
    while off + 12 <= total:
        (length,) = struct.unpack(">I", data[off:off + 4])
        ctype = data[off + 4:off + 8]
        end = off + 8 + length
        if end + 4 > total:
            raise ValueError(f"PNG chunk {ctype!r} 越界,文件损坏")
        yield ctype, data[off + 8:end]
        off = end + 4


def _chunk(ctype: bytes, payload: bytes) -> bytes:
    """按 PNG 规范打包一个 chunk:长度 + 类型 + 数据 + CRC32(类型||数据)。"""
    return (
        struct.pack(">I", len(payload))
        + ctype
        + payload
        + struct.pack(">I", zlib.crc32(ctype + payload) & 0xFFFFFFFF)
    )


def _fctl(seq: int, w: int, h: int, delay_num: int, delay_den: int) -> bytes:
    """APNG 帧控制 chunk。x/y 偏移为 0,dispose=NONE,blend=SOURCE(整帧替换)。"""
    return _chunk(
        b"fcTL",
        struct.pack(">IIIIIHHBB", seq, w, h, 0, 0, delay_num, delay_den,
                    _DISPOSE_NONE, _BLEND_SOURCE),
    )


def _scan_png(path: Path) -> tuple[bytes | None, bytes]:
    """读一个 spool PNG,返回 ``(IHDR payload, 拼接后的 IDAT 数据)``。"""
    data = path.read_bytes()
    ihdr: bytes | None = None
    idat: list[bytes] = []
    for ctype, payload in _png_chunks(data):
        if ctype == b"IHDR":
            ihdr = payload
        elif ctype == b"IDAT":
            idat.append(payload)
    return ihdr, b"".join(idat)


def _assemble_apng(frame_paths: list[Path], out_path: Path,
                   duration_ms: int, loop: int) -> None:
    """把一串**单帧 PNG** 流式组装成一个 APNG。

    内存:同时只保留一帧的压缩数据(arena 画面每帧约 2 KB)。
    """
    n = len(frame_paths)
    if n == 0:
        raise ValueError("_assemble_apng 需要至少一帧")

    ihdr0, idat0 = _scan_png(frame_paths[0])
    if ihdr0 is None:
        raise ValueError(f"spool 帧缺少 IHDR: {frame_paths[0]}")
    width, height = struct.unpack(">II", ihdr0[:8])

    with open(out_path, "wb") as fp:
        fp.write(_PNG_SIG)
        fp.write(_chunk(b"IHDR", ihdr0))
        # acTL 必须在第一个 IDAT 之前
        fp.write(_chunk(b"acTL", struct.pack(">II", n, loop)))
        # 首帧:fcTL 必须在首帧 IDAT 之前
        fp.write(_fctl(0, width, height, duration_ms, 1000))
        fp.write(_chunk(b"IDAT", idat0))
        del idat0

        seq = 1
        for k in range(1, n):
            path = frame_paths[k]
            ihdr, idat = _scan_png(path)
            if ihdr != ihdr0:
                raise ValueError(
                    f"spool 第 {k} 帧的 IHDR 与首帧不一致,无法合并为 APNG"
                    f"({path.name});请确认所有帧尺寸/色彩模式相同"
                )
            fp.write(_fctl(seq, width, height, duration_ms, 1000))
            seq += 1
            # fdAT = 4 字节序号 + 与 IDAT 相同的图像数据
            fp.write(_chunk(b"fdAT", struct.pack(">I", seq) + idat))
            seq += 1
            del ihdr, idat
        fp.write(_chunk(b"IEND", b""))


# ------------------------------------------------------------------ 录制器


class FrameRecorder:
    """把逐帧画面**流式**落盘的录制器。

    可直接作为 ``run_episode`` 的 ``on_frame`` 回调::

        rec = FrameRecorder("run/episode.png", every=2)
        runner.run_episode(..., on_frame=rec)
        info = rec.close()

    Parameters
    ----------
    out_path:
        输出文件路径(父目录会自动创建)。APNG 建议用 ``*.png`` / ``*.apng``;
        GIF 建议用 ``*.gif``。
    every:
        抽帧间隔,``every=2`` 表示每 2 帧录 1 帧(30 fps 靶场 → 15 fps 回放)。
        必须 >= 1。
    max_frames:
        最多录制多少帧(抽帧**之后**的数量);``None`` 表示不限制。
    fmt:
        ``"apng"``(默认,流式、无损、峰值内存恒定)或 ``"gif"``(备选,见下)。
    duration_ms:
        每帧回放时长(毫秒)。``None`` 时按 ``round(1000/fps*every)`` 推算。
    fps:
        源画面帧率,仅用于推算 ``duration_ms``(默认 30)。
    keep_frames:
        为 ``True`` 时保留 spool 目录(单帧 PNG 序列)便于排查,默认清理。

    Notes
    -----
    **内存**:``__call__`` 每帧立刻写盘,**绝不累积到内存**。这点很关键 ——
    ``rollout.run_episode(collect_frames=True)`` 对 900 帧 640x480 会吃
    **约 829 MB**,录制整场 Phase 3(4 臂 x 10 seed)则完全不可行。
    selftest 用「峰值 RSS 不随帧数增长」+「故意累积的对照组」双向验证了这一点。

    **GIF 的内存代价(已知限制)**:Pillow 的 GIF 写入器必须拿到完整帧列表,
    实测峰值内存按 **约 307 KB/帧** 增长(450 帧 ≈ +138 MB)。因此 ``fmt="gif"``
    只适合短片段;长时间录制请用默认的 ``apng``(本模块手写组装,内存恒定)。

    **失败必须可见**:任何磁盘写入/组装错误都会**抛出**(``OSError`` /
    ``ValueError``),不会被静默吞掉 —— 「录屏缺了但实验报成功」是最坏情况。
    """

    def __init__(
        self,
        out_path: str | Path,
        every: int = 2,
        max_frames: int | None = None,
        fmt: str = "apng",
        *,
        duration_ms: int | None = None,
        fps: float = 30.0,
        keep_frames: bool = False,
    ) -> None:
        fmt = str(fmt).lower()
        if fmt not in FORMATS:
            raise ValueError(f"fmt 必须是 {FORMATS} 之一,收到 {fmt!r}")
        every = int(every)
        if every < 1:
            raise ValueError(f"every 必须 >= 1,收到 {every}")
        if max_frames is not None:
            max_frames = int(max_frames)
            if max_frames < 1:
                raise ValueError(f"max_frames 必须 >= 1 或 None,收到 {max_frames}")

        if duration_ms is None:
            duration_ms = max(1, int(round(1000.0 / float(fps) * every)))
        duration_ms = int(duration_ms)
        if duration_ms < 1:
            raise ValueError(f"duration_ms 必须 >= 1,收到 {duration_ms}")

        self.out_path = Path(out_path)
        self.every = every
        self.max_frames = max_frames
        self.fmt = fmt
        self.duration_ms = duration_ms
        self.fps = float(fps)
        self.keep_frames = bool(keep_frames)

        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        self.spool_dir = self.out_path.parent / (self.out_path.name + ".frames")
        # 立刻建 spool:路径不可写要在**实验开始前**就暴露,而不是跑完 900 帧才炸
        if self.spool_dir.exists() and not self.spool_dir.is_dir():
            raise NotADirectoryError(f"spool 路径被同名文件占用: {self.spool_dir}")
        if self.spool_dir.exists():
            shutil.rmtree(self.spool_dir)
        self.spool_dir.mkdir(parents=True)

        self._frame_paths: list[Path] = []
        self._n_seen = 0
        self._n_skipped = 0
        self._auto_n = 0          # 单参数 on_step(res) 形式使用的内部帧序号
        self._t0 = time.perf_counter()
        self._closed = False
        self._result: dict | None = None

    # ------------------------------------------------------------ 录制

    @property
    def n_frames(self) -> int:
        """已录制的帧数(= 抽帧之后)。"""
        return len(self._frame_paths)

    def __call__(self, i_or_res, frame: np.ndarray | None = None, res=None) -> None:
        """录制回调。支持两种钩子签名(两种在代码库里都存在):

        1. ``on_frame(i, frame, step_result)`` —— ``flyaim.runner.run_episode``
           用的形式,直接传本对象即可。
        2. ``on_step(step_result)`` —— ``flyaim.arena.rollout.run_episode``
           用的形式,只传 ``StepResult``。此时帧从 ``res.frame`` 取,
           ``i`` 用录制器内部的单调计数器(0,1,2,...),抽帧语义与形式 1 一致。

        两种形式都按**调用序号**做 ``every`` 抽帧,所以 ``every=k`` 时
        ``N`` 帧 episode 得到 ``ceil(N/k)`` 帧。
        """
        if self._closed:
            raise RuntimeError("FrameRecorder 已 close(),不能继续录制")

        if frame is None:
            # 形式 2:on_step(res) —— 第 1 个位置参数其实是 StepResult
            step = i_or_res
            fr = getattr(step, "frame", None)
            if fr is None:
                raise TypeError(
                    "FrameRecorder 单参数调用时,参数必须带 .frame 属性"
                    "(rollout.run_episode 的 on_step 形式);"
                    "若要按 on_frame(i, frame, res) 调用请传满三个参数"
                )
            i = self._auto_n
            self._auto_n += 1
            frame = fr
            res = step
        else:
            i = int(i_or_res)
        del res  # 仅为匹配回调签名,录制不需要 StepResult 内容

        self._n_seen += 1
        if i % self.every != 0:
            return
        if self.max_frames is not None and len(self._frame_paths) >= self.max_frames:
            self._n_skipped += 1
            return

        img = self._to_image(frame)
        path = self.spool_dir / f"f{len(self._frame_paths):06d}.png"
        # 单帧 PNG 立刻落盘;这里若失败必须抛出(不 try/except 吞掉)
        img.save(path, format="PNG", compress_level=6)
        self._frame_paths.append(path)

    @staticmethod
    def _to_image(frame: np.ndarray) -> Image.Image:
        arr = np.asarray(frame)
        if arr.dtype != np.uint8:
            raise TypeError(f"frame dtype 必须是 uint8,收到 {arr.dtype}")
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)
        elif arr.ndim == 3 and arr.shape[2] == 3:
            pass
        elif arr.ndim == 3 and arr.shape[2] == 4:
            arr = arr[:, :, :3]
        else:
            raise ValueError(f"frame 形状必须是 (H,W) / (H,W,3) / (H,W,4),收到 {arr.shape}")
        if not arr.flags.writeable:
            arr = arr.copy()
        if not arr.flags.c_contiguous:
            arr = np.ascontiguousarray(arr)
        return Image.fromarray(arr, mode="RGB")

    # ------------------------------------------------------------ 收尾

    def close(self) -> dict:
        """写盘并返回结果 dict(幂等:重复调用返回同一结果)。

        Returns
        -------
        dict
            仅含 str/int/float/None,可直接 ``metrics.add_extra("recording", info)``。
            键:``path`` ``frames`` ``bytes`` ``seconds`` ``format`` ``every``
            ``duration_ms`` ``fps_playback`` ``seen`` ``dropped`` ``status``
            ``spool_bytes``。
            空 episode(0 帧)时 ``path=None``、``status="empty"``,并且**不会**
            生成文件(但会发一条 warning,不静默)。
        """
        if self._closed:
            assert self._result is not None
            return self._result

        seconds = time.perf_counter() - self._t0
        n = len(self._frame_paths)
        spool_bytes = sum(p.stat().st_size for p in self._frame_paths)

        if n == 0:
            self._cleanup_spool()
            warnings.warn(
                f"FrameRecorder 没录到任何帧(every={self.every}, seen={self._n_seen});"
                f"未生成 {self.out_path}",
                RuntimeWarning,
                stacklevel=2,
            )
            self._closed = True
            self._result = {
                "path": None,
                "frames": 0,
                "bytes": 0,
                "seconds": round(seconds, 3),
                "format": self.fmt,
                "every": self.every,
                "duration_ms": self.duration_ms,
                "fps_playback": round(1000.0 / self.duration_ms, 3),
                "seen": int(self._n_seen),
                "dropped": int(self._n_skipped),
                "spool_bytes": int(spool_bytes),
                "status": "empty",
            }
            return self._result

        if self.fmt == "apng":
            _assemble_apng(self._frame_paths, self.out_path, self.duration_ms, loop=0)
        else:
            self._assemble_gif()

        size = self.out_path.stat().st_size
        if not self.keep_frames:
            self._cleanup_spool()

        self._closed = True
        self._result = {
            "path": str(self.out_path),
            "frames": int(n),
            "bytes": int(size),
            "seconds": round(seconds, 3),
            "format": self.fmt,
            "every": self.every,
            "duration_ms": self.duration_ms,
            "fps_playback": round(1000.0 / self.duration_ms, 3),
            "seen": int(self._n_seen),
            "dropped": int(self._n_skipped),
            "spool_bytes": int(spool_bytes),
            "status": "ok",
        }
        return self._result

    def _assemble_gif(self) -> None:
        """GIF 备选路径。

        **已知限制**:Pillow 的 GIF 写入器需要完整帧列表,峰值内存按
        ~307 KB/帧 增长(实测)。本方法把该代价显式记录下来,不做隐藏。
        """
        imgs = [Image.open(p).convert("RGB") for p in self._frame_paths]
        try:
            imgs[0].save(
                self.out_path,
                save_all=True,
                append_images=imgs[1:],
                duration=self.duration_ms,
                loop=0,
                optimize=False,
            )
        finally:
            for im in imgs:
                im.close()
            del imgs

    def _cleanup_spool(self) -> None:
        if self.spool_dir.exists():
            shutil.rmtree(self.spool_dir, ignore_errors=True)

    # ------------------------------------------------------------ 结果 / 上下文

    @property
    def result(self) -> dict | None:
        """``close()`` 之后的结果;未 close 时为 ``None``。"""
        return self._result

    def __enter__(self) -> "FrameRecorder":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        if exc_type is not None:
            # 主体异常:仍尽力把已录到的帧收尾(便于排查),但不让收尾错误
            # 掩盖原始异常。
            try:
                self.close()
            except Exception as close_err:  # noqa: BLE001
                print(f"[FrameRecorder] 主体异常期间收尾失败: {close_err!r}", file=sys.stderr)
            return False
        self.close()
        return False

    def __repr__(self) -> str:  # pragma: no cover - 诊断用
        state = "closed" if self._closed else "recording"
        return (f"FrameRecorder({self.out_path.name!r}, fmt={self.fmt!r}, "
                f"every={self.every}, frames={self.n_frames}, {state})")
