"""Aim Lab HUD 分数读取(轻量 OCR):POINTS / TIME / ACCURACY 三框。

**为什么需要它**(2026-10-06,用户指令「aimlab上方有分数,观察运行前后的
分数继续调试」):战报的「推断命中」只是几何判定(err<=门限即记 hit),游戏
真实分数才是地面真值 —— Gridshot 计分 = 命中按「距上一球间隔」加权(越快
单球分越高)+ **miss 扣分**,分数全程显示在屏幕顶部。读 HUD 才能把调试闭环
接到真实目标上。目标基准(用户手机理想实录):**130,709 分 / 96% ACC / 60s**。

布局(1920x1080 实测校准,screen_now.png 逐像素验证):
    三框固定顶部中间;数字行 y∈[64,86](字高约 22px,白色粗体,深灰底)。
    POINTS x∈[595,850] | TIME x∈[850,1078] | ACCURACY x∈[1078,1330]

OCR 方案(无 cv2 / tesseract 依赖):
    二值化(RGB 均 >190)→ 列投影按 >=3px 空列分段 → 每段 bbox 归一化
    20x28 → 与模板库逐字符最小 MSE 匹配。任何字符置信度不足 → 该框 None
    (宁可缺测不可错读 —— 错读会污染分数闭环)。

模板自举(免人工标注,两条路):
  1. **内嵌种子**:0 / 1 / % 三条(PC 真实帧提取,见 _SEED_*);
  2. **TIME 在线自校准**:任务局 TIME 倒计时「MM:SS」内容已知(由局内
     计时推),前 11 秒秒个位 9→0 全出现 + 分钟位 0/1 —— 0-9 模板自动凑齐。
     只在秒中段(elapsed%1 ∈ [0.25,0.75])学习,防跨秒边界误学。

⚠️ **已知缺陷(D42.5,尚未修)**:第 2 条路的 `remain_s` 由**宿主时钟**推算,
而游戏要等 autostart + 开场动画之后才起跳,两者天然差几秒。**一旦 TIME 真的
开始倒计时,这段代码会把字形按错误的数字打标签学进去,污染整个模板库 ——
触发可能比不触发更糟。** 实测 d42-score 局 TIME 全程 `∞`(分段只有 1 个 glyph,
`len(glyphs)!=5` 直接 return),自校准**从未触发**,所以这个雷还没炸。

⇒ 因此:① 冷启动只有 `{0,1,%}` 时,POINTS 任何含 2~9 的数都拒读、ACC 只有
由 `{0,1}` 组成的 `"100"` 能读 —— 实测 score 非空 1 次 / acc 非空 10 次,与此
完全吻合;② **在校准锚点问题解决前不要打开在线学习**(调用方可控)。
"""

from __future__ import annotations

import time as _time

import numpy as np

# ---------------------------------------------------------------- 布局常量
GRAB_BBOX = (595, 60, 1330, 90)     # ImageGrab bbox (left, top, right, bottom)
_DIGIT_Y = (4, 26)                  # 数字行(相对 GRAB_BBOX 原点)
_BOX_X = {"points": (0, 255),       # 三框 x(相对 GRAB_BBOX 原点)
          "time": (255, 483),
          "acc": (483, 735)}
_GW, _GH = 20, 28                   # 归一化 glyph 尺寸
_WHITE = 190                        # 白字阈值
_GAP = 3                            # 字符间最小空列数
_MAX_SAMPLES = 8                    # 每字符最多保留样本数
_MSE_OK = 0.075                     # 认领阈值(实测余量,见冒烟 T23)

# ---------------------------------------------------------------- 内嵌种子模板
# (PC 1920x1080 真实帧提取;20x28,'#'=白)
_SEED_0 = """\
....................
....................
....................
....................
.....#########......
....###########.....
...#############....
..###############...
.######......#####..
.#####........####..
#####.........#####.
#####..........####.
#####..........####.
####...........####.
####...........####.
####...........#####
####...........#####
####...........####.
####...........####.
#####..........####.
#####.........#####.
#####.........#####.
.#####........####..
.######......#####..
..###############...
...#############....
....###########.....
.....#########......"""

_SEED_1 = """\
....................
....................
....................
....................
.....###############
####################
####################
####################
########...#########
...........#########
...........#########
...........#########
...........#########
...........#########
...........#########
...........#########
...........#########
...........#########
...........#########
...........#########
...........#########
...........#########
...........#########
...........#########
...........#########
...........#########
...........#########
...........#########"""

_SEED_PCT = """\
....................
....................
....................
....................
..#####........##...
.#######......###...
###..###......###...
###...###....###....
###....##...###.....
###....##...###.....
###....##...###.....
###...###..###......
###..###..###.......
.#######..##........
..#####...##........
.........###...#....
........###...####..
........##...#####..
........##..#######.
.......###.###...###
......###..###...###
.....###...##....###
.....###...##....###
.....##....##....###
....###....###...###
...###......########
...###.......######.
...##........#####.."""

# 2 / 4 / 8:2026-10-06 从**本次实盘落下的真实 PC HUD 裁剪**提取
# (run 20261006-024357-d42-hudfix,hud_00 / hud_01 两张同一次渲染)。
# 该帧真值:POINTS=10820 / ACC=48% → 字形集合 {0,1,2,4,8,%}。
# 提取脚本 `tools/extract_hud_seeds.py`,其中对 0/1/% 做了**交叉验证**:
# 新提字形与上方内嵌种子的 MSE = 0.0137~0.0494,全部低于阈值 0.075 ——
# 这同时证明了「提取链路正确」与「内嵌种子确实来自同字体同尺度」。
# 为什么必须用真实帧:T24 已证手画字形(如 ring)**不匹配**真实 0(超阈)。
_SEED_2 = """\
....................
....................
....................
.........####.......
.....############...
..################..
..##################
.###################
######........######
..###..........#####
...............#####
...............#####
...............#####
..............######
..............#####.
.............#####..
............######..
...........######...
........#######.....
......########......
.....########.......
....########........
...######...........
..######............
####################
####################
####################
####################"""

_SEED_4 = """\
....................
....................
....................
....................
............####....
...........#####....
...........#####....
..........######....
.........#######....
........########....
.......#####.###....
......#####..###....
......####...###....
.....####....###....
.....###.....###....
....###......###....
...####......###....
..#####......###....
.#####.......###....
####################
####################
####################
.###################
.............###....
.............###....
.............###....
.............###....
.............###...."""

_SEED_8 = """\
....................
....................
....................
........####........
....###########.....
...##############...
..######....#####...
.#######....######..
.####........######.
#####..........####.
.####..........####.
.####.........#####.
.#####.......#####..
.######.....#####...
...#############....
...#############....
..###############...
.#################..
#####........######.
####...........####.
####...........#####
####...........#####
####...........####.
#####........######.
.#######...########.
.#################..
..###############...
....############...."""


def _parse_seed(text: str, name: str = "?") -> np.ndarray:
    """把 ASCII 种子串解析成 (_GH, _GW) 的 0/1 浮点阵。

    **先逐行查宽度再 np.array**:否则行宽不齐时 numpy 只会抛一句
    "inhomogeneous shape",要靠回溯才找得到是哪一行 —— 实测手抄种子时
    多打一个点就踩过(2026-10-06)。
    """
    lines = text.strip().splitlines()
    bad = [(i, len(ln)) for i, ln in enumerate(lines) if len(ln) != _GW]
    if bad:
        raise ValueError(
            f"种子 {name!r} 每行必须 {_GW} 字符,异常行(行号,长度): {bad[:4]}"
            + (f" …共 {len(bad)} 行" if len(bad) > 4 else ""))
    a = np.array([[1.0 if c == "#" else 0.0 for c in ln] for ln in lines],
                 dtype=np.float32)
    if a.shape != (_GH, _GW):
        raise ValueError(f"种子 {name!r} 尺寸错: {a.shape} != {(_GH, _GW)}")
    return a


# ---------------------------------------------------------------- 窗口 -> HUD 框

_HUD_REF_WH = (1920, 1080)   # GRAB_BBOX 的参考客户区尺寸


def hud_region_for(
    full_region: tuple[int, int, int, int],
    ref: tuple[int, int] = _HUD_REF_WH,
) -> tuple[int, int, int, int]:
    """把参考分辨率下校准出的 HUD 框映射到实际客户区,返回**绝对屏幕坐标**
    `(left, top, width, height)`。

    Aim Lab 的 HUD 是**顶部居中、随渲染分辨率缩放**的固定 UI,所以按客户区
    宽高线性缩放即可。客户区恰为 1920x1080 时缩放系数为 1,映射退化为恒等
    —— 即逐像素实测的校准值原样生效(D42)。
    """
    fl, ft, fw, fh = (int(v) for v in full_region)
    rw, rh = (int(v) for v in ref)
    if rw <= 0 or rh <= 0:
        raise ValueError(f"参考分辨率非法: {ref}")
    sx, sy = fw / rw, fh / rh
    left = fl + int(round(GRAB_BBOX[0] * sx))
    top = ft + int(round(GRAB_BBOX[1] * sy))
    w = int(round((GRAB_BBOX[2] - GRAB_BBOX[0]) * sx))
    h = int(round((GRAB_BBOX[3] - GRAB_BBOX[1]) * sy))
    return (left, top, max(1, w), max(1, h))


class HudStrip:
    """HUD 专用取帧 —— 走**与视觉管线相同的 DXGI 层**(dxcam / bettercam)。

    ===========================================================================
    为什么不复用主循环已经抓到的 frame(D42.4 修正 3)
    ===========================================================================
    `DualWindowSource` 在**跟踪态**抓的是中心 900x900 裁窗,上沿 y=90 恰好切掉
    HUD(数字行 y∈[64,86] 全在窗外)。实测 d42-score 局 `n_full_reads=22` /
    `n_center_reads=6905` —— **全屏帧只占 0.318%**,"从已有帧里裁 HUD"这条路
    在双窗下基本永远裁不到。独立小区域抓取与当前用哪个窗无关,故对双窗免疫。

    为什么不用 PIL.ImageGrab:GDI BitBlt 在 flip-model / 独占全屏下可能取到
    非游戏内容。dxcam 走 Desktop Duplication —— 既然主捕获能看到游戏画面,
    它就一定能看到 HUD。**这是"与视觉管线同层"而不是"换个 API 试试"。**

    代价:每读一次是一次小区域抓帧(默认约 735x30),`--score-every 10`
    下约 6Hz,均摊远小于 1ms/拍。
    """

    def __init__(
        self,
        full_region: tuple[int, int, int, int],
        ref: tuple[int, int] = _HUD_REF_WH,
        backend: str = "auto",
        grab_retry: int = 4,
        grab_retry_ms: float = 4.0,
    ) -> None:
        from flyaim.bridge.capture import ScreenCapture

        self.bbox = hud_region_for(full_region, ref)
        left, top, w, h = self.bbox
        # out_size=None:不缩放 —— OCR 依赖原始像素形状,缩放会改变字重与笔画宽度
        #
        # grab_retry:D42.7d 实测 —— 捕获线程以 ~180Hz 抢占同一台 dxcam 相机,
        # 而 Desktop Duplication 只在**有新帧**时返回图像,于是低频的 HUD 条
        # 会被高频方抢光新帧(实测主线程 59 次读里 12 次拿不到首帧)。
        # 默认重试 4 次 × 4ms ≈ 覆盖一个 60Hz 刷新周期。
        self._cap = ScreenCapture(region=(left, top, w, h), out_size=None,
                                  backend=backend, grab_retry=grab_retry,
                                  grab_retry_ms=grab_retry_ms)
        self.backend = f"screen:{self._cap.backend_name}"
        self.n_reads = 0
        self.n_empty = 0
        self.n_stale = 0     # 复用上一帧的次数(HUD 会因此"看到旧画面")

    def read(self) -> np.ndarray:
        """返回 (H, W, 3) int32 RGB,尺寸 == (bbox 高, bbox 宽)。

        直接喂给 `ScoreHUD.read(img)`。int32 是为与旧的 ImageGrab 路径
        (`np.asarray(...).astype(np.int32)`)保持同一 dtype —— `_split_glyphs`
        的比较虽不受影响,但值域比较必须同型以免踩 uint8 回绕。
        """
        self.n_reads += 1
        frame, meta = self._cap.read()
        if meta.get("grab_stale"):
            self.n_stale += 1
        if frame is None or frame.size == 0:
            self.n_empty += 1
            raise RuntimeError("HUD 抓帧为空")
        return np.asarray(frame, dtype=np.int32)

    @property
    def stale_pct(self) -> float:
        """复用旧帧的占比。长期偏高 ⇒ HUD 读数不是"与渲染同步"的,需查。"""
        return 100.0 * self.n_stale / self.n_reads if self.n_reads else 0.0

    def close(self) -> None:
        self._cap.close()


# ---------------------------------------------------------------- 分段(模块级)

def split_glyphs(img: np.ndarray, x0: int, x1: int) -> list[np.ndarray]:
    """对某个值框做字符分段;返回归一化 (_GH, _GW) 的 glyph 列表(可能为空)。

    **模块级且不依赖模板库** —— 所以"某个框里有几段"这个判据永远可用,哪怕
    一个字都没学过。`time_glyph_count()` 就建立在这一点上,而它正是判断
    「回合到底开始了没有」最可靠的信号。
    """
    reg = img[_DIGIT_Y[0]:_DIGIT_Y[1], x0:x1]
    white = ((reg[:, :, 0] > _WHITE) & (reg[:, :, 1] > _WHITE)
             & (reg[:, :, 2] > _WHITE))
    cols = white.sum(axis=0)
    xs = np.where(cols > 0)[0]
    if xs.size == 0:
        return []
    segs: list[tuple[int, int]] = []
    s = int(xs[0])
    for i in range(1, xs.size):
        if int(xs[i]) - int(xs[i - 1]) >= _GAP:
            segs.append((s, int(xs[i - 1]) + 1))
            s = int(xs[i])
    segs.append((s, int(xs[-1]) + 1))
    out = []
    from PIL import Image
    for a, b in segs:
        g = white[:, a:b].astype(np.uint8) * 255
        g = np.asarray(Image.fromarray(g).resize((_GW, _GH), Image.BILINEAR),
                       dtype=np.float32) / 255.0
        out.append(g)
    return out


def time_glyph_count(img: np.ndarray) -> int:
    """TIME 框的分段数 —— **不依赖任何模板**,永远可用。

        1 = 练习/自由模式的 `∞`  → **没开局**
        5 = `MM:SS` 倒计时        → **计分局进行中**
        其它 = 该怀疑抓帧或布局
    """
    return len(split_glyphs(img, *_BOX_X["time"]))


class ScoreHUD:
    """读 Aim Lab 顶部 HUD 三框分数。grab_fn 注入便于测试(默认 PIL.ImageGrab)。"""
    def __init__(self, grab_fn=None) -> None:
        self._grab_fn = grab_fn or self._default_grab
        # 字符 -> 样本列表(20x28 float);内嵌种子冷启动。
        # {0,1,2,4,8,%} 来自真实 PC 帧(见各 _SEED_* 的注释);
        # 缺 3/5/6/7/9 与 ':' —— 由 TimeCalibrator 在**真实倒计时**上补齐。
        self._seeds = {
            "0": _SEED_0, "1": _SEED_1, "2": _SEED_2,
            "4": _SEED_4, "8": _SEED_8, "%": _SEED_PCT,
        }
        self.templates: dict[str, list[np.ndarray]] = {}
        self.reset_templates()
        self.n_reads = 0
        self.n_unknown = 0       # 有字符未认出的次数(诊断:模板缺口)
        self.last_raw: dict = {}

    def reset_templates(self) -> None:
        """把模板库还原到**冷启动种子**状态(供 TimeCalibrator 回滚)。

        D42.7 说过:同帧回读是循环论证,证明不了标签对不对。所以回滚的触发
        必须来自**跨帧**校验(见 TimeCalibrator),而不是"学完立刻读同一帧"。
        """
        self.templates = {c: [_parse_seed(t, c)] for c, t in self._seeds.items()}

    @property
    def n_templates(self) -> int:
        return sum(len(v) for v in self.templates.values())

    # ------------------------------------------------------------ 抓帧
    @staticmethod
    def _default_grab() -> np.ndarray:
        from PIL import ImageGrab
        return np.asarray(ImageGrab.grab(bbox=GRAB_BBOX)).astype(np.int32)

    # ------------------------------------------------------------ 分段
    def _split_glyphs(self, img: np.ndarray, x0: int, x1: int) -> list[np.ndarray]:
        """对一个值框做字符分段;薄封装 `split_glyphs`(保留旧调用点)。"""
        return split_glyphs(img, x0, x1)

    # ------------------------------------------------------------ 匹配
    def _match(self, glyph: np.ndarray) -> tuple[str | None, float]:
        best_c, best_d = None, 1e9
        for c, samples in self.templates.items():
            for t in samples:
                d = float(np.mean((glyph - t) ** 2))
                if d < best_d:
                    best_c, best_d = c, d
        if best_d <= _MSE_OK:
            return best_c, best_d
        return None, best_d

    def _learn(self, glyph: np.ndarray, char: str) -> None:
        lst = self.templates.setdefault(char, [])
        # 与已有样本几乎相同则不重复存(防样本库被单一字形塞满)
        for t in lst:
            if float(np.mean((glyph - t) ** 2)) < 0.01:
                return
        lst.append(glyph)
        if len(lst) > _MAX_SAMPLES:
            lst.pop(0)

    # ------------------------------------------------------------ 读框
    def _read_number(self, img: np.ndarray, box: str,
                     strip_pct: bool = False) -> int | None:
        glyphs = self._split_glyphs(img, *_BOX_X[box])
        if not glyphs:
            return None
        if strip_pct and glyphs:
            c, _ = self._match(glyphs[-1])
            if c == "%":
                glyphs = glyphs[:-1]
            elif len(glyphs) >= 3:
                # 末段不是 % 但段数 >=3:可能是 % 未识别 —— ACC 必然带 %,拒读
                return None
        chars = []
        for g in glyphs:
            c, d = self._match(g)
            if c is None or not c.isdigit():
                self.n_unknown += 1
                return None
            chars.append(c)
        try:
            return int("".join(chars))
        except ValueError:
            return None

    # ------------------------------------------------------------ TIME 自校准
    def calibrate_from_time(self, img: np.ndarray, remain_s: int) -> bool:
        """用已知的倒计时显示值学习字形。remain_s = 游戏 TIME 框当前应显示的
        秒数(0-60);格式 %02d:%02d(如 01:00 / 00:45)。段数!=5 不学习
        (练习模式 ∞ / 结算闪烁等)。返回是否学到。"""
        if not (0 <= remain_s <= 5999):
            return False
        text = f"{remain_s // 60:02d}:{remain_s % 60:02d}"
        glyphs = self._split_glyphs(img, *_BOX_X["time"])
        if len(glyphs) != 5:
            return False
        ok = False
        for g, ch in zip(glyphs, text):
            self._learn(g, ch)
            ok = True
        return ok

    # ------------------------------------------------------------ 主入口
    def read(self, img: np.ndarray | None = None) -> dict:
        """读三框。返回 {points, acc_pct, time_s, time_glyphs, n_templates}。
        任何框置信不足 → 对应字段 None。

        `time_glyphs` 是 TIME 框的**分段数**,是分辨"抓帧坏"与"内容变了"的
        关键诊断量(D42.6):`1` = 练习态 `∞`,`5` = 正常倒计时 `MM:SS`,
        其它 = 该怀疑抓帧或布局。**它不依赖任何模板,永远有效。**

        img 可注入(测试);None 时自行抓屏。
        """
        if img is None:
            img = self._grab_fn()
        self.n_reads += 1
        # TIME 框:5 段 MM:SS 才读(练习模式 ∞ 返回 None)
        glyphs = self._split_glyphs(img, *_BOX_X["time"])
        out = {
            "points": self._read_number(img, "points"),
            "acc_pct": self._read_number(img, "acc", strip_pct=True),
            "time_s": None,
            "time_glyphs": len(glyphs),
            "n_templates": sum(len(v) for v in self.templates.values()),
        }
        if len(glyphs) == 5:
            chars = []
            for i, g in enumerate(glyphs):
                c, _ = self._match(g)
                if c is None:
                    chars = None
                    break
                chars.append(c)
            if chars is not None and chars[2] == ":":
                try:
                    out["time_s"] = int(chars[0] + chars[1]) * 60 + int(chars[3] + chars[4])
                except ValueError:
                    pass
        self.last_raw = out
        return out

    # ------------------------------------------------------------ 便捷类方法
    @classmethod
    def grab_and_calibrate(cls, hud: "ScoreHUD", remain_s: int) -> dict:
        """抓一帧:先自校准(若给了 remain_s),再读。供主循环每 N 拍调用。"""
        img = hud._grab_fn()
        if remain_s >= 0:
            hud.calibrate_from_time(img, remain_s)
        return hud.read(img)


def demo() -> None:  # pragma: no cover - 手动调试入口
    hud = ScoreHUD()
    for _ in range(3):
        print(hud.read())
        _time.sleep(0.5)


# ---------------------------------------------------------------- TIME 在线校准

class TimeCalibrator:
    """TIME 倒计时在线校准 —— **锚定在观测到的 ∞→倒计时切换**,不用宿主时钟。

    ===========================================================================
    为什么宿主时钟不行(D42.5 的缺陷)
    ===========================================================================
    原实现:`remain = round_len - (本进程 elapsed)`。但游戏要等「进入任务 + 开场
    动画」之后时钟才起跳,两者天然差几秒。按错位的秒数学习 = 给字形打上**错误
    的数字标签**存进模板库 → 之后读什么都错。**触发过一次可能比从不触发更糟。**

    本类改成三点:

      1. **锚点来自观测**:先见到 `∞`(TIME 只有 1 段)= 未开局;此后第一次见到
         `5 段` 的那一刻就是**回合真正开始** → 以此建锚,与宿主时钟无关。
      2. **标签 = `round_len - (now - anchor)`**,且只在**秒中段**
         (`remain % 1 ∈ [0.25, 0.75]`)学习,防跨秒边界误学。
      3. **跨帧校验**:模板齐了之后,每帧把读回来的 TIME 与预测值比对;连续
         `CONFIRM` 次不符 ⇒ 判定锚点或标签有错 → **回滚模板库并停用学习**,
         并留下 `note` 说明原因。

    ⚠️ **必须用另一帧校验**。用刚学的模板读同一帧是循环论证(距离恒 0,必然
    "通过"),证明不了标签对不对 —— 这就是 D42.7 里明确记为"未做的事"的那条。
    这里的校验发生在**后续帧**上,标签错了它就会露馅。
    """

    VERIFY_TOL = 2      # 允许 ±2 秒(取整 + 采样时差)
    CONFIRM = 3         # 连续多少次不符才停用
    NEED = "0123456789:"

    def __init__(self, hud: ScoreHUD, round_len_s: float = 60.0) -> None:
        self.hud = hud
        self.round_len = float(round_len_s)
        self.anchor_t: float | None = None
        self.saw_infinity = False
        self.n_learn = 0
        self.n_verify_ok = 0
        self.n_verify_bad = 0
        self.disabled = False
        self.note = ""

    def missing(self) -> list[str]:
        return [c for c in self.NEED if c not in self.hud.templates]

    def observe(self, img: np.ndarray, now: float) -> dict:
        """喂一帧 HUD 裁剪。返回诊断字典(不抛异常)。"""
        glyphs = self.hud._split_glyphs(img, *_BOX_X["time"])
        ng = len(glyphs)
        miss = self.missing()
        out: dict = {"time_glyphs": ng, "time_s": None, "learned": not miss,
                     "missing": miss, "disabled": self.disabled, "note": ""}
        if self.disabled:
            out["note"] = "已停用"
            return out
        # 练习态 ∞:记住"我见过未开局",这是锚点成立的前提
        if ng == 1:
            self.saw_infinity = True
            return out
        if ng != 5:
            return out
        if not self.saw_infinity:
            out["note"] = "本进程未见过 TIME=∞ → 无锚点,不学习(防错标)"
            return out
        if self.anchor_t is None:
            self.anchor_t = now
            self.note = f"锚点已建立 t={now:.2f}s(首次见到 5 段倒计时)"
        remain_f = self.round_len - (now - self.anchor_t)
        if remain_f < -2.0:
            self.disabled = True
            self.note = (f"倒计时已超局时长 {self.round_len:.0f}s → 锚点可疑,"
                         f"停用校准")
            out["disabled"] = True
            out["note"] = self.note
            return out
        if self.missing():
            # 只在秒中段学习:跨秒边界时显示的可能是下一秒的字形
            if remain_f > 0 and 0.25 <= (remain_f % 1.0) <= 0.75:
                txt = f"{int(remain_f) // 60:02d}:{int(remain_f) % 60:02d}"
                for g, ch in zip(glyphs, txt):
                    self.hud._learn(g, ch)
                self.n_learn += 1
            out["missing"] = self.missing()
            out["learned"] = not out["missing"]
            return out
        # 模板齐了 → 跨帧校验(用**这一帧**读,与预测比 —— 非循环)
        v = self._decode(glyphs)
        pred = int(round(remain_f))
        out["time_s"] = v
        if v is not None and abs(v - pred) <= self.VERIFY_TOL:
            self.n_verify_ok += 1
            self.n_verify_bad = 0
        else:
            self.n_verify_bad += 1
        if self.n_verify_bad >= self.CONFIRM:
            self.hud.reset_templates()
            self.disabled = True
            self.note = (f"跨帧校验连续 {self.CONFIRM} 次不符(读出 {v} vs 预测 "
                         f"{pred})→ **已回滚模板并停用学习**")
            out["disabled"] = True
            out["note"] = self.note
            out["learned"] = False
        return out

    def _decode(self, glyphs: list[np.ndarray]) -> int | None:
        chars = []
        for g in glyphs:
            c, _ = self.hud._match(g)
            if c is None:
                return None
            chars.append(c)
        if chars[2] != ":":
            return None
        try:
            return int(chars[0] + chars[1]) * 60 + int(chars[3] + chars[4])
        except ValueError:
            return None


def round_running(glyphs: int) -> bool:
    """TIME 分段数 -> 是否在**计分局**里。

    这是本模块最有用的一个判据,而且**不依赖任何模板**(只数分段):

        1 段 = 练习/自由模式的 `∞`  → 没开局
        5 段 = `MM:SS` 倒计时        → 计分局进行中
        其它 = 该怀疑抓帧或布局

    **为什么不能用"画面里有青靶"当判据**(2026-10-06 实盘教训):大厅/UI 上也有
    青色元素。实测 `autostart` 在 (1664,43) 检出一个 R=11.3px 的青色块就断定
    "已在任务内",于是整局 60 秒 TIME 全是 `∞` —— 分数 10,820、ACC 48% 都真实
    记录,但**没有任何计时器**,和 130,335 那个目标根本不在同一个量纲上。
    """
    return glyphs == 5


if __name__ == "__main__":
    demo()
