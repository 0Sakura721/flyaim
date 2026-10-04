"""复眼视觉伺服臂(eye-servo)—— 「让果蝇百发百中」的正当实现。

================================================================================
这是什么,以及它诚实地替换了什么
================================================================================
保留:
    真实的 **6,098 个复眼感光细胞**(MaleCNS `ol_sensory`),真实的
    R1-R6 亮度通路 / R7-R8 色觉通路的**分工**(靶是红的 → 色觉通道是靶的专属
    探测器;准星是白的 → 亮度通道是准星的专属探测器)。视觉入口与果蝇一致:
    `frame -> Retina.frame_to_spikes -> (6098, 2) 驱动`。

替换:
    **连接组 + 2 维全局求和读出**。改为 `Retina.receptor_maps()` 把感光细胞驱动
    铺回 (24, 32) 小眼网格,在这张图上取"靶心 - 准星"的像素误差,再用一条
    刹停律(全速逼近 + 末端停车)输出鼠标增量。

为什么必须替换(不是调参能解决的):
    1. 6,098 维驱动经**随机稀疏投影**进 166,700 神经元 —— 空间位置在进入网络前
       就被随机混合抹掉;
    2. 读出只有 **2 维线性组合 + 率编码**,没有方向选择性(T4/T5 EMD 未建模)。
    结论见 DECISIONS.md D10–D24:六个学习体制全部阴性。

边界声明(与 CONTRACT 的关系)
    本臂**不读** `Arena.get_state()`,输入只有帧 —— 不违反禁止事项 1。
    但它也**不是**果蝇臂:连接组被绕过了。因此它**不得**替代 `fly` 进入
    CONTRACT 第 3 节的 fly vs shuffle 判定。它的用途是回答另一个问题:
    「这个任务对**果蝇的眼睛** + 一个 2 维读出,有多难?」
    答:大约 0 难度 —— 见 tools/aim_perfect.py 的实测。
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

from flyaim.config import RetinaConfig

__all__ = ["EyeServoArm"]


class EyeServoArm:
    """帧 -> 复眼编码 -> 小眼空间图 -> 误差 -> 刹停律 -> action。

    Parameters
    ----------
    retina:
        已构造好的 `flyaim.retina.encoder.Retina`(输入靶点 = 6,098 感光细胞)。
    v_px:
        执行器速度(px/帧),即 arena 的 `speed_px_per_action`。**这是执行器
        参数,不是环境状态** —— 刹停律必须知道"一帧能走多远"才能不冲过头。
    k_aim:
        准星质心的保护系数:亮度图上取 `max - k_aim*(max-min)` 以上的部分。
    aim_decay:
        检测不到准星时,把上一帧的准星估计向画面中心回拉的系数
        (纯粹是传感器记忆策略,不引入任何外部信息)。

    诚实提示:本臂对**色觉通路**有硬依赖(靶必须与背景有彩色对比)。
    靶场设计已保证(`bg=(18,18,22)` 近灰、靶 `(235,70,70)` 高饱和红)。
    换配色会失效 —— 这一点与 `detect.py` 的颜色阈值法同性质,如实标注。
    """

    name = "eye-servo"
    blind = True  # 输入只有帧(架构级:act 只接受 frame)

    def __init__(self, retina, v_px: float = 14.0, k_aim: float = 0.10,
                 aim_decay: float = 0.0, cell_px: float = 20.0,
                 freeze_px: float = 0.0, slow_px: float = 0.0,
                 slow_frac: float = 0.30,
                 chroma: str = "r7", chroma_min_hz: float = 6.0,
                 aim_mode: str = "detect",
                 search: str = "none", search_amp: float = 1.0,
                 search_pitch: float = 0.25,
                 search_pitch_period: int = 170) -> None:
        """
        chroma:
            用哪一条色觉通路当靶探测器 —— **由靶的色相决定,不是可调参数**:
              "r7": 靶比背景更偏红(R > G)。离线靶场红靶 (235,70,70) 走这条;
              "r8": 靶比背景更偏绿/青(G > R)。Aim Lab 默认青靶 (48,224,224) 走这条。
            两条通路来自真实 R7(长波)/R8(短波)的**对立极性**,所以这不是作弊开关:
            果蝇的色觉系统本来就把"更红"与"更绿"分给不同的感光细胞。
        chroma_min_hz:
            色觉通道"看到靶了"的门限(Hz,与 retina 的驱动同量纲)。
            **必须有**,否则把噪声当靶:实测(模拟 FPS,青靶)
              靶在画面内 11° -> r8 峰值 33.4 Hz;4° -> 12.9 Hz;
              靶在画面外    -> r8 峰值 2.2 Hz(红准星/红枪模型漏过来的串扰)。
            默认 6.0 Hz(= max_drive_hz 的 3%)把两者分得很开。低于门限就判定
            "视野里没有靶",交给 search。接别的游戏/配色时这个门限要重测。
        aim_mode:
            "detect": 准星在画面里**会动**(离线 2D 靶场:准星动、背景静止),
                      瞄准点靠检测 + 传出拷贝外推。
            "center": 准星**钉在画面中心不动**(真实 FPS:动的是相机,世界在流)。
                      此时瞄准点是常量,外推必须关掉 —— 把上一拍命令加到
                      "准星位置"上是错的(准星没动,动的是靶)。
                      ⚠️ 这个语义差异是我在接入实战时发现的最容易搞错的一处。
        search:
            看不到靶时怎么办。"none" = 输出 0 停住;"scan" = **扫掠式搜索**
            (绕 yaw 持续转 + 缓慢改变 pitch),把整个视野扫过一遍后重新捕获。
            ⚠️ 必须是"持续转",不能是"小圈微动":action 是**角速度**指令,
            小振幅振荡只让视线在 ±1° 内晃,永远扫不到 103° 的视野外面。
            **实战必需**:D18/D19 实测过"视角一转离靶区就再也回不来"
            (fly 臂全程 staring at wall,20 张预览里只有 1 张看得到靶)。
            搜索只依赖"我现在看不到靶"这一个观测,不偷看靶位。
        search_amp / search_pitch:
            搜索时的 (yaw, pitch) 角速度指令,单位是 action(1.0 = 满量程)。
            默认 (1.0, 0.15):yaw 满速转一圈 ≈ 360/deg_per_action 拍;
            pitch 慢速上漂,于是球面在若干圈内被扫遍。
        """
        self.retina = retina
        self.v_px = float(v_px)
        self.k_aim = float(k_aim)
        self.aim_decay = float(aim_decay)
        self.freeze_px = float(freeze_px)
        self.eye_rows = int(retina.eye_rows)
        self.eye_cols = int(retina.eye_cols)
        # cell_px:一个小眼接受角对应多少像素(靶场 640x480 / 24x32 = 20px)。
        # slow_px:近距限速区半径,缺省取 2 格(靶直径 ~2.2 格,即"快到时就减速")。
        self.cell_px = float(cell_px)
        self.slow_px = float(slow_px) if slow_px > 0 else 2.0 * self.cell_px
        self.slow_frac = float(slow_frac)
        if str(chroma) not in ("r7", "r8"):
            raise ValueError(f"chroma 只能是 r7(偏红靶)或 r8(偏青靶): {chroma}")
        if str(aim_mode) not in ("detect", "center"):
            raise ValueError(f"aim_mode 只能是 detect / center: {aim_mode}")
        if str(search) not in ("none", "scan"):
            raise ValueError(f"search 只能是 none / scan: {search}")
        self.chroma = str(chroma)
        self.chroma_min_hz = float(chroma_min_hz)
        self.aim_mode = str(aim_mode)
        self.search = str(search)
        self.search_amp = float(search_amp)
        self.search_pitch = float(search_pitch)
        # pitch 换向周期(拍)。**必须有界且居中**,否则 pitch 会一路漂到
        # ±88° 的钳位处卡死 —— 实测(2026-10-04):无界漂移时靶在画面外的
        # 再捕获率是 0%,与"完全不动"一模一样,即搜索形同虚设。
        self.search_pitch_period = max(4, int(search_pitch_period))
        self._aim: tuple[float, float] | None = None
        self._pending: np.ndarray | None = None
        self.n_no_target = 0
        self.n_no_aim = 0
        self.n_frames = 0
        self.last_err_px = 0.0
        self.last_maps: dict | None = None
        self.last_target_px: tuple[float, float] | None = None
        self.last_target_radius_px: float = 0.0

    # ------------------------------------------------------------------ 生命周期

    def reset(self, seed: int = 0) -> None:
        del seed
        self.retina.reset()
        self._aim = None
        self._pending = None
        self.n_no_target = 0
        self.n_no_aim = 0
        self.n_frames = 0
        self.last_err_px = 0.0
        self.last_target_px = None
        self.last_target_radius_px = 0.0

    # ------------------------------------------------------------------ 主接口

    def act(self, frame: np.ndarray, state=None) -> np.ndarray:
        del state  # 架构级:本臂永不消费环境状态
        self.n_frames += 1
        h, w = frame.shape[:2]

        drive = self.retina.frame_to_spikes(frame)          # (6098, 2)
        maps = self.retina.receptor_maps(drive, channel=0)
        self.last_maps = maps
        lum = maps["lum"]
        cw, ch = w / float(self.eye_cols), h / float(self.eye_rows)

        # ---- 靶:选定的色觉通路的正响应 ----
        # ⚠️ 掩膜必须**再膨胀一格**:粒度为 20px 时,靶盘的边缘格只被覆盖几个像素
        # (实测 seed=4 靶在 x=582:格 28/29/30 的覆盖是 20/20/4 px),尾格色差
        # 只有峰值的一成不到 → 不膨胀就漏掉 → 靶心估计自己就偏格子中心。
        chroma = np.clip(maps[self.chroma], 0.0, None)
        cmax = float(chroma.max())
        if cmax <= self.chroma_min_hz:
            self.n_no_target += 1
            self.last_target_px = None
            return self._search_action()
        tgt_core = chroma >= 0.10 * cmax
        tgt_mask = ndimage.binary_dilation(tgt_core, iterations=1)
        tgt = self.retina.centroid(chroma * tgt_mask)
        if tgt is None:
            self.n_no_target += 1
            self.last_target_px = None
            return self._search_action()
        # 供遥测/可视化:靶在**像素**坐标下的位置与视半径(半径由色觉斑块面积推)
        self.last_target_px = (float(tgt[0] * cw), float(tgt[1] * ch))
        self.last_target_radius_px = float(
            np.sqrt(float((chroma > 0).sum()) * cw * ch / np.pi)
        )

        # ---- 瞄准点 ----
        if self.aim_mode == "center":
            # 真实 FPS:准星钉在画面中心,**动的是相机**。此时瞄准点是常量,
            # 「外推上一拍命令」是错的(准星根本没动)。
            self._aim = ((self.eye_cols - 1) / 2.0, (self.eye_rows - 1) / 2.0)
        else:
            # 离线 2D:准星在画面里移动 —— 先按上一拍命令外推,再尽量用检测纠正。
            # 靶是**面**、准星是**细十字**(被 20x20 小眼积分稀释),必须先把靶的
            # 整个亮度足迹(膨胀版)挖掉,剩下的正响应才只有准星。
            self._dead_reckon()
            lum_m = lum.copy()
            lum_m[tgt_mask] = 0.0
            aim = self.retina.centroid(np.clip(lum_m, 0.0, None))
            if aim is not None:
                self._aim = aim                      # 检测到就纠正外推漂移
            elif self._aim is None:
                self.n_no_aim += 1
                self._aim = ((self.eye_cols - 1) / 2.0, (self.eye_rows - 1) / 2.0)
            else:
                self.n_no_aim += 1
        aim = self._aim

        # ---- 小眼单位 -> 像素(小眼接受角 ≈ 一格)----
        ex = (tgt[0] - aim[0]) * cw
        ey = (tgt[1] - aim[1]) * ch
        err = np.array([ex, ey], dtype=np.float64)
        r = float(np.linalg.norm(err))
        self.last_err_px = r
        if r <= self.freeze_px:
            self._pending = np.zeros(2, dtype=np.float64)   # 已在靶心:停住
            return np.zeros(2, dtype=np.float32)

        # ---- 刹停律:全速逼近 + 近距限速 ----
        # ⚠️ 近距必须限速:估计残差在靶附近有 ~±1 格(20px)的系统偏差(准星遮挡靶、
        # 小眼 20px 粒度),若继续满速,单帧 14px 的步长会把系统顶成一个
        # 7px <-> 21px 的极限环(实测稳靶率从 98% 掉到 55%)。
        # 限速后振荡幅度 < 5px,稳稳落在 22px 判定圈内。
        mag = min(1.0, r / self.v_px)
        if r < self.slow_px:
            mag = min(mag, self.slow_frac)
        a = err / r * mag
        if self.aim_mode == "center":
            self._pending = None                     # 准星不动 -> 无需外推
        else:
            self._pending = a * np.array([self.v_px / cw, self.v_px / ch])
        return a.astype(np.float32)

    # ------------------------------------------------------------------ 内部

    def _search_action(self) -> np.ndarray:
        """看不到靶时的搜索动作(search="scan" 时慢速画圆扫视)。

        为什么实战必须有(D18/D19 实测):视角一旦转离靶区,**没有任何机制能把
        相机带回来** —— fly 臂全程 staring at wall,20 张预览里只有 1 张看得到靶。
        搜索只依赖"我现在看不到靶"这一个观测,不偷看靶位,是环境无关的行为。
        """
        if self.search != "scan":
            self._pending = None
            return np.zeros(2, dtype=np.float32)
        # 扫掠:yaw 满量程**持续转**(一圈 ≈ 360/deg_per_action 拍),
        # pitch 走**对称三角波**(先走半程把中心找平,再每整周期换向)——
        # 于是"全方位角 × ±(半周期转角) 的俯仰"被扫遍,而不会漂到钳位卡死。
        n, P = self.n_frames, self.search_pitch_period
        if n <= P // 2:
            sgn = 1.0                       # 先向下走半程
        else:
            sgn = -1.0 if ((n - P // 2) // P) % 2 == 0 else 1.0
        a = np.array([self.search_amp, self.search_pitch * sgn], dtype=np.float64)
        self._pending = None
        return np.clip(a, -1.0, 1.0).astype(np.float32)

    def _dead_reckon(self) -> None:
        """把上一拍命令的位移加到瞄准点估计上(传出拷贝 / corollary discharge)。

        为什么必须有:准星压在靶上时,靶区掩膜会把准星整个挖掉 → 瞄准点只能靠
        记忆。而**我们自己刚刚把它移走了** —— 用旧估计算误差会得到一串方向错误的
        指令,实测形成 7px <-> 21px 的极限环(±14px,正好是执行器满速)。
        外推一拍后误差估计与实际一致,极限环消失。
        """
        if self._aim is not None and self._pending is not None:
            self._aim = (self._aim[0] + float(self._pending[0]),
                         self._aim[1] + float(self._pending[1]))
            self._aim = (float(np.clip(self._aim[0], 0.0, self.eye_cols - 1.0)),
                         float(np.clip(self._aim[1], 0.0, self.eye_rows - 1.0)))


def make_eye_servo(data_dir: str, retina_cfg: RetinaConfig | None = None,
                   v_px: float = 14.0, **kwargs) -> EyeServoArm:
    """便捷构造:从数据产物里取 6,098 个感光细胞装配一条 eye-servo。"""
    from flyaim.io import ARTIFACT_ROLES, load_roles
    from flyaim.retina.encoder import make_retina

    roles = load_roles(f"{data_dir}/{ARTIFACT_ROLES}")
    retina = make_retina(retina_cfg or RetinaConfig(), roles)
    return EyeServoArm(retina, v_px=v_px, **kwargs)
