"""桥接控制器:接在 FrameSource 与 GainModel 之间的决策体。

===============================================================================
协议
===============================================================================
controller.act(frame) -> (2,) float32,归一化到 [-1,1](与 runner.Arm 同约定,
但输入只有帧,没有 state —— 桥接层里没人能拿到"环境状态")。

可选属性:
    .name        str,遥测/报告用
    .brain       有 .spikes/.rates 属性的引擎(FlyController 才有),供遥测热图
    .blind       bool,该控制器是否保证只看像素(遥测侧须据此决定能否用检测)

诚实声明之二:EyeController(见文件末尾)保留真实复眼(6,098 感光细胞),
但**绕过了连接组** —— 它是 D25 的实战版,不是果蝇臂。

诚实声明之一:SeekController 是**管道校验臂**
    它用 detect.find_target 的像素误差做 PD 控制,本质是「偷看答案的经典
    控制器」。它的用途只有一个:在接入真实鼠标/真实屏幕前,验证
    捕获->决策->注入整条链路的时序与方向正确性。它**不是**实验对照臂,
    不能出现在任何判定里(判定规则冻结在 CONTRACT.md 第 3 节)。
"""

from __future__ import annotations

import logging

import numpy as np

from flyaim.bridge.detect import Detection, find_target, find_targets
from flyaim.config import RetinaConfig

logger = logging.getLogger(__name__)


class FlyController:
    """真实链路:Retina -> Connectome -> Readout。与离线实验唯一合法的果蝇臂同源。

    system: flyaim.pipeline.FlySystem(装配参数必须与正式实验一致:
    indeg 工作点 + trained 读出权重,见 tools/aimlab_bridge.py 的装配函数)。
    """

    name = "fly"
    blind = True  # FlySystem 只收帧;架构级保证见 pipeline.FlyArm

    def __init__(self, system) -> None:
        self.system = system
        self.last_act_ms = 0.0

    @property
    def brain(self):
        return self.system.brain

    def act(self, frame: np.ndarray) -> np.ndarray:
        import time as _t

        t0 = _t.perf_counter()
        a = np.asarray(self.system.act(frame), dtype=np.float32).reshape(2)
        self.last_act_ms = (_t.perf_counter() - t0) * 1000.0
        return a

    def reset(self) -> None:
        self.system.reset()

    def close(self) -> None:
        pass


class SeekController:
    """纯视觉 PD 管道校验臂:像素误差 -> 归一化 action(见模块 docstring)。

    瞄准点(aim point)的确定:
        use_aim_detect=True 时检测帧内的**准星标记**(默认白色)作为瞄准点;
        检测不到则回退画面中心。为什么不用画面中心:
        - 2D 靶场彩排里,准星移动不改变靶的屏幕位置,若以画面中心为瞄准点,
          误差信号与准星位置脱钩,PD 会一直朝同一边推(实测:准星贴墙振荡);
        - 真实 FPS 里捕获区域若不完全居中,画面中心 ≠ 准星,同样会引入恒定
          偏置误差。检测真实准星标记对两种场景都更正确。
        注意 Aim Lab 里白色 UI 文字可能干扰准星检测 —— 用 preview 帧目检。

    kp:每「半屏误差」输出多少 action;kd:对上帧误差变化的阻尼。
    未检出靶时输出 0(静止),并记 not_found 计数。
    """

    name = "seek(plumbing)"
    blind = False  # 依赖检测器(观测者侧信息),不可当实验臂

    def __init__(self, ref_color=(70, 170, 255), tolerance: float = 60.0,
                 kp: float = 1.2, kd: float = 0.15,
                 aim_color=(240, 240, 240), aim_tolerance: float = 50.0,
                 use_aim_detect: bool = True, lock: bool = True) -> None:
        self.ref_color = tuple(ref_color)
        self.tolerance = float(tolerance)
        self.kp = float(kp)
        self.kd = float(kd)
        self.aim_color = tuple(aim_color)
        self.aim_tolerance = float(aim_tolerance)
        self.use_aim_detect = bool(use_aim_detect)
        self.lock = bool(lock)  # 目标锁定:治"两目标间摇摆"(2026-10-04 用户观察)
        self._locked_xy: tuple[float, float] | None = None
        self._prev_err = np.zeros(2, dtype=np.float32)
        self.last_detection: Detection | None = None
        self.aim_detection: Detection | None = None
        self.n_not_found = 0

    def act(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        if self.lock:
            cands = find_targets(frame, ref_color=self.ref_color,
                                 tolerance=self.tolerance)
            if not cands:
                self._locked_xy = None
                det = Detection(ok=False)
            else:
                if self._locked_xy is not None:
                    det = min(cands, key=lambda d: (d.cx - self._locked_xy[0]) ** 2
                              + (d.cy - self._locked_xy[1]) ** 2)
                else:
                    det = cands[0]  # 默认最大块
                self._locked_xy = (det.cx, det.cy)
            self.last_detection = det
        else:
            det = find_target(frame, ref_color=self.ref_color, tolerance=self.tolerance)
            self.last_detection = det
        if not det.ok:
            self.n_not_found += 1
            self._prev_err = np.zeros(2, dtype=np.float32)
            return np.zeros(2, dtype=np.float32)
        aim = None
        if self.use_aim_detect:
            aim = find_target(frame, ref_color=self.aim_color,
                              tolerance=self.aim_tolerance, min_area_px=4)
        self.aim_detection = aim
        if aim is not None and aim.ok:
            ax, ay = aim.cx, aim.cy
        else:  # 回退:画面中心
            ax, ay = (w - 1) / 2.0, (h - 1) / 2.0
        err = np.array([det.cx - ax, det.cy - ay], dtype=np.float32)
        a = self.kp * err / (w / 2.0) - self.kd * (err - self._prev_err) / (w / 2.0)
        self._prev_err = err
        return np.clip(a, -1.0, 1.0).astype(np.float32)

    def reset(self) -> None:
        self._prev_err = np.zeros(2, dtype=np.float32)
        self._locked_xy = None
        self.n_not_found = 0

    def close(self) -> None:
        pass


class EyeController:
    """桥接版复眼视觉伺服:帧 -> 6,098 感光细胞 -> 小眼空间图 -> 刹停律。

    ===========================================================================
    它是什么、不是什么(接真实游戏前必须读)
    ===========================================================================
    是:`flyaim.baselines.eye_servo.EyeServoArm` 的桥接外壳 —— 与离线
    (D25,首达 12.5 帧、稳靶后 100%)**同一份代码**,只是把输入从
    `Arena.render()` 换成屏幕捕获帧。视觉入口仍是果蝇自己的复眼:
    R1-R6 亮度通路 + R7/R8 色觉通路,没有手搓假眼睛。

    不是:它**绕过了连接组**。所以它不能进任何"果蝇会不会瞄准"的判定,
    它回答的是「这个任务有多难」。这条边界与 D25 完全一致。

    实战相关的三处语义开关(都在 EyeServoArm 里,这里只做装配与遥测):
      * `chroma="r8"` —— Aim Lab 默认靶是**青色**(R<G),走 R8 短波通路;
        离线靶场的**红靶**走 R7 长波通路。这不是换参数,是换感光细胞群。
      * `aim_mode="center"` —— 真实 FPS 里准星**钉在画面中心不动**,动的是相机;
        离线 2D 靶场里准星自己会动。两者的"瞄准点"语义完全不同,
        错用会把上一拍命令外推到没动的准星上(详见 arm 的 docstring)。
      * `search="scan"` —— 看不到靶时慢速扫视。实战必备:D18/D19 实测
        视角一转离靶区就再也回不来。

    另:准星颜色在实战里**不可靠**(Aim Lab 的枪模型也是红的,检测会锁到枪上,
    见 AIMLAB.md §4 Phase B),所以实战默认 aim_mode="center",不依赖准星检测。
    """

    name = "eye"
    blind = True  # 输入只有帧(架构级:act 只接受 frame)

    def __init__(self, arm, keep_detection: bool = True) -> None:
        self.arm = arm
        self.keep_detection = bool(keep_detection)
        self.last_act_ms = 0.0
        self.last_detection: Detection | None = None

    def act(self, frame: np.ndarray) -> np.ndarray:
        import time as _t

        t0 = _t.perf_counter()
        a = np.asarray(self.arm.act(frame), dtype=np.float32).reshape(2)
        self.last_act_ms = (_t.perf_counter() - t0) * 1000.0
        if self.keep_detection:
            # 遥测/预览用:把复眼的估计转成像素 Detection。
            # ⚠️ 这只进文件与画面标注,**不回流给控制器**(控制器只吃 frame)。
            px = self.arm.last_target_px
            if px is None:
                self.last_detection = Detection(ok=False)
            else:
                self.last_detection = Detection(
                    ok=True, cx=float(px[0]), cy=float(px[1]),
                    radius_px=float(self.arm.last_target_radius_px),
                    score=float(self.arm.last_err_px),
                )
        return a

    def reset(self) -> None:
        self.arm.reset(0)
        self.last_detection = None

    def close(self) -> None:
        pass

    def describe(self) -> dict:
        d = {
            "name": self.name,
            "chroma_channel": self.arm.chroma,
            "aim_mode": self.arm.aim_mode,
            "search": self.arm.search,
            "v_px": self.arm.v_px,
            "eye_grid": [self.arm.eye_rows, self.arm.eye_cols],
            "n_receptors": int(self.arm.retina.input_neuron_ids.size),
            "retinotopy": getattr(self.arm.retina, "_retinotopy_used", "?"),
            "n_no_target": int(self.arm.n_no_target),
            "n_no_aim": int(self.arm.n_no_aim),
        }
        return d


class RandomController:
    """均匀随机(离线实验的随机臂,用于桥接层空转对照)。"""

    name = "random"
    blind = True

    def __init__(self, seed: int = 0) -> None:
        self.rng = np.random.default_rng(seed)

    def act(self, frame: np.ndarray) -> np.ndarray:
        del frame
        return self.rng.uniform(-1.0, 1.0, size=2).astype(np.float32)

    def reset(self) -> None:
        pass

    def close(self) -> None:
        pass


class ZeroController:
    """恒零(纯冻结:验证捕获与遥测在无动作时也正常)。"""

    name = "zero"
    blind = True

    def act(self, frame: np.ndarray) -> np.ndarray:
        del frame
        return np.zeros(2, dtype=np.float32)

    def reset(self) -> None:
        pass

    def close(self) -> None:
        pass


class HybridController:
    """果蝇 + 视觉伺服混合:action = α·seek + (1-α)·fly。

    **诚实声明(这不是"果蝇会瞄准了")**:瞄准由 seek(经典视觉 PD,
    偷看检测结果)承担;果蝇网络以权重 (1-α) 真实叠加在控制回路里,
    其以权重 β 加性地叠加在 seek 输出上(α 稀释版实测会把均衡点附近的
    矫正力稀释到无法收敛:准星停在离靶 21px 处 55 秒进不了 10px 触发圈)。
    果蝇的因果贡献用 β 消融量化:β=0(纯 seek)vs β=0.25 的命中数与
    轨迹偏差差,就是"接上连接组后行为变了多少"的测量。

    为什么需要它:CONTRACT 冻结连接组、学习只允许在读出层,而两轮域内
    实验(离线 D11 + 游戏域 holdout r²=-0.67)一致表明 DN 群不编码
    目标方向 —— 果蝇独立瞄准在该架构下不可达。混合器让"果蝇在回路里"
    与"可交付的瞄准行为"同时成立,且不掩盖任何一方。
    """

    name = "hybrid"
    blind = False  # seek 部分用检测

    def __init__(self, fly_controller, beta: float = 0.25,
                 ref_color=(48, 224, 224), tolerance: float = 60.0,
                 fly_every: int = 2, kp: float = 2.2) -> None:
        self.fly = fly_controller
        self.seek = SeekController(ref_color=ref_color, tolerance=tolerance,
                                   use_aim_detect=False, kp=kp)
        self.fly_every = max(1, int(fly_every))  # 果蝇每 N 拍步进一次(扰动不需高频)
        self._tick = 0
        # β = 果蝇扰动权重:a = seek + β·fly。seek 不稀释(保住已验证的收敛),
        # 果蝇作为加性扰动真实进入控制回路;贡献 = 有/无扰动的命中差(消融)。
        self.beta = float(np.clip(beta, 0.0, 1.0))
        self.last_fly = np.zeros(2, dtype=np.float32)
        self.last_seek = np.zeros(2, dtype=np.float32)

    @property
    def brain(self):
        return self.fly.brain

    def act(self, frame: np.ndarray) -> np.ndarray:
        if self._tick % self.fly_every == 0:
            self._last_fly_out = np.asarray(self.fly.act(frame), dtype=np.float32).reshape(2)
        self._tick += 1
        a_fly = self._last_fly_out
        a_seek = np.asarray(self.seek.act(frame), dtype=np.float32).reshape(2)
        self.last_fly, self.last_seek = a_fly, a_seek
        self.last_detection = self.seek.last_detection  # 供 TriggerOnTarget 复用
        a = a_seek + self.beta * a_fly
        return np.clip(a, -1.0, 1.0).astype(np.float32)

    def reset(self) -> None:
        self.fly.reset()
        self.seek.reset()

    def close(self) -> None:
        self.fly.close()


def make_retina_probe_config() -> RetinaConfig:
    """返回默认视网膜配置(桥接层不改视网膜参数;留作显式入口以便未来调)。"""
    return RetinaConfig()


class TriggerOnTarget:
    """自动开火层:准星压住靶时左键点击 —— 靶场「每帧自动开火」的游戏版。

    **边界声明(与 CONTRACT 一致)**:arena 的设计是把"何时开火"从控制问题
    里拿掉(每帧自动判定)。游戏里对应的语义就是:几何上准星与靶重叠即点击。
    开火是环境层的固定规则,**不是网络的输出**;被测的永远只是瞄准。

    包装任意 controller(seek / fly);靶位来自 detect.find_target(观测者侧),
    只进开火判定与遥测。click_fn 由调用方注入(通常是 SendInputSink.click)。

    命中推断:Gridshot 里"准星在靶盘内点击 = 必命中"(hitscan 无散布),
    所以 err <= radius*err_frac 的点击直接计为 inferred hit。
    """

    name = "trigger"
    blind = False  # 用了检测(仅开火层);内层 controller 保持自己的 blind 属性

    def __init__(self, inner, click_fn, ref_color=(48, 224, 224), tolerance: float = 60.0,
                 err_frac: float = 0.9, cooldown_s: float = 0.22,
                 fire_frac: float | None = None, min_radius_px: float = 0.0,
                 sticky_px: float = 0.0) -> None:
        """err_frac 保留作向后兼容别名;fire_frac 是显式命名的开火门限。

        **为什么要分开"瞄准"与"开火"两个判据**(2026-10-05 实测修正):

        原实现用 `err <= radius * 1.15` 同时承担两个职责,结果在真机上
        **瞄得准却不开火**。原因实测很清楚(640x480 缩放口径):

            靶半径 radius  = 7.0 ~ 8.4 px   (靶越靠画面边缘,透视上越小)
            开火门限        = 8.1 ~ 9.7 px
            收敛后的 err   = 8.1 ~ 9.4 px   <- 与门限同量级,擦边

        即 err 与门限**共用了同一个数量级**,差 0.5px 就失手。而半径还会
        随靶的位置变化,使门限跟着抖 —— 这是"有时开有时不开"的直接成因。

        修法:门限改为 `max(radius * fire_frac, min_radius_px)`。两个旋钮
        各管一件事:
          * `fire_frac`(默认 1.4)放宽几何门限,吸收"擦边"抖动;
          * `min_radius_px` 给缩到很小的靶兜底,避免门限随透视缩到 0。
        """
        self.inner = inner
        self.click_fn = click_fn
        self.ref_color = tuple(ref_color)
        self.tolerance = float(tolerance)
        self.err_frac = float(err_frac)
        self.fire_frac = float(err_frac if fire_frac is None else fire_frac)
        self.min_radius_px = float(min_radius_px)
        self.sticky_px = float(sticky_px)  # >0 时启用目标粘滞(见 act 注释)
        self._prev_xy: tuple[float, float] | None = None
        self.cooldown_s = float(cooldown_s)
        self._last_fire = -1e9
        self.n_fires = 0
        self.n_hits_inferred = 0
        self.n_blocked_by_cooldown = 0
        self.n_geometric_miss = 0
        self.n_in_gate = 0
        self.last_detection: Detection | None = None

    @property
    def brain(self):
        return getattr(self.inner, "brain", None)

    def act(self, frame: np.ndarray) -> np.ndarray:
        import time as _t

        a = np.asarray(self.inner.act(frame), dtype=np.float32).reshape(2)
        # 内层(seek/hybrid)若本拍已检测过,直接复用,省一次 ~25ms 的连通域
        det = getattr(self.inner, "last_detection", None)
        if det is None or not det.ok:
            det = find_target(frame, ref_color=self.ref_color, tolerance=self.tolerance)
        if self.sticky_px > 0.0 and det is not None and det.ok:
            # 目标粘滞(治「换靶导致 err 跳变」):内层 find_target 永远选最大连通块,
            # 多靶并存时准星刚靠近 A,A 稍微变小就切到 B,err 每一拍都在两级之间跳,
            # 永远进不了开火门限 —— 这是「瞄得准却不开火」的主因(2026-10-05 实测:
            # seq 15 dx=-28.0 一路收敛,seq 17 突跳成 dy=+35.5,即换了靶)。
            # 修法:本拍在全图候选里挑离上一拍最近的那个,让 err 连续可比。
            cands = find_targets(frame, ref_color=self.ref_color, tolerance=self.tolerance)
            if cands and self._prev_xy is not None:
                det = min(cands, key=lambda d: (d.cx - self._prev_xy[0]) ** 2
                          + (d.cy - self._prev_xy[1]) ** 2)
                near = (det.cx - self._prev_xy[0]) ** 2 + (det.cy - self._prev_xy[1]) ** 2
                if near > self.sticky_px ** 2:
                    det = find_target(frame, ref_color=self.ref_color,
                                      tolerance=self.tolerance)  # 跳太远则信全局最大
        self.last_detection = det
        if det is None or not det.ok:
            return a
        self._prev_xy = (det.cx, det.cy)
        h, w = frame.shape[:2]
        err = float(np.hypot(det.cx - (w - 1) / 2.0, det.cy - (h - 1) / 2.0))
        limit = max(det.radius_px * self.fire_frac, self.min_radius_px)
        now = _t.perf_counter()
        if err <= limit:
            self.n_in_gate += 1
            if now - self._last_fire >= self.cooldown_s:
                if self.click_fn():
                    self._last_fire = now
                    self.n_fires += 1
                    self.n_hits_inferred += 1  # Gridshot:盘内点击必中
            else:
                self.n_blocked_by_cooldown += 1
        else:
            self.n_geometric_miss += 1
        return a

    def reset(self) -> None:
        self.inner.reset()
        self._last_fire = -1e9
        self.n_fires = 0
        self.n_hits_inferred = 0
        self.n_blocked_by_cooldown = 0
        self.n_geometric_miss = 0
        self.n_in_gate = 0
        self._prev_xy = None

    def close(self) -> None:
        self.inner.close()


class TeacherCollectController:
    """导师采集器:seek 当导师驱动相机,同时用真实链路步进网络并记录数据对。

    记录 (X=DN 放电率 [n_features], Y=seek 的归一化 action) —— 与离线
    fit_readout 的 collect_training_set 完全同构,只是"靶场帧"换成了
    "真实游戏帧"、"PID 导师"换成了"seek 导师"。

    数据在 close() 时落盘 npz(X, Y, 元信息)。注意 tick 成本 ≈ seek(25ms)
    + 视网膜+脑(45ms)≈ 70ms → ~14 Hz,可接受。
    """

    name = "collect(seek-teacher)"
    blind = False  # seek 内层用检测;数据本身只含 (DN rates, seek action)

    def __init__(self, system, ref_color=(48, 224, 224), tolerance: float = 60.0,
                 kp: float = 1.2, kd: float = 0.15, use_aim_detect: bool = False,
                 out_npz=None) -> None:
        self.system = system
        self._seek = SeekController(ref_color=ref_color, tolerance=tolerance,
                                   kp=kp, kd=kd, use_aim_detect=use_aim_detect)
        self._out_npz = out_npz
        self._X: list[np.ndarray] = []
        self._Y: list[np.ndarray] = []
        self.last_detection = self._seek.last_detection

    @property
    def brain(self):
        return self.system.brain

    def act(self, frame: np.ndarray) -> np.ndarray:
        a = np.asarray(self._seek.act(frame), dtype=np.float32).reshape(2)
        # 平行步进真实链路(不参与控制,只为采集 DN 特征)
        drive = self.system.retina.frame_to_spikes(frame)
        n = self.system.brain.cfg.steps_per_frame
        if hasattr(self.system.brain, "step_many"):
            self.system.brain.step_many(n, drive[:, 0], drive[:, 1],
                                        self.system.brain.cfg.dt_ms)
        else:
            for _ in range(n):
                self.system.brain.step(drive[:, 0], drive[:, 1], self.system.brain.cfg.dt_ms)
        x = np.asarray(self.system.brain.rates, dtype=np.float32)[self.system.readout.feature_ids]
        self._X.append(x.astype(np.float32))
        self._Y.append(a.astype(np.float32))
        self.last_detection = self._seek.last_detection
        return a

    def reset(self) -> None:
        self.system.reset()

    def close(self) -> None:
        if self._out_npz and self._X:
            X = np.stack(self._X)
            Y = np.stack(self._Y)
            np.savez_compressed(self._out_npz, X=X, Y=Y,
                                feature_ids=self.system.readout.feature_ids.astype(np.int64))
            logger.info("TeacherCollect: 保存 %d 样本 -> %s", X.shape[0], self._out_npz)
        self._seek.close()
