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
                 use_aim_detect: bool = True, lock: bool = True,
                 downsample: int = 1,
                 xhair_px: float = 12.0, xhair_max_r: float = 8.0,
                 scan_amp: float = 0.35, scan_period_s: float = 1.6,
                 err_scale_px: float | None = None,
                 assoc_px: float = 0.0, switch_gain: float = 1.0,
                 lost_tol: int = 3,
                 fixate_px: float = 14.0, saccade_px: float = 20.0,
                 fire_gate_frac: float = 0.75) -> None:
        self.ref_color = tuple(ref_color)
        self.tolerance = float(tolerance)
        # >1 时检测前把帧整数倍降采样(坐标自动放大回原尺寸)。
        # 实测 700x700:/2 快 2.1x 且质心零偏差,/3 快 3.5x 偏 1px。
        self.downsample = max(1, int(downsample))
        # 准星排除(默认开):准星恒在画面中心且半径很小,放宽 tolerance 后会被
        # 误检为靶;它 err≈0 又静止,一旦被 sticky 选中就永远粘住 → 对着自己开枪。
        self.xhair_px = float(xhair_px)          # 离中心多近算"疑似准星"
        self.xhair_max_r = float(xhair_max_r)    # 半径多小才算"疑似准星"
        self.n_xhair_excluded = 0
        self.kp = float(kp)
        self.kd = float(kd)
        self.aim_color = tuple(aim_color)
        self.aim_tolerance = float(aim_tolerance)
        self.use_aim_detect = bool(use_aim_detect)
        self.lock = bool(lock)  # 目标锁定:治"两目标间摇摆"(2026-10-04 用户观察)
        self._locked_xy: tuple[float, float] | None = None
        self._lock_miss = 0                              # 连续未关联拍数
        # ---- 目标关联(assoc)与换靶迟滞(switch hysteresis)(2026-10-05) --------
        # 旧实现"每拍取离上拍位置最近的候选"有两个致命缺陷(用户反馈"晃动太大"
        # 的真因,preview 帧可见 3 个靶同时在屏):
        #   1. **无关联门限**:两个靶相距较近时,"最近"会在它们之间翻转 —— 准星
        #      被推到一个靶上,另一个立刻变成"更近",于是准星在靶间来回弹。
        #   2. **无换靶代价**:只要另一个候选略近一点就换,没有任何迟滞。
        # 新实现:只在"关联半径"内认自己是原靶;超出则记为丢靶(miss),连续
        # `lost_tol` 拍都关联不上才允许重新选靶。这样靶的**身份**是连续的,
        # 准星会稳稳跟一个靶,而不是追"当前最近的那个"。
        # assoc_px<=0 时自动 = err_scale_px*0.75(约一个靶的直径量级)。
        self.assoc_px = float(assoc_px)
        # switch_gain:允许换靶的"更近"倍数 —— 新靶到准星的距离必须小于
        # 当前靶的 1/switch_gain。>1 越保守(越不愿换),=1 只要更近就换。
        self.switch_gain = max(1.0, float(switch_gain))
        self.lost_tol = max(0, int(lost_tol))
        self.n_switch = 0      # 目标身份切换次数(诊断:每拍都在换=有问题)
        self._prev_err = np.zeros(2, dtype=np.float32)
        self.last_detection: Detection | None = None
        self.last_candidates: list[Detection] | None = None  # 本拍全部候选(供开火层复用)
        self.aim_detection: Detection | None = None
        self.n_not_found = 0
        # 搜靶(scan):看不到靶时慢速扫视(2026-10-05)。**这是命中率的命门**:
        # 原实现"未检出靶 -> 输出 0(静止)"形成死锁 —— 看不到靶就不动,不动
        # 就更看不到靶。实测 play-cov900 那局 2152 帧里 **det-ok=0、|act| 全程 0**,
        # 相机对着空墙站了整整 60 秒,开火 0 次。EyeController 早有 search="scan"
        # (D18/D19:"视角一转离靶区就再也回不来"),SeekController 此前没有。
        # scan_amp:扫视幅度(action 单位);scan_period_s:完成一轮扫视的秒数。
        self.scan_amp = float(scan_amp)
        self.scan_period_s = float(scan_period_s)
        self._scan_t0: float | None = None
        self.n_scans = 0
        # 误差归一化尺度(像素)。None = 旧行为(用 w/2,窗大小会改变增益);
        # 给定时用固定像素尺度,让闭环增益与捕获窗无关(dual 双窗必需)。
        self.err_scale_px = None if err_scale_px is None else float(err_scale_px)
        # ---- D41 扫视-固视双模态(2026-10-05 深夜) ----------------------------
        # 量化对比(相位相关逐帧角速度,同一脚本跑用户手机实录 vs 程序录像):
        #   理想(人手):运动占空比 5%,转动段 p50 33ms,**驻留 p50 629ms**
        #   程序(连续追踪):运动占空比 27%,驻留 p50 108ms —— 永远在闭环微调,
        #   相机"停不住",这就是用户「幅度较大的晃动」的定量本质。
        # 人手动与果蝇注视行为共享同一节律:**扫视(fast saccade)-> 固视
        # (hold still)**交替,而不是连续比例追踪。果蝇视觉系统天然以
        # saccade-fixation 方式稳定注视,这正是"发挥果蝇优势"的行为学落点。
        # 实现:err < fixate_px 进入固视(动作 0,完全停住,让球撞进准星/开火);
        # err > saccade_px 才重启追踪 —— 中间 10px 迟滞带防止检测噪声(±3px)
        # 引起状态抖动。固视期间检测/关联锁照常运行(纯视觉:固视是"不动",
        # 不注入任何预测)。fixate_px=0 可关闭(回退连续追踪)。
        self.fixate_px = float(fixate_px)
        self.saccade_px = float(saccade_px)
        self._fixating = False
        self.n_fixate = 0      # 固视拍数(战报:固视占比 = 停得住的程度)
        self.n_saccades = 0    # 完成的扫视次数(固视->追踪跳转)
        # ---- D41b 固视带动态收缩(d41-confirm 死锁,2026-10-05 深夜) --------
        # 开火门限 = fire_gate_frac × 靶半径(TriggerOnTarget 同参数,装配时
        # 回填)。**固视带必须整体落在该门限内侧**,否则出现「固视着(不动)
        # 但永远进不了门限」的死锁:实测 Gridshot 小靶 r=20.66 → 门限 15.5px
        # < 静态保持带 20px,滑行把 err 停在 17.5px 冻结 1153 拍,19 次重开
        # 也逃不掉(静止画面 + 确定性 PD 每次都滑回同一停点)。
        # fix_enter = min(静态带, 0.6×门限):切入留 40% 滑行余量,固视后惯性
        #   滑行(指针管道存货)不致滑出门限;
        # fix_hold  = min(静态带, 0.85×门限):保持带顶在门限内侧 85%,滑出
        #   即退固视恢复追踪 —— 门限内永远允许开火。
        self.fire_gate_frac = float(fire_gate_frac)
        self._fix_cooldown = 0  # 自救/扫视后的固视禁入冷却(拍),防立即重新锁死

    def _scan_action(self) -> np.ndarray:
        """无靶时的慢速扫视:Lissajous 轨迹覆盖画面(两个不同周期的正弦)。

        用不同周期避免"来回只扫一条线",能覆盖一块区域。幅度受 scan_amp 限制,
        所以扫视是**慢**的 —— 太快会晃、也会错过短暂出现的靶。
        """
        import time as _t

        if self._scan_t0 is None:
            self._scan_t0 = _t.perf_counter()
            self.n_scans += 1
        t = _t.perf_counter() - self._scan_t0
        w = 2.0 * np.pi / max(self.scan_period_s, 1e-3)
        return np.array([self.scan_amp * np.sin(w * t),
                         self.scan_amp * np.sin(w * 0.618 * t + 1.1)],
                        dtype=np.float32)

    def _pick_downsample(self, h: int, w: int) -> int:
        """按帧边长选降采样倍数:让检测像素量大致恒定(≈0.3-0.5 MP)。

        **为什么必须自适应**(2026-10-05 dual 双窗暴露的问题):
        原来 `downsample` 是固定值(默认 2)。中心窗 900 -> 检测 450x450=0.2MP,
        很快;但搜靶时换成全屏 1920 -> 检测 960x960=0.9MP,慢 4 倍,拍频掉一半。
        反过来,若为全屏调大固定值(如 4),切回中心窗又会欠采样(靶半径只剩 6px)。
        所以按**当前窗边长**动态选:大窗多降、小窗少降。用户显式给的
        `--downsample` 作为**下界**(不会比它更狠),避免配置被悄悄覆盖。
        """
        base = max(1, int(self.downsample))
        side = min(int(h), int(w))
        # 目标:降采样后边长 >= 256px。320 -> 256(2026-10-05 晚):实测 900 窗
        # ds=2 检测 10.6ms / ds=3 只要 6.1ms,而局内(游戏抢 CPU)act_p50 从
        # 空载 11.8ms 涨到 31ms,拍频 54 -> 29Hz —— 拍频减半=相位滞后加倍,
        # 振荡更凶。降到 ds=3 后质心偏差 ~1px(实测),远小于 28px 的开火门限,
        # 换回拍频划算。靶半径 25px 在 ds=3 下仍有 ~8px,足够质心估计。
        ds = 1
        while side // (ds + 1) >= 256 and ds < 8:
            ds += 1
        return max(base, ds)

    def _select_locked(self, cands: list[Detection], w: int, h: int) -> Detection:
        """从候选里选"仍是原来那个靶"的一个,带关联门限 + 换靶迟滞。

        规则(按优先级):
        1. 有锁定位置时,在 `assoc_px` 内找**离上拍观测位置最近**的候选 ——
           这就是"同一个靶"。
        2. 找不到(超出门限)则 `_lock_miss += 1`;只有连续丢 `lost_tol` 拍
           才允许重新选靶。期间**保持在最后观测位置**(不动,等它回来)。
        3. 重新选靶时也带迟滞:新靶必须比"若延续的旧靶"显著更近
           (`switch_gain` 倍),否则宁可不换 —— 避免两靶间来回弹。

        纯视觉纪律(2026-10-05 教训,D37):**动作只能落在有观测证据的位置上**。
        曾经用"上拍位置+速度"外推幻影来接力,结果靶飞出视野后幻影继续按旧速度
        滑行,准星跟着幻影一直往左上角推 —— 用户实测「靶出视野再入视野就一直
        往左上角滑」。速度外推已彻底删除:丢靶=停在最后看见的地方等,
        关联门限足够宽(240px),靶短暂遮挡/漂移后仍能重续身份。
        """
        cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
        gate = self.assoc_px if self.assoc_px > 0.0 else max(
            30.0, 0.75 * (self.err_scale_px or 320.0))

        # -- 1) 关联:离上拍观测位置最近的候选(无外推 —— 见 docstring 教训)
        if self._locked_xy is not None:
            lx, ly = self._locked_xy
            best = None; best_d2 = gate * gate
            for d in cands:
                dd = (d.cx - lx) ** 2 + (d.cy - ly) ** 2
                if dd <= best_d2:
                    best = d; best_d2 = dd
            if best is not None:
                self._locked_xy = (best.cx, best.cy)
                self._lock_miss = 0
                return best
            self._lock_miss += 1

        # -- 2) 还没到"允许重选"的程度:保持在最后观测位置(不追幻影、不改追别的靶)
        if self._locked_xy is not None and self._lock_miss <= self.lost_tol:
            lx, ly = self._locked_xy
            r = cands[0].radius_px if cands else 20.0
            return Detection(ok=True, cx=float(lx), cy=float(ly),
                             radius_px=float(r), area_px=0, score=0.0)

        # -- 3) 允许重选:挑离准星最近的候选。
        #    迟滞只对"旧靶还在附近、只是不那么近了"生效;若真丢了(`lost_tol`
        #    拍都没关联上),说明旧靶已消失,必须无条件改锁新靶 —— 否则会
        #    永久钉在一个已经不存在的坐标上(实测:靶跑掉后准星停在原位不动)。
        far = min(cands, key=lambda d: (d.cx - cx) ** 2 + (d.cy - cy) ** 2)
        if (self._locked_xy is not None and self._lock_miss <= self.lost_tol):
            old_d = np.hypot(self._locked_xy[0] - cx, self._locked_xy[1] - cy)
            new_d = np.hypot(far.cx - cx, far.cy - cy)
            if new_d * self.switch_gain >= old_d:
                return Detection(ok=True, cx=float(self._locked_xy[0]),
                                 cy=float(self._locked_xy[1]),
                                 radius_px=float(far.radius_px), area_px=0, score=0.0)
        self.n_switch += 1
        self._locked_xy = (far.cx, far.cy)
        self._lock_miss = 0
        return far

    def act(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        ds = self._pick_downsample(h, w)
        if self.lock:
            cands = find_targets(frame, ref_color=self.ref_color,
                                 tolerance=self.tolerance,
                                 downsample=ds,
                                 exclude_center_px=self.xhair_px,
                                 exclude_center_max_r=self.xhair_max_r)
            self.last_candidates = cands
            if not cands:
                self._locked_xy = None
                self._lock_miss = self.lost_tol + 1
                det = Detection(ok=False)
            else:
                det = self._select_locked(cands, w, h)
            self.last_detection = det
        else:
            det = find_target(frame, ref_color=self.ref_color, tolerance=self.tolerance,
                              downsample=ds,
                              exclude_center_px=self.xhair_px,
                              exclude_center_max_r=self.xhair_max_r)
            self.last_candidates = [det] if det.ok else []
            self.last_detection = det
        if not det.ok:
            self.n_not_found += 1
            self._prev_err = np.zeros(2, dtype=np.float32)
            # 搜靶:看不到靶不能傻站着(否则死锁),慢速扫视直到重新捕获。
            if self.scan_amp > 0.0:
                return self._scan_action()
            return np.zeros(2, dtype=np.float32)
        self._scan_t0 = None  # 一旦看到靶就结束扫视
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
        # ---- 误差归一化:用**像素**而非半屏宽 --------------------------------
        # 原实现 `err / (w/2)` 会让同一物理误差在不同窗下得到不同 action:
        # 全屏窗 w=1920 -> 除以 960;中心窗 w=900 -> 除以 450。等于"搜靶时
        # 增益自动减半,锁上后增益翻倍" —— dual 双窗一切换就改变闭环增益,
        # 是抖动与新振荡的来源。这里改用 `self.err_scale_px`(像素域固定值),
        # 让 action 只取决于"靶离准星多少像素",与窗大小无关。语义:
        # `kp` 每 err_scale_px 像素输出 kp 的 action(默认 320px,约 ±12.6° @FOV)。
        denom = (max(1.0, float(self.err_scale_px)) if self.err_scale_px is not None
                 else max(1.0, w / 2.0))
        # ---- D41 扫视-固视双模态 ------------------------------------------
        # 固视态:动作 0(真正停住)。迟滞:进固视 err<fixate_px,退出需
        # err>saccade_px(>fixate_px)。退出即记一次扫视。
        # D41b:进/保持带动态收缩到开火门限内侧(见 __init__ 注释)——
        # 静态带只对大靶(r≈28,门限 21px)成立,小靶门限收缩后会与保持带
        # 交形成死锁带。禁入冷却期间固视不进,让 PD 把准星重新推进门限。
        if self.fixate_px > 0.0:
            err_mag = float(np.hypot(err[0], err[1]))
            gate = self.fire_gate_frac * float(det.radius_px)
            fix_enter = min(self.fixate_px, 0.6 * gate)
            fix_hold = min(self.saccade_px, 0.85 * gate)
            if self._fix_cooldown > 0:
                self._fix_cooldown -= 1
            elif self._fixating:
                if err_mag > fix_hold:
                    self._fixating = False
                    self.n_saccades += 1
            elif err_mag < fix_enter:
                self._fixating = True
            if self._fixating:
                self.n_fixate += 1
                self._prev_err = err
                return np.zeros(2, dtype=np.float32)
        a = self.kp * err / denom - self.kd * (err - self._prev_err) / denom
        self._prev_err = err
        return np.clip(a, -1.0, 1.0).astype(np.float32)

    def force_unfixate(self, cooldown_beats: int = 8) -> None:
        """强制退出固视并短暂禁入(外层看门狗自救,2026-10-05 d41-confirm)。

        固视死锁(err 冻结在保持带内但 > 开火门限)靠重开**逃不掉** ——
        静止画面 + 确定性 PD 每次都滑回同一停点(实测 17.5px × 19 次重开)。
        正确自救:踢回追踪态重新收敛,并给几拍「禁入固视」冷却,防止下一拍
        err 仍 < 进入带时立即重新锁死。冷却期间 PD 正常输出,把准星推进
        开火门限;门限内开火层自然接管。
        """
        if self._fixating:
            self._fixating = False
            self.n_saccades += 1
        self._fix_cooldown = max(self._fix_cooldown, int(cooldown_beats))

    def reset(self) -> None:
        self._prev_err = np.zeros(2, dtype=np.float32)
        self._locked_xy = None
        self._lock_miss = 0
        self.last_candidates = None
        self.n_not_found = 0
        self._scan_t0 = None
        self.n_scans = 0
        self.n_switch = 0
        self._fixating = False
        self.n_fixate = 0
        self.n_saccades = 0
        self._fix_cooldown = 0

    def on_window_switch(self) -> None:
        """捕获窗在「搜靶全屏」与「跟踪中心窗」之间切换时被桥接层调用。

        两种窗的**局部坐标原点不同**(全屏窗原点在全屏左上,中心窗原点在
        屏幕中心减半窗),所以切换那一拍同一物理靶的坐标会突变。若不重置,
        锁定判定 `_locked_xy` 会把切换后的靶当成"另一个靶"从而重新锁,
        阻尼项 `_prev_err` 也会产生一个巨大的假导数脉冲(实测会让 action
        瞬间顶到 ±1,是"抽搐"的一个来源)。重置这三项即可,代价是丢掉
        一拍阻尼 —— 可忽略。
        """
        self._prev_err = np.zeros(2, dtype=np.float32)
        self._locked_xy = None
        self._lock_miss = 0
        self.last_candidates = None
        self._scan_t0 = None
        self._fixating = False  # 切窗坐标突变,固视判据失效,回扫视态
        self._fix_cooldown = 0

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
        self.last_candidates = getattr(self.seek, "last_candidates", None)
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
                 sticky_px: float = 0.0, downsample: int = 1,
                 xhair_px: float = 12.0, xhair_max_r: float = 8.0,
                 jump_reset_px: float = 0.0,
                 fire_confirm_beats: int = 2) -> None:
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
        self.downsample = max(1, int(downsample))
        # 准星排除(默认开):准星恒在画面中心且半径很小,放宽 tolerance 后会被
        # 误检为靶;它 err≈0 又静止,一旦被 sticky 选中就永远粘住 → 对着自己开枪。
        self.xhair_px = float(xhair_px)          # 离中心多近算"疑似准星"
        self.xhair_max_r = float(xhair_max_r)    # 半径多小才算"疑似准星"
        self.n_xhair_excluded = 0
        self._prev_xy: tuple[float, float] | None = None
        # >0 时:靶心比上一拍移动超过该像素数,视为换靶并清零冷却(见 act 注释)
        self.jump_reset_px = float(jump_reset_px)
        self.n_target_switches = 0
        self.cooldown_s = float(cooldown_s)
        self._last_fire = -1e9
        # D41 开火确认:连续 N 拍进门限才扣扳机(默认 2)。**为什么要确认**:
        # 扫视(saccade)刚到位的那一拍,前几拍的大额注入还在系统指针管道里
        # 消化(「提高指针精确度」EPP 会非线性放大高速段位移) —— 画面判定
        # err<门限,但**真实准星还在惯性滑行**,此刻点击必 miss。推迟 1 拍
        # (64Hz 下 15ms)等相机真正停稳:Gridshot 靶静止,代价可忽略。
        # 首局实测 231 发/~45 球 ≈ 5 发/球,首发命中率是最后的短板。
        self.fire_confirm_beats = max(1, int(fire_confirm_beats))
        self._gate_streak = 0
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
        # 复用内层的**候选列表**,避免 sticky 再扫一遍全图(2026-10-05 性能):
        # 内层 lock=True 时已经算过 find_targets,这里直接用同一份结果。
        reuse = getattr(self.inner, "last_candidates", None)
        # 内层是否已自带"目标锁定"(SeekController.lock / HybridController)。
        # 若自带,**开火层不得再用自己的 sticky 另选一个靶** —— 否则会出现
        # 「瞄准锁 A、开火看 B」的分歧:瞄准把 A 拖到中心(action→0)后相机停住,
        # 而开火层盯着离中心很远的 B,err 永远进不了门限 → 瞄得准却不开火
        # (2026-10-05 实盘定位)。此时开火层只做"信使",直接吃内层锁定的那个靶。
        inner_locked = bool(getattr(self.inner, "lock", False)) and det is not None and det.ok
        if (det is None or not det.ok) and not reuse:
            det = find_target(frame, ref_color=self.ref_color, tolerance=self.tolerance,
                              downsample=self.downsample,
                              exclude_center_px=self.xhair_px,
                              exclude_center_max_r=self.xhair_max_r)
        if self.sticky_px > 0.0 and det is not None and det.ok and not inner_locked:
            # 目标粘滞(治「换靶导致 err 跳变」):内层 find_target 永远选最大连通块,
            # 多靶并存时准星刚靠近 A,A 稍微变小就切到 B,err 每一拍都在两级之间跳,
            # 永远进不了开火门限 —— 这是「瞄得准却不开火」的主因(2026-10-05 实测:
            # seq 15 dx=-28.0 一路收敛,seq 17 突跳成 dy=+35.5,即换了靶)。
            # 修法:本拍在候选里挑离上一拍最近的那个,让 err 连续可比。
            cands = reuse if reuse else find_targets(
                frame, ref_color=self.ref_color, tolerance=self.tolerance,
                downsample=self.downsample,
                exclude_center_px=self.xhair_px,
                exclude_center_max_r=self.xhair_max_r)
            if cands and self._prev_xy is not None:
                det = min(cands, key=lambda d: (d.cx - self._prev_xy[0]) ** 2
                          + (d.cy - self._prev_xy[1]) ** 2)
                near = (det.cx - self._prev_xy[0]) ** 2 + (det.cy - self._prev_xy[1]) ** 2
                if near > self.sticky_px ** 2:
                    det = find_target(frame, ref_color=self.ref_color,
                                      tolerance=self.tolerance,
                                      downsample=self.downsample,
                                      exclude_center_px=self.xhair_px,
                                      exclude_center_max_r=self.xhair_max_r)  # 跳太远则信全局最大
        self.last_detection = det
        if det is None or not det.ok:
            return a
        # 换靶检测(2026-10-05):靶被打爆后画面会切到下一个靶,此时**不应**再受
        # 上一个靶的冷却限制 —— 冷却的本意是"别对同一个靶连点",不是"别打新靶"。
        # 实测 46.6Hz 下一拍 21.5ms,0.10s 冷却 = 4.65 拍,192 个进门限拍里
        # 有 124 拍被冷却挡掉(64%)—— 其中相当一部分是"新靶刚出现就被挡"。
        # 判据:靶心移动超过 jump_px 视为换靶,清零冷却。
        if self._prev_xy is not None and self.jump_reset_px > 0.0:
            moved = np.hypot(det.cx - self._prev_xy[0], det.cy - self._prev_xy[1])
            if moved > self.jump_reset_px:
                self._last_fire = -1e9
                self._gate_streak = 0   # 换靶:确认计数清零,新靶重新确认
                self.n_target_switches += 1
        self._prev_xy = (det.cx, det.cy)
        h, w = frame.shape[:2]
        err = float(np.hypot(det.cx - (w - 1) / 2.0, det.cy - (h - 1) / 2.0))
        limit = max(det.radius_px * self.fire_frac, self.min_radius_px)
        now = _t.perf_counter()
        if err <= limit:
            self.n_in_gate += 1
            self._gate_streak += 1
            if self._gate_streak >= self.fire_confirm_beats:
                # 确认期满:此时才谈得上开火/被冷却挡下
                if now - self._last_fire >= self.cooldown_s:
                    if self.click_fn():
                        self._last_fire = now
                        self.n_fires += 1
                        self.n_hits_inferred += 1  # Gridshot:盘内点击必中
                else:
                    self.n_blocked_by_cooldown += 1
            # streak < confirm:处于确认期(等相机停稳),不计入冷却挡下
        else:
            self.n_geometric_miss += 1
            self._gate_streak = 0
        return a

    def reset(self) -> None:
        self.inner.reset()
        self._last_fire = -1e9
        self.n_fires = 0
        self.n_hits_inferred = 0
        self.n_blocked_by_cooldown = 0
        self.n_geometric_miss = 0
        self.n_in_gate = 0
        self.n_target_switches = 0
        self._prev_xy = None
        self._gate_streak = 0

    def close(self) -> None:
        self.inner.close()

    def on_window_switch(self) -> None:
        """捕获窗切换:清 sticky 的"上一拍靶位",并转发给内层。

        开火层的 `_prev_xy` 与 `sticky_px` 是**按窗内局部坐标**存的;全屏窗
        与中心窗原点不同,切换后同一物理靶的坐标会整体平移(实测可达数百 px),
        若不清掉,`sticky_px` 会把这次平移误判成"靶跳走了"→ 触发换靶/冷却,
        白白吃掉一次开火窗口。清掉 `_prev_xy` 后下一拍直接用当前帧选靶,
        代价是一拍不做空间一致性,可忽略。
        """
        self._prev_xy = None
        hook = getattr(self.inner, "on_window_switch", None)
        if callable(hook):
            hook()

    def __getattr__(self, item):
        # 透传内层属性(n_limited/n_deadband/blind 等统计量),避免包装层
        # 把内层信息挡住 —— 这正是 D32「瞄准/开火分歧」的教训。
        # 显式声明的属性和 self.inner 不受影响(__getattr__ 仅在查找失败时触发)。
        # 防御:`inner` 自身尚未赋值时(极端情形)直接报错,避免无限递归。
        if item == "inner":
            raise AttributeError(item)
        return getattr(self.inner, item)


class ActionSmoother:
    """动作整形层:治「瞄得抖、太激进」(2026-10-05 用户反馈)。

    原始 seek 是纯 PD + **硬裁剪**(`np.clip(a, -1, 1)`)。实测两个毛病:

      1. **饱和 → 来回冲**:kp=2.2 时 err > (w/2)/2.2 ≈ 159px 就顶到 ±1.0。
         700px 窗口里靶只要离开中心 1/4 屏,控制器就满舵 —— 典型的 bang-bang。
         实测 12.3% 的帧 |action|>0.9,37.4% >0.5,err 变化符号翻转率 25%,
         表现为准星在靶两侧来回扫(用户说的"晃动太大")。
      2. **无速率限制 → 单拍猛跳**:增益标定是给"每拍转一点点"设计的,
         55ms/拍的旧 tick 下一拍满舵会转过头,下一拍再反向满舵 → 自激。

    本层做三件正交的事:
      * `soft`  —— 用 tanh 软饱和替代硬裁剪:误差小时近似线性(保精度),
                    误差大时渐进逼近 ±1(不顶死,留出纠偏余量)。tanh(a/k)*k
                    在 |a|<<k 时 ≈ a,所以小误差行为不变。
      * `max_delta` —— 每拍动作变化量的上限(slew-rate)。把"满舵跳变"限速成
                    "每拍最多动一点",从根上掐掉自激振荡。这是最有效的一个旋钮。
      * `deadband` —— 靶已进入中心死区时把动作拉零,避免"快到中心还在抖"
                     造成的过冲。**死区必须严格小于开火门限**,否则会把自己
                     卡在"进不了门限"的死区里(这正是 D30 的坑,别重蹈)。
    """

    name = "smooth"

    def __init__(self, inner, max_delta: float = 0.40, soft: float = 1.5,
                 deadband_px: float = 0.0, center_px: float = 0.0) -> None:
        """max_delta: 每拍动作最大变化量(<=0 关闭);
        soft: 软饱和拐点 k(k 越大越接近线性/越激进;<=0 关闭,退回硬裁剪);
        deadband_px: 靶离中心多近就输出 0(<=0 关闭);
        center_px: 画面中心坐标(用于 deadband;<=0 时从帧推断)。
        """
        import numpy as _np

        self.inner = inner
        self.max_delta = float(max_delta)
        self.soft = float(soft)
        self.deadband_px = float(deadband_px)
        self.center_px = float(center_px)
        self._prev = _np.zeros(2, dtype=_np.float32)
        self.n_limited = 0  # 被速率限制削过的拍数
        self.n_deadband = 0

    @property
    def brain(self):
        return getattr(self.inner, "brain", None)

    @property
    def name(self):
        return getattr(self.inner, "name", "?") + "+smooth"

    @property
    def last_detection(self):
        return getattr(self.inner, "last_detection", None)

    @property
    def last_candidates(self):
        return getattr(self.inner, "last_candidates", None)

    def __getattr__(self, item):
        if item == "inner":
            raise AttributeError(item)
        return getattr(self.inner, item)

    def on_window_switch(self) -> None:
        """捕获窗切换:重置限速的历史,并把事件转发给内层。

        `_prev` 是"上一拍真正发出去的动作";但切窗时**动作语义没变**
        (仍是归一化 action,与窗无关),所以这里其实**不该**清零 `_prev` ——
        清零反而会让限速器把切换那一拍当"从 0 起步"从而人为减速。
        真正要转发的是内层(SeekController)的锁定/阻尼重置。
        """
        hook = getattr(self.inner, "on_window_switch", None)
        if callable(hook):
            hook()

    def act(self, frame):
        import numpy as _np

        a = _np.asarray(self.inner.act(frame), dtype=_np.float32).reshape(2).copy()

        # 1) 死区:靶已足够近则不再输出(遏制中心附近的抖)
        if self.deadband_px > 0.0:
            det = getattr(self.inner, "last_detection", None)
            if det is not None and det.ok:
                h, w = frame.shape[:2]
                cx = self.center_px if self.center_px > 0 else (w - 1) / 2.0
                cy = self.center_px if self.center_px > 0 else (h - 1) / 2.0
                if _np.hypot(det.cx - cx, det.cy - cy) <= self.deadband_px:
                    a *= 0.0
                    self.n_deadband += 1

        # 2) 软饱和:`k*tanh(a/k)` 替代硬裁剪。
        #    性质(a 归一化到 [-1,1]):
        #      * |a| 小时 ≈ a(一阶展开),小误差精度不丢;
        #      * 中间段被**压缩**(a=0.5 落在 <0.5),削掉临界附近的 bang-bang;
        #      * 满舵被温和收敛(a=1 -> k*tanh(1/k) < 1),等于给"最高速"降一档,
        #        这正是用户要的"别太激进"。k 越大越接近线性(越激进)。
        #    实测 k=1.5:0.5->0.482,满舵->0.874(最高速降约 13%,温和去激进化)。
        if self.soft > 0.0:
            k = self.soft
            a = k * _np.tanh(a / k)

        # 3) 速率限制:每拍变化量不超过 max_delta
        if self.max_delta > 0.0:
            da = a - self._prev
            mag = float(_np.max(_np.abs(da)))
            if mag > self.max_delta:
                da *= self.max_delta / mag
                a = self._prev + da
                self.n_limited += 1
            a = _np.clip(a, -1.0, 1.0)

        self._prev = a.astype(_np.float32)
        return self._prev.copy()

    def reset(self) -> None:
        self.inner.reset()
        self._prev = __import__("numpy").zeros(2, dtype=__import__("numpy").float32)
        self.n_limited = 0
        self.n_deadband = 0

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
        self.last_candidates = getattr(self._seek, "last_candidates", None)

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
        self.last_candidates = getattr(self._seek, "last_candidates", None)
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
