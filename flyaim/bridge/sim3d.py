"""SimFPS3D —— 用代码复刻一个**真实 FPS 的视觉语义**的帧源(实战彩排用)。

===============================================================================
为什么不能拿自建 2D 靶场当彩排(AIMLAB.md §6 的差异清单)
===============================================================================
                            自建 2D 靶场           真实 FPS
    准星                     在画面里**会动**       钉在画面中心**不动**
    动的是什么               准星                   相机(整个世界在流)
    action 语义              准星像素速度           鼠标计数 -> 视转角
    背景                     完全静止               全域光流
    边角                     无畸变                 透视投影,越靠边 px/度越大

四行里前三行都会**改变控制律的正确形式**,所以在 2D 靶场上调通的控制器
直接搬到游戏里是错的。这个源用最小的代价把上面四行都实现出来:

  * 针孔投影 `u = W/2 + f·tan(Δyaw)`,`v = H/2 − f·tan(Δpitch)`, `f = (W/2)/tan(FOV/2)`
    —— 不是线性近似,所以 GainModel 那条「屏幕像素比例 ≈ 角度比例」的假设
    会**真的受到检验**(越靠画面边缘,同样的 action 转过的像素越多/越少)。
  * 准星固定在画面中心(红色小十字,与 Aim Lab 一致),相机由**注入计数**驱动
    (`push(action)` 走的是同一个 GainModel),因此旋转是真实的。
  * 世界是**静止点阵**(不是纯色背景):相机一转,整幅画面的纹理一起流 ——
    这正是 D18 指出的"同一链路的换域检验"里最关键的一项统计差异。
  * 靶是青色的球(默认 `(48,224,224)`,与 Aim Lab 实测靶色一致),
    角直径可配;命中判定 = 角误差 <= 靶角半径(准星钉死在中心)。

它**不**模仿的东西(写报告时必须说):真实渲染延迟、真实鼠标加速度、
真实 HUD/枪模型遮挡、真实靶的球面明暗。这些只在真机上才能验。

用法见 `tools/aimlab_sim3d.py`。
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np

__all__ = ["SimFPS3DSource", "SimFPS3DConfig"]


@dataclass
class SimFPS3DConfig:
    """模拟器参数。默认值对齐 Aim Lab 全屏捕获的桥接配置。"""

    width: int = 640              # 捕获后尺寸(与 bridge 的 out_size 一致)
    height: int = 480
    fov_h_deg: float = 103.0      # 水平 FOV(Aim Lab 常见值)
    target_color: tuple[int, int, int] = (48, 224, 224)   # 青靶(实测 Aim Lab 色)
    crosshair_color: tuple[int, int, int] = (240, 60, 60)  # 红色小十字
    bg_color: tuple[int, int, int] = (26, 26, 32)
    dot_color: tuple[int, int, int] = (120, 120, 132)
    gun_color: tuple[int, int, int] = (150, 55, 55)
    target_ang_deg: float = 7.0   # 靶的**角直径**(度);半径 = 一半
    spawn_ang_deg: tuple[float, float] = (6.0, 34.0)  # 新靶出现的角距范围
    respawn_on_hit: bool = True   # Gridshot 语义:命中即换位置
    draw_crosshair: bool = True
    draw_gun: bool = True         # 红枪模型:用来复现"红色找准星会锁到枪上"
    dot_step_deg: float = 9.0     # 世界点阵间距(全域光流的纹理)
    dot_span_deg: float = 72.0
    seed: int = 0


class SimFPS3DSource:
    """FrameSource 协议:`read() -> (H,W,3) uint8`, `push(action)` 消费指令。

    与 `ArenaSource` 的关键差别:`ArenaSource.push` 让准星走 14px;
    这里 `push` 把 action 经 **GainModel 换成鼠标计数**,再换成**视角旋转**,
    靶因此在画面里反向移动 —— 与真实 FPS 的因果链完全一致。
    """

    backend = "sim3d"

    def __init__(self, cfg: SimFPS3DConfig | None = None, gain=None) -> None:
        self.cfg = cfg or SimFPS3DConfig()
        if gain is None:  # 允许独立使用:默认沿用靶场语义(2.19%/拍)
            from flyaim.bridge.gain import GainConfig, GainModel

            gain = GainModel(GainConfig())
        self.gain = gain
        self._rng = np.random.default_rng(self.cfg.seed)
        self.seq = -1
        self.t_grab = 0.0

        # 相机与靶的世界朝向(度)
        self.yaw = 0.0
        self.pitch = 0.0
        self.target_yaw = 0.0
        self.target_pitch = 0.0

        # 世界点阵(相机不动时它是静止的;相机一转整幅在流)
        step, span = float(self.cfg.dot_step_deg), float(self.cfg.dot_span_deg)
        g = np.arange(-span, span + 1e-9, step)
        gy, gp = np.meshgrid(g, g, indexing="ij")
        self._dots = np.stack([gy.ravel(), gp.ravel()], axis=1)
        self._dot_shade = self._rng.uniform(0.55, 1.0, size=self._dots.shape[0])

        self._f = (self.cfg.width / 2.0) / math.tan(math.radians(
            max(1.0, float(self.cfg.fov_h_deg)) / 2.0))
        self._r_target_px = self._f * math.tan(math.radians(
            max(0.05, float(self.cfg.target_ang_deg)) / 2.0))
        self._r_target_deg = float(self.cfg.target_ang_deg) / 2.0

        self.history: list[dict] = []
        self.n_respawns = 0
        self._last: dict = {"err_deg": -1.0, "hit": False, "dist_px": -1.0,
                            "on_screen": True, "counts": (0, 0)}
        self._respawn()

    # ---------------------------------------------------------------- 内部

    def _respawn(self) -> None:
        lo, hi = self.cfg.spawn_ang_deg
        a = self._rng.uniform(lo, hi)
        th = self._rng.uniform(0.0, 2.0 * math.pi)
        self.target_yaw = float(self.yaw + a * math.cos(th))
        self.target_pitch = float(np.clip(
            self.pitch + a * math.sin(th), -55.0, 55.0))
        self.n_respawns += 1

    def _project(self, yaw: float, pitch: float) -> tuple[float, float]:
        """世界方向 -> 屏幕像素(针孔;返回 (u, v),可能落在画面外)。"""
        dy = math.radians(yaw - self.yaw)
        dp = math.radians(pitch - self.pitch)
        u = self.cfg.width / 2.0 + self._f * math.tan(dy)
        v = self.cfg.height / 2.0 - self._f * math.tan(dp)
        return float(u), float(v)

    def _ang_err_deg(self) -> float:
        """相机朝向与靶方向的角距(度)。

        ⚠️ dyaw 必须**环绕到 ±180°**(2026-10-04 修):搜索会连续转很多圈,
        不环绕的话误差会跑到几千度,指标失去意义(实测 mean=2209°)。
        """
        dy = (self.target_yaw - self.yaw + 180.0) % 360.0 - 180.0
        dp = self.target_pitch - self.pitch
        return float(math.hypot(dy, dp))

    # ---------------------------------------------------------------- 渲染

    def _render(self) -> np.ndarray:
        cfg = self.cfg
        W, H = int(cfg.width), int(cfg.height)
        f = np.empty((H, W, 3), dtype=np.uint8)
        f[:, :] = np.asarray(cfg.bg_color, dtype=np.uint8)

        # 世界点阵(2x2 小点,亮度带随机差异 -> 视网膜看得到纹理在流)
        uv = np.stack([self._project(a, b) for a, b in self._dots])
        base = np.asarray(cfg.dot_color, dtype=np.float64)
        on = ((uv[:, 0] >= 1) & (uv[:, 0] < W - 2)
              & (uv[:, 1] >= 1) & (uv[:, 1] < H - 2))
        for k in np.flatnonzero(on):
            x, y = int(uv[k, 0]), int(uv[k, 1])
            c = np.clip(base * self._dot_shade[k], 0, 255).astype(np.uint8)
            f[y:y + 2, x:x + 2] = c

        # 靶:青色圆盘 + 一道亮边(与 2D 靶场同样的做法:让降采样后仍有信噪比)
        u, v = self._project(self.target_yaw, self.target_pitch)
        r = self._r_target_px
        x0, x1 = int(math.floor(u - r)) - 1, int(math.ceil(u + r)) + 2
        y0, y1 = int(math.floor(v - r)) - 1, int(math.ceil(v + r)) + 2
        if x1 > 0 and y1 > 0 and x0 < W and y0 < H:
            xx0, xx1 = max(0, x0), min(W, x1)
            yy0, yy1 = max(0, y0), min(H, y1)
            yy, xx = np.mgrid[yy0:yy1, xx0:xx1]
            d2 = (xx - u) ** 2 + (yy - v) ** 2
            inside = d2 <= r * r
            rim = inside & (d2 > (r * 0.72) ** 2)
            patch = f[yy0:yy1, xx0:xx1]
            patch[inside] = np.asarray(cfg.target_color, dtype=np.uint8)
            base_c = np.asarray(cfg.target_color, dtype=np.float64)
            patch[rim] = np.clip(base_c + (255.0 - base_c) * 0.45, 0, 255).astype(np.uint8)

        # 枪模型(红色,画面下方中央)—— 复现"红色找准星会锁到枪上"这个真实陷阱
        if cfg.draw_gun:
            gx = W // 2
            gy = H - 1
            gw = max(6, W // 40)
            gh = max(18, H // 9)
            f[gy - gh:gy, gx - gw:gx + gw] = np.asarray(cfg.gun_color, dtype=np.uint8)

        # 准星:钉在画面中心的红色小十字(与 Aim Lab 一致)
        if cfg.draw_crosshair:
            cx, cy = W // 2, H // 2
            half, th = max(6, W // 48), max(1, W // 320)
            f[cy - th:cy + th + 1, cx - half:cx + half + 1] = np.asarray(
                cfg.crosshair_color, dtype=np.uint8)
            f[cy - half:cy + half + 1, cx - th:cx + th + 1] = np.asarray(
                cfg.crosshair_color, dtype=np.uint8)
        return f

    # ---------------------------------------------------------------- FrameSource

    def read(self) -> tuple[np.ndarray, dict]:
        """捕获线程高频调用:只渲染当前状态(与真实屏幕一样,没有 push 就不动)。

        ⚠️ 判定与记账**不在这里**做 —— 捕获线程的调用频率与网络拍无关
        (ArenaSource 的教训:把环境步进放进捕获线程会让等效增益暴增)。
        一拍的因果链是:read() 给出**动作前**的画面 -> push(action) 施加动作。
        """
        frame = self._render()
        self.seq += 1
        self.t_grab = time.perf_counter()
        return frame, {
            "seq": self.seq, "backend": self.backend, "t_grab": self.t_grab,
            "target_dist": float(self._last.get("dist_px", -1.0)),
            "hit": bool(self._last.get("hit", False)),
            "err_deg": float(self._last.get("err_deg", -1.0)),
        }

    def push(self, action: np.ndarray) -> None:
        """消费线程每拍一次:action -> (与真机同一条)增益路径 -> 视角旋转 -> 判定。

        ⚠️ 这里必须走 GainModel(而不是像 ArenaSource 那样直接加 14px):
        真实 FPS 的因果链是「注入计数 -> 视角转 -> 靶在画面里反向移动」,
        也只有这样才检验得了「屏幕像素 ≈ 角度」这条线性近似。
        """
        dx, dy = self.gain.to_counts(action)
        dpc = self.gain.deg_per_count()
        self.yaw = float(self.yaw + dx * dpc)          # 鼠标右移 -> 视角右转
        self.pitch = float(np.clip(self.pitch - dy * dpc, -88.0, 88.0))

        u, v = self._project(self.target_yaw, self.target_pitch)
        dist_px = float(math.hypot(u - self.cfg.width / 2.0,
                                   v - self.cfg.height / 2.0))
        err = self._ang_err_deg()
        hit = bool(err <= self._r_target_deg)
        on_screen = (-1.0 < u < self.cfg.width + 1.0
                     and -1.0 < v < self.cfg.height + 1.0)
        self._last = {"err_deg": err, "hit": hit, "dist_px": dist_px,
                      "on_screen": on_screen, "counts": (dx, dy)}
        self.history.append({
            "tick": len(self.history), "err_deg": round(err, 4), "hit": hit,
            "on_screen": bool(on_screen), "dist_px": round(dist_px, 2),
            "counts": (int(dx), int(dy)),
            "yaw": round(self.yaw, 4), "pitch": round(self.pitch, 4),
        })
        if hit and self.cfg.respawn_on_hit:
            self._respawn()

    def close(self) -> None:
        pass
