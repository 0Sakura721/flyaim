"""验证:「百发百中」到底能不能做到、以及要做到它需要什么。

================================================================================
背景
================================================================================
用户命题(2026-10-04):「靶静止 + 每帧自动开火 + 命中即重生成 ⇒ 只要准星朝
误差方向持续收敛并停住,就能 100%。用连接组做不到,换个网络几小时就有。」

本脚本用实测回答两个可分离的问题:

  Q1. **控制问题本身有多难?** 一个纯像素的 P/PD 视觉伺服(不看环境状态,
      只用颜色检测从帧里找靶与准星)能在多少个 seed 上收敛?首达几帧?
  Q2. **"100%" 是哪个口径的 100%?** 靶场有两种 respawn 语义:
        respawn_on_hit=True  : 命中即刻重生 → 每次命中都要重新走一段路,
                               命中率上限 ≈ 1/(平均路程/速度)  ← 与瞄准精度无关
        respawn_on_hit=False : 靶不动 → 收敛后每帧都命中 → 命中率 → 100%
      D12 已测出前者上限 5.56%(4 个作弊级控制器得分完全相同)。本脚本复测,
      并把「稳靶帧占比」这个能真正表达"百发百中"的量一并报出。

================================================================================
诚实边界
================================================================================
* servo-pixel 是**纯像素**臂:输入只有 frame(H,W,3)。它不用 get_state(),
  不读靶位坐标。与 SeekController 同构(颜色距离 + 连通域 + PD),
  只是跑在离线 Arena 上而不是真实屏幕上。
* 它**不是**实验对照臂,不能进 CONTRACT 第 3 节的 fly vs shuffle 判定 ——
  它是"任务难度刻度尺",用来界定:如果连接组做不到,差的是任务难度还是架构。
* oracle 臂直接读 state(作弊),仅作上界刻度。

运行::

    <捆绑 python> tools/verify_servo_100.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from flyaim.arena.arena import Arena  # noqa: E402
from flyaim.arena.metrics import Metrics  # noqa: E402
from flyaim.bridge.detect import find_target  # noqa: E402
from flyaim.config import ArenaConfig  # noqa: E402
from flyaim.runner import run_episode  # noqa: E402

SEEDS = tuple(range(10))
FRAMES = 900


def _brighten(c: tuple[int, int, int], t: float) -> tuple[int, int, int]:
    """复刻 arena._brighten:命中帧的靶会被整体提亮,检测必须覆盖这两种颜色。"""
    a = np.asarray(c, dtype=np.float64)
    return tuple(int(x) for x in np.clip(np.round(a + (255.0 - a) * t), 0, 255))


class ServoPixel:
    """纯像素视觉伺服:帧 -> 颜色检测 -> 控制律 -> action。不接触环境状态。

    两种控制律(都在像素误差上算,都不碰环境状态):
        mode="pd"   : a = kp·e/(w/2) − kd·(e−e_prev)/(w/2)
                      —— 与 SeekController 同款。比例项意味着越近越慢,
                      末端收敛被时间常数拖长(实测明显低于上界)。
        mode="brake": a = û · min(1, |e|/v) —— 全速逼近,最后一步刹停在靶心。
                      v 是控制器已知的执行器速度(px/帧),不是环境状态。
    """

    name = "servo-pixel"

    def __init__(self, kp: float = 1.2, kd: float = 0.15,
                 mode: str = "brake", v_px: float = 14.0) -> None:
        self.kp = float(kp)
        self.kd = float(kd)
        self.mode = str(mode)
        self.v_px = float(v_px)
        self._prev_err = np.zeros(2, dtype=np.float32)
        self.n_not_found = 0
        self.n_frames = 0

    def reset(self, seed: int) -> None:
        del seed
        self._prev_err = np.zeros(2, dtype=np.float32)
        self.n_not_found = 0
        self.n_frames = 0

    def act(self, frame: np.ndarray, state=None) -> np.ndarray:
        del state  # 架构级保证:本臂不消费环境状态
        self.n_frames += 1
        h, w = frame.shape[:2]
        base = (235, 70, 70)                      # 常态靶色
        hitc = _brighten(base, 0.60)              # 命中帧提亮后的靶色
        det = find_target(frame, ref_color=base, tolerance=60.0)
        if not det.ok:
            det = find_target(frame, ref_color=hitc, tolerance=60.0)
        if not det.ok:
            self.n_not_found += 1
            self._prev_err = np.zeros(2, dtype=np.float32)
            return np.zeros(2, dtype=np.float32)

        # 瞄准点 = 检测到的白色准星(tol 收紧到 30,排除提亮靶的外描边)
        aim = find_target(frame, ref_color=(240, 240, 240), tolerance=30.0,
                          min_area_px=4)
        if aim.ok:
            ax, ay = aim.cx, aim.cy
        else:
            ax, ay = (w - 1) / 2.0, (h - 1) / 2.0

        err = np.array([det.cx - ax, det.cy - ay], dtype=np.float32)
        r = float(np.linalg.norm(err))
        if self.mode == "brake" and r > 1e-9:
            mag = min(1.0, r / self.v_px)          # 全速逼近 + 末端刹停
            a = (err / r) * mag
        else:
            a = self.kp * err / (w / 2.0) - self.kd * (err - self._prev_err) / (w / 2.0)
        self._prev_err = err
        return np.clip(a, -1.0, 1.0).astype(np.float32)


class OraclePID:
    """作弊上界:直接读靶位与准星坐标(仅作刻度,不进任何判定)。"""

    name = "oracle-pid"

    def __init__(self) -> None:
        self.arena = None

    def attach_arena(self, a) -> None:
        self.arena = a

    def reset(self, seed: int) -> None:
        del seed

    def act(self, frame: np.ndarray, state=None) -> np.ndarray:
        del frame
        st = self.arena.get_state()
        ch = np.asarray(st["crosshair"], dtype=np.float64)
        tg = np.asarray(st["targets"], dtype=np.float64).reshape(-1, 2)
        d = tg - ch
        dist = np.linalg.norm(d, axis=1)
        k = int(np.argmin(dist))
        r = float(dist[k])
        if r <= 22.0:
            return np.zeros(2, np.float32)     # 已在靶内:停住(最大化贴靶帧)
        if r < 1e-9:
            return np.zeros(2, np.float32)
        u = d[k] / r
        mag = min(1.0, r / float(st["speed_px_per_action"]))
        return (u * mag).astype(np.float32)


class RandomArm:
    name = "random"

    def __init__(self, seed: int = 0) -> None:
        self.rng = np.random.default_rng(seed)

    def reset(self, seed: int) -> None:
        self.rng = np.random.default_rng(seed)

    def act(self, frame: np.ndarray, state=None) -> np.ndarray:
        del frame, state
        return self.rng.uniform(-1.0, 1.0, size=2).astype(np.float32)


def _run(arm_name: str, respawn: bool, seed: int) -> dict:
    cfg = ArenaConfig()
    cfg.max_frames = FRAMES
    cfg.respawn_on_hit = respawn

    if arm_name == "servo-brake":
        arm = ServoPixel(mode="brake")
    elif arm_name == "servo-pd":
        arm = ServoPixel(mode="pd")
    elif arm_name == "oracle-pid":
        arm = OraclePID()
    else:
        arm = RandomArm(seed)

    on_target = 0
    n = 0

    def observer(i, frame, res, arena, a):
        nonlocal on_target, n
        n += 1
        if float(res.target_dist) <= float(cfg.target_radius):
            on_target += 1

    m = Metrics(n_bins=10, frames_budget=FRAMES)
    res = run_episode(arm, lambda c, s: Arena(c, s), cfg, lambda: m,
                      seed=seed, frames=FRAMES, observer=observer)
    s = res.summary
    out = {
        "hit_rate": s["hit_rate"],
        "mean_dist": s["mean_target_dist_px"],
        "on_target_frac": on_target / max(n, 1),
        "ttff_frames": s["time_to_first_hit_frames"],
        "hits": s["hits"],
    }
    if arm_name.startswith("servo-"):
        out["detect_fail_frac"] = arm.n_not_found / max(arm.n_frames, 1)
    return out


def _agg(rows: list[dict]) -> dict:
    keys = [k for k in rows[0]]
    out = {}
    for k in keys:
        v = np.array([r[k] for r in rows], dtype=np.float64)
        out[k] = (float(v.mean()), float(v.std(ddof=1)) if v.size > 1 else 0.0)
    return out


def main() -> int:
    print("=" * 96)
    print("「百发百中」可行性验证:纯像素视觉伺服 vs 作弊上界 vs 随机下界")
    print("=" * 96, flush=True)
    cfg = ArenaConfig()
    print(f"靶场: {cfg.width}x{cfg.height}, r={cfg.target_radius}px, "
          f"v={cfg.speed_px_per_action}px/帧, target_speed={cfg.target_speed_px_per_frame}")
    print(f"预算: {len(SEEDS)} seeds x {FRAMES} 帧/臂")
    print("判定: 稳靶帧占比 = dist<=22px 的帧比例(这才是「贴住靶」的直接度量)\n",
          flush=True)

    table: dict[str, dict] = {}
    for respawn in (True, False):
        tag = "respawn=True(命中即重生)" if respawn else "respawn=False(靶不动)"
        print("-" * 96)
        print(f"### {tag}")
        print(f"{'臂':<14} | {'每帧命中率':>18} | {'稳靶帧占比':>18} | {'平均靶距':>16} | "
              f"{'首达帧':>10} | {'检出失败率':>10}")
        print("-" * 96, flush=True)
        for arm_name in ("servo-brake", "servo-pd", "oracle-pid", "random"):
            rows = [_run(arm_name, respawn, s) for s in SEEDS]
            agg = _agg(rows)
            table[f"{arm_name}|respawn={respawn}"] = agg
            df = agg.get("detect_fail_frac", (float("nan"), 0.0))
            ttff = agg["ttff_frames"][0]
            print(f"{arm_name:<14} | "
                  f"{agg['hit_rate'][0]*100:>8.2f}% ±{agg['hit_rate'][1]*100:>6.2f} | "
                  f"{agg['on_target_frac'][0]*100:>8.2f}% ±{agg['on_target_frac'][1]*100:>6.2f} | "
                  f"{agg['mean_dist'][0]:>9.1f}px ±{agg['mean_dist'][1]:>5.1f} | "
                  f"{('%.1f' % ttff) if ttff == ttff else 'n/a':>10} | "
                  f"{df[0]*100:>9.2f}%", flush=True)
        print()

    print("=" * 96)
    print("判读")
    print("=" * 96)
    a = table["servo-brake|respawn=True"]
    b = table["servo-brake|respawn=False"]
    print(f"1) respawn=True  下 servo-brake 稳靶帧占比 {a['on_target_frac'][0]*100:.2f}%,"
          f"每帧命中率 {a['hit_rate'][0]*100:.2f}% —— 与作弊上界 "
          f"{table['oracle-pid|respawn=True']['hit_rate'][0]*100:.2f}% 同量级。")
    print("   ⇒ 该口径的「100%」不存在:上限由「命中后必须重新走一段路」决定,"
          "与控制器质量无关。")
    print(f"2) respawn=False 下 servo-brake 稳靶帧占比 {b['on_target_frac'][0]*100:.2f}%,"
          f"每帧命中率 {b['hit_rate'][0]*100:.2f}% —— 这才是「百发百中」的口径。")
    print(f"3) 纯像素伺服 vs 作弊上界(respawn=False):"
          f" {b['hit_rate'][0]*100:.2f}% vs "
          f"{table['oracle-pid|respawn=False']['hit_rate'][0]*100:.2f}% —— "
          f"差距越小,说明任务对「只许看像素」的控制器几乎没有额外难度。")
    print(f"4) 比例项 vs 刹停律(respawn=True):"
          f" pd {table['servo-pd|respawn=True']['hit_rate'][0]*100:.2f}% vs "
          f"brake {a['hit_rate'][0]*100:.2f}% —— 末端收敛律决定能否贴近上界。")
    print("=" * 96)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
