"""让果蝇百发百中 —— 四臂实测:复眼视觉伺服 / 像素伺服 / 作弊上界 / 随机下界。

================================================================================
要回答的问题
================================================================================
「百发百中」在本靶场里必须**先说清口径**,否则数字没有意义(D12):

  respawn_on_hit=True  (命中即重生): 每命中一次,靶立刻 teleport,
      下一次命中必须重新走完平均 ~271px 的路 → 命中率上限
      ≈ 1/(271/14 + 1) ≈ 4.9%,**与控制器质量无关**
      (D12 实测:4 个行为完全不同的完美控制器得分完全相同 5.56%)

  respawn_on_hit=False (靶不动): 收敛后每一帧都命中 → 命中率 → 100%
      **这才是「百发百中」能被定义出来的口径。**

================================================================================
四个臂
================================================================================
  eye-servo   : **果蝇自己的复眼**(6,098 感光细胞 → R1-R6 亮度 / R7-R8 色觉)
                → 铺回 (24,32) 小眼网格 → 靶心-准星误差 → 刹停律。
                连接组被绕过(那是本项目的核心发现:D10–D24 六个体制全阴性)。
                输入只有帧,不读 `Arena.get_state()`。
  pixel-servo : 直接对像素做颜色检测 + 刹停律(与 SeekController 同构)。
                用来回答"换成纯像素伺服是不是更容易"。
  oracle-pid  : 直接偷看靶位坐标(上界刻度,禁止当成绩表述)。
  random      : 均匀随机(下界)。

两个后端都要报 fps 与"墙钟"只作参考:本脚本的目的是**任务难度刻度**,
不是 fly vs shuffle 的实验判定(那个已按预注册归档)。

运行::

    <捆绑 python> tools/aim_perfect.py [--frames 900] [--seeds 10]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from flyaim.arena.arena import Arena  # noqa: E402
from flyaim.arena.metrics import Metrics  # noqa: E402
from flyaim.baselines.eye_servo import make_eye_servo  # noqa: E402
from flyaim.bridge.detect import find_target  # noqa: E402
from flyaim.config import ArenaConfig  # noqa: E402
from flyaim.runner import run_episode  # noqa: E402

DATA_DIR = ROOT / "flyaim" / "data" / "build"

# eye-servo 用类默认参数(标定过程与三次被实测推翻的中间版本见 DECISIONS.md D25):
#   * 掩膜膨胀 1 格     —— 不膨胀会有"尾格"污染,seed=4 永久停在离靶 32px
#   * 无 freeze 门限    —— freeze 是单向闩锁,一旦在估计偏差下提前锁死就再也回不来
#   * 近距限速 0.3×v    —— 不限速会出现 ±14px(=满速)极限环,稳靶率 98%→55%
#   * 传出拷贝外推      —— 准星压在靶上时被掩膜挖掉,只能靠命令外推
EYE_KWARGS: dict = {}


# ---------------------------------------------------------------- 像素伺服(下界参照)


def _brighten(c, t: float):
    a = np.asarray(c, dtype=np.float64)
    return tuple(int(x) for x in np.clip(np.round(a + (255.0 - a) * t), 0, 255))


class PixelServo:
    """纯像素颜色检测 + 刹停律(不碰环境状态)。"""

    name = "pixel-servo"

    def __init__(self, v_px: float = 14.0, k_aim: float = 0.30,
                 freeze_px: float = 16.0) -> None:
        self.v_px = float(v_px)
        self.k_aim = float(k_aim)
        self.freeze_px = float(freeze_px)
        self._aim = None
        self._pending = None
        self.n_no_target = 0

    def reset(self, seed: int = 0) -> None:
        del seed
        self._aim = None
        self._pending = None
        self.n_no_target = 0

    def act(self, frame: np.ndarray, state=None) -> np.ndarray:
        del state
        h, w = frame.shape[:2]
        base = (235, 70, 70)
        det = find_target(frame, ref_color=base, tolerance=60.0)
        if not det.ok:
            det = find_target(frame, ref_color=_brighten(base, 0.60), tolerance=60.0)
        if not det.ok:
            self.n_no_target += 1
            return np.zeros(2, dtype=np.float32)
        if self._aim is not None and self._pending is not None:
            self._aim = (self._aim[0] + float(self._pending[0]),
                         self._aim[1] + float(self._pending[1]))
        aim = find_target(frame, ref_color=(240, 240, 240), tolerance=30.0,
                          min_area_px=4)
        if aim.ok:
            self._aim = (aim.cx, aim.cy)
        elif self._aim is None:
            self._aim = ((w - 1) / 2.0, (h - 1) / 2.0)
        err = np.array([det.cx - self._aim[0], det.cy - self._aim[1]],
                       dtype=np.float64)
        r = float(np.linalg.norm(err))
        if r <= self.freeze_px:
            self._pending = np.zeros(2, dtype=np.float64)
            return np.zeros(2, dtype=np.float32)
        mag = min(1.0, r / self.v_px)
        a = err / r * mag
        self._pending = a * self.v_px
        return a.astype(np.float32)


class OracleArm:
    """作弊上界:直接读靶位与准星坐标。"""

    name = "oracle-pid"

    def __init__(self, v_px: float = 14.0, freeze_px: float = 16.0) -> None:
        self.arena = None
        self.v_px = float(v_px)
        self.freeze_px = float(freeze_px)

    def attach_arena(self, a) -> None:
        self.arena = a

    def reset(self, seed: int = 0) -> None:
        del seed

    def act(self, frame: np.ndarray, state=None) -> np.ndarray:
        del frame, state
        st = self.arena.get_state()
        ch = np.asarray(st["crosshair"], dtype=np.float64)
        tg = np.asarray(st["targets"], dtype=np.float64).reshape(-1, 2)
        d = tg - ch
        dist = np.linalg.norm(d, axis=1)
        k = int(np.argmin(dist))
        r = float(dist[k])
        if r <= self.freeze_px or r < 1e-9:
            return np.zeros(2, np.float32)
        u = d[k] / r
        return (u * min(1.0, r / self.v_px)).astype(np.float32)


class RandomArm:
    name = "random"

    def __init__(self) -> None:
        self.rng = np.random.default_rng(0)

    def reset(self, seed: int = 0) -> None:
        self.rng = np.random.default_rng(seed)

    def act(self, frame: np.ndarray, state=None) -> np.ndarray:
        del frame, state
        return self.rng.uniform(-1.0, 1.0, size=2).astype(np.float32)


# ---------------------------------------------------------------- 单 episode


def run_one(arm_name: str, seed: int, frames: int, respawn: bool,
            eye_kwargs: dict | None = None) -> dict:
    cfg = ArenaConfig()
    cfg.max_frames = int(frames)
    cfg.respawn_on_hit = bool(respawn)

    if arm_name == "eye-servo":
        arm = make_eye_servo(str(DATA_DIR), **(eye_kwargs or EYE_KWARGS))
    elif arm_name == "pixel-servo":
        arm = PixelServo()
    elif arm_name == "oracle-pid":
        arm = OracleArm()
    else:
        arm = RandomArm()

    dists: list[float] = []

    def observer(i, frame, res, arena, a):
        dists.append(float(res.target_dist))

    m = Metrics(n_bins=10, frames_budget=int(frames))
    t0 = time.perf_counter()
    res = run_episode(arm, lambda c, s: Arena(c, s), cfg, lambda: m,
                      seed=seed, frames=int(frames), observer=observer)
    wall = time.perf_counter() - t0

    d = np.asarray(dists, dtype=np.float64)
    ok = d <= float(cfg.target_radius)
    lock = int(np.argmax(ok)) if bool(ok.any()) else -1
    post = d[lock:] if lock >= 0 else d[~np.isfinite(d)]
    s = res.summary
    out = {
        "arm": arm_name,
        "seed": int(seed),
        "respawn": bool(respawn),
        "frames": int(s["frames"]),
        "hits": int(s["hits"]),
        "hit_rate": float(s["hit_rate"]),
        "mean_dist_px": float(s["mean_target_dist_px"]),
        "on_target_frac": float(ok.mean()),
        "lock_frame": int(lock),                       # 首次进入 22px 判定圈的帧
        "post_lock_on_target_frac": float((post <= 22.0).mean()) if post.size else 0.0,
        "post_lock_max_dist_px": float(post.max()) if post.size else float("nan"),
        "wall_s": float(wall),
        "fps": float(s["frames"] / max(wall, 1e-9)),
    }
    if arm_name == "eye-servo":
        out["no_aim_frames"] = int(arm.n_no_aim)
        out["no_target_frames"] = int(arm.n_no_target)
    if arm_name == "pixel-servo":
        out["no_target_frames"] = int(arm.n_no_target)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=900)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--out", type=str, default="")
    args = ap.parse_args()

    seeds = list(range(int(args.seeds)))
    cfg = ArenaConfig()
    print("=" * 104)
    print("让果蝇百发百中:复眼视觉伺服(6,098 感光细胞)vs 像素伺服 vs 作弊上界 vs 随机")
    print("=" * 104)
    print(f"靶场 {cfg.width}x{cfg.height} | 靶半径 {cfg.target_radius}px | "
          f"执行器 {cfg.speed_px_per_action}px/帧 | 靶速 {cfg.target_speed_px_per_frame} | "
          f"{len(seeds)} seeds x {args.frames} 帧")
    print("稳靶帧占比 = dist<=22px 的帧比例;首达帧 = 首次进入判定圈的帧号\n", flush=True)

    arms = ("eye-servo", "pixel-servo", "oracle-pid", "random")
    rows: list[dict] = []
    for respawn in (False, True):
        tag = ("respawn=False  ← 「百发百中」口径(靶不动)" if not respawn
               else "respawn=True   ← 上限被旅行时间压到 ~5%,与控制器质量无关")
        print("-" * 104)
        print(f"### {tag}")
        print(f"{'臂':<12} | {'每帧命中率':>17} | {'稳靶帧占比':>17} | {'稳靶后占比':>12} | "
              f"{'平均靶距':>14} | {'首达帧':>8} | {'FPS':>7}")
        print("-" * 104, flush=True)
        for arm_name in arms:
            sub = [run_one(arm_name, s, args.frames, respawn) for s in seeds]
            rows.extend(sub)
            hr = np.array([r["hit_rate"] for r in sub])
            ot = np.array([r["on_target_frac"] for r in sub])
            pl = np.array([r["post_lock_on_target_frac"] for r in sub])
            md = np.array([r["mean_dist_px"] for r in sub])
            lk = np.array([r["lock_frame"] for r in sub], dtype=np.float64)
            fps = np.array([r["fps"] for r in sub])
            lks = f"{lk.mean():.1f}" if lk.min() >= 0 else "n/a"
            print(f"{arm_name:<12} | {hr.mean()*100:>8.2f}% ±{hr.std(ddof=1)*100:>5.2f} | "
                  f"{ot.mean()*100:>8.2f}% ±{ot.std(ddof=1)*100:>5.2f} | "
                  f"{pl.mean()*100:>10.2f}% | "
                  f"{md.mean():>7.1f}px ±{md.std(ddof=1):>4.1f} | {lks:>8} | "
                  f"{fps.mean():>7.1f}", flush=True)
        print()

    # ---- 逐 seed 明细(eye-servo, respawn=False)----
    print("=" * 104)
    print("逐 seed 明细:eye-servo / respawn=False")
    print("=" * 104)
    print(f"{'seed':>4} | {'首达帧':>7} | {'稳靶帧占比':>10} | {'稳靶后最差靶距':>14} | "
          f"{'平均靶距':>9} | {'未检出准星帧':>12}")
    print("-" * 104, flush=True)
    for r in rows:
        if r["arm"] == "eye-servo" and not r["respawn"]:
            print(f"{r['seed']:>4} | {r['lock_frame']:>7} | "
                  f"{r['on_target_frac']*100:>9.2f}% | "
                  f"{r['post_lock_max_dist_px']:>12.1f}px | "
                  f"{r['mean_dist_px']:>7.1f}px | {r.get('no_aim_frames', 0):>12}", flush=True)

    out_dir = Path(args.out) if args.out else (
        ROOT / "flyaim" / "runs"
        / (time.strftime("%Y%m%d-%H%M%S") + "-aim-perfect"))
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": {"frames": args.frames, "seeds": seeds,
                   "arena": {k: str(v) for k, v in vars(ArenaConfig()).items()}},
        "rows": rows,
    }
    (out_dir / "summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n产物 -> {out_dir / 'summary.json'}")
    print("=" * 104)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
