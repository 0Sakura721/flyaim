"""诊断:命中率为什么这么低?分解到设计机制。

Lead 持有。背景:PID(上限参照)实测 6.33%,而「旅行时间」解析上限只有 4.91%
—— 反超说明模型有缺。本脚本用实测回答三个问题:

    Q1. 一次"命中事件"实际能刷几帧?(若命中后不 respawn,持续贴靶会刷满整帧)
    Q2. respawn 是否让命中变得"必须一次一次重新搜"?
    Q3. 速度 v / 判定半径 r 的比值(v/r)如何决定命中率上限?

判据:构造 4 个"作弊级"控制器(都直接读靶位,故都是上界),实测它们的命中率,
就能把靶场的**设计上限**和"果蝇为什么没命中"分开:

    oracle_hold   : 直接把准星放到靶心,并按速度限制慢慢趋近 → 刷满贴靶帧
    oracle_travel : 朝靶心全速直线移动(不停留)→ 旅行时间主导
    oracle_vprop  : 速度按剩余距离比例缩小(不冲过头)→ 接近 PID
    constant_speed: 朝靶心但保持满速不减速 → 模拟"果蝇读出不减速"的失败模式

运行::

    <捆绑 python> tools/diag_hitrate_ceiling.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from flyaim.arena.arena import Arena  # noqa: E402
from flyaim.arena.metrics import Metrics  # noqa: E402
from flyaim.config import ArenaConfig  # noqa: E402
from flyaim.runner import run_episode  # noqa: E402


def make_arm(name: str, mode: str, speed: float):
    class _Arm:
        def __init__(self):
            self.arena = None

        def attach_arena(self, a):
            self.arena = a

        def reset(self, seed):
            del seed

        def act(self, frame, state=None):
            del frame, state
            st = self.arena.get_state()
            ch = np.asarray(st["crosshair"], dtype=np.float64)
            tg = np.asarray(st["targets"], dtype=np.float64).reshape(-1, 2)
            d = tg - ch
            dist = np.linalg.norm(d, axis=1)
            k = int(np.argmin(dist))
            r = float(dist[k])
            if r < 1e-9:
                return np.zeros(2, np.float32)
            u = d[k] / r

            if mode == "hold":
                # 已在靶内就不动(最大化贴靶帧)
                if r <= 22.0:
                    return np.zeros(2, np.float32)
                # 否则慢慢趋近,避免冲过头
                mag = min(1.0, r / speed)
            elif mode == "travel":
                # 全速冲,不减速 -> 一定冲过头
                mag = 1.0
            elif mode == "vprop":
                # 速度按距离比例:到靶心正好停下
                mag = min(1.0, r / speed)
            elif mode == "const":
                # 恒定幅度 1.0(模拟"读出层不看距离就输出满量程")
                mag = 1.0
            else:
                raise ValueError(mode)
            return (u * mag).astype(np.float32)

    return _Arm()


def main() -> int:
    print("=" * 92)
    print("命中率上限分解:设计机制 vs 控制质量")
    print("=" * 92, flush=True)

    cfg = ArenaConfig()
    print(f"靶半径 r = {cfg.target_radius} px, 速度 v = {cfg.speed_px_per_action} px/帧,"
          f" respawn_on_hit = {cfg.respawn_on_hit}")
    print(f"判定区域 = pi*r^2 = {np.pi*cfg.target_radius**2:,.0f} px2 "
          f"占画布 {100*np.pi*cfg.target_radius**2/(cfg.width*cfg.height):.3f}%")
    print()

    modes = [
        ("hold",     "抵达后停在靶内(作弊上界)"),
        ("travel",   "全速直线冲,不减速"),
        ("vprop",    "按距离比例减速(理想控制)"),
        ("const",    "恒定满量程动作(果蝇失败模式)"),
    ]
    cfg2 = ArenaConfig()
    cfg2.max_frames = 900

    print(f"{'模式':>8}  {'说明':<24} | {'hits':>6} {'hit_rate':>9} {'mean_dist':>10} {'ttff':>8}")
    print("-" * 92, flush=True)
    for mode, desc in modes:
        arm = make_arm(mode, mode, cfg2.speed_px_per_action)
        res = run_episode(arm, lambda c, s: Arena(c, s), cfg2, Metrics, seed=0, frames=900)
        s = res.summary
        ttff = s.get("time_to_first_hit_frames")
        print(f"{mode:>8}  {desc:<24} | {s['hits']:>6} {s['hit_rate']:>9.4f} "
              f"{s['mean_target_dist_px']:>10.1f} {str(ttff):>8}", flush=True)

    # ---- Q1/Q2:命中后不 respawn 时能刷几帧 ----
    print("\n" + "=" * 92)
    print("Q1/Q2: resp 机制的影响")
    print("=" * 92)
    cfg3 = ArenaConfig()
    cfg3.max_frames = 900
    cfg3.respawn_on_hit = False
    arm = make_arm("hold", "hold", cfg3.speed_px_per_action)
    res = run_episode(arm, lambda c, s: Arena(c, s), cfg3, Metrics, seed=0, frames=900)
    s = res.summary
    print(f"  respawn_on_hit=False, hold 控制器: hits={s['hits']} hit_rate={s['hit_rate']:.4f}")
    print("  → 若该值远高于 respawn=True 的 hold,则证明 respawn 是主要上限来源。")

    # ---- Q3: 速度扫描 ----
    print("\n" + "=" * 92)
    print("Q3: 速度扫描(v 越大,判定窗口越容易一帧内跳出)")
    print("=" * 92)
    print(f"{'v (px/帧)':>10} {'v/2r':>8} | {'vprop hit_rate':>15} {'const hit_rate':>15}")
    print("-" * 60, flush=True)
    r_px = float(ArenaConfig().target_radius)
    for v in [3.0, 7.0, 14.0, 21.0, 28.0]:
        row = []
        for mode in ("vprop", "const"):
            c = ArenaConfig()
            c.max_frames = 900
            c.speed_px_per_action = v
            arm = make_arm(mode, mode, v)
            r = run_episode(arm, lambda cc, s: Arena(cc, s), c, Metrics, seed=0, frames=900)
            row.append(r.summary["hit_rate"])
        print(f"{v:>10.1f} {v/(2*r_px):>8.2f} | {row[0]:>15.4f} {row[1]:>15.4f}",
              flush=True)

    print()
    print("结论指向:hit_rate 的绝对值由**靶场设计**(r, v, respawn)决定,"
          "不是控制器优劣的直接度量。")
    print("  - 若要让 hit_rate 对控制器更敏感,应增大 r/减小 v/取消 respawn_on_hit,"
          "或改用 mean_target_dist_px(本项目报告的主力指标)。")
    print("=" * 92)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
