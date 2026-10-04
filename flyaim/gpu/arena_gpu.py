"""批量 GPU 靶场(向量化,与 CPU Arena 同几何、同判定)。

    - B 个环境**锁步**推进,一帧同时产出 B 张画面;
    - 渲染用 scatter(只写圆内像素),不物化 (B,H,W) 距离场,省带宽;
    - 判定/位移/碰壁反射与 CPU ``Arena`` 逐条对应;
    - 教师标签走**特权信息**(靶心坐标),省掉 CPU 版每帧 18ms 的
      颜色检测 + scipy.label。

确定性说明:CuPy 的 RNG 与 numpy PCG64 不同,故 GPU 靶场的采样序列
与 CPU 版**不逐点相同**(两者都是合法独立同分布采样)。需要与 CPU 版
逐帧对齐时用 ``sync_from_cpu``(见 tools/verify_gpu_arena.py)。
"""

from __future__ import annotations

import numpy as np

from flyaim.arena.arena import _SPAWN_EDGE_PAD, _SPAWN_MARGIN
from flyaim.config import ArenaConfig


def _brighten_u8(color, t: float):
    c = np.asarray(color, dtype=np.float64)
    return np.clip(np.round(c + (255.0 - c) * t), 0, 255).astype(np.uint8)


class GPUArenaBatch:
    """B 个环境锁步推进的 GPU 靶场。"""

    def __init__(self, cfg: ArenaConfig, n_env: int, seed: int = 0):
        import cupy as cp

        self.cp = cp
        self.cfg = cfg
        self.B = int(n_env)
        self.H = int(cfg.height)
        self.W = int(cfg.width)
        self.K = int(cfg.n_targets)
        self.r = float(cfg.target_radius)
        self.speed = float(cfg.speed_px_per_action)
        self.tgt_speed = float(cfg.target_speed_px_per_frame)
        self.seed = int(seed)
        self._rng = cp.random.default_rng(seed)

        lo_r = self.r + _SPAWN_EDGE_PAD
        self._lo = cp.asarray([lo_r, lo_r], dtype=cp.float32)
        self._hi = cp.asarray([self.W - 1.0 - lo_r, self.H - 1.0 - lo_r], dtype=cp.float32)
        self._ch_lo = cp.asarray([_SPAWN_EDGE_PAD, _SPAWN_EDGE_PAD], dtype=cp.float32)
        self._ch_hi = cp.asarray(
            [self.W - 1.0 - _SPAWN_EDGE_PAD, self.H - 1.0 - _SPAWN_EDGE_PAD], dtype=cp.float32)
        self._center = cp.asarray([(self.W - 1) / 2.0, (self.H - 1) / 2.0], dtype=cp.float32)
        self._bg = np.asarray(cfg.bg_color, dtype=np.uint8)
        self._base = np.asarray(cfg.target_color, dtype=np.uint8)
        self._hit_col = _brighten_u8(cfg.target_color, 0.60)
        self._out_col = _brighten_u8(cfg.target_color, 0.45)
        self._ch_col = np.asarray(cfg.crosshair_color, dtype=np.uint8)

        self._X = None

        self.crosshair = None
        self.targets = None
        self.target_vel = None
        self.frame_idx = cp.zeros(self.B, dtype=cp.int32)
        self._prev_err = cp.zeros((self.B, 2), dtype=cp.float32)

    # ------------------------------------------------------------------ 采样

    def _sample_targets(self):
        cp = self.cp
        # 拒绝采样(与 CPU 同几何);循环上限内几乎总能成功
        pos = []
        for k in range(self.K):
            p = None
            for _ in range(400):
                cand = self._rng.uniform(self._lo, self._hi, size=(self.B, 2)).astype(cp.float32)
                ok = cp.linalg.norm(cand - self.crosshair, axis=1) >= (
                    self.r + float(self.cfg.crosshair_radius) + _SPAWN_MARGIN)
                if p is not None:
                    ok &= cp.linalg.norm(cand - p, axis=1) >= (2 * self.r + _SPAWN_MARGIN)
                if bool(cp.all(ok)):
                    p = cand
                    break
            if p is None:
                p = cp.clip(self._center[None, :] + cp.float32(0.42) * self.H, self._lo, self._hi)
            pos.append(p)
        self.targets = (cp.stack(pos, axis=1) if self.K
                        else cp.zeros((self.B, 0, 2), dtype=cp.float32))

    def reset(self):
        cp = self.cp
        self.crosshair = cp.tile(self._center[None, :], (self.B, 1)).astype(cp.float32)
        self.targets = None
        self.frame_idx[:] = 0
        self._prev_err[:] = 0
        self._sample_targets()
        if self.tgt_speed > 0.0 and self.K:
            ang = self._rng.uniform(0.0, 2 * np.pi, size=(self.B, self.K)).astype(cp.float32)
            self.target_vel = cp.stack(
                [self.tgt_speed * cp.cos(ang), self.tgt_speed * cp.sin(ang)], axis=2)
        else:
            self.target_vel = cp.zeros((self.B, self.K, 2), dtype=cp.float32)
        return self.render()

    # ------------------------------------------------------------------ 推进

    def step(self, action):
        """action (B,2) in [-1,1];返回 (frame (B,H,W,3) uint8, dist (B,), hit (B,))。"""
        cp = self.cp
        a = cp.clip(cp.asarray(action, dtype=cp.float32), -1.0, 1.0)
        self.crosshair = cp.clip(self.crosshair + a * cp.float32(self.speed),
                                 self._ch_lo, self._ch_hi)
        if self.tgt_speed > 0.0 and self.K:
            self._advance_targets()
        d = cp.linalg.norm(self.targets - self.crosshair[:, None, :], axis=2)  # (B,K)
        j = cp.argmin(d, axis=1)
        dist = d[cp.arange(self.B), j]
        hit = dist <= cp.float32(self.r)
        frame = self.render(hit_j=j, hit_mask=hit)
        if bool(cp.any(hit)):
            self._respawn(hit, j)
        self.frame_idx += 1
        return frame, dist, hit

    def _advance_targets(self):
        cp = self.cp
        self.targets = self.targets + self.target_vel
        for axis, limit in ((0, float(self.W) - 1.0), (1, float(self.H) - 1.0)):
            lo, hi = self.r, limit - self.r
            col = self.targets[..., axis]
            under, over = col < lo, col > hi
            col[under] = lo + (lo - col[under])
            col[over] = hi - (col[over] - hi)
            flip = under | over
            self.target_vel[..., axis] = cp.where(flip, -self.target_vel[..., axis],
                                                  self.target_vel[..., axis])
            cp.clip(col, lo, hi, out=col)

    def _respawn(self, hit, j):
        """命中环境重采样该靶(向量化,只处理命中 env)。"""
        cp = self.cp
        idx = cp.flatnonzero(hit)
        if idx.size == 0 or self.K == 0:
            return
        m = int(idx.size)
        cand = cp.zeros((m, 2), dtype=cp.float32)
        todo = cp.ones(m, dtype=bool)   # 尚未采到合法点的 env
        for _ in range(400):
            if not bool(cp.any(todo)):
                break
            ti = cp.flatnonzero(todo)
            trial = self._rng.uniform(self._lo, self._hi, size=(int(ti.size), 2)).astype(cp.float32)
            ok = cp.linalg.norm(trial - self.crosshair[idx[ti]], axis=1) >= (
                self.r + float(self.cfg.crosshair_radius) + _SPAWN_MARGIN)
            if self.K > 1:
                kk = j[idx[ti]]
                others = self.targets[idx[ti]]                 # (t,K,2)
                for k in range(self.K):
                    sel = kk == k
                    if bool(cp.any(sel)):
                        dd = cp.linalg.norm(others[sel] - trial[sel][:, None, :], axis=2)
                        ok[sel] &= cp.all(dd >= (2 * self.r + _SPAWN_MARGIN), axis=1)
            good = ti[ok]
            cand[good] = trial[ok]
            todo[good] = False
        self.targets[idx, j[idx]] = cand

    # ------------------------------------------------------------------ 渲染
    # 向量化距离场:一次算完 (B,H,W) 的最近靶距离,再按掩码上色。无 scatter、
    # 无逐 env Python 循环 —— scatter 版实测 B=1 就 27ms,比 CPU 还慢。

    def _grids(self):
        cp = self.cp
        if getattr(self, "_X", None) is None:
            # CPU _draw_disc 用像素中心 (k+0.5) 判定,这里必须一致,否则圆边缘
            # 会差一圈像素。
            self._X = cp.arange(self.W, dtype=cp.float32) + cp.float32(0.5)
            self._Y = cp.arange(self.H, dtype=cp.float32) + cp.float32(0.5)
            self._X2 = self._X * self._X
            self._Y2 = self._Y * self._Y
        return self._X, self._Y

    def _min_dist2(self):
        """(B,H,W) 最近靶的距离平方(K=0 时全大值)。

        全帧向量化 —— 实测比"矩孔补丁 scatter"更快且无 Python 循环:
        每个 target 一次广播的三项式展开,像素中心 (k+0.5) 与真实浮点圆心,
        与 CPU ``_draw_disc`` 逐像素一致。
        """
        cp = self.cp
        X, Y = self._grids()
        if self.K == 0:
            return cp.full((self.B, self.H, self.W), 1e18, dtype=cp.float32)
        d2 = None
        for k in range(self.K):
            cx = self.targets[:, k, 0][:, None, None]
            cy = self.targets[:, k, 1][:, None, None]
            dk = (self._X2[None, None, :] + self._Y2[None, :, None]
                  - cp.float32(2.0) * cx * self._X[None, None, :]
                  - cp.float32(2.0) * cy * self._Y[None, :, None]
                  + (cx * cx + cy * cy))
            d2 = dk if d2 is None else cp.minimum(d2, dk)
        return d2

    def render(self, hit_j=None, hit_mask=None):
        cp = self.cp
        d2 = self._min_dist2()
        inside = d2 <= cp.float32(self.r * self.r)
        inner = max(1.0, self.r - max(1.0, self.r * 0.25))
        ring = inside & (d2 > cp.float32(inner * inner))
        hit_region = None
        if hit_mask is not None:
            hit_region = inside & cp.asarray(hit_mask)[:, None, None]

        frame = cp.empty((self.B, self.H, self.W, 3), dtype=cp.uint8)
        for c in range(3):
            ch = cp.full((self.B, self.H, self.W), int(self._bg[c]), dtype=cp.uint8)
            ch[inside] = int(self._base[c])
            ch[ring] = int(self._out_col[c])   # 描边最后画,覆盖该靶基底
            if hit_region is not None:
                # 命中靶:整盘换成高亮色,但描边仍是通用 out_col(与 CPU 一致)
                ch[hit_region & ~ring] = int(self._hit_col[c])
            frame[..., c] = ch
        self._draw_crosshair(frame)
        return frame

    def _draw_crosshair(self, frame):
        """向量化准星(无 Python 循环):加号两条矩形。"""
        cp = self.cp
        cx = cp.rint(self.crosshair[:, 0]).astype(cp.int32)[:, None, None]
        cy = cp.rint(self.crosshair[:, 1]).astype(cp.int32)[:, None, None]
        half = int(max(4, self.cfg.crosshair_radius * 3))
        th = int(max(1, self.cfg.crosshair_radius // 2))
        yy = cp.arange(self.H, dtype=cp.int32)[None, :, None]
        xx = cp.arange(self.W, dtype=cp.int32)[None, None, :]
        hx = ((yy >= cy - th // 2) & (yy < cy - th // 2 + th)
              & (xx >= cx - half) & (xx < cx + half + 1))
        vx = ((yy >= cy - half) & (yy < cy + half + 1)
              & (xx >= cx - th // 2) & (xx < cx - th // 2 + th))
        mask = hx | vx
        for c in range(3):
            ch = frame[..., c]
            ch[mask] = int(self._ch_col[c])

    # ------------------------------------------------------------------ 特权教师

    def teacher_labels(self, kp: float = 1.2, kd: float = 0.15,
                       aim_center: bool = True):
        """特权信息教师:直接用靶心坐标算 PD 动作(省掉图像检测)。

        与 CPU ``SeekController(use_aim_detect=False)`` 同公式:err = 靶心 - 画面中心。
        返回 (B,2) float32。**标签用特权信息、学生输入仍是像素**(标准 DAgger 做法)。
        """
        cp = self.cp
        if self.K == 0:
            return cp.zeros((self.B, 2), dtype=cp.float32)
        j = cp.argmin(cp.linalg.norm(self.targets - self.crosshair[:, None, :], axis=2), axis=1)
        tgt = self.targets[cp.arange(self.B), j]
        aim = self._center[None, :]
        err = (tgt - aim).astype(cp.float32)
        a = (cp.float32(kp) * err / cp.float32(self.W / 2.0)
             - cp.float32(kd) * (err - self._prev_err) / cp.float32(self.W / 2.0))
        self._prev_err = err
        return cp.clip(a, -1.0, 1.0).astype(cp.float32)
