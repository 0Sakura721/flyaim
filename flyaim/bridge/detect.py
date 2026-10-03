"""纯像素的靶标检测(供遥测与管道校验,**不是**果蝇臂的输入)。

===============================================================================
边界声明(重要,与 CONTRACT 禁止事项 1 的关系)
===============================================================================
本模块从捕获帧里找靶标位置,属于「观测者」侧信息 —— 与 runner.py 的
observer 同一性质:**输出只进遥测/报告,绝不回流给 FlyController**。
允许消费它的是:

    - 遥测(画框、算像素误差曲线,供报告分析);
    - SeekController(管道校验臂,见 controllers.py 的诚实声明)。

把检测接进 FlyController 就是让果蝇偷看答案,Forbidden。

===============================================================================
实现
===============================================================================
方法:RGB 颜色距离阈值 + scipy.ndimage.label 连通域,取最大连通块的质心。
Aim Lab 的靶是高饱和纯色球(默认偏蓝),颜色距离法比 HSV 阈值更少参数、
对光照渐变更稳。3D 场景里靶呈圆形投影,质心即瞄点参考。

局限(诚实版):
    - 颜色阈值对本项目的暗背景靶场帧与 Aim Lab 默认靶色均实测可用,
      但**换皮肤/换地图配色就会失效** —— 用前先看 preview 帧上的检测框;
    - 多目标时只返回最大块(本阶段任务设定为单靶);
    - 靶被准星/血条遮挡时质心会偏,不做任何修补。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

# 默认参考色:Aim Lab 经典靶的亮蓝;以及本项目的红靶。
# 实际用哪个由调用方传入;两者都只是起点,必须用 preview 帧核验。
DEFAULT_BLUE = (70, 170, 255)
DEFAULT_RED = (235, 70, 70)


@dataclass
class Detection:
    """单帧检测结果。not_found 时 ok=False。"""

    ok: bool
    cx: float = -1.0
    cy: float = -1.0
    radius_px: float = -1.0
    area_px: int = 0
    score: float = 0.0  # 匹配像素占比(掩膜内),调试阈值用


def find_target(
    frame: np.ndarray,
    ref_color: tuple[int, int, int] = DEFAULT_BLUE,
    tolerance: float = 60.0,
    min_area_px: int = 12,
) -> Detection:
    """在 RGB 帧里找与 ref_color 颜色距离 < tolerance 的最大连通块。

    tolerance 是 RGB 欧氏距离(0-441 量纲)。太小->漏检,太大->背景误检;
    可以用返回的 score(掩膜内平均颜色距离的补)粗调。
    """
    a = np.asarray(frame, dtype=np.float32)
    if a.ndim != 3 or a.shape[2] < 3:
        return Detection(ok=False)
    rgb = a[:, :, :3]
    ref = np.asarray(ref_color, dtype=np.float32).reshape(1, 1, 3)
    d2 = np.sum((rgb - ref) ** 2, axis=2)
    mask = d2 <= float(tolerance) ** 2
    n = int(mask.sum())
    if n < int(min_area_px):
        return Detection(ok=False, area_px=n)

    labels, nlab = ndimage.label(mask)
    if nlab <= 0:
        return Detection(ok=False, area_px=n)
    sizes = np.bincount(labels.reshape(-1))
    sizes[0] = 0  # 背景
    best = int(np.argmax(sizes))
    if sizes[best] < int(min_area_px):
        return Detection(ok=False, area_px=n)
    sel = labels == best
    yy, xx = np.nonzero(sel)
    score = float(np.mean(np.sqrt(d2[sel])))
    return Detection(
        ok=True,
        cx=float(xx.mean()),
        cy=float(yy.mean()),
        radius_px=float(np.sqrt(float(sizes[best]) / np.pi)),
        area_px=int(sizes[best]),
        score=round(1.0 - score / max(tolerance, 1e-6), 4),
    )


def annotate(frame: np.ndarray, det: Detection, color=(0, 255, 90)) -> np.ndarray:
    """把检测框画到帧副本上(preview 目检用):十字 + 外接方框。"""
    out = np.array(frame, dtype=np.uint8, copy=True)
    if not det.ok:
        return out
    h, w = out.shape[:2]
    r = max(2.0, float(det.radius_px))
    x0, x1 = int(max(0, det.cx - r)), int(min(w - 1, det.cx + r))
    y0, y1 = int(max(0, det.cy - r)), int(min(h - 1, det.cy + r))
    out[y0 : y1 + 1, x0] = color
    out[y0 : y1 + 1, x1] = color
    out[y0, x0 : x1 + 1] = color
    out[y1, x0 : x1 + 1] = color
    cx, cy = int(round(det.cx)), int(round(det.cy))
    if 0 <= cx < w:
        out[max(0, cy - 3) : min(h, cy + 4), cx] = color
    if 0 <= cy < h:
        out[cy, max(0, cx - 3) : min(w, cx + 4)] = color
    return out


def find_targets(
    frame: np.ndarray,
    ref_color: tuple[int, int, int] = DEFAULT_BLUE,
    tolerance: float = 60.0,
    min_area_px: int = 12,
    top_k: int = 6,
) -> list[Detection]:
    """多目标版:返回按面积降序的最多 top_k 个色块(目标锁定用)。"""
    a = np.asarray(frame, dtype=np.float32)
    if a.ndim != 3 or a.shape[2] < 3:
        return []
    rgb = a[:, :, :3]
    ref = np.asarray(ref_color, dtype=np.float32).reshape(1, 1, 3)
    d2 = np.sum((rgb - ref) ** 2, axis=2)
    mask = d2 <= float(tolerance) ** 2
    labels, nlab = ndimage.label(mask)
    if nlab <= 0:
        return []
    sizes = np.bincount(labels.reshape(-1))
    sizes[0] = 0
    out: list[Detection] = []
    for i in np.argsort(sizes)[::-1][: int(top_k)]:
        if sizes[i] < int(min_area_px):
            break
        yy, xx = np.nonzero(labels == i)
        score = float(np.mean(np.sqrt(d2[yy, xx])))
        out.append(Detection(
            ok=True, cx=float(xx.mean()), cy=float(yy.mean()),
            radius_px=float(np.sqrt(float(sizes[i]) / np.pi)),
            area_px=int(sizes[i]), score=round(1.0 - score / max(tolerance, 1e-6), 4),
        ))
    return out
