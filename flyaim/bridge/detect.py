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


def _detect_all(
    frame: np.ndarray,
    ref_color: tuple[int, int, int],
    tolerance: float,
    min_area_px: int,
    max_n: int,
    downsample: int = 1,
    exclude_center_px: float = 0.0,
    exclude_center_max_r: float = 0.0,
) -> list[Detection]:
    """核心检测:一次颜色距离 + 一次连通域,产出所有合格色块(按面积降序)。

    **为什么重写成这一个函数**(2026-10-05,为"又快又准"):
    原来 `find_target` 与 `find_targets` 各写一份,且 `find_targets` 对每个
    候选做一次 `labels == i` 全图比较(top_k=6 就是 6 遍 200 万像素扫描),
    1920x1080 下单次 99ms。这里改成:

      1. 颜色距离一次性算完(`d2`);
      2. `ndimage.label` 一次;
      3. 用 `ndimage.find_objects` / `bincount` + `argsort` 取前 N 个,
         **只对入选的 N 个做局部切片求质心**,不再全图比较。

    **downsample**(>1 时):先把帧整数倍降采样再检测,坐标/半径/面积乘回原尺寸。
    实测 700x700 Aim Lab 真实帧(BOX 滤波):

        /1 17.5ms -> cx=498, cy=20
        /2  8.2ms -> cx=498, cy=20   (质心零偏差,快 2.1x)
        /3  5.0ms -> cx=497, cy=19   (1px 偏差,快 3.5x)

    对半径 27px 的靶,1px 偏差远小于门限,换来 2~3.5 倍速度 —— 这是
    「又快又准」的实际取法。BOXA 平均是降采样的正确滤波(与视网膜分块均值同语义)。

    返回按面积降序的列表;`find_target` 取 [0],`find_targets` 要全部。
    """
    a = np.asarray(frame)
    if a.ndim != 3 or a.shape[2] < 3:
        return []
    ds = max(1, int(downsample))
    if ds > 1:
        h, w = a.shape[:2]
        th, tw = max(1, h // ds), max(1, w // ds)
        from PIL import Image

        small = np.asarray(Image.fromarray(a[:, :, :3]).resize((tw, th), Image.BOX))
        src = small
    else:
        src = a
        th, tw = a.shape[:2]
    rgb = src[:, :, :3]
    tol = float(tolerance)
    tol2 = tol * tol

    # ---- 整型快速路径(2026-10-05,为"又快又准") --------------------------
    # 原实现 rgb.astype(np.float32) 会为 700x700 分配 5.9MB 并逐像素转换,
    # 是 act 里最大的一块。颜色距离平方和可以直接用 int16 算:
    #   每个通道差 |Δ|<=255,Δ²<=65025,int32 累加 3 通道最大 195075,不溢出;
    #   比较 d2 <= tol² 时 tol<=441 -> tol²<=194481,同样在 int32 内。
    # 实测 700x700:整型比 float32 快 ~1.6x,且**阈值判定逐位等价**
    # (两边都是精确整数平方和,没有浮点舍入)。
    if rgb.dtype == np.uint8 and tol <= 441.0:
        # 逐通道 abs 差:直接在 uint8 上算,结果 max 255 不溢出 —— 省掉 3 次 int16 分配
        r = rgb[:, :, 0]
        g = rgb[:, :, 1]
        b = rgb[:, :, 2]
        dr = np.subtract(r, np.uint8(ref_color[0]), dtype=np.int16)
        dg = np.subtract(g, np.uint8(ref_color[1]), dtype=np.int16)
        db = np.subtract(b, np.uint8(ref_color[2]), dtype=np.int16)
        np.abs(dr, out=dr)
        np.abs(dg, out=dg)
        np.abs(db, out=db)
        # 单次 int32 分配:先算累加和(每项 <=255),再一次性平方。
        # 关键:(dr+dg+db)² 不等于 dr²+dg²+db²,所以这里**不能**合并 ——
        # 仍要三次平方,但可以复用同一块 int32 缓冲区,避免峰值内存与多次分配。
        acc = dr.astype(np.int32)
        acc *= acc
        tmp = dg.astype(np.int32)
        tmp *= tmp
        acc += tmp
        tmp = db.astype(np.int32)
        tmp *= tmp
        acc += tmp
        d2 = acc
        mask = d2 <= int(round(tol2))
        d2f = None  # score 用时再开方,避免全图 float
    else:
        ref = np.asarray(ref_color, dtype=np.float32).reshape(1, 1, 3)
        diff = rgb.astype(np.float32) - ref
        d2 = np.einsum("ijk,ijk->ij", diff, diff)
        mask = d2 <= tol2
        d2f = d2

    n = int(mask.sum())
    # 降采样下面积/阈值都要按比例折算
    min_area = int(min_area_px) / (ds * ds) if ds > 1 else int(min_area_px)
    if n < min_area:
        return []

    labels, nlab = ndimage.label(mask)
    if nlab <= 0:
        return []
    sizes = np.bincount(labels.reshape(-1))
    sizes[0] = 0  # 背景

    order = np.argsort(sizes)[::-1][: max(1, int(max_n) + 4)]
    # 一次拿到每个标签的像素坐标(比 labels==i 全图比较快得多)
    slices = ndimage.find_objects(labels, max_label=int(sizes.shape[0]) - 1)
    # 画面几何中心(原尺寸口径):准星恒在此处,见下 exclude_center_*
    cx_c = (a.shape[1] - 1) / 2.0
    cy_c = (a.shape[0] - 1) / 2.0
    out: list[Detection] = []
    n_excluded_xhair = 0
    for i in order:
        i = int(i)
        if sizes[i] < min_area:
            break
        if len(out) >= int(max_n):
            break
        sl = slices[i - 1] if i - 1 < len(slices) else None
        if sl is None:
            continue
        yy0, xx0 = sl[0].start, sl[1].start
        sub = labels[sl] == i
        yy, xx = np.nonzero(sub)
        if yy.size == 0:
            continue
        yy = yy + yy0
        xx = xx + xx0
        cx = float(xx.mean()) * ds
        cy = float(yy.mean()) * ds
        r_px = float(np.sqrt(float(sizes[i]) / np.pi)) * ds
        # ---- 排除准星自身(2026-10-05,「瞄了不开火」的最后一层成因) -------
        # 准星恒在画面几何中心,且是一个**很小**的色块。tolerance 放宽后(95)
        # 会把准星十字也匹配进来:它 err≈0、又永远静止,一旦 sticky 选中它就
        # 永远粘住 —— 系统对着自己开枪,真靶被无视。
        # 判据:离中心足够近 AND 半径小于上限(真靶在中心附近时半径远大于此)。
        if (exclude_center_px > 0.0 and exclude_center_max_r > 0.0
                and r_px <= exclude_center_max_r
                and abs(cx - cx_c) <= exclude_center_px
                and abs(cy - cy_c) <= exclude_center_px):
            n_excluded_xhair += 1
            continue
        d2sel = d2f[yy, xx] if d2f is not None else d2[yy, xx].astype(np.float32)
        score = float(np.mean(np.sqrt(d2sel)))
        out.append(Detection(
            ok=True,
            cx=cx,
            cy=cy,
            radius_px=r_px,
            area_px=int(round(float(sizes[i]) * (ds * ds))),
            score=round(1.0 - score / max(tol, 1e-6), 4),
        ))
    return out


def find_target(
    frame: np.ndarray,
    ref_color: tuple[int, int, int] = DEFAULT_BLUE,
    tolerance: float = 60.0,
    min_area_px: int = 12,
    downsample: int = 1,
    exclude_center_px: float = 0.0,
    exclude_center_max_r: float = 0.0,
) -> Detection:
    """在 RGB 帧里找与 ref_color 颜色距离 < tolerance 的**最大**连通块。

    tolerance 是 RGB 欧氏距离(0-441 量纲)。太小->漏检,太大->背景误检;
    可以用返回的 score(掩膜内平均颜色距离的补)粗调。

    多靶时只返回最大块 —— 这正是「换靶」的根源,需要空间一致性时用
    `find_targets` 自己挑(见 controllers.TriggerOnTarget 的 sticky)。
    """
    cands = _detect_all(frame, ref_color, tolerance, min_area_px, max_n=1,
                        downsample=downsample, exclude_center_px=exclude_center_px,
                        exclude_center_max_r=exclude_center_max_r)
    if cands:
        return cands[0]
    a = np.asarray(frame)
    if a.ndim != 3 or a.shape[2] < 3:
        return Detection(ok=False)
    return Detection(ok=False)


def find_targets(
    frame: np.ndarray,
    ref_color: tuple[int, int, int] = DEFAULT_BLUE,
    tolerance: float = 60.0,
    min_area_px: int = 12,
    top_k: int = 6,
    downsample: int = 1,
    exclude_center_px: float = 0.0,
    exclude_center_max_r: float = 0.0,
) -> list[Detection]:
    """多目标版:返回按面积降序的最多 top_k 个色块(目标锁定用)。"""
    return _detect_all(frame, ref_color, tolerance, min_area_px, max_n=top_k,
                       downsample=downsample, exclude_center_px=exclude_center_px,
                       exclude_center_max_r=exclude_center_max_r)


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


