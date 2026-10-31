"""定位并点击 Aim Lab 的「开始任务」按钮(标题行里那个青色 ▶)。

**为什么需要它**(2026-10-06 实测):游戏停在任务预览态时 HUD 显示
`POINTS 0 / TIME ∞ / ACCURACY 100%`,画面上**没有**「点击开始」白字、也没有
大白块按钮 —— 所以 `autostart_task` 原有两条探测都落空,只能退回按热键;
而 `R` 与 `F6` 实测都开不了局(F6 是「重新设置」,不是「开始」)。

真正的开始入口是**标题行里那个青色圆形 ▶**(屏幕抓图目检确认):
    `⬡ Gridshot Ultimate Tuned  [▶]  ⌄`

做法:在标题行带内按颜色找最大的青色连通块 → 取质心 → 点击 → 立刻监视
HUD 的 TIME 分段数,看是否从 1(∞) 变成 5(倒计时)。

**安全约束**:只在 y∈[Y0,Y1] 的标题带内找,绝不下探到左列菜单
(那里有 `退出游戏`,误点代价高)。找不到就不点,并明确报告。

用法::

    & $py -u tools/diag_click_start.py              # 只定位,不点
    & $py -u tools/diag_click_start.py --click      # 定位并点击,再监视 20s
"""
import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from flyaim.bridge.capture import ScreenCapture, find_window_region  # noqa: E402
from flyaim.bridge.inject import click_at  # noqa: E402
from flyaim.bridge.score_ocr import hud_region_for, time_glyph_count  # noqa: E402

OUT = ROOT / ".cache" / "screen_shot"

# 只在这条带里找 ▶ —— 标题行。下探会碰上左列菜单的「退出游戏」,风险不可接受。
BAND_Y = (60, 170)
BAND_X = (700, 1300)
# Aim Lab 的青色强调色:明显偏绿蓝、红通道低
TEAL_MIN_GB = 140
TEAL_MAX_R = 130


def find_start_button(img: np.ndarray) -> tuple[int, int, int] | None:
    """在标题带里找最大青色连通块。返回 (cx, cy, 像素数) 或 None。"""
    y0, y1 = BAND_Y
    x0, x1 = BAND_X
    reg = img[y0:y1, x0:x1]
    r, g, b = reg[:, :, 0].astype(np.int16), reg[:, :, 1].astype(np.int16), \
        reg[:, :, 2].astype(np.int16)
    mask = (g > TEAL_MIN_GB) & (b > TEAL_MIN_GB) & (r < TEAL_MAX_R)
    if not mask.any():
        return None
    # 简单连通分量(4 邻域,BFS)—— 不引 scipy
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    best = (0, 0, 0)
    ys, xs = np.nonzero(mask)
    for sy, sx in zip(ys, xs):
        if seen[sy, sx]:
            continue
        stack = [(sy, sx)]
        seen[sy, sx] = True
        comp = []
        while stack:
            cy, cx = stack.pop()
            comp.append((cy, cx))
            for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                ny, nx = cy + dy, cx + dx
                if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not seen[ny, nx]:
                    seen[ny, nx] = True
                    stack.append((ny, nx))
        if len(comp) > best[2]:
            cy = sum(p[0] for p in comp) / len(comp)
            cx = sum(p[1] for p in comp) / len(comp)
            best = (int(round(cx)) + x0, int(round(cy)) + y0, len(comp))
    return best if best[2] > 0 else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--click", action="store_true", help="定位后真的点击")
    ap.add_argument("--watch", type=float, default=20.0)
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    region = find_window_region("aimlab")
    cap = ScreenCapture(region=region, out_size=None, backend="dxcam",
                        grab_retry=4, grab_retry_ms=4.0)
    frame, _m = cap.read()
    img = np.asarray(frame, dtype=np.uint8)
    print(f"窗口 {region}  帧 {img.shape}")

    hit = find_start_button(img)
    # ⚠️ **最小像素门槛是必须的**(2026-10-06 实测):首版没有门槛,结果把标题行里
    # 零散的抗锯齿像素凑成 10 px 就打印"✅ 找到青色块"。而那个 ▶ 圆点至少几百像素
    # —— 也就是说它当时**报的是假阳性**。左列就有「退出游戏」,这种假阳性一旦被
    # 信任并点击,代价极高。宁可报"没找到",也不能报"找到了"而其实是噪点。
    MIN_PX = 200
    if hit is None or hit[2] < MIN_PX:
        got = "无" if hit is None else f"{hit[2]} 像素(低于门槛 {MIN_PX})"
        print(f"❌ 标题带 y{BAND_Y} x{BAND_X} 内没找到可信的青色按钮:{got}")
        print("   可能:不在任务预览页 / 该页配色不同 / 带范围要调。")
        print("   **不会点击** —— 假阳性点错(左列有「退出游戏」)代价太高。")
        cap.close()
        return 1
    cx, cy, n = hit
    print(f"✅ 找到青色块:{n} 像素(≥{MIN_PX}),质心 (cx={cx}, cy={cy})"
          f"  ← 应该是那个 ▶ 播放键")

    # 存一张带标记的图供人核对
    marked = img.copy()
    marked[max(0, cy - 30):cy + 30, max(0, cx - 30):cx + 30] = [255, 0, 0]
    Image.fromarray(marked).crop((600, 40, 1400, 200)).resize((1600, 320),
                                                              Image.LANCZOS).save(
        OUT / "start_button_marked.png")
    print(f"   标记图 -> {OUT / 'start_button_marked.png'}(红框=识别到的位置)")

    if not args.click:
        print("(--click 未给,只定位不动手)")
        cap.close()
        return 0

    print(f"👉 点击 ({cx},{cy}) …")
    click_at(cx, cy)
    time.sleep(1.0)

    # 立刻看 HUD 的 TIME 分段:1=∞ 没开局 / 5=倒计时
    left, top, w, h = hud_region_for(region)
    hud_cap = ScreenCapture(region=(left, top, w, h), out_size=None,
                            backend="dxcam", grab_retry=4, grab_retry_ms=4.0)
    t0 = time.perf_counter()
    last = None
    seen5 = 0
    while time.perf_counter() - t0 < args.watch:
        el = time.perf_counter() - t0
        try:
            f2, _ = hud_cap.read()
            ng = time_glyph_count(np.asarray(f2, dtype=np.int32))
        except Exception as exc:
            print(f"[{el:5.1f}s] 抓帧失败 {type(exc).__name__}")
            time.sleep(0.5)
            continue
        mark = ""
        if last is not None and ng != last:
            mark = f"  <<<< 分段 {last} -> {ng}"
        if ng == 5:
            seen5 += 1
            mark += "  ★★★ 计分局开始了!"
        print(f"[{el:5.1f}s] 分段={ng}{mark}", flush=True)
        last = ng
        time.sleep(0.5)
    hud_cap.close()
    cap.close()
    print("=" * 70)
    print("✅ 点击成功开局!" if seen5 else
          "❌ 点了但 TIME 仍未变成倒计时 —— 那个青色块不是开始键,或还有一步。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
