"""Aim Lab 任务内会话:自动开始任务 -> 控制器打满一局 -> 命中数入账。

模式:
    collect  seek 当导师打靶(带开火),同时步进真实网络记录 (DN rates, action) 对
    seek     纯 seek + 开火(导师上限参照,也是"能瞄准会开枪"的工程基线)
    fly      果蝇网络 + 开火(被测臂;--readout 指定重训权重)

运行(一局 Gridshot ≈ 60s,工具自动点开始):

    & $py tools/aimlab_play.py --mode collect --npz-out flyaim/runs/bridge/collect_1.npz
    & $py tools/aimlab_play.py --mode seek            # 导师上限
    & $py tools/aimlab_play.py --mode fly --readout flyaim/runs/bridge/readout_ingame.npz

命中入账规则(Gridshot = hitscan 无散布):准星压住靶盘(err <= r*0.9)时
点击 = 必中。inferred_hits 即该口径的命中数。

⚠️ 运行期间会真实移动/点击鼠标。请保持 Aim Lab 前台、手离鼠标。
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from flyaim.bridge.capture import ScreenCapture, focus_window, find_window_region  # noqa: E402
from flyaim.bridge.controllers import (  # noqa: E402
    FlyController,
    TeacherCollectController,
    TriggerOnTarget,
)
from flyaim.bridge.detect import find_target  # noqa: E402
from flyaim.bridge.gain import GainModel  # noqa: E402
from flyaim.bridge.inject import SendInputSink, click_at, press_key  # noqa: E402
from flyaim.bridge.loop import BridgeLoop  # noqa: E402
from flyaim.config import BrainConfig, ReadoutConfig, RetinaConfig  # noqa: E402

D = ROOT / "flyaim" / "data" / "build"
NORM = ROOT / "flyaim" / "runs" / "norm"
READOUT_OFFLINE = ROOT / "flyaim" / "runs" / "readout" / "readout_weights.npz"
TEAL = (48, 224, 224)
CAL_W_SCALE = 16.0
CAL_INPUT_GAIN = 1.5
CAL_NORM = "indeg"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["collect", "seek", "fly", "hybrid"])
    ap.add_argument("--beta", type=float, default=0.25,
                    help="hybrid 模式:action = seek + β·fly(果蝇扰动权重)")
    ap.add_argument("--kp", type=float, default=2.6,
                    help="seek 比例增益(实测标定)。语义 = 每 --err-scale-px 像素"
                         "误差输出 kp 的 action。2026-10-05 晚实测上调:靶速中位 "
                         "17px/拍需要 act 0.86,kp2.2 只输出 0.40(追不上);"
                         "kp3.2 实测 err 中位 66->59.6、换靶率最低(0.8%%/拍)")
    ap.add_argument("--err-scale-px", type=float, default=320.0,
                    help="误差归一化尺度(像素,默认 320)。**dual 双窗必需**:"
                         "否则同一物理误差在全屏窗(w=1920)、中心窗(w=900)下会得到"
                         "相差 2.1 倍的 action —— 切窗即改增益,引起抖动/振荡。"
                         "0=用旧行为(w/2,随窗变)")
    ap.add_argument("--assoc-px", type=float, default=0.0,
                    help="目标关联半径(像素,0=自动≈0.75*err_scale_px)。**治「晃动太大」"
                         "的关键旋钮**:靶场常同屏 3 个靶,准星会在「当前最近的那个」之间"
                         "来回弹(实测 err 方向翻转率 40、每 50 拍一次大于 150px 的靶坐标"
                         "瞬移 = 换靶)。给出关联半径后,只在半径内认自己是原靶,同屏多靶"
                         "不再互相抢;超出半径连续 --lost-tol 拍才允许重选")
    ap.add_argument("--switch-gain", type=float, default=1.3,
                    help="换靶迟滞倍数(默认 1.3):新靶到准星的距离必须小于当前靶的"
                         "1/switch-gain 才换,否则维持现有目标。越大越「专一」(越不愿换)")
    ap.add_argument("--lost-tol", type=int, default=3,
                    help="丢靶容忍拍数(默认 3):关联不上连续这么多拍才允许重选目标。"
                         "期间沿用外推位置继续追,避免靶被短暂遮挡就丢锁")
    ap.add_argument("--fixate-px", type=float, default=14.0,
                    help="固视门限 px(D41 扫视-固视双模态):err 低于此值动作归零,完全停住等开火 —— "
                         "量化自用户理想实录(运动占空比 5%%/驻留 629ms vs 程序 27%%/108ms)。"
                         "实际进入带=min(此值, 0.6×开火门限)、保持带=min(--saccade-px, 0.85×开火门限)"
                         "**动态收缩**(D41b:小靶门限 15.5px 时静态带 20px 会造成「固视着但开不了火」死锁)。0=关")
    ap.add_argument("--saccade-px", type=float, default=20.0,
                    help="退出固视 px(默认 20;与 fixate-px 组成迟滞带防检测噪声抖动。"
                         "实际保持带=min(此值, 0.85×开火门限)—— 动态收缩见 --fixate-px)")
    ap.add_argument("--npz-out", default=None, help="collect 模式的数据落盘路径")
    ap.add_argument("--readout", default=None, help="fly 模式的读出权重(默认离线权重)")
    ap.add_argument("--gain-json", default=str(ROOT / "flyaim/runs/bridge/gain.json"))
    ap.add_argument("--max-counts", type=float, default=150.0)
    ap.add_argument("--seconds", type=float, default=52.0,
                    help="打靶时长(留出任务开场与收尾余量)")
    ap.add_argument("--no-autostart", action="store_true",
                    help="不自动点「点击开始」(已手动进入任务时用)")
    ap.add_argument("--no-trigger", action="store_true", help="关闭开火层(纯瞄准)")
    ap.add_argument("--fire-frac", type=float, default=0.75,
                    help="开火门限 = 靶半径 × 该系数(默认 0.75:准星进入球体 3/4 才开火。"
                         "D40:1.0 时 25%% 的开火贴着球边(err/r>0.8),球在点击生效的"
                         "几十 ms 里移出即打空;0.75 过滤贴边弹。历史:1.4 会在球外"
                         " 40%% 就开火,大量擦边弹。太紧会「瞄了不开火」)")
    ap.add_argument("--min-radius-px", type=float, default=4.0,
                    help="开火门限的像素下限(默认 4.0;防小靶/远靶时门限塌缩到 0)")
    ap.add_argument("--cooldown", type=float, default=0.15,
                    help="两次开火的最小间隔秒(默认 0.15,约 10 拍 @71Hz)。"
                         "D40 根因修复:0.03 让同一颗球被连泻 5-10 枪(52s 开火 425 次,"
                         "而球只有 ~45 个)—— Gridshot 每球只记 1 hit,多打的枪全是 "
                         "miss,游戏内准确率被砸到 ~10%%。新球出现会触发换靶检测"
                         "(jump-reset)自动清零冷却,故 0.15 **不**拖累新靶首枪。"
                         "历史:0.22 在 21Hz 下占 4.7 拍会吃掉刚刷新的靶,当时降到 0.03"
                         "是矫枉过正")
    ap.add_argument("--fire-confirm", type=int, default=2,
                    help="开火确认拍数(D41,默认 2):连续 N 拍进门限才扣扳机。"
                         "扫视刚停时前几拍的大额注入还在指针管道里消化"
                         "(EPP 放大高速段),画面 err 达标但真实准星在惯性滑行,"
                         "此刻点击必 miss。推迟 1 拍等相机停稳:Gridshot 靶静止,"
                         "代价 15ms 可忽略。1=关(旧行为)")
    ap.add_argument("--det-tolerance", type=float, default=95.0,
                    help="色块匹配容差(默认 95,原来 60)。容差越大掩膜覆盖越完整的靶盘,"
                         "det.radius_px 越接近靶的真实视觉半径 —— 半径偏小是"
                         "「瞄了不开火」的第三成因")
    ap.add_argument("--sticky-px", type=float, default=90.0,
                    help="目标粘滞半径(默认 90,0=关)。>0 时开火层在候选靶里挑离上一拍"
                         "最近的那个,防止「永远选最大连通块」导致换靶、err 跳变。"
                         "这是「瞄得准却不开火」的主因")
    ap.add_argument("--capture", choices=["center", "full", "dual"], default="dual",
                    help="center=只抓准星周围的小窗不缩放(read 快 7 倍,但**搜靶阶段"
                         "看不到窗外的靶**);full=抓全屏再缩到 640x480(旧路径);"
                         "dual(默认)=**搜靶用全屏、锁定后用中心窗**自动切换 —— "
                         "这是「命中率低」的正解:实测靶在全屏 (144,48) 时,中心窗 700/900"
                         "全都 ok=False(靶在窗外左 366px、上 42px),而 dual 能先在全屏"
                         "看见它、把相机转过去、再切小窗精跟")
    ap.add_argument("--capture-size", type=float, default=900.0,
                    help="dual/center 模式的**跟踪窗**边长(默认 900)。dual 模式下"
                         "搜靶窗恒为全屏不受此影响,故不必为了'看得远'而放大它")
    ap.add_argument("--track-lock-px", type=float, default=260.0,
                    help="dual 模式:靶离窗中心在该像素数以内即认为'已锁上',切到中心窗"
                         "(默认 260;应小于 capture-size/2,否则一进窗就抖着切)")
    ap.add_argument("--full-down", type=int, default=960,
                    help="dual 模式搜靶全屏帧的降采样边长(默认 960,0=不缩放)。"
                         "全屏 1920->960 让检测快 ~4 倍;坐标由 detect 的 downsample"
                         "逻辑还原,不影响精度(靶半径 25px 仍是 12px,远超 min_area)")
    ap.add_argument("--xhair-px", type=float, default=12.0,
                    help="准星排除:离画面中心该像素数以内、且半径 <= --xhair-max-r 的色块"
                         "判为准星自身,不当作靶(默认 12)。0=关。**这是「瞄了不开火」"
                         "最后一层成因**:准星 err≈0 又静止,被 sticky 选中就永远粘住")
    ap.add_argument("--xhair-max-r", type=float, default=8.0,
                    help="准星排除的半径上限(默认 8;真靶在中心附近时半径远大于此)")
    ap.add_argument("--jump-reset-px", type=float, default=45.0,
                    help="靶心比上一拍移动超过该像素数即视为换靶,清零开火冷却(默认 45,"
                         "0=关)。冷却本意是防对同一靶连点,换靶后不该继续被挡 —— "
                         "实测 46.6Hz 下 64%% 的进门限拍被冷却吃掉,不少是新靶刚出现")
    ap.add_argument("--downsample", type=int, default=2,
                    help="检测前把帧整数倍降采样(默认 2,0/1=关)。坐标自动放大回原"
                         "尺寸。实测 700x700: /2 快 2.1x 且质心零偏差,/3 快 3.5x 偏 1px")
    ap.add_argument("--action-ema", type=float, default=0.7,
                    help="动作 EMA 平滑系数(默认 0.8;0=关)。**越高滞后越小**:"
                         "α=0.35 在 54Hz 下时间常数 2.9 拍,在实测的 1.9Hz 振荡频点"
                         "引入 34.6° 相位滞后 —— 纯消耗相位裕度、加剧极限环,而动作"
                         "本身**从未饱和**(实测 |raw|>1 占 0%%,EMA 没在压裁剪,只是加滞后)。"
                         "实测 α=0.35→0.6:振荡从 1.92Hz 明显 → 无明显,一局开火 263→550")
    ap.add_argument("--smooth-delta", type=float, default=0.60,
                    help="每拍动作最大变化量(速率限制,治「晃得太激」)。"
                         "实测 0.40 会把 10%% 的拍削顶、拖慢追靶(靶速 17px/拍需要"
                         "大动作);0.80 只削 3%% 且保留 bang-bang 抑制。0=关")
    ap.add_argument("--smooth-soft", type=float, default=1.5,
                    help="软饱和拐点 k:用 k*tanh(a/k) 替代硬裁剪(小误差近线性、中间段压缩、"
                         "满舵温和收敛)。k 越大越激进。默认 1.5;0=关(退回硬裁剪)")
    ap.add_argument("--scan-amp", type=float, default=0.35,
                    help="搜靶扫视幅度(action 单位):看不到靶时慢速扫视而不是傻站。"
                         "实测不开扫视→0.35 可救回『对着墙站 60 秒开火 0 次』的死锁。"
                         "默认 0.35;0=关(退回『没靶就静止』)")
    ap.add_argument("--scan-period-s", type=float, default=1.6,
                    help="扫视周期(秒):越短扫得越快越晃。默认 1.6")
    ap.add_argument("--deadband-px", type=float, default=6.0,
                    help="中心死区:靶离画面中心在此像素内则动作清零(遏制中心附近抖动)。"
                         "**必须小于开火门限**,否则会卡在进不了门限的死区。默认 6;0=关")
    ap.add_argument("--stall-frames", type=int, default=20,
                    help="画面停滞看门狗:连续 N 帧靶位不动且无新开火,判定任务已结束并"
                         "自动按 R 重开(默认 20,0=关)。实测任务几十秒就打完了,之后空跑"
                         "会让战报看起来像「瞄了不开火」(rep-3: 96%% 帧画面静止)")
    ap.add_argument("--stall-px", type=float, default=3.0,
                    help="停滞判据:靶心在此像素数以内视为不动(默认 3)")
    ap.add_argument("--record", type=int, default=0, metavar="W",
                    help="进程内录像,值为视频宽度像素(推荐 480;0=关)。D39:"
                         "独立进程录屏与桌面捕获冲突只会得到冻结帧,录像必须"
                         "挂主循环。产出 runs/<dir>/rec.mp4 + beats.csv")
    ap.add_argument("--record-every", type=int, default=2,
                    help="录像抽帧:每 N 拍存一帧(默认 2,80Hz 拍频下 ≈40fps)")
    ap.add_argument("--beat-log", action="store_true",
                    help="只记逐拍 CSV 不录像(--record 时 CSV 自动开)")
    ap.add_argument("--scratch-dir", default=None, metavar="DIR",
                    help="录像 raw 中间流落盘目录(默认与局目录同盘)。**D 盘满时"
                         "指定到 C 盘**(如 C:/Users/Admin/AppData/Local/Temp):"
                         "60s 局 raw 约 250MB,转码成 MP4 后仅 ~10MB 自动移回")
    ap.add_argument("--no-telemetry", action="store_true",
                    help="跳过遥测写盘(磁盘满/快速跑分时用;战报统计不受影响)")
    ap.add_argument("--score-every", type=int, default=10,
                    help="HUD 分数读取间隔拍数(D42,默认 10≈6Hz@60Hz);0=关闭。"
                         "读 POINTS/ACCURACY 两框,TIME 框用于字形自校准")
    ap.add_argument("--hud-backend", choices=["auto", "dxcam", "mss", "pil"],
                    default="auto",
                    help="HUD 取帧后端(D42.7,默认 auto=dxcam→mss→pil)。"
                         "**不要用 pil**:实测 d42-score 局 335 次读分仅 11 次产出"
                         "数值。dxcam 走 Desktop Duplication —— 既然主捕获看得见"
                         "游戏,它就看得见 HUD")
    ap.add_argument("--hud-learn-time", action="store_true",
                    help="打开**遗留的** TIME 在线字形自校准(D42.5,**默认关闭**)。"
                         "remain_s 由宿主时钟推算,与游戏时钟差几秒时会把字形打上"
                         "错误标签存进模板库、污染后续全部读数。"
                         "**新默认走锚定式 TimeCalibrator**(见 --no-time-calib)")
    ap.add_argument("--no-time-calib", action="store_true",
                    help="关闭锚定式 TIME 校准(TimeCalibrator)。默认**开启**:"
                         "它锚在**观测到的 ∞→倒计时切换**上(不用宿主时钟),"
                         "且带跨帧校验 —— 标签错了会回滚并停用,不污染模板库")
    ap.add_argument("--round-seconds", type=float, default=60.0,
                    help="一局计分任务的时长(D42,Gridshot 默认 60s)。锚定式校准用"
                         "它预测 TIME 的显示值;填错会导致校验失败并停用校准")
    ap.add_argument("--round-wait", type=float, default=45.0, metavar="S",
                    help="开局门闸等待上限(默认 45s)。**判据是 HUD 的 TIME 框显示 "
                         "MM:SS**;等不到就中止 —— 避免整局跑在 TIME=∞ 的练习模式里"
                         "却毫无告警(2026-10-06 实盘教训)")
    ap.add_argument("--no-round-gate", action="store_true",
                    help="关掉开局门闸(退回旧行为:**不保证在计分局里**)")
    ap.add_argument("--allow-practice", action="store_true",
                    help="门闸超时后仍然开跑(练习局数据会被标注**不可与目标比较**)")
    ap.add_argument("--runs-dir", default=None, metavar="DIR",
                    help="局目录根(默认 flyaim/runs)。**D 盘满时指定到别的盘**"
                         "(如 E:/flyaim_runs):局目录/战报/录像/逐拍日志全部落那里,"
                         "代码与模型仍在原处")
    ap.add_argument("--tag", default=None)
    return ap.parse_args()


def _task_running(cap_region) -> bool:
    """任务是否在进行:画面里能找到青色靶。"""
    cap = ScreenCapture(region=cap_region, out_size=None)
    frame, _ = cap.read()
    cap.close()
    return bool(find_target(frame, ref_color=TEAL, tolerance=60.0).ok)


def autostart_task(cap_region, tries: int = 3, require_target: bool = False,
                   hud_running=None) -> bool:
    """尝试进入一局**计分**任务。

    ===========================================================================
    2026-10-06 修订:原「检出青靶 = 已在任务内」的早退判据**是错的**
    ===========================================================================
    实测(run 20261006-024357-d42-hudfix):大厅/UI 上也有青色元素。`autostart`
    在 (1664,43) 检出一个 **R=11.3px** 的青色块(任务靶 p50 可是 27.7px),就打印
    "确认已在任务内"并**立即 return** —— 一次都没去点开始。于是整局 60 秒跑在
    练习模式(TIME=∞),分数 10,820 / ACC 48% 全是真的,**但没有计时器**,
    和目标 130,335 根本不在同一量纲。

    **修订后的判据:唯一可信的"已开局"信号是 HUD 的 TIME 框显示 `MM:SS`**
    (即 `time_glyph_count() == 5`)。它不依赖任何 OCR 模板,永远可用。
    青靶只用来区分"在大厅"还是"在靶场",**不再当作开局证据**。

    **为什么在靶场里只按 R、不点击**:任务内的白色 HUD 文字会被误判成"大按钮",
    点一下就把刚开的一局点没(原实现吃过这个亏)。靶场里没有「点击开始」那种
    大厅 UI,所以按热键 R 是安全的;点击只留给大厅/结算页。

    require_target:保留参数以兼容旧调用点,但语义已变(见上)。
    """
    saw_target = False
    for i in range(tries):
        if hud_running is not None and hud_running():
            print(f"  autostart[{i}]: TIME 已在倒计时 —— 计分局进行中")
            return True
        cap = ScreenCapture(region=cap_region, out_size=None)
        frame = None
        # dxcam 的区域首帧有时返回 None(「尚无首帧」)—— 重试几次再放弃。
        for _ in range(5):
            try:
                frame, _ = cap.read()
                break
            except RuntimeError:
                time.sleep(0.12)
        cap.close()
        if frame is None:
            continue
        l, t, w, h = cap_region
        tgt = find_target(frame, ref_color=TEAL, tolerance=60.0)
        if tgt.ok:
            saw_target = True
        # 0) 在靶场里但还没开局 → **只按 R**(不点击,避免把 HUD 当按钮点掉)
        if tgt.ok and tgt.radius_px >= 15.0:
            print(f"  autostart[{i}]: 靶场已就位(青靶 R{tgt.radius_px:.1f}),按 R 开局")
            press_key(0x52)  # VK R = 再练一次/开始
            time.sleep(2.0)
            continue
        # 1) 大厅的「点击开始」固定文字(细笔画,面积阈值要小)
        txt = find_target(frame, ref_color=(245, 245, 245), tolerance=45.0,
                          min_area_px=80)
        if txt.ok and 0.18 * h < txt.cy < 0.40 * h and 0.30 * w < txt.cx < 0.70 * w:
            sx, sy = l + int(txt.cx), t + int(txt.cy)
            print(f"  autostart[{i}]: 检测到「点击开始」({sx},{sy}),点击")
            click_at(sx, sy)
            time.sleep(2.0)
            continue
        # 2) 结算页大按钮(大面积白块)
        btn = find_target(frame, ref_color=(245, 245, 245), tolerance=45.0,
                          min_area_px=1500)
        if btn.ok and 0.30 * h < btn.cy < 0.85 * h:
            sx, sy = l + int(btn.cx), t + int(btn.cy)
            print(f"  autostart[{i}]: 检测到大按钮/重开键 ({sx},{sy}),点击")
            click_at(sx, sy)
            time.sleep(2.0)
            continue
        # 3) 结算页/未知页:按 R(再练一次热键)
        print(f"  autostart[{i}]: 按 R(再练一次)")
        press_key(0x52)
        time.sleep(2.0)
    if hud_running is not None:
        return bool(hud_running())
    return saw_target


def wait_for_round(strip, region, timeout_s: float = 45.0, auto_start: bool = True,
                   allow_practice: bool = False) -> bool:
    """等**真正的计分局**开始(HUD 的 TIME 显示 `MM:SS`)才返回 True。

    ===========================================================================
    为什么要有这道闸(D42 实盘教训)
    ===========================================================================
    上一版把「全屏检出青靶」当"已在任务内",于是 60 秒整局跑在练习模式里
    (TIME=∞)而**毫无告警** —— 分数/准确率都在动,看起来像正常工作,实际
    与目标完全不可比。**没有这道闸,任何"分数闭环调优"都是在错量纲上做。**

    判据用 `time_glyph_count()`(TIME 框分段数,不依赖模板):
        5 → 计分局进行中,放行
        1 → 练习态 `∞`,没开局 → 尝试自动开局,并**在屏幕上提示人工介入**
    超时:allow_practice=True 时仅告警放行(明确记为"非计分局");否则返回 False,
    由调用方**中止**而不是跑一局废数据。
    """
    from flyaim.bridge.score_ocr import time_glyph_count

    def _state() -> int:
        if strip is None:
            return -1
        try:
            return time_glyph_count(strip.read())
        except Exception:
            return -1

    ng = _state()
    if ng == 5:
        print("  ✅ TIME 显示倒计时 —— 计分局已在进行")
        return True
    if ng == 1:
        print("  ⏳ TIME 显示 ∞ —— **练习态,尚未开局**。"
              "正在尝试按 R 开局…")
    else:
        print(f"  ⏳ TIME 分段异常({ng})—— 可能在加载/大厅。正在尝试开局…")

    t0 = time.perf_counter()
    nxt_try = time.perf_counter()
    last_msg = 0.0
    while time.perf_counter() - t0 < timeout_s:
        if time.perf_counter() >= nxt_try:
            if auto_start:
                autostart_task(region, tries=1)
            nxt_try = time.perf_counter() + 3.0
        ng = _state()
        if ng == 5:
            print(f"  ✅ 检出计分局(TIME 倒计时)—— 用时 "
                  f"{time.perf_counter() - t0:.1f}s")
            return True
        el = time.perf_counter() - t0
        if el - last_msg >= 6.0:
            last_msg = el
            print(f"  ⏳ 仍在等开局({el:.0f}/{timeout_s:.0f}s)—— "
                  f"TIME 分段={ng}。**如脚本开不了,请人工在 Aim Lab 里点开始/"
                  f"按空格**;脚本会一直等。")
        time.sleep(0.4)
    if allow_practice:
        print(f"  ⚠️ {timeout_s:.0f}s 内未检出计分局(TIME 分段={ng}),"
              f"但 --allow-practice 已开 → **本局是非计分局,数据不可与目标比较**")
        return True
    print(f"  ❌ {timeout_s:.0f}s 内始终没有计分局(TIME 分段={ng})。"
          f"**中止,不跑废数据。** 请先在 Aim Lab 里进入 Gridshot 并点开始,"
          f"或用 --allow-practice 强制跑练习局。")
    return False


class ActionEMA:
    """动作 EMA 平滑:把 17-30 Hz 的阶跃指令变成连续轨迹(治抽搐,不改语义)。

    ⚠️ **必须透传 `last_detection` / `last_candidates`**(2026-10-05 实测教训):
    本类夹在 SeekController 与 TriggerOnTarget 之间。若它把内层的检测结果挡住,
    开火层就会**另起炉灶**重新检测、并用**自己那份 sticky 状态**选靶 ——
    于是"瞄准锁的靶"与"开火看的靶"是两个不同的靶:SeekController 把它选中的
    靶拖到画面中心(action→0,相机停住),而 TriggerOnTarget 盯着另一个离中心
    193px 的靶,err 永远进不了门限 → 表象就是「瞄得挺准,就是不开火」。
    实测铁证:同一帧(535,398,R26)独立回放 seek.act=[1.0,0.28],而实盘 act≈0.005,
    因为实盘里 seek 锁的其实是 (349,350) 那个靶,与开火层看的不是同一个。

    修法:`__getattr__` 兜底透传(下面属性显式声明只是为了可读与 IDE 提示)。
    """

    def __init__(self, inner, alpha: float) -> None:
        self.inner = inner
        self.alpha = float(alpha)  # 新样本权重;0.35 = 温和平滑
        self._prev: np.ndarray | None = None
        self.raw_action = np.zeros(2, dtype=np.float32)

    @property
    def brain(self):
        return getattr(self.inner, "brain", None)

    @property
    def name(self):
        return getattr(self.inner, "name", "?") + "+ema"

    @property
    def last_detection(self):
        """透传内层本拍检测到的靶(开火层必须看到**瞄准用的同一个靶**)。"""
        return getattr(self.inner, "last_detection", None)

    @property
    def last_candidates(self):
        """透传内层本拍全部候选(开火层复用,避免重复检测 + 选靶分歧)。"""
        return getattr(self.inner, "last_candidates", None)

    def __getattr__(self, item):
        # 未知属性一律转发到内层(统计量、blind 等),避免再次出现"属性被挡住"
        # 这类隐蔽分歧。注意 __getattr__ 只在正常查找失败时触发,不会遮蔽上面
        # 显式声明的属性,也不会遮蔽 self.inner/self.alpha。
        return getattr(self.inner, item)

    def act(self, frame: np.ndarray) -> np.ndarray:
        a = np.asarray(self.inner.act(frame), dtype=np.float32).reshape(2)
        self.raw_action = a
        if self.alpha <= 0 or self._prev is None:
            self._prev = a
        else:
            self._prev = self.alpha * a + (1 - self.alpha) * self._prev
        return self._prev.copy()

    def reset(self) -> None:
        self.inner.reset()
        self._prev = None

    def on_window_switch(self) -> None:
        """捕获窗切换:转发给内层(EMA 的 `_prev` 是 action 语义,与窗无关,不清)。"""
        hook = getattr(self.inner, "on_window_switch", None)
        if callable(hook):
            hook()

    def close(self) -> None:
        self.inner.close()

    @property
    def n_hits_inferred(self):
        return getattr(self.inner, "n_hits_inferred", 0)

    @property
    def n_fires(self):
        return getattr(self.inner, "n_fires", 0)

    @property
    def n_blocked_by_cooldown(self):
        return getattr(self.inner, "n_blocked_by_cooldown", 0)

    @property
    def n_geometric_miss(self):
        return getattr(self.inner, "n_geometric_miss", 0)

    @property
    def n_in_gate(self):
        """落在开火几何门限内的拍数(未被冷却挡下的才开火)。"""
        return getattr(self.inner, "n_in_gate", 0)


def _build_fly(args):
    rp = Path(args.readout) if args.readout else READOUT_OFFLINE
    if not Path(rp).exists():
        sys.exit(f"❌ 读出权重不存在: {rp}")
    ro = ReadoutConfig(mode="trained", weights_path=str(rp))
    core = FlyController(build_system())
    core.system.readout.load(rp)
    print(f"  fly 读出权重: {rp.name}")
    return core


def build_system(device: str = "cuda"):
    from flyaim.pipeline import FlySystem

    rc = RetinaConfig()
    bc = BrainConfig(weight_scale_exc=CAL_W_SCALE, weight_scale_inh=CAL_W_SCALE,
                     input_gain=CAL_INPUT_GAIN)
    ro = ReadoutConfig(mode="trained", weights_path=str(READOUT_OFFLINE))
    return FlySystem(D, rc, bc, ro, weights_override=NORM / f"connectome_{CAL_NORM}.npz",
                     device=device)


class StallRestart:
    """画面停滞看门狗:任务打完/停在结算页时自动重开(2026-10-05)。

    **为什么必须有这一层**:实测 rep-3 那局 1299 帧里 **1241 帧(96%)画面完全
    静止** —— Aim Lab 的 3 靶任务几十秒就打完了,之后脚本还在空跑:相机不动
    → det 恒定 → err 卡在某个值 → 永远进不了开火门限。战报里看起来就是
    「未进门限 1296 拍,开火 2 次」,和「瞄了不开火」的表象一模一样,
    但真正原因只是**任务已经结束了**。

    判据:连续 `stall_frames` 帧里,检测到的靶位置几乎不动(<= `stall_px` 像素),
    则认为画面冻结 → 调 `restart_fn()` 重开一局,并重置计数。

    注意:这**不会**误伤"准星稳稳压在靶上"的正常情况 —— 那种情况下
    TriggerOnTarget 会持续开火(n_fires 在涨),而这里是**开火数也不涨**。
    """

    def __init__(self, inner, restart_fn, stall_frames: int = 20,
                 stall_px: float = 3.0, fixate_kick_beats: int = 18) -> None:
        self.inner = inner
        self.restart_fn = restart_fn
        self.stall_frames = int(stall_frames)
        self.stall_px = float(stall_px)
        # D41b 固视自救:固视连续 N 拍无开火 → 踢回追踪(force_unfixate),
        # **不重开**。d41-confirm 实测:固视死锁靠重开逃不掉 —— 静止画面 +
        # 确定性 PD 每次都滑回同一停点(err 恒 17.5px × 19 次重开,整局白耗)。
        # kick 让内层退出固视 + 禁入冷却,PD 把准星重新推进开火门限;
        # stall_frames*3 拍重开只作最后兜底(防真死锁如检测坏死)。
        self.fixate_kick_beats = max(1, int(fixate_kick_beats))
        self.n_kicks = 0
        self._last_xy: tuple[float, float] | None = None
        self._still = 0
        self._fix_still = 0
        self._fires_at_last_check = 0
        self.n_restarts = 0

    @property
    def brain(self):
        return getattr(self.inner, "brain", None)

    @property
    def name(self):
        return getattr(self.inner, "name", "?") + "+stall"

    def act(self, frame: np.ndarray) -> np.ndarray:
        a = self.inner.act(frame)
        det = getattr(self.inner, "last_detection", None)
        if det is None or not det.ok:
            self._last_xy = None
            self._still = 0
            self._fix_still = 0
            return a
        # D41 固视豁免:固视态动作 0 → 画面必然静止,这是**预期行为**不是
        # 任务结束。首局实测:迟滞带卡门外 + 固视静止,被本看门狗误判重开
        # 19 次,1053 拍全耗在重启循环(拍频 74→17.5Hz)。固视拍单独计数,
        # 3 倍宽限内(且无新开火)不触发重开 —— 兜底防真死锁(如检测坏死)。
        if bool(getattr(self.inner, "_fixating", False)):
            fires_now = int(getattr(self.inner, "n_fires", 0))
            if fires_now > self._fires_at_last_check:
                self._fires_at_last_check = fires_now
                self._fix_still = 0
            else:
                self._fix_still = getattr(self, "_fix_still", 0) + 1
            self._still = 0
            self._last_xy = (float(det.cx), float(det.cy))
            # D41b 轻自救:固视 N 拍无开火 → 踢回追踪(不重开)。1 秒内 kick 过
            # 仍死锁 → 升级重开(防 kick 无限循环;kick 用时间窗而非计数器,
            # 避免跨正常期残留累积导致误重开)。
            if self._fix_still >= self.fixate_kick_beats:
                import time as _t
                now = _t.perf_counter()
                kick = getattr(self.inner, "force_unfixate", None)
                self._fix_still = 0
                if callable(kick) and (now - getattr(self, "_last_kick_t", -1e9)) >= 1.0:
                    kick()
                    self.n_kicks += 1
                    self._last_kick_t = now
                    print(f"  [stall] 固视 {self.fixate_kick_beats} 拍无开火 → 踢回追踪(自救,不重开)")
                    return a
            if self._fix_still >= self.stall_frames * 3:
                print(f"  [stall] 固视连续 {self._fix_still} 拍无开火 → 疑似死锁,重开")
                try:
                    self.restart_fn()
                except Exception as exc:
                    print(f"  [stall] 重开失败: {type(exc).__name__}: {exc}")
                self.n_restarts += 1
                self._fix_still = 0
                self._last_xy = None
                self._fires_at_last_check = 0
            return a
        self._fix_still = 0
        xy = (float(det.cx), float(det.cy))
        if self._last_xy is not None:
            moved = ((xy[0] - self._last_xy[0]) ** 2 + (xy[1] - self._last_xy[1]) ** 2) ** 0.5
            self._still = self._still + 1 if moved <= self.stall_px else 0
        self._last_xy = xy
        # 停滞判据:画面不动 **且** 这段时间没有新开火(排除"稳稳压住靶"的情况)
        fires_now = int(getattr(self.inner, "n_fires", 0))
        if self._still >= self.stall_frames:
            if fires_now > self._fires_at_last_check:
                self._fires_at_last_check = fires_now
                self._still = 0
            else:
                print(f"  [stall] 画面连续 {self._still} 帧不动且无新开火 → 判定任务结束,重开")
                try:
                    self.restart_fn()
                except Exception as exc:
                    print(f"  [stall] 重开失败: {type(exc).__name__}: {exc}")
                self.n_restarts += 1
                self._still = 0
                self._last_xy = None
                self._fires_at_last_check = 0
        return a

    def reset(self) -> None:
        self.inner.reset()
        self._last_xy = None
        self._still = 0

    def on_window_switch(self) -> None:
        """捕获窗切换:清停滞判据的历史坐标,并转发给内层。

        `_last_xy` 是窗内局部坐标;切窗后同一物理靶坐标会整体平移,会被
        误判成"画面动了"从而把 `_still` 清零 —— 这方向是**安全**的(不会
        误触重开),但仍应清掉以保持语义干净。
        """
        self._last_xy = None
        self._still = 0
        hook = getattr(self.inner, "on_window_switch", None)
        if callable(hook):
            hook()

    def close(self) -> None:
        self.inner.close()

    def __getattr__(self, item):
        # 透传 n_fires / n_in_gate / n_target_switches 等统计量到内层
        return getattr(self.inner, item)


class BeatRecorder:
    """进程内逐拍录像 + 开火诊断日志(D39,2026-10-05)。

    **为什么不用独立进程录屏**:独立进程 mss 抓帧与 BridgeLoop 的桌面复制
    捕获互相冲突,实测 74 帧画面完全冻结(球/HUD 纹丝不动 8 秒)—— 抓到的
    是同一缓存帧,毫无观察价值。修法:录像挂进主循环回调 —— 每拍本来就
    有 frame,缩帧 [::k,::k] 后追加原始字节流(零编码开销,~8MB/s),
    局末用 imageio_ffmpeg 自带的 ffmpeg 一次性转 MP4。

    同时记逐拍 CSV(beat/t/err/r/fired),**开火拍由 n_fires 差分判定**,
    可精确回答「开火瞬间准星离靶心多远」—— 这是"开了火不一定准"的定量
    证据:若开火时 err/r 比值常在 0.9+(贴球边缘),球在弹道时间内移出
    即打空;若比值健康(<0.5)还打空,问题就不在开火时机而在别处。
    """

    def __init__(self, run_dir, width: int = 480, every: int = 2,
                 fire_frac: float = 1.0, min_radius_px: float = 0.0,
                 fires_fn=None, scratch_dir=None,
                 score_hud=None, score_every: int = 10,
                 total_seconds: float = 60.0, learn_time: bool = False,
                 calibrator=None) -> None:
        import csv as _csv
        self.width = max(0, int(width))
        self.every = max(1, int(every))
        self.fire_frac = float(fire_frac)
        self.min_radius_px = float(min_radius_px)
        self._fires_fn = fires_fn
        # D42 分数闭环(2026-10-06):每 score_every 拍抓 HUD 读真实分数/准确率。
        # total_seconds 用于 TIME 自校准的「已知倒计时」推算(秒中段才学)。
        self._hud = score_hud
        self.score_every = max(1, int(score_every))
        self._total_s = float(total_seconds)
        # D42.5 修正:在线学 TIME 字形有两条路 ——
        #   ① **推荐**:`calibrator`(TimeCalibrator)—— 锚点来自**观测到的
        #      ∞→倒计时切换**,不用宿主时钟,且带跨帧校验/回滚;
        #   ② 遗留:`learn_time=True` —— 用宿主时钟 `total - elapsed` 当标签,
        #      与游戏时钟天然差几秒,**会把字形错标污染模板库**。仅为复现旧行为保留。
        self._calibrator = calibrator
        self.learn_time = bool(learn_time)
        self.cal_diag: dict = {}
        self.score_first: int | None = None   # 第一个有效读数(基线)
        self.score_last: int | None = None
        self.acc_last: int | None = None
        self.n_score_reads = 0
        self.n_score_ok = 0                   # 至少一个数值产出的读分次数
        self.n_score_fail = 0                 # 抓帧/解析异常次数
        self.glyph_hist: dict[int, int] = {}  # TIME 分段数 -> 次数(不依赖模板)
        self._prev_glyphs: int | None = None
        self._last_crop = None                # 最近一次 HUD 裁剪(局末兜底落盘)
        self.n_hud_dumps = 0
        self.max_hud_dumps = 12               # 落图上限(每张 ~10KB)
        self.run_dir = Path(run_dir)
        # scratch_dir:D 盘满时把 raw 中间流写到别的盘(如 C:/Temp);
        # 转码成小体积 MP4 后自动移回 run_dir 归档。
        self._scratch = Path(scratch_dir) if scratch_dir else self.run_dir
        if scratch_dir:
            self._scratch.mkdir(parents=True, exist_ok=True)
        self.raw_path = self._scratch / "frames.raw"   # 兼容旧字段(未用)
        self.csv_path = self.run_dir / "beats.csv"
        self._seg_idx = 0
        self._seg_sizes: dict[int, tuple[int, int]] = {}  # 段号 -> (w,h)
        self._fh = self._open_segment() if self.width > 0 else None
        self._csv_fh = open(self.csv_path, "w", newline="", encoding="utf-8")
        self._cw = _csv.writer(self._csv_fh)
        self._cw.writerow(["beat", "t_s", "err_px", "r_px", "limit_px", "fired",
                           "n_fires", "score", "acc_pct", "time_s", "n_glyphs"])
        self.n_video_frames = 0
        self._h = self._w = 0
        self._prev_fires = 0
        self.fire_errs: list[float] = []   # 每次开火瞬间的 err(px)
        self.fire_ratio: list[float] = []  # err / r(0~fire_frac,越低越居中)
        self._t0 = time.perf_counter()
        self._t_last_beat = self._t0

    def _open_segment(self):
        fh = open(self._scratch / f"frames_{self._seg_idx}.raw", "wb")
        self._h = self._w = 0
        return fh

    def _roll_segment(self) -> None:
        """dual 双窗切窗时帧尺寸会变(全屏 960x540 / 中心 900x900),必须分段!

        首版把两种尺寸的帧混写一个 raw 流,MP4 按第一帧尺寸切分,行全部
        错位 —— 画面呈「行级撕裂」假象,差点误判成捕获撕裂(2026-10-05)。
        检测不受影响是因为 dual source 的每帧各自尺寸正确,只有**混写**才坏。
        """
        if self._fh is not None:
            self._fh.close()
        self._seg_idx += 1
        self._fh = self._open_segment()

    def _dump_hud(self, img, tag: str) -> None:
        """把一次 HUD 裁剪落成 PNG(3x 最近邻放大,像素无损)。

        D42.7 自证基建:两个假说——H1「HUD 内容变了(任务真的开局,OCR 读不懂
        新内容)」与 H2「抓帧坏了(GDI 取不到 D3D 表面)」——**在日志里长得一模
        一样**(都是"读不出数"),只有看图能分开:

            图是清晰 HUD + 读数仍全拒  ⇒ H1,问题在模板/时钟锚点;
            图是花屏 / 桌面 / 黑块     ⇒ H2,问题在抓帧路径。

        3x 放大是为了一眼看清:裁剪只有约 30px 高,数字约 22px。
        """
        if self.n_hud_dumps >= self.max_hud_dumps:
            return
        try:
            from PIL import Image
            a = np.clip(np.asarray(img), 0, 255).astype(np.uint8)
            im = Image.fromarray(a).resize((a.shape[1] * 3, a.shape[0] * 3),
                                           Image.NEAREST)
            p = self.run_dir / f"hud_{self.n_hud_dumps:02d}_{tag}.png"
            im.save(p)
            self.n_hud_dumps += 1
            print(f"  [hud] 落图 {p.name}  ({a.shape[1]}x{a.shape[0]})")
        except Exception as exc:
            print(f"  [hud] 落图失败(忽略): {type(exc).__name__}: {exc}")

    def __call__(self, beat: int, frame, det) -> None:
        h, w = frame.shape[:2]
        cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
        err = r = limit = -1.0
        if det is not None and getattr(det, "ok", False):
            err = float(np.hypot(det.cx - cx, det.cy - cy))
            r = float(det.radius_px)
            limit = max(r * self.fire_frac, self.min_radius_px)
        # 开火判定:n_fires 差分(act 内已判定,on_beat 在其后)
        fired = 0
        if self._fires_fn is not None:
            nf = int(self._fires_fn())
            fired = max(0, nf - self._prev_fires)
            self._prev_fires = nf
            if fired > 0 and err >= 0.0:
                self.fire_errs.append(err)
                self.fire_ratio.append(err / r if r > 0 else 9.9)
        # ---- D42 分数读取(每 score_every 拍;HUD 小区域抓帧,均摊 <1ms/拍) ----
        # 取帧源由调用方注入(ScoreHUD.grab_fn):D42.4 起默认走 HudStrip ——
        # 与视觉管线同一 DXGI 层(dxcam/bettercam),不再走 GDI,且不依赖
        # 主循环当前用的是全屏窗还是中心窗(双窗下全屏帧仅占 0.3%)。
        score_v = acc_v = time_v = glyph_v = ""
        if self._hud is not None and beat % self.score_every == 0:
            try:
                img = self._hud._grab_fn()
                self._last_crop = img
                # TIME 在线校准:优先走**锚定+跨帧校验**的校准器(D42.5 修正)
                if self._calibrator is not None:
                    self.cal_diag = self._calibrator.observe(
                        img, time.perf_counter())
                elif self.learn_time:
                    # 遗留路径:宿主时钟当标签 —— 已知会错标,仅为复现旧行为
                    elapsed = time.perf_counter() - self._t0
                    remain = self._total_s - elapsed
                    frac = remain % 1.0
                    if 0.25 <= frac <= 0.75 and remain > 0:
                        self._hud.calibrate_from_time(img, int(remain))
                rd = self._hud.read(img)
                self.n_score_reads += 1
                if rd["points"] is not None:
                    score_v = str(rd["points"])
                    if self.score_first is None:
                        self.score_first = rd["points"]
                    self.score_last = rd["points"]
                if rd["acc_pct"] is not None:
                    acc_v = str(rd["acc_pct"])
                    self.acc_last = rd["acc_pct"]
                ng = rd.get("time_glyphs")
                if ng is not None:
                    glyph_v = str(ng)
                    self.glyph_hist[ng] = self.glyph_hist.get(ng, 0) + 1
                if rd["time_s"] is not None:
                    time_v = str(rd["time_s"])
                if score_v or acc_v:
                    self.n_score_ok += 1
                # 自证(D42.7):分段数一变化就落图 —— 那正是「∞ → 倒计时」的
                # 转变瞬间,是分辨「抓帧坏」与「内容变了」的关键帧。
                if ng is not None and (self.n_score_reads == 1
                                       or ng != self._prev_glyphs):
                    self._dump_hud(img, f"b{beat}_g{ng}")
                self._prev_glyphs = ng
            except Exception as exc:  # 读分失败不拖垮主循环
                self.n_score_fail += 1
                if self.n_score_fail == 1:
                    print(f"  [score] 抓帧失败(忽略): {type(exc).__name__}: {exc}")
        self._cw.writerow([beat, f"{time.perf_counter()-self._t0:.3f}",
                           f"{err:.2f}", f"{r:.2f}", f"{limit:.2f}", fired,
                           self._prev_fires, score_v, acc_v, time_v, glyph_v])
        # 录像:每 every 拍一帧;尺寸变化(切窗)即滚动新段
        if self._fh is not None and beat % self.every == 0:
            k = max(1, round(w / self.width))
            small = frame[::k, ::k]
            if self._h == 0:
                self._h, self._w = small.shape[:2]
                self._seg_sizes[self._seg_idx] = (self._w, self._h)
                print(f"  [rec] 段{self._seg_idx}: {self._w}x{self._h} 每 {self.every} 拍一帧")
            elif small.shape[:2] != (self._h, self._w):
                self._roll_segment()
                self._h, self._w = small.shape[:2]
                self._seg_sizes[self._seg_idx] = (self._w, self._h)
                print(f"  [rec] 切窗换尺寸 → 段{self._seg_idx}: {self._w}x{self._h}")
            self._fh.write(small.tobytes())
            self.n_video_frames += 1

    def close(self, tick_hz: float = 0.0):
        """落盘并把每个分段转 MP4;返回最长段的 mp4 路径(失败返回 None)。"""
        self._csv_fh.close()
        # 局末兜底落一张:HUD 若全程只在某一态,上面的"变化即落图"不会触发
        if self._last_crop is not None:
            self._dump_hud(self._last_crop, "last")
        if self._fh is None:
            return None
        self._fh.close()
        try:
            import imageio_ffmpeg
            import subprocess
            fps = (tick_hz / self.every) if tick_hz > 0 else 30.0
            fps = min(max(fps, 5.0), 60.0)
            exe = imageio_ffmpeg.get_ffmpeg_exe()
            best_mp4, best_n = None, -1
            for i in range(self._seg_idx + 1):
                raw = self._scratch / f"frames_{i}.raw"
                if not raw.exists():
                    continue
                nbytes = raw.stat().st_size
                wh = self._seg_sizes.get(i)
                if wh is None or nbytes == 0:
                    raw.unlink(missing_ok=True)
                    continue
                w_, h_ = wh
                n = nbytes / (w_ * h_ * 3)
                mp4 = raw.with_suffix(".mp4")
                subprocess.run(
                    [exe, "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
                     "-s", f"{w_}x{h_}", "-r", f"{fps:.2f}",
                     "-i", str(raw), "-pix_fmt", "yuv420p",
                     "-crf", "23", str(mp4)],
                    check=True, capture_output=True)
                raw.unlink(missing_ok=True)
                final = self.run_dir / mp4.name  # MP4 移回局目录归档
                if mp4.resolve() != final.resolve():
                    mp4.replace(final)
                    mp4 = final
                if n > best_n:
                    best_mp4, best_n = mp4, n
            self._best_mp4 = best_mp4
            return best_mp4
        except Exception as e:
            print(f"  [warn] MP4 转换失败({e});原始帧保留在 {self.run_dir}")
            return None

    def report_lines(self) -> list[str]:
        """开火质量诊断 + 真实分数(D42)(战报用)。"""
        lines: list[str] = []
        if self._hud is not None:
            if self.score_first is not None and self.score_last is not None:
                delta = self.score_last - self.score_first
                lines.append(
                    f"  真实分数(HUD): {self.score_first} → {self.score_last}"
                    f"  Δ{delta:+d}  ACC {self.acc_last if self.acc_last is not None else '?'}%")
            else:
                lines.append("  真实分数(HUD): 未读到有效分数")
            # D42.7 自证:拒读率 + TIME 分段直方图(后者不依赖任何模板,永远有效)
            n = max(self.n_score_reads, 1)
            rej = 100.0 * (1.0 - self.n_score_ok / n) if self.n_score_reads else 0.0
            hist = " ".join(f"{k}段×{v}" for k, v in sorted(self.glyph_hist.items()))
            lines.append(
                f"  读分自证: 读 {self.n_score_reads} 次 / 产出数值 {self.n_score_ok} 次"
                f" / 拒读 {rej:.0f}% / 抓帧失败 {self.n_score_fail} 次"
                f" / TIME 分段 {hist or '无'}"
                f" / 模板 {self._hud.last_raw.get('n_templates', '?')} 个")
            if self._calibrator is not None:
                c = self._calibrator
                anc = f"t={c.anchor_t:.1f}s" if c.anchor_t else "未建立(本局未见 ∞)"
                lines.append(
                    f"  TIME 校准: 学习 {c.n_learn} 次 / 锚点 {anc}"
                    f" / 跨帧校验 ok {c.n_verify_ok} bad {c.n_verify_bad}"
                    f" / 缺字符 {''.join(c.missing()) or '无'}"
                    + (f" / ⛔ 已停用({c.note})" if c.disabled else ""))
            # 🔴 高分两种病要分开:**不是计分局** ≠ **OCR 读不出**。混在一起会让
            # 下一轮又在错误的地方调参(这正是 D42.4 踩过的坑)。
            n_inf = self.glyph_hist.get(1, 0)
            if self.n_score_reads >= 20 and rej > 90.0:
                if n_inf > self.n_score_reads * 0.9:
                    lines.append(
                        f"  🔴 拒读率 {rej:.0f}% 且 TIME 全程 {n_inf} 段(=∞)"
                        f" —— **本局根本不是计分局**(练习/自由模式),"
                        f"分数与 130,335 目标不可比。开局门闸本应拦住,"
                        f"若见到此行说明门闸被绕过或超时被放行")
                else:
                    lines.append(
                        f"  🔴 拒读率 {rej:.0f}% —— 看局目录 hud_*.png 分辨:"
                        f"图是清晰 HUD ⇒ 模板/时钟锚点问题(H1);"
                        f"图是花屏/桌面 ⇒ 抓帧问题(H2)")
        if not self.fire_errs:
            return lines
        arr = np.asarray(self.fire_errs)
        rat = np.asarray(self.fire_ratio)
        edge = float((rat > 0.8).mean() * 100.0)
        lines += [
            f"  开火质量: 开火时准星-靶心距 p50 {np.percentile(arr,50):.1f}px"
            f" / p90 {np.percentile(arr,90):.1f}px / max {arr.max():.1f}px",
            f"  开火位置: err/r 比值 p50 {np.percentile(rat,50):.2f}"
            f" / p90 {np.percentile(rat,90):.2f}"
            f" / 贴边开火(>0.8)占 {edge:.0f}%(越高越易打空)",
        ]
        return lines


def _make_restart_fn(full_region):
    """返回一个"重开一局"的可调用:按 R(再练一次)并短暂等待。

    必须传**全屏** region —— 见 autostart_task 的说明。
    """
    def _restart() -> None:
        for _ in range(3):
            if autostart_task(full_region, tries=1):
                break
            time.sleep(0.4)
    return _restart


def main() -> int:
    args = parse_args()
    print("=" * 84)
    print(f"Aim Lab 任务会话: mode={args.mode} seconds={args.seconds}")
    print("  ⚠️ 将真实控制鼠标。请保持 Aim Lab 前台、手离鼠标键。")
    print("=" * 84, flush=True)

    from flyaim.bridge.capture import primary_monitor_size, centered_region
    from flyaim.io import new_run_dir, save_json

    region = find_window_region("aimlab")
    if not focus_window("aimlab"):
        print("❌ 无法聚焦 Aim Lab 窗口")
        return 1
    print(f"  窗口已聚焦 region={region}")

    # ---- 捕获区:dual(默认) / center(旧) / full(旧) ----------------------
    # 双窗是「命中率」的正解:center 单窗只在**相机已对准靶**时才有意义,
    # 而搜靶阶段靶可能在全屏任意位置(实测 (144,48),中心窗 900 是 x[510,1410]
    # y[90,990] → 完全在窗外,det-ok 全程 0、60 秒开火 0 次)。dual 让:
    #   搜靶 → 全屏(看得见任意位置的靶)   跟踪 → 中心窗(快 7 倍 + 原始精度)
    cap_size = int(args.capture_size)
    cap_region, cap_out = None, None
    dual_source = None
    if args.capture == "dual":
        from flyaim.bridge.capture import DualWindowSource

        cap_region = region  # 兼容旧路径(握手/重启用全屏)
        cap_out = None
        dual_source = DualWindowSource(
            full_region=region,
            center_size=cap_size,
            lock_switch_px=args.track_lock_px,
            full_downsample_size=(None if args.full_down <= 0 else args.full_down),
        )
        print(f"  捕获: 双窗(搜靶=全屏{region[2]}x{region[3]}→{args.full_down or '原尺寸'},"
              f"跟踪=中心窗{cap_size}x{cap_size}@{dual_source._center_region[:2]}),"
              f"锁定阈值 {args.track_lock_px:.0f}px")
    elif args.capture == "center":
        cl, ct, cw, ch = region
        cx, cy = cl + cw // 2, ct + ch // 2
        cap_region = (cx - cap_size // 2, cy - cap_size // 2, cap_size, cap_size)
        cap_out = None  # 不缩放
        print(f"  捕获: 中心窗 {cap_size}x{cap_size} @({cap_region[0]},{cap_region[1]})"
              f"  不缩放(≈±{cap_size/2/12.57:.1f}° 视野)")
    else:
        cap_region, cap_out = region, (640, 480)
        print(f"  捕获: 全屏 {region[2]}x{region[3]} -> 640x480")

    # ---- HUD 取帧(D42.7):**提前建**,因为下面「有没有真的开局」要靠它判 ----
    from flyaim.bridge.score_ocr import HudStrip, ScoreHUD, TimeCalibrator, \
        time_glyph_count

    hud_strip = None
    if args.score_every > 0:
        try:
            hud_strip = HudStrip(region, backend=args.hud_backend)
            print(f"  HUD 取帧: {hud_strip.backend}  bbox={hud_strip.bbox} "
                  f"(绝对屏幕坐标 left,top,w,h)")
        except Exception as exc:
            print(f"  ⚠️ HUD 专用取帧不可用({type(exc).__name__}: {exc});"
                  f"退回默认 PIL/GDI 路径 —— 读数可能不可靠")

    def _hud_running() -> bool:
        """计分局进行中?判据 = TIME 框 5 段(不依赖 OCR 模板,永远可用)。"""
        if hud_strip is None:
            return False
        try:
            return time_glyph_count(hud_strip.read()) == 5
        except Exception:
            return False

    # ---- 开局门闸(2026-10-06,D42 实盘教训)--------------------------------
    # 上一版把「全屏检出青靶」当"已在任务内"(而大厅/UI 也有青靶,实测在
    # (1664,43) 检出 R=11.3px 的块就早退),于是整局 60s 跑在 TIME=∞ 的练习
    # 模式里却毫无告警 —— 分数/ACC 都在动,看着像正常,实际与目标不可比。
    # 现在**只有 TIME 显示 MM:SS 才算开局**;开不了就等人工,等不到就中止。
    #
    # ⚠️ 门闸**故意放在最后**(紧挨 loop.run()):建捕获源、开局握手都会真实
    # 动鼠标,那些都该在 ∞ 练习态里做完,别吃掉计分局的时间。
    if args.no_round_gate and not args.no_autostart:
        if not autostart_task(region):
            print("⚠️ autostart 未确认进入任务。仍继续(已显式关掉开局门闸)。")
        time.sleep(1.2)  # 等任务开场动画
    # dual 模式下 cap 就是 dual_source(它内部管两个窗);否则退化为单窗
    cap = dual_source if dual_source is not None else ScreenCapture(
        region=cap_region, out_size=cap_out)
    # 开局握手:注入模式必须实测「相机响应注入」。失灵(锁定丢失/焦点被夺)
    # 时注入会 100% 静默丢失 —— 与其跑一局废数据,不如当场失败。
    if not args.no_trigger or args.mode != "collect":
        from flyaim.bridge.inject import SendInputSink as _S

        probe = _S()
        ok = False
        for i in range(3):
            f0, _ = cap.read()
            probe.send(300, 0)
            time.sleep(0.15)
            f1, _ = cap.read()
            d = float(np.abs(f1.astype(np.int16) - f0.astype(np.int16)).mean())
            if d > 1.5:
                ok = True
                print(f"  握手: 300 计数 -> 画面差 {d:.1f}/255,相机响应正常")
                break
            print(f"  握手[{i}]: 画面差 {d:.1f} —— 相机未响应,点击中央重建锁定...")
            click_at(region[0] + region[2] // 2, region[1] + int(region[3] * 0.28))
            time.sleep(2.0)
        probe.close()
        if not ok:
            print("❌ 三次握手失败:相机对注入无响应。请人工点进游戏窗口后再跑。")
            return 1

    sink = SendInputSink(max_counts_per_tick=int(args.max_counts))
    gain = GainModel.load(args.gain_json)

    if args.mode == "collect":
        out_npz = args.npz_out or str(ROOT / "flyaim/runs/bridge/collect_1.npz")
        Path(out_npz).parent.mkdir(parents=True, exist_ok=True)
        core = TeacherCollectController(build_system(), out_npz=out_npz)
    elif args.mode == "seek":
        from flyaim.bridge.controllers import SeekController

        core = SeekController(ref_color=TEAL, tolerance=args.det_tolerance,
                              use_aim_detect=False, kp=args.kp,
                              downsample=args.downsample,
                              xhair_px=args.xhair_px, xhair_max_r=args.xhair_max_r,
                              scan_amp=args.scan_amp,
                              scan_period_s=args.scan_period_s,
                              err_scale_px=(args.err_scale_px or None),
                              assoc_px=args.assoc_px,
                              switch_gain=args.switch_gain,
                              lost_tol=args.lost_tol,
                              fixate_px=args.fixate_px,
                              saccade_px=args.saccade_px,
                              fire_gate_frac=args.fire_frac)
    elif args.mode == "fly":
        core = _build_fly(args)
    else:  # hybrid
        from flyaim.bridge.controllers import HybridController

        core = HybridController(_build_fly(args), beta=args.beta, kp=args.kp)
        print(f"  hybrid: β={args.beta}(seek + β·fly 加性扰动)")

    stack = core
    # 动作整形**必须在 TriggerOnTarget 之内**(先整形后判定开火):
    #   * 死区/软饱和作用在"真正发出去的动作"上,开火层看到的就是最终瞄准;
    #   * 若放在外层,开火层会基于未整形的高频动作判 err,反而更容易误开火。
    if args.smooth_delta > 0 or args.smooth_soft > 0 or args.deadband_px > 0:
        from flyaim.bridge.controllers import ActionSmoother

        stack = ActionSmoother(stack, max_delta=args.smooth_delta,
                               soft=args.smooth_soft, deadband_px=args.deadband_px)
    if args.action_ema > 0:
        stack = ActionEMA(stack, args.action_ema)
    if not args.no_trigger:
        stack = TriggerOnTarget(stack, sink.click, ref_color=TEAL,
                                tolerance=args.det_tolerance,
                                fire_frac=args.fire_frac, min_radius_px=args.min_radius_px,
                                cooldown_s=args.cooldown, sticky_px=args.sticky_px,
                                downsample=args.downsample,
                                xhair_px=args.xhair_px, xhair_max_r=args.xhair_max_r,
                                jump_reset_px=args.jump_reset_px,
                                fire_confirm_beats=args.fire_confirm)
    if args.stall_frames > 0:
        stack = StallRestart(stack, _make_restart_fn(region),
                             stall_frames=args.stall_frames, stall_px=args.stall_px)
    print(f"  控制栈: {core.name}"
          f"{' +smooth' if (args.smooth_delta > 0 or args.smooth_soft > 0 or args.deadband_px > 0) else ''}"
          f"{' +EMA' if args.action_ema > 0 else ''}{' +trigger' if not args.no_trigger else ''}"
          f"{' +stall' if args.stall_frames > 0 else ''}")
    # dual source 需要读 stack 的 last_detection 来判断"锁上了没",故在 stack
    # 装配完成后回灌。注意 stack 顶层是 StallRestart/TriggerOnTarget,两者的
    # last_detection 都会一路透传到 SeekController,口径一致。
    if dual_source is not None:
        dual_source.set_controller(stack)

    runs_root = Path(args.runs_dir) if args.runs_dir else ROOT / "flyaim" / "runs"
    if args.runs_dir:
        runs_root.mkdir(parents=True, exist_ok=True)
    run_dir = new_run_dir(runs_root, tag=args.tag or f"play-{args.mode}")
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    writer = None
    if not args.no_telemetry:
        try:
            from telemetry import TelemetryWriter, make_sample_plan  # type: ignore

            import pandas as pd

            from flyaim.io import load_roles

            idx = pd.read_parquet(D / "neuron_index.parquet")
            roles = load_roles(D / "roles.json")
            plan = make_sample_plan(int(idx.shape[0]), roles, neuron_index=idx)
            writer = TelemetryWriter(run_dir / "live", sample_shape=plan["sample_shape"],
                                     sample_idx=plan["sample_idx"], dn_idx=plan["dn_idx"],
                                     layer_labels=plan["layer_labels"], preview_every=20,
                                     run_tag=f"play-{args.mode}")
        except Exception as e:
            print(f"  [warn] 遥测不可用({e})")
            writer = None

    # D39 逐拍录像/诊断:开火拍用 n_fires 差分,stack 顶层是
    # StallRestart/TriggerOnTarget,n_fires 会一路透传上来,直接读即可。
    # D42:ScoreHUD 读游戏真实分数(Gridshot 计分=命中间隔加权+miss 扣分,
    # 「推断命中」只是几何判定,分数才是地面真值)。
    # **取帧走 HudStrip = 与视觉管线同一 DXGI 层**(D42.7):主捕获看得见游戏帧,
    # 它就看得见 HUD;且 HUD 框固定在屏幕顶部,与 dual 当前用哪个窗无关 ——
    # 实测双窗下全屏帧仅占 0.3%,"从已有帧里裁 HUD"根本裁不到。
    # hud_strip 已在开局门闸之前建好并复用(避免开第二台相机互相抢帧)。
    hud = None
    calibrator = None
    if args.score_every > 0:
        hud = ScoreHUD(grab_fn=(hud_strip.read if hud_strip is not None else None))
        if not args.no_time_calib:
            calibrator = TimeCalibrator(hud, round_len_s=args.round_seconds)
    rec = None
    if args.record > 0 or args.beat_log or hud is not None:
        rec = BeatRecorder(run_dir, width=args.record, every=args.record_every,
                           fire_frac=args.fire_frac, min_radius_px=args.min_radius_px,
                           fires_fn=lambda: int(getattr(stack, "n_fires", 0)),
                           scratch_dir=args.scratch_dir,
                           score_hud=hud, score_every=args.score_every,
                           total_seconds=args.seconds,
                           learn_time=args.hud_learn_time,
                           calibrator=calibrator)

    # ---- 最后一刻才等计分局(见上方门闸说明)--------------------------------
    if not args.no_round_gate:
        if not wait_for_round(hud_strip, region, timeout_s=args.round_wait,
                              auto_start=not args.no_autostart,
                              allow_practice=args.allow_practice):
            if hud_strip is not None:
                hud_strip.close()
            return 1
        time.sleep(0.3)          # 让刚起跳的倒计时稳定一拍,再开始测量
        if calibrator is not None:
            print(f"  TIME 校准: 锚定式已启用(局时长 "
                  f"{args.round_seconds:.0f}s,未见过 ∞ 则拒绝学习)")
    loop = BridgeLoop(source=cap, sink=sink, controller=stack, gain=gain,
                      telemetry=writer, max_seconds=args.seconds,
                      focus_watchdog_title="aimlab", on_beat=rec)
    summary = loop.run()
    summary["mode"] = args.mode
    summary["n_fires"] = int(getattr(stack, "n_fires", 0))
    summary["n_hits_inferred"] = int(getattr(stack, "n_hits_inferred", 0))
    summary["n_in_gate"] = int(getattr(stack, "n_in_gate", 0))
    summary["n_blocked_by_cooldown"] = int(getattr(stack, "n_blocked_by_cooldown", 0))
    summary["n_geometric_miss"] = int(getattr(stack, "n_geometric_miss", 0))
    summary["n_target_switches"] = int(getattr(stack, "n_target_switches", 0))
    summary["n_restarts"] = int(getattr(stack, "n_restarts", 0))
    summary["n_limited"] = int(getattr(stack, "n_limited", 0))
    summary["n_deadband"] = int(getattr(stack, "n_deadband", 0))
    summary["n_scans"] = int(getattr(stack, "n_scans", 0))
    summary["n_aim_switches"] = int(getattr(stack, "n_switch", 0))
    summary["n_fixate_beats"] = int(getattr(stack, "n_fixate", 0))
    summary["n_saccades"] = int(getattr(stack, "n_saccades", 0))
    if dual_source is not None:
        summary["n_window_switches"] = int(dual_source.n_switches)
        summary["n_full_reads"] = int(dual_source.n_full_reads)
        summary["n_center_reads"] = int(dual_source.n_center_reads)
    save_json(str(run_dir / "bridge_summary.json"), summary)

    print("\n---- 本局战报 ----")
    print(f"  模式 {args.mode}  帧数 {summary['frames']}  拍频 {summary['tick_hz']} Hz")
    print(f"  开火 {summary['n_fires']} 次  推断命中 {summary['n_hits_inferred']}")
    if dual_source is not None:
        # 注意:分母用**捕获线程实际读帧数**(full+center),不是主拍数 frames ——
        # 捕获线程通常比决策拍跑得快(实测 6090 读 vs 2893 拍),用 frames 当
        # 分母会算出 >100% 的荒谬占比(2026-10-05 实测 211%)。
        n_read = summary["n_full_reads"] + summary["n_center_reads"]
        _tot = max(n_read, 1)
        print(f"  双窗: 切窗 {summary['n_window_switches']} 次 | "
              f"搜靶(全屏) {summary['n_full_reads']} 读"
              f"({summary['n_full_reads']/_tot*100:.0f}%),"
              f"跟踪(中心窗) {summary['n_center_reads']} 读"
              f"({summary['n_center_reads']/_tot*100:.0f}%)"
              f"  [共 {n_read} 读 / {summary['frames']} 拍]")
    if summary["n_scans"]:
        print(f"  扫视(未检出靶时): {summary['n_scans']} 段")
    _asw = summary.get("n_aim_switches", 0)
    _fr = max(summary["frames"], 1)
    if _asw or args.assoc_px >= 0:
        print(f"  瞄准锁定: 目标身份切换 {_asw} 次"
              f"({_asw/_fr*100:.1f}%/拍,越低越稳;关联半径 "
              f"{args.assoc_px if args.assoc_px > 0 else 'auto'}"
              f"/迟滞 {args.switch_gain}/容忍 {args.lost_tol} 拍)")
    if summary["n_in_gate"] or summary["n_geometric_miss"]:
        print(f"  开火层诊断: 进门限 {summary['n_in_gate']} 拍"
              f"(冷却挡下 {summary['n_blocked_by_cooldown']} 拍),"
              f"未进门限 {summary['n_geometric_miss']} 拍,"
              f"换靶 {summary.get('n_target_switches', 0)} 次,"
              f"自动重开 {summary.get('n_restarts', 0)} 次")
    _nfix = int(getattr(stack, "n_fixate", 0))
    _nsac = int(getattr(stack, "n_saccades", 0))
    if args.fixate_px > 0:
        print(f"  扫视-固视(D41): 固视 {_nfix} 拍({_nfix/_fr*100:.0f}%"
              f",理想实录 ~72% 静止)/ 完成扫视 {_nsac} 次"
              f"({args.fixate_px:.0f}/{args.saccade_px:.0f}px 迟滞)")
    if summary["n_limited"] or summary["n_deadband"]:
        tot = max(summary["frames"], 1)
        print(f"  动作整形: 速率限制削过 {summary['n_limited']} 拍"
              f"({summary['n_limited']/tot*100:.0f}%),"
              f"死区内输出 0 占 {summary['n_deadband']} 拍"
              f"({summary['n_deadband']/tot*100:.0f}%)")
    if rec is not None:
        mp4 = rec.close(tick_hz=float(summary.get("tick_hz", 0) or 0))
        for line in rec.report_lines():
            print(line)
        if mp4 is not None:
            print(f"  录像: {mp4}({rec.n_video_frames} 帧)"
                  f"  逐拍日志: {rec.csv_path}")
        summary["record_video_frames"] = rec.n_video_frames
        if rec.fire_ratio:
            summary["fire_err_ratio_p50"] = float(np.percentile(rec.fire_ratio, 50))
            summary["fire_edge_pct"] = float((np.asarray(rec.fire_ratio) > 0.8).mean() * 100.0)
        # D42.7 读分自证落进 summary,便于跨局对比(人工只读战报容易漏)
        if hud is not None:
            summary["hud_backend"] = getattr(hud_strip, "backend", "pil(GDI)")
            summary["hud_bbox"] = list(getattr(hud_strip, "bbox", ()))
            summary["n_score_reads"] = int(rec.n_score_reads)
            summary["n_score_ok"] = int(rec.n_score_ok)
            summary["n_score_fail"] = int(rec.n_score_fail)
            summary["n_hud_dumps"] = int(rec.n_hud_dumps)
            summary["hud_glyph_hist"] = {str(k): int(v) for k, v in rec.glyph_hist.items()}
            summary["score_first"] = rec.score_first
            summary["score_last"] = rec.score_last
            summary["acc_last"] = rec.acc_last
    if hud_strip is not None:
        hud_strip.close()   # 相机是进程级单例,close 只解绑本对象的引用
    print(f"  目录 {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
