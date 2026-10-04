# FlyAim → Aim Lab 接入准备(bridge 层)

> 状态:**Phase A 完成**(管道层已建、离线验证通过)。本文档是接入真实
> Aim Lab 的总计划与验收标准。决定记录见 `DECISIONS.md` D18(推翻 D6)。

---

## 0. 先说清楚:为什么明知结论是负面的还要接

正式实验(DECISIONS.md D10/D11)已经证明:**LIF 简化模型 + 静态连接组在
2D 自建靶场上不承载可用的视觉伺服信号**(fly 显著差于随机,预注册判定
「接线图无因果贡献」)。接真实 Aim Lab **不会自动让果蝇会瞄准**。

接入的真实目的有三个,都与科学问题有关:

1. **换输入统计**:Aim Lab 是 3D 透视场景,鼠标一动**整幅画面都在流**。
   当前视网膜的时间差分通道(`temporal_diff=0.7`)在这种全域光流下的
   行为与 2D 靶场(准星动、背景完全静止)完全不同 —— 这是同一链路的
   一次真正换域检验,而不是重复实验。
2. **换输出语义**:2D 里 action 是「准星像素速度」;3D 里鼠标增量是
   「相机角速度」,两者只经 `GainModel` 的线性假设相连。这个假设本身
   就是被测对象。
3. **端到端真实性**:补上真实渲染延迟、真实帧率抖动、真实判分,
   让「该配置不承载视觉伺服信号」的结论在工业级环境里也有一个数据点。
   (公开的 DOOMFLY 跑 3,000+ 轮无改善;我们在受控靶场也得到一致结论。)

---

## 1. 架构:两端替换,中间一行不改

```
┌─────────────────────────── 离线实验(现状)───────────────────────────┐
│  Arena.render() ──► Retina ──► Connectome ──► Readout ──► Arena.step() │
└────────────────────────────────────────────────────────────────────────┘
                    │ 替换首尾                          ▲ 替换末端
┌─────────────────────────── 桥接闭环(新增)───────────────────────────┐
│  FrameSource      ──► Retina ──► Connectome ──► Readout ──► GainModel │
│  (屏幕捕获线程)                                        │              │
│                                              ActionSink(SendInput)◄──┘
└────────────────────────────────────────────────────────────────────────┘
```

| 模块 | 文件 | 职责 | 替代了什么 |
|---|---|---|---|
| FrameSource | `flyaim/bridge/capture.py` | 屏幕/靶场/数组 → RGB uint8 帧 | `Arena.render()` |
| ActionSink | `flyaim/bridge/inject.py` | 鼠标计数注入(Win32 SendInput) | `Arena.step()` 的准星位移 |
| GainModel | `flyaim/bridge/gain.py` | action[-1,1] → 鼠标计数 | `speed_px_per_action=14` |
| 控制器 | `flyaim/bridge/controllers.py` | fly / seek / random / zero | runner.Arm |
| 闭环 | `flyaim/bridge/loop.py` | 捕获线程 + 网络拍 + 注入 + 遥测 + 延迟记账 | `runner.run_episode` |
| 检测 | `flyaim/bridge/detect.py` | 纯像素找靶(**只进遥测,不回流 fly**) | observer 侧 |

**Retina / Connectome / Readout 零改动。** 视网膜本来就容忍任意分辨率帧
(`_ommatidial_mean` 的 bincount 路径),读出层权重原样装回。

工具(`tools/`):

| 工具 | 用途 |
|---|---|
| `tools/aimlab_bridge.py` | 主入口:三种模式(见下) |
| `tools/aimlab_gain.py` | 增益标定(灵敏度/cm/360 → counts/360,见 D27),写 gain.json |
| `tools/aimlab_calibrate.py` | 端到端定标向导:3 个数 → counts/360 + 指针探针 + 可选游戏内校验 + PASS/FAIL |
| `tools/aimlab_smoke.py` | 64 项冒烟检查(零依赖 GPU/游戏/注入) |

---

## 2. 本轮已验证的事实(Phase A,全部实测)

| 项 | 结果 |
|---|---|
| 冒烟测试 | **30/30 通过**(`tools/aimlab_smoke.py`) |
| seek 控制器闭环 | ArenaSource 彩排 300 帧命中 **7 次**(判定半径 22px)→ 管道收敛性成立 |
| 真实屏幕闭环(seek) | **28.4 Hz**(Phase B8);fly 接真实画面 **17.6 Hz**(bridge-fly-live) |
| 捕获后端 | bettercam(DXGI)→ mss → PIL 自动回退;BGRA 全链路,零 cv2 依赖 |
| SendInput 结构 | x64 `sizeof(INPUT)=40` 正确;指针加速度检测可用(**用户决定保留开启,见 Phase C**) |
| 读出权重装回 | `runs/readout/readout_weights.npz` → FlySystem CPU 装配 1s,has_weights=True |
| 遥测兼容 | 桥接的 JSONL 与 `tools/live_view.py` / `live_web.py` 直接兼容 |

一个被测试抓住并修正的设计缺陷(留档):seek 最初用「画面中心」当瞄准点,
在 2D 靶场里准星移动不改变靶的屏幕位置 → 误差信号与准星脱钩,PD 把准星
一路推到墙上。改为**检测帧内白色准星标记**作瞄准点(检测不到回退画面中心)
后收敛。这个修正对真实 FPS 同样重要:捕获区域若不完全居中,画面中心 ≠ 准星。

---

## 3. 关键语义转换(接入前必须理解的三件事)

### 3.1 action → 鼠标计数(增益层)

靶场语义:action=1.0 → 每拍扫过画面宽度的 14/640 = **2.19%**。
游戏语义:注入 C 计数 → 视线转 C/counts_per_360 × 360°。
桥接保留比例含义:`counts_per_action = speed_fraction × counts_per_360`。

```
counts_per_360 = DPI × cm360 / 2.54        # 800DPI × 40cm → 12598
deg_per_action(默认增益) ≈ 7.9°           # 满量程 action 转过的角度
```

两个**已知近似**(不要悄悄修掉,它们是被测对象的一部分):
- 「画面比例 ≈ 角度比例」是线性近似,透视边缘失真、FOV 与窗口不匹配时更差;
- 同样的 action 序列,**tick 越快转得越快**(deg/s = deg_per_action × tick_hz)。
  训练闭环 ~4-10 Hz;桥接 tick 频率必须连同结果一起报告。

### 3.2 瞄准点 = 检测到的准星,不是画面中心

见 §2 缺陷留档。`SeekController(use_aim_detect=True)` 检测白色准星标记;
FlyController 不消费任何检测(架构上只见像素)。

### 3.3 闭环里自带 ≥1 帧传输延迟

`Arena.step` 是「先动再渲染」,帧与动作严格因果;真实屏幕做不到 —— 注入的
移动要等游戏渲染完才出现在下一帧。`BridgeLoop` 把**画面年龄**(grab→消费)
单独记账进 `bridge_summary.json`,以后所有结果都要带这个数。

---

## 4. 分阶段计划与验收标准

### Phase A — 管道层(✅ 本轮完成)
验收:冒烟 30/30;ArenaSource 彩排 seek 命中 ≥3;CLI 三模式可跑。
**已完成,证据见 §2。**

### Phase B — 真实屏幕管道校验(不注入,需要 Aim Lab 就位)
```powershell
& $py tools/aimlab_probe.py --find aimlab          # 先探测:定位窗口+找靶色+目检标注图
& $py tools/aimlab_bridge.py --controller seek --source screen --window aimlab \
      --no-aim-detect --frames 300 --live
```
**本机实测(2026-10-04,全屏 1920×1080):**
- 窗口标题 `aimlab_tb`;`--window aimlab` 自动定位,无需手填 region;
- 捕获后端装了 mss 后自动选用:全屏 grab 30.2ms(PIL 50.7ms);
- **靶色是青色** `ref_color=(48,224,224)`(CLI 默认已配好),实测检出率 100%;
- **准星是中央红色小十字**,恰在窗口几何中心(实测 (960,534) vs 中心 (959.5,539.5))
  → 全屏捕获时用 `--no-aim-detect`(画面中心)最稳;**别用红色找准星**,
  枪模型也是红色,会锁到枪上;
- 白色检测会命中 Aimlab 的 Logo/文字,不能当准星;
- Phase B 实跑:150 帧 6.4 Hz(PIL 时代),画面年龄 p50 0.8ms,遥测检出率 100%。

Phase B 验收清单:
- [x] 窗口定位 + 捕获区域正确(顶部窗口诊断 = aimlab_tb);
- [x] 靶色检测 100%(run 20261004-010155-bridge-phaseB);
- [ ] **在任务内**(非大厅/详情页)复跑一次,确认打分 HUD 在场时检出率仍高;
- [ ] `bridge_summary.json` 里 capture_age_p95 < 200ms(已满足)。

### Phase C — 注入方向与增益标定(**会动鼠标**)
```powershell
# 0) 用户决定(2026-10-04)保留「提高指针精确度」,不关闭。后果与对策:
#    - EPP 只影响系统光标路径;浏览器 Pointer Lock 与 Unity raw input 读原始
#      增量,大概率不受影响 —— 以 Phase C 实测为准;
#    - 若受影响,增益随注入速度非线性:标定必须在意向工作幅度附近做,
#      bridge_summary.json 已自动记录 pointer_accel_enabled 状态。
# 1) 生成增益(D27:也可用 --from-sens <sens> --engine aimlab,不必量桌面)
& $py tools/aimlab_gain.py --from-sens 2 --engine aimlab --aimlab-dpi 800 --fov 106.26
# 2) 光标微移注入自检(动 3px 后自动复位)
& $py tools/aimlab_smoke.py --cursor-check
# 3) 真实注入先小幅试方向(可临时 --max-counts 150)
& $py tools/aimlab_bridge.py --controller seek --source screen --sink sendinput --gain-json flyaim/runs/bridge/gain.json --frames 120
```
- [ ] 方向:准星朝靶动的方向与检测误差一致;不对用 `--invert-x/y` 修;
- [ ] 标定 gain.json:路线 B(`--from-sens --engine --aimlab-dpi`,不必量桌面)或
  路线 A(`--from-cm360`),两条对拍后取一致值 —— 见 D27;
- [ ] seek 在真实游戏里能命中(它偷看答案,只作管道验收,**不得进判定**)。

### Phase D — fly 闭环科学实验
```powershell
& $py tools/aimlab_bridge.py --controller fly --source screen --sink sendinput --gain-json flyaim/runs/bridge/gain.json --live --seconds 180
```
**先补预注册**:真实游戏没有同 seed 重放,CONTRACT.md 第 3 节的配对判定
**不能沿用**。上线前须在 CONTRACT 附录写死新判定(建议:固定时长分块 A/B,
fly vs seek vs zero;主指标沿用平均角误差而非命中率 —— 理由同 D12)。
- [ ] 判定规则预注册后才准跑正式数据;
- [ ] 每次运行留存 gain.json / bridge_summary.json / 遥测;
- [ ] 报告必须带 tick_hz、capture_age、FOV、灵敏度四件套。

---

## 5. 性能预算(实测,2026-10-04,本机,Phase B 六轮优化后)

| 环节 | 耗时 | 备注 |
|---|---:|---|
| 抓帧 1920×1080(bettercam/DXGI,BGRA) | ~6 ms | mss/GDI 32ms → bettercam 6ms |
| resize→640×480(PIL BOX + 小图重排) | ~20 ms | BGRA 先 resize 后重排(2x),BOX 滤波 |
| 捕获周期合计(捕获线程,并行) | **~28 ms** | 与网络拍并行,可隐藏 |
| 视网膜编码(CPU,640×480) | ~15.7 ms | D17 |
| 脑仿真 8 步(GPU) | ~23-30 ms | 随活跃度变化 |
| seek 检测(act,640×480) | ~25 ms | scipy 连通域 |
| fly act 合计 | **~48 ms** | 视网膜+脑+读出 |
| **seek 闭环实测** | **28.4 Hz** | run 20261004-012234-bridge-phaseB8 |
| **fly 闭环实测(真实游戏画面)** | **17.6 Hz** | run 20261004-012305-bridge-fly-live |

(旧离线闭环 4.2 FPS;接入真实屏幕后反而快 4 倍 —— 脑仿真不再是唯一瓶颈。)

**Phase B 六轮优化日志(每轮都有实测证据,细节见 git/运行目录):**
1. mss 替换 PIL:全屏 grab 50.7→30.2ms;
2. **修复消费端忙等自旋**(sleep 被错误放在 2s 停滞告警门槛后,自旋占住 GIL
   把捕获线程饿到 10 FPS):5.5→10 Hz;
3. **BGRA 重排与 resize 交换**(PIL 对负步长数组走慢路径;数值逐位一致):
   10→15.4 Hz;
4. bettercam(Desktop Duplication,BGRA 输出零 cv2 依赖;region 校验需
   SetProcessDPIAware——本机 125% 缩放):15.4→23.8 Hz;
5. BOX 滤波(面积平均,与视网膜分块均值同语义):23.8→28.4 Hz;
6. 进程内 timeBeginPeriod(1):Windows 默认 15.6ms 定时器精度会放大所有
   Event.wait —— BridgeLoop 启动时提升、退出时恢复。

---

## 6. 与离线实验的差异清单(写报告必读)

| 维度 | 离线靶场 | Aim Lab 桥接 |
|---|---|---|
| 确定性 | 同 seed 逐位复现 | **不可重放**,无同 seed 配对 |
| 帧因果 | 先动后渲染,零延迟 | 注入→渲染→捕获,≥1 帧延迟 |
| 输入统计 | 背景静止,只有准星/靶动 | 全域光流(相机一动全画幅都动) |
| action 语义 | 像素速度(14px/拍) | 角速度(经线性增益折算) |
| hit_rate | 已被证伪无分辨力(D12) | 更不可用(判定半径/命中音效均不可见),主指标必须是角误差 |
| 开火 | 每帧自动 | 桥接**不碰开火**(SendInput 只发移动;要射击需显式加按键,默认关闭) |

## 7. 合规与安全边界(DECISIONS.md D18)

- 只用 Win32 `SendInput` 标准注入(与 AutoHotkey 同路径);**不做**驱动级
  注入、不做反作弊检测规避;
- Aim Lab 是单机训练器,但自动化仍可能违反其服务条款 —— 使用者自担风险,
  建议只用本地/自定义任务、**不提交排行榜成绩**;
- 真实注入永远显式(`--sink sendinput`),默认 NullSink;
- 每次注入累计计数写进 summary,全程可审计。

## 8. 命令速查

```powershell
$py = "C:\Users\Admin\.dsh\dsh-runtimes\dsh-primary-runtime\dependencies\python\python.exe"
$env:PYTHONIOENCODING = "utf-8"

& $py tools/aimlab_smoke.py                    # 冒烟(离线,30 项)
& $py tools/aimlab_smoke.py --screen           # + 真实截屏 5 帧(只读)
& $py tools/aimlab_smoke.py --cursor-check     # + 光标微移注入(显式)
& $py tools/aimlab_gain.py --from-sens 2 --engine aimlab --fov 106.26  # 增益标定(D27/D28)→ gain.json& $py tools/aimlab_bridge.py --controller fly --source arena --frames 50   # 无头彩排
& $py tools/aimlab_bridge.py --controller seek --source screen --frames 300 --live  # 管道校验
& $py tools/aimlab_bridge.py --controller fly --source screen --sink sendinput --gain-json flyaim/runs/bridge/gain.json --live  # 真实闭环
& $py tools/live_web.py --dir flyaim/runs/<ts>-bridge/live   # 实时可视化
```

## 9. GPU rollout 管线(DECISIONS.md D24)

训练数据生成与反传已可全部跑在 GPU(视网膜编码、靶场推进、网络前向/反传),
见 `flyaim/gpu/` 与 `tools/train_ann2_gpu.py`。

```powershell
# GPU 战役(默认 64 环境并行;rollout ~560 帧/s,GPU 利用率 100%)
& $py tools/train_ann2_gpu.py --envs 64 --frames 12000 --tbptt 16
# 等价性校验(GPU 版必须与 CPU 版数值一致才可用于结论)
& $py tools/verify_gpu_retina.py       # 视网膜,相对误差 ~8e-6
& $py tools/verify_gpu_backward.py     # 批量反传,W/Wz/g/gz 逐位一致
```

**硬件约束(实测,GTX 1660 Ti Max-Q / 6GB)**:本环境 **无 cuBLAS**,
cuSPARSE 的 `csrmv`/`csrmm2` 不可用且 `spmm` 按列线性退化,故 SpMM 由自写
CSR kernel(`flyaim/gpu/csr_spmm.py`)承担。TBPTT 长度 `--tbptt` 控制显存占用
(每帧 traces 约 `0.171 GB × B/64`);若同时运行游戏(如 CS2 占 4GB 显存),
需下调 `--envs` 或 `--tbptt` 以免颠簸。

**结论提醒**:GPU 只改吞吐,**不改科学结论**。A2 战役(第六体制)在 GPU 上
重跑后 real 375.6px 仍劣于 random 286.5px(C-G1/G2 失败;C-G3 效应仅 2.4px,
名义显著而实际无意义)。

---

## 10. Phase E — 复眼视觉伺服接入实战(DECISIONS D26)

> §1–§9 是**连接组**臂接入真实游戏的准备。工程上那套(捕获/注入/增益/闭环)现在
> 原样复用,但接的控制器换成了 **eye-servo**:保留 6,098 个真实复眼感光细胞,
> 把连接组 + 2 维全局求和读出换成 (24×32) 小眼空间图 + 刹停律(D25)。
> **它绕过了连接组,因此不得当成果蝇臂**;它回答的是"这个任务对果蝇的眼睛有多难"。

### 10.1 已经做完的(离线,全部零风险)

| 步骤 | 命令 | 结果 |
|---|---|---|
| 模拟 FPS 闭环彩排(不动鼠标) | `tools/aimlab_sim3d.py --seconds 20` | 角误差 **0.85°**、锁定后 100%、首达 3 拍 |
| 经完整桥接管道彩排 | `tools/aimlab_bridge.py --controller eye --source arena --frames 300 --target-color 235,70,70 --eye-aim detect --arena-no-respawn` | 稳靶率 **97.0%**,58.8 Hz |
| 同一命令(respawn 开) | 去掉 `--arena-no-respawn` | **5.33%** —— 与离线逐位一致,管道零回归 |

模拟器的语义清单(为什么它能替代"先把游戏打开试一下"):针孔投影、准星钉在画面
中心、相机由**注入计数**驱动、青色靶、世界点阵光流、命中判定按角半径。
它**不**复刻:真实渲染延迟、鼠标加速度、HUD/枪模遮挡、球的明暗。

### 10.2 🔴 上线前必须知道的三条(都是实测,不是经验)

**(1) `--fov` 不是可选项,而且填哪个数有讲究(见 D28)。**
`GainModel` 旧实现把"扫过画面宽度的 2.19%"当成"整圈的 2.19%"。FOV=103° 时
7.875°/拍 vs 正确 3.151°/拍。对带刹停律的 eye 臂只是精度损失(0.85°→2.37°);
对**恒定量程**的控制器是致命的(锁定 100.0% → **0.3%**)—— 这正是 D19 "一开跑就
转离靶区 staring at wall"的成因。命令里必须带 `--fov <你的FOV>`。

🔴 **要填的是「渲染视野的水平 FOV」,不是游戏设置里那个 fov 值。**
这两个数在 Valve 系游戏里**不一样**:

| 游戏 | 设置里的 fov | `--fov` 该填(16:9) |
|---|---|---|
| CS2 / CS:GO | 90 | **106.26** |
| Valorant | — | 103 |
| Apex | 90(竖直) | 按 `2·atan(tan(v/2)·16/9)` 换算 |

原因是 Valve 的 fov 值恒为 90(与纵横比无关),实际按纵横比放大;90 在 4:3 下
等于 16:9 的 106.26°。**若照 Aim Lab 界面上的「视野范围 90°」填,每拍少转 33%**
(2.5067 vs 3.3422 °/action),表现为"收敛慢、总停在靶前面"。
判据:你的 `speed_fraction` 来自"靶在画面上移了多少**像素比例**",像素↔角度
必须用渲染 FOV。拿不准就用 Aim Lab 里**把 Game 选成你的游戏**后的那张场景,
用 `aimlab_probe.py` 量一个已知角直径的靶做标定。

**(2) 任务的靶必须够大 —— 这是能不能接的硬边界。**
24×32 复眼要靶角直径 **≥ 11°**(半径 5.5°);48×64 降到 **≥ 7°**。
Aim Lab Gridshot 默认球约 2~3° 角直径,**接不了**。做法:自定义任务把靶调大,
或站近一点。网格也不能无限加密:R1-R6 只有 3,377 个感光细胞,
超过 ~58×58 就会有格子分不到细胞、空间图出现空洞,性能反而崩。

**(3) cm/360 不必在桌面上量 —— 用游戏内灵敏度算即可(见 D27)。**
误差反馈闭环对增益不敏感:乘 0.5x~2x 锁定率都还是 99.7%。它影响的是**稳态残差**
与抖动。所以 EPP(指针加速)带来的非线性是"精度问题"而非"可行性问题"。
这意味着**标定的容错带很宽**,你不需要把 cm/360 量得准 —— 但你需要它**不错一个
数量级的量级**(引擎系数给错 3.2 倍、Aim Lab DPI 缩放漏补 4 倍,都会真的坏掉)。

### 10.3 真机三步(第 3 步才会动鼠标)

**示例参数已按本机实测填好**(CS2 / sens=2 / 800 DPI / 16:9,见 D28):

```bash
# 解释器:见 README 的 $py 定义
# --- 第 1 步:定标(只需一次)。用端到端向导,3 个数就够 ---
#   它会:①算 counts_per_360 ②跑指针路径探针(自动)③可选游戏内手测 ④给 PASS/FAIL
$py tools/aimlab_calibrate.py --sens 2 --engine aimlab --aimlab-dpi 800 --fov 106.26
#     --sens 填界面上的灵敏度;--engine aimlab/source(同 0.022)/unreal
#     --aimlab-dpi 填界面上那个 DPI(高 DPI 模式会自动补缩放,漏补会过转 2~4 倍)
#     --fov 填**渲染视野**水平 FOV:CS2@16:9 = 106.26(不是界面上的 90!)
#     --no-probe 可跳过探针(不动鼠标)
#   也可用 --edpi 1600 --dpi 800(社区 eDPI 口径),或 --cm360 25.9773 --dpi 800
#   本机结果:counts_per_360 = 8181.8,cm/360 = 25.98cm,判定 PASS
#
#   只想要 counts_per_360 不想探针,用老入口:
$py tools/aimlab_gain.py --from-sens 2 --engine aimlab --fov 106.26
#
#   最可信的一层:游戏内手测(注入已知计数 N,量准星移过画面宽度百分比 p)
$py tools/aimlab_calibrate.py --sens 2 --engine aimlab --fov 106.26 \
      --verify-counts <你量出的等效 counts/360>
#   差异 ≤5% 才放行;>5% 会按「加速没关 → 引擎系数 → DPI缩放 → 标称DPI」排查

$py tools/aimlab_smoke.py --cursor-check                 # 光标微移自检(动 3px)

# --- 第 2 步:只看不注入。Aim Lab 进任务内(不是大厅!),无边框窗口 ---
$py tools/aimlab_probe.py --find aimlab                  # 定位窗口 + 找靶色 + 目检标注图
$py tools/aimlab_bridge.py --controller eye --source screen --window aimlab \
      --sink null --fov 106.26 --target-hue auto --eye 24x32 \
      --eye-aim center --eye-search scan --frames 300 --live
#   看 runs/<ts>-*/live/preview/ 的检测框:靶有没有被框住、框在哪。
#   ⚠️ 靶色不是青/红时用 --target-color R,G,B 覆盖
#   ⚠️ 若遥测里靶检测率 < 90%:说明色觉响应低于门限(6 Hz),即"复眼其实没看见靶",
#      先目检确认靶色/对比度,再考虑注入

# --- 第 3 步:真实注入(会动鼠标;先小幅度试方向)---
$py tools/aimlab_bridge.py --controller eye --source screen --window aimlab \
      --sink sendinput --gain-json flyaim/runs/bridge/gain.json \
      --fov 106.26 --max-counts 150 --frames 120 --live
#   方向不对 -> --invert-x / --invert-y
#   确认方向与幅度后去掉 --max-counts 限制,拉长 --seconds
```

### 10.4 验收标准与必须随结果报告的四件套

- 验收:`--sink null` 阶段遥测里**靶检测率 ≥ 90%**,且 `act` 方向与检测误差同向;
  `--sink sendinput` 阶段角误差单调下降并稳定在**靶角半径以内**。
- 每次运行必须连同以下四项一起报(理由同 §3.1 与 D26.7):
  **`tick_hz`(拍频)、`capture_age_p95`(画面年龄)、`FOV`、`counts_per_360`(灵敏度)**,
  外加 `pointer_accel_enabled`(EPP 状态,`bridge_summary.json` 自动记录)。
- 主指标用**角误差**,不用 hit_rate(理由同 D12:`hit_rate` 无分辨力)。
- 真机结果与 §10.1 数字的差异应归因于**环境**(渲染延迟/加速度/HUD),
  而不是控制律 —— 两者已在模拟域分开。
