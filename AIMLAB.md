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
| `tools/aimlab_gain.py` | 增益标定(cm/360 × DPI → counts/360),写 gain.json |
| `tools/aimlab_smoke.py` | 30 项冒烟检查(零依赖 GPU/游戏/注入) |

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
# 1) 生成增益
& $py tools/aimlab_gain.py --from-cm360 800 40 --fov 103
# 2) 光标微移注入自检(动 3px 后自动复位)
& $py tools/aimlab_smoke.py --cursor-check
# 3) 真实注入先小幅试方向(可临时 --max-counts 150)
& $py tools/aimlab_bridge.py --controller seek --source screen --sink sendinput --gain-json flyaim/runs/bridge/gain.json --frames 120
```
- [ ] 方向:准星朝靶动的方向与检测误差一致;不对用 `--invert-x/y` 修;
- [ ] 量 cm/360(游戏内转一整圈量桌面位移)重算 gain.json;
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
& $py tools/aimlab_gain.py --from-cm360 800 40 # 增益标定 → flyaim/runs/bridge/gain.json
& $py tools/aimlab_bridge.py --controller fly --source arena --frames 50   # 无头彩排
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
