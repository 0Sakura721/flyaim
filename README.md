# FlyAim — 用果蝇连接组驱动闭环瞄准控制

用 **MaleCNS v1.0 果蝇全神经连接组**(166,700 神经元 / 25,582,938 条边)驱动一个闭环瞄准控制器,
在自建靶场中与**打乱接线的零模型**、**PID 上限**、**随机下界**做同 seed 同帧预算的对照实验。

---

## 结论先说(TL;DR)

> **该 LIF 简化模型 + 静态连接组配置,不承载可用的视觉伺服信号。**
>
> **2026-10-04 更新:结论已扩展到可塑性条件** —— 三因子可塑性(连接组权重
> 可学习)+ 冻结随机读出,36k 帧训练 + 四臂 10 seeds 对照,三道预注册验证门
> 全部未通过,四臂统计不可区分(D21)。阴性边界现覆盖三种学习体制:
> 冻结读出 / 游戏域重训读出 / 网络可塑性。
>
> **2026-10-04 终版:梯度训练体制同样阴性** —— 连接组-RNN 行为克隆闭环
> 坍缩为零动作(D22);DAgger 修复轮后 391px 仍显著劣于随机游走 278px
> (p=0.012,D23)。阴性边界现覆盖**五种学习体制**,项目按预注册最终归档。
>
> **2026-10-04 建设性对偶(D25):同一任务换掉连接组 = 复眼视觉伺服 100% 稳靶。**
> `eye-servo` 保留 **6,098 个真实复眼感光细胞**(R1-R6 亮度通路找准星 /
> R7-R8 色觉通路找靶),把连接组 + 2 维全局求和读出换成 (24×32) 小眼空间图 + 刹停律。
> 实测 10 seeds × 900 帧:**首达 12.5 帧,之后稳靶帧占比逐 seed 100.00%**,
> 且与**直接偷看靶位坐标**的作弊上界打平(98.61% vs 98.66%)、闭环 39.7~54.7 FPS(纯 CPU)。
> ⇒ **该任务不难,难的是用连接组做。** 详见 `DECISIONS.md` D25。
> ⚠️ 两条必须一起引用的边界:①这是**绕过连接组**的结果,不得当成果蝇臂;
> ②稳态残差最大 21.9px、判定圈 22px —— **余量只有 0.1px**。
>
> **2026-10-05 实战接入(D26):同一份 arm 接进桥接层并跑通模拟 FPS 与真机前全链。**
> `--controller eye` 已进桥接主入口;在**语义复刻真实 FPS** 的模拟域
> (针孔投影 / 准星钉在中心 / 青靶 / 世界点阵光流)里:角误差 **0.85°**、
> 锁定后 100%、首达 3 拍。顺带查出并修掉一处**真实的标定缺陷**:
> `GainModel` 把"扫过画面宽度的比例"当成"整圈的比例",FOV=103° 时
> **每拍过转 2.50 倍**。详见 `DECISIONS.md` D26。
>
> **2026-10-05 定标免测量(D27):cm/360 不必在桌面上量。**
> 新增路线 B:`counts_per_360 = 360 / (sens × 引擎系数)` ——
> **DPI 根本不在这个式子里**(它只决定手滑 1cm 产生多少计数;
> 400DPI×2sens ≡ 800DPI×1sens,这就是 eDPI)。同时钉死两个会让注入量
> **系统性偏 2~4 倍**的坑:Aim Lab 高 DPI 模式把计数归一化回 800 基准
> (3200 → ×0.25),以及各引擎 `yaw_coef` 不同(Source 0.022 vs UE 0.0703)。
> 附 `--verify-counts` 实机对拍:差异 >5% 直接拒上线。详见 `DECISIONS.md` D27。
>
> **2026-10-05 FOV 口径(D28):填渲染 FOV,不是游戏设置里的 fov 值。**
> 从两张 Aim Lab 截图交叉验证:`counts_per_360 = 8181.8`(sens=2),
> 与截图「360°转身距离 25.9773cm」逐位吻合(差 0.0001%),顺带确认 DPI=800。
> 但 `--fov` 要填的是**渲染视野水平 FOV** —— CS2 在 16:9 下是 **106.26°**,
> 不是界面上的 90(Aim Lab 的 90 是 sens 换算基准,CS2 官方 fov 也是 90)。
> 照界面填 90 会**每拍少转 33%**。命令里的示例已全部按本机参数更新。

**正式实验(10 seed × 900 帧,80.3 分钟):**

| 控制臂 | 命中率 mean±std | 平均靶距 | 闭环 FPS |
|---|---:|---:|---:|
| 果蝇连接组 + 读出 | 0.0000 ± 0.0000 | **412.5 px** | 3.5 |
| 打乱接线(零模型) | 0.0000 ± 0.0000 | **430.9 px** | 5.4 |
| PID(偷看靶位,上限参照) | 0.0633 ± 0.0044 | 155.8 px | 567.7 |
| 均匀随机(下界) | 0.0002 ± 0.0007 | **251.7 px** | 583.7 |

**预注册判定:`【接线图无因果贡献】fly 相对零模型 shuffle 无显著优势 (p=1.0000)`**

学习曲线 10 箱全部平坦为 0.00 —— 没有学习发生。

### ⚠️ 为什么命中率这么低?——这是靶场设计,不是控制器差

`hit_rate` 是个**被设计压平的指标**,不能用来评判瞄准水平:

**(a) 上限 ~6%,由旅行时间决定,与瞄准精度无关。**
`respawn_on_hit=True` + 命中前靶不动 ⇒ 任务其实是「走到一个**固定点**」;
命中一次后靶立刻 teleport 到别处,**无法贴靶刷分**。

| 情形 | hit_rate |
|---|---:|
| `respawn_on_hit=False` + 完美控制器 | **99.22%** |
| `respawn_on_hit=True` + 完美控制器 | **5.56%** |
| 解析上限 `1/(271px÷14px + 1)` | 4.91% |
| **PID 实测** | **6.33%** |

4 个行为完全不同的完美控制器(停靶内 / 全速冲 / 比例减速 / 恒定满量程)
**得分完全相同** —— 唯一决定命中率的是移动速度。PID 的 6.33% **就是上限**。

**(b) 判定窗口只占画布 0.495%**(π·22² = 1,521 px² / 307,200 px²)。

**(c) 失败臂是「从未抵达」,不是「瞄得不准」。** 逐臂实测(900 帧):

| 臂 | hits | respawns | min_dist | mean_dist | 墙贴靠 | 覆盖格 |
|---|---:|---:|---:|---:|---:|---:|
| pid | 56 | **56** | 22.1 | 181.3 | 0.3% | **199** |
| fly | 1 | 1 | 28.3 | 402.0 | 25.3% | 55 |
| random | 0 | 0 | **24.1** | 189.4 | 18.7% | 75 |
| const_bias | 0 | 0 | 68.9 | 267.3 | **96.2%** | **17** |

**random 的 min_dist = 24.1px —— 差 2.1px 就能命中,却一次没中。**
真正的区别是**机动性**:PID 覆盖 199 格、fly 55 格、恒定偏置只有 17 格。

> **⇒ 因此本项目的主力指标是 `mean_target_dist_px`,不是 `hit_rate`。**
> `fly vs random` 在 hit_rate 上不显著(p=0.34)**不是表现相近,是指标无分辨率**;
> 用靶距则显著:`+160.8px`(p=0.0069, dz=1.10)。
> 这是 Lead 的设计缺陷(选主指标前没做分辨率检验),完整留档在 `DECISIONS.md` D12。

**⚠️ 这个负面结果比"没帮上忙"更强:** 两个网络臂的平均靶距(412.5 / 430.9 px)
都**显著差于均匀随机的 251.7 px**。用「平均靶距」做配对检验(越低越好):

| 比较 | 差值 | p | dz | 判定 |
|---|---:|---:|---:|---|
| fly vs random | **+160.8 px** | 0.0069 | +1.10 | **显著更差** |
| shuffle vs random | **+179.2 px** | 0.0080 | +1.07 | **显著更差** |
| fly vs pid | +256.7 px | 0.00028 | +1.82 | 显著更差 |
| fly vs shuffle | −18.4 px | 0.66 | −0.14 | 不显著 |

即:接上连接组后不是"无效果",而是**让准星比乱动更远离目标**。
(注:`hit_rate` 指标上 fly vs random 不显著,因为两者都接近 0 命中 ——
命中率在 0 附近**没有分辨力**,靶距才是有效指标。这个细节本身也值得记录。)

四方法收敛:

| 方法 | 结果 |
|---|---|
| 方差分解(同时控制刺激与初始状态) | 刺激/状态 比值 **0.48 ~ 0.94**(< 1) |
| 独立复现(「同画面不同随机历史」作噪声底) | 比值 **0.96 ~ 1.04** |
| 端到端实验 | 训练 **r²=0.75** → 评估 **0 命中**,且劣于随机 |
| 时间积分补救(1~24 帧窗口) | 比值**不随窗口改善**,无效 |

**⚠️ 但结论的正确表述是「我们的简化仿真不承载视觉信号」,而不是「果蝇不会瞄准」。**
被证伪的是 LIF 简化模型 + 静态权重(缺少突触动力学、神经调质、可塑性、lamina 分层处理),
不是果蝇的神经回路。真实果蝇当然能追踪目标。

**机制:** 网络对初始状态混沌敏感,轨迹主要由自身历史决定;
而视觉信号太弱 —— 实测感光细胞 → 一级靶的**中位突触前伙伴只有 1 个、总权重仅 8**,
信号在抵达下行神经元前已被混沌涨落淹没。时间积分也无法提取,
因为残差是**确定性混沌轨迹**而非零均值噪声,平均消不掉。

一个被推翻的中间假设留档在 `DECISIONS.md` D11:曾以为是"读出层 bias 导致准星顶死墙角",
诊断显示动作确实在变化、r² 远超常数预测器 —— **真实机制是读出层在单条轨迹上过拟合**。

这与公开结果吻合:网络热传的 DOOMFLY 跑 3,000+ 轮,**存活时间同样没有改善**。

---

## 这个项目做对了什么

1. **视觉输入是真实的**:驱动的不是手搓的假眼睛,而是数据集里**真实存在的 6,098 个复眼感光细胞**
   (`superclass == 'ol_sensory'`:R1-R6 3,377 / R7* 1,385 / R8* 1,329 / HBeyelet 7),
   `visual_input_fallback = False`。
2. **零模型是严格的**:`shuffle` 保持每行非零数、权重数组逐位不变、出权重和严格不变,
   只打乱「谁连谁」——即摧毁拓扑信息而保留全部统计量。
3. **学习被限制在读出层**:连接组权重在训练前后**逐位校验不变**
   (`assert_connectome_frozen` 比对 `indptr`/`indices`/`data`)。
4. **判定规则预先注册**:在实验前写死在 `CONTRACT.md` 第 3 节,不允许事后修改。
5. **失败方向如实报告**:`DECISIONS.md` 留档了 **5 处被证伪的中间结论**,
   包括 Lead 自己的两次错误判断(过早告警数据损坏、动作恒定偏置假设)。
6. **录屏可查**:`flyaim/arena/recorder.py` 流式 APNG 录制(内存 O(1),
   实测 N=60→600 增长 0.0 MB,对照组 +801 MB)。实验画面留在 `runs/<ts>/episode_*.png`,
   可直接目视核对"准星到底动没动"。

---

## 快速开始

### 环境

本机 `python` 是 Microsoft Store 占位 stub,**必须用捆绑解释器**:

```powershell
$py = "C:\Users\Admin\.dsh\dsh-runtimes\dsh-primary-runtime\dependencies\python\python.exe"
$env:PYTHONIOENCODING = "utf-8"   # 否则控制台中文/符号乱码
Set-Location D:\dsh\autoaim
```

依赖:`numpy` `pandas` `pyarrow` `scipy` `matplotlib` `Pillow`(**不需要 torch**)。

### 冻结地基回归测试(先跑这个)

```powershell
& $py flyaim/tests/test_core.py
```
覆盖 33 项检查,包括两个曾导致**静默失效**的缺陷(见 `DECISIONS.md` D7)。

### 查看数据是否就绪

```powershell
& $py -m flyaim.main check
```

### 用显卡训练(GPU)

LIF 引擎有 **CPU** 与 **GPU** 两套实现,数值等价、可切换:

```powershell
# 默认走 GPU(有卡就自动用)
& $py tools/train_readout.py --seeds 2 --frames 700 --device cuda
& $py tools/run_experiment.py --seeds 2 --frames 300 --device cuda

# 复现历史结果走 CPU
& $py tools/train_readout.py --seeds 2 --frames 700 --device cpu
```

**实测(GTX 1660 Ti Max-Q 6GB,全量 166,700,dt=4ms/8 步,同期受控 A/B):**

| 引擎 | 100 帧耗时 | 每帧 | 速率 |
|---|---:|---:|---:|
| CPU `Connectome`(scipy 稀疏,单线程) | 154.6 s | **1546 ms** | 0.65 帧/秒 |
| **GPU `ConnectomeGPU`(cuSPARSE)** | **25.2 s** | **252 ms** | **3.97 帧/秒** |
| | | | **6.1x** |

**数值等价性已证明**(`tools/bench_gpu.py`):

| 指标 | 结果 |
|---|---|
| 起始逐位一致步数 | **6 步**(第 7 步首次分岔) |
| 脉冲逐元素一致率 | **99.28%** |
| 平均放电率相对差 | **0.009%** |
| 中位数放电率 | 99.9997 / 99.9997 Hz |

> **为什么"前 6 步完全一致、之后分岔"就是等价?**
> cuSPARSE 与 scipy 的**求和顺序不同**,阈值边缘的神经元会翻转,而混沌循环动力学
> 把这种浮点微差指数放大。真 bug 会从**第 1 步**就分岔。
> 因此判据是「短时程精确一致 + 长时程统计等价」,而不是逐位相等
> —— 混沌网络里逐位相等**既不可能也不必要**。

GPU 引擎的两处**有意差异**(数学等价,为效率):合成单一带符号矩阵
(`W = W_exc·w_e − W_inh·w_i`)把每步两次 SpMV 降为一次;放弃脉冲驱动的列收集
(GPU 上不规则 gather 反而不如全量 SpMV 用带宽换规整)。

> ⚠️ **口径提醒:绝对帧率随网络活跃度大幅变化**(单步成本 ∝ 活跃神经元数)。
> 低活跃时可达 ~17 帧/秒,高活跃(82% 发放)时降到 ~4 帧/秒。
> **6.1x 这个比例才是可靠结论**(同条件受控 A/B),绝对值必须连同活跃度一起报。

**安装(本机没有 CUDA toolkit,用 pip wheel)**

```powershell
# 注意 PyPI 在此网络下被限速到 0.02~0.07 MB/s,必须用并发下载器
& $py tools/fetch_fast.py --batch .cache/wheels/batch.txt --dest .cache/wheels --conn 64
& $py -m pip install --no-index --find-links .cache/wheels `
     cupy-cuda12x cuda-pathfinder nvidia-cuda-runtime-cu12 `
     nvidia-cusparse-cu12 nvidia-nvjitlink-cu12 nvidia-cuda-nvrtc-cu12
```

需要的 CUDA 库(来自 pip,非 toolkit):

| 包 | 大小 | 用途 |
|---|---:|---|
| `nvidia-cuda-runtime-cu12` | 3.4 MB | `cudart64_12.dll` |
| `nvidia-cusparse-cu12` | 345.7 MB | SpMV |
| `nvidia-nvjitlink-cu12` | 33.9 MB | cusparse 依赖 |
| `nvidia-cuda-nvrtc-cu12` | 76.4 MB | **运行时 JIT 编译 kernel(必需)** |
| `cupy-cuda12x` | 94.2 MB | CuPy 本体 |

`cublas / cufft / cusolver` **用不到**(LIF 不需要),省掉 ~1.1 GB。

---



**方式一:直接实时渲染(推荐,零延迟)**
训练进程自己画窗口,边跑边刷新,不需要另开进程:

```powershell
# 训练读出层,窗口直接弹出(默认开渲染)
& $py tools/train_readout.py --seeds 2 --frames 700 --ridge 10

# 评估四臂,同样直接渲染
& $py tools/run_experiment.py --seeds 2 --frames 300

# 关掉窗口(无头模式)
& $py tools/train_readout.py --seeds 2 --frames 700 --no-render
```

窗口是 2×2 四格:

| 格 | 内容 |
|---|---|
| 左上 | **靶场画面**(实时帧,能看到准星朝靶移动) |
| 右上 | **166,700 神经元活动热图**,按解剖层次分块(ol_sensory / ol_intrinsic / visual_projection / descending_neuron …) |
| 左下 | **1,360 个下行神经元放电率**的滚动时序 |
| 右下 | **指标曲线**:平均靶距、spike_rate、读出层 R² |

标题栏实时显示:帧号 / spike_rate / 活跃神经元数 / 靶距 / R² / 样本数。
窗口被关闭不会让训练崩溃 —— 自动降级为无窗口继续跑。

**方式二:遥测 + 独立查看器(训练不受影响)**
训练写 JSONL,另一个进程读文件刷新。适合训练要跑很久、想关掉窗口再看的情况:

```powershell
# 终端 1:训练并写遥测
& $py tools/run_experiment.py --seeds 10 --frames 900 --live

# 终端 2:实时查看(可随时开/关)
& $py tools/live_view.py --dir flyaim/runs/<时间戳>-phase3/live/fly_seed0
```

---



```powershell
# 1) 诊断:连接组拓扑可达性
& $py tools/diag_reachability.py

# 2) 标定:找生理工作点(约 10 分钟)
& $py tools/calibrate_lif.py
& $py tools/test_normalization.py

# 3) 标定:验证 DN 是否承载视觉信息(约 3 分钟)
& $py tools/test_dn_variance.py

# 4) 端到端对照实验(10 seed × 900 帧,约 80 分钟)
& $py tools/run_experiment.py --seeds 10 --frames 900

# 加 --record 可录出各臂画面(APNG),用于目视核对
& $py tools/run_experiment.py --seeds 2 --frames 200 --record

# 5) 机制诊断:动作是恒定偏置还是有效控制?
& $py tools/diag_readout_mechanism.py

# 6) 机制诊断:为什么命中率这么低?(上限分解 + 逐臂因果链)
& $py tools/diag_hitrate_ceiling.py
& $py tools/diag_hitrate_chain.py

# 7) 「百发百中」:复眼视觉伺服 vs 纯像素伺服 vs 作弊上界 vs 随机(约 22 分钟)
& $py tools/aim_perfect.py
#   口径验证(比例项 vs 刹停律、两种 respawn 语义,约 22 分钟)
& $py tools/verify_servo_100.py
#   可视化:复眼怎么看靶(四格图)
& $py tools/plot_aim_perfect.py

# 8) 实战接入(D26):模拟 FPS 里的复眼闭环彩排 + 增益语义/角分辨率/再捕获实测
& $py tools/aimlab_sim3d.py --seconds 20
#   经**完整桥接管道**彩排(捕获线程→复眼→增益→注入),不动真实鼠标
& $py tools/aimlab_bridge.py --controller eye --source arena --frames 300 `
      --target-color 235,70,70 --eye-aim detect --arena-no-respawn
```

产物落在 `flyaim/runs/<时间戳>-phase3/`:`raw_results.json` `stats.json` `REPORT.md`。

---

## 目录结构

```
CONTRACT.md                 三方接口契约 + 预注册判定规则(冻结)
DECISIONS.md                关键决策与**被证伪的假设**留档 ← 建议先读
flyaim/
  config.py                 配置数据类                    [Lead]
  neuron_index.py           全局索引空间 + 角色选择        [Lead]
  io.py                     产物 schema 与读写            [Lead]
  stats.py                  配对检验 + 效应量 + 预注册判定  [Lead]
  runner.py                 实验编排(Arm 协议)           [Lead]
  report.py                 Markdown 报告生成             [Lead]
  pipeline.py               跨线集成(FlySystem / FlyArm) [Lead]
  fit_readout.py            冻结前提下的读出层训练         [Lead]
  main.py                   CLI                          [Lead]
  data/                     连接组下载/解析/导出           [A 线]
    build/                  ← 产物:connectome.npz 等
  retina/encoder.py         复眼编码器(驱动真实感光细胞)  [B 线]
  brain/lif.py              稀疏 LIF 引擎 + 行归一化       [B 线]
  brain/readout.py          DN → 鼠标增量                 [B 线]
  arena/                    自建靶场 + 指标 + 绘图         [C 线]
  baselines/                pid / random / shuffle        [C 线]
  bridge/                   Aim Lab 桥接(捕获/注入/增益) [接入线] ← 见 AIMLAB.md
  tests/test_core.py        冻结地基回归测试              [Lead]
tools/                      Lead 的诊断与标定脚本
  aimlab_bridge.py          桥接主入口(屏幕→网络→鼠标)
  aimlab_gain.py            增益标定(灵敏度/cm/360 → counts/360,D27)
  aimlab_calibrate.py       端到端定标向导(+指针路径探针,PASS/FAIL,D27/D28)
  aimlab_probe.py           定位窗口 + 找靶色 + 目检标注图(只读)
  aimlab_play.py            任务内会话:collect/seek/fly/hybrid 真游戏闭环(D29)
  aimlab_train_ingame.py    游戏域读出层重训(岭回归,连接组冻结)
  aimlab_sim3d.py           模拟 FPS 语义彩排(不动鼠标,正负对照 + 增益容差)
  aimlab_smoke.py           桥接层冒烟测试(183 项)
```

---

## 数据

**MaleCNS v1.0**(HHMI Janelia + 剑桥大学 + MRC LMB + Google Research),CC-BY-4.0。
下载自 `gs://flyem-male-cns/v1.0/connectome-data/flat-connectome/`。

| 文件 | 字节 |
|---|---:|
| `body-annotations-...-minconf-0.5.feather` | 14,483,314 |
| `connectome-weights-...-minconf-0.5.feather` | 1,051,241,946 |
| `body-neurotransmitters-...feather` | 43,282,834 |
| `body-stats-...-minconf-0.5.feather` | 778,062,826 |

产物:`N = 166,700` / 边 `25,582,938` / 突触 `124,177,616`。
正确性经官方 `-traced-only` 神经元级版本交叉验证(差异 +0.095%,本产物为超集)。

**关键实测事实:**
- 功能分类主字段是 **`superclass`**,不是 `class`(`class` 有 84% 为 NaN,且无 `descending`/`motor`)
- 过滤掉 126M 条边**不是数据损失**:全量文件把突触摊到了未追踪的碎片对象上,
  `weight>=10` 的边 98.3% 两端都是真神经元
- `~1.25e8` 指的是**突触**不是边;实测突触 124,177,616 与之吻合

---

## 已知限制与偏差(必须随结论一起引用)

1. **histamine 符号近似**:7,905 个组胺能细胞(含**全部 R1-R6**)因 `INHIBITORY_NTS` 不含组胺
   而归入 `W_exc`。生物学上组胺在 lamina 是**抑制性**的 → 视觉通路第一级极性反转、
   ON/OFF 对比度响应失真。
2. **神经元模型是 LIF 简化模型**:连接组只给「谁连谁、连多强」,不含真实膜电导、
   递质释放动力学、神经调质扩散。
3. **动作语义是人为定义的**:DN 活动 → 「左右/上下」的映射由开发者设定,
   无行为学证据支持。
4. **视拓扑是近似的**:感光细胞没有 `assignedOlHex` 坐标(实测仅 `ol_intrinsic` 有值),
   小眼映射实际回退为索引分块,无生物学含义。
5. **无方向选择性**:真实 T4/T5 是 4 方向 EMD,本实现的编码器只有无方向时间差分。
6. **30 FPS 与「DN 有生理活动」不可兼得**(实测,全量 166,700):

   | 网络状态 | 单步耗时 | 每帧(33 步) | 闭环 FPS |
   |---|---:|---:|---:|
   | 近静默(默认参数,无发放) | 1.14 ms | 37.7 ms | **26.5** |
   | **工作点(indeg ws=16,DN 20 Hz)** | **7.25 ms** | **239 ms** | **4.2** |
   | raw 0.12(癫痫态) | 17.3 ms | 571 ms | 1.8 |

   单步耗时 ∝ 活跃神经元数 × 出度,所以**只要 DN 真的在放电,就掉到 4~6 FPS**。
   本项目的对照实验是在 4.2 FPS 的慢闭环下跑的 —— 各臂帧预算相同,结论仍然公平,
   但**不能把 26.5 FPS 当作"系统性能"对外表述**。
7. 未涉及任何真实游戏客户端或在线对战,全部为离线自建靶场。

---

## 如果要做下去,最有希望的方向

> **先看 D25 的修正**:本项目此前认为"块映射把图像空间结构第一步就打乱了"——
> 实测不成立。那个映射单调且全覆盖,**光栅空间结构可以逐位还原**
> (`Retina.receptor_maps()`)。所以下面前两条的优先级要改:
> 真正致命的是**进入网络之后的随机投影**与**2 维全局求和读出**。

按预期收益排序:

1. **建模突触动力学而非静态权重** —— 当前最大的简化。加入短期可塑性(STP)或
   突触后电导动力学,可能让信号在 3 跳内不被混沌淹没。
2. **读出层按视野分区(优先级已上调)** —— 把 DN 群按视网膜覆盖区域分组读出,
   而不是 1,360 维全局求和压成 2 个数;这是 D25.3 指出的、最被低估的一处架构缺陷。
3. **加入 lamina 分层与 LMC 中间层** —— 感光细胞直连中央脑是本项目最大的结构性偏离。
4. **只用运动通路子网**(T4/T5 + lobula plate + DN),而非全脑。
   全脑 99.3% 可达带来的混沌可能淹没了目标信号。
5. **把视觉输入从「逐帧稳态」改为「事件驱动」** —— 复眼本来就更像边缘/变化检测器。
6. **换任务**:本项目选的是对精度要求很高的瞄准闭环。若改测
   「趋光性」「逃逸反应」这类果蝇本征行为,成功概率会高得多。

> 已交付的替代方案(不保留连接组):`tools/aim_perfect.py` 的 **eye-servo**
> —— 复眼 + 2 维刹停读出,首达 12.5 帧、稳靶后 100%,见 D25。

---

## 开源说明(许可 / 数据 / 大文件 / 合规)

- **代码**:MIT(见 LICENSE)。
- **数据与衍生工件**:MaleCNS v1.0 原始数据(CC-BY-4.0)来自
  `gs://flyem-male-cns/v1.0/`,用 `flyaim/data/_download.py` 或 README 前文的
  下载流程获取;**仓库不内置原始数据与 >100MB 的二进制**(GitHub 硬限制),
  大件(connectome.npz、indeg 权重等)放
  [GitHub Releases](../../releases),下载后放回 `flyaim/data/build/` 与
  `flyaim/runs/norm/` 即可复现。
- **实验产物**:`flyaim/runs/` 的大二进制不入库;各次实验的
  `REPORT.md / stats.json / verdict.json` 等小文件保留,可追溯。
- **Aim Lab 桥接(`flyaim/bridge/`、`tools/aimlab_*.py`)的合规边界**
  (DECISIONS.md D18-D20):仅使用 Win32 SendInput 标准注入,不做驱动级
  注入、不做反作弊规避;Aim Lab 服务条款下自动化可能受限,使用者自担
  风险;建议只用本地/自定义任务、不提交排行榜成绩。项目的科学结论
  (连接组在该架构下不承载视觉伺服信号)不依赖该桥接。
