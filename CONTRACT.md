# FlyAim 接口契约 v1 (Phase 0,冻结)

> 本文件是三条并行工作线(A 数据 / B 仿真 / C 靶场)之间的**唯一隔离面**。
> 任何一方不得单方面修改以下签名。若确需变更,先在 `DECISIONS.md` 记录并同步 Lead。

---

## 0. 术语与硬事实

| 事实 | 值 | 影响 |
|---|---|---|
| 数据集 | MaleCNS v1.0 (Janelia/Cambridge/Google) | CC-BY |
| 神经元总数 | 166,700 | 全局索引空间 |
| 连接边数 | **151,856,684**(实测边表行数) | 稀疏度 **0.546%** |
| 术语区分 | **边(edge)≠ 突触(synapse)** | 一个 pre→post 对可含多个突触;`connectome-weights` 给的是**边的突触权重合计**。文献常引的「约 1.25 亿突触」是突触计数,与 1.519 亿条边不矛盾 |
| **感光细胞** | **6,098 个**(`superclass == 'ol_sensory'`) | **视觉输入用真实生物接线** |
| 分类主字段 | **`superclass`**(28 取值) | `class` 列 87% 为 NaN,不可用 |
| 递质预测 | `body-neurotransmitters` 1,835,518 行 | 覆盖 164,600/166,700 |
| 显存 | 6 GB (GTX 1660 Ti Max-Q) | 不支持 bf16,Turing |
| 后端 | `scipy.sparse` (CPU) 起步 | torch 未安装,按需再加 |

### ⚠️ 已实测修正的两项早期判断

1. **MaleCNS 确实包含复眼感光细胞**(早期判断错误,已修正)。
   实测 `ol_sensory = 6,098`,type 分布:

   | type | 数量 | 含义 |
   |---|---:|---|
   | `R1-R6` | 3,377 | 运动/明暗检测(与 DOOMFLY 用的 3,335 量级一致) |
   | `R7y/p/d/unclear` | 1,385 | 色觉 |
   | `R8y/p/d/unclear` | 1,329 | 色觉 |
   | `R7R8_unclear` | 85 | — |
   | `HBeyelet` | 7 | 眼点(非复眼) |

   → **`visual_input_fallback = False`**。视觉编码器可以驱动**真实的感光细胞群**,
   这在生物学真实性上显著优于"手搓一个假眼睛"。

2. **数据集的功能分类字段是 `superclass`,不是 `class`。**
   `class` 列 140,187/166,700 为 NaN,`class=='descending'` 与 `class=='motor'` 均为 0
   —— 早期按 `class` 写的选择逻辑**一条都打不中**,已改为 superclass 优先。

### 实测角色规模(166,700 神经元上验证通过)

| 角色 | 数量 | 策略 |
|---|---:|---|
| visual_input | **6,098** | `superclass == 'ol_sensory'` |
| descending (DN) | **1,360** | `superclass=='descending_neuron'`(1314) ∪ type 前缀 `DN`(1342) |
| motor (MN) | **708** | `superclass == 'vnc_motor'` |
| inhibitory | 0(当前) | 依赖 `body-neurotransmitters` 文件,尚未下载 |
| *(参照)* ol_intrinsic | 89,403 | 含 **T4 6,865 / T5 6,720** 运动检测神经元 |
| *(参照)* visual_projection | 9,201 | 视叶投射 |

**已知限制:感光细胞的 `somaSide` 几乎全为 NaN(6,098 中仅 36 个有 L/R 标注)**,
因此不能靠 side 划分左右视野;左右分组需依据 `assignedOlHex1/2`(小眼六边形坐标)或索引顺序。
注意 `assignedOlHex1/2` **只有 ol_intrinsic 有值(23,720 个),感光细胞自身没有**。

### ⚠️ 必须写入报告的生物学偏差:组胺被当作兴奋性

实测递质分布:`acetylcholine 104,049 / glutamate 29,622 / gaba 22,122 / histamine 7,905 /
unknown 2,100 / dopamine 395 / serotonin 375 / octopamine 132`。

`INHIBITORY_NTS = {gaba, glycine}` **不含 histamine**,因此
**全部 R1-R6 感光细胞 → lamina(LMC)的输出边被归入 `W_exc`(兴奋性)**。

生物学上组胺在果蝇 lamina 是**抑制性**的。这是一个已知的符号近似,影响:

- 视觉通路第一级的极性被反转,ON/OFF 对比度响应会失真;
- 可能出现整片视觉通路同步兴奋(缺少第一级抑制),需在调参时留意。

**处理方式:** 不修改判定规则(改了就不再是"标准做法"),但必须
(a) 写入 `manifest.notes`,(b) 写入最终报告的「限制与偏差」章节。



---

## 1. 神经元索引空间 (`NeuronIndex`)

- 全局索引是 `[0, N)` 的整数,`N = 166700`(以实际解析出的 body 数为准,记录到 `manifest.json`)。
- 索引 ↔ body ID 的双向映射由 A 线产出,存 `flyaim/data/build/neuron_index.parquet`。
- **所有模块共享同一索引空间。** B 线内部可以做子网重排,但**对外暴露的 `rates` / `spikes` 数组必须回到全局索引空间**(长度 N 的数组),除非显式声明 `compact=True`。

细胞类型选择器(按优先级,缺失则降级并在报告里标注):

| 角色 | 选择策略 |
|---|---|
| 视觉输入靶点 | 优先视叶投射/VPN 类细胞;若数据集无,则由 A 线提供**最优可用下游群**,并在 `manifest.json` 标 `visual_input_fallback=true` |
| 下行神经元 (DN) | `class == 'descending'` 或 type 前缀 `DN`;这是**唯一合法的控制读出群** |
| 运动神经元 (MN) | VNC 内 `class == 'motor'`(辅助读出,非主路径) |
| 抑制性 | 由 `body-neurotransmitters` 的 GABA/Glycine 预测判定 |

---

## 2. 三组核心接口

### 2.1 `Retina`(B 线实现,`flyaim/retina/`)

```python
class Retina:
    def __init__(self, cfg: RetinaConfig) -> None: ...
    def frame_to_spikes(self, frame: np.ndarray) -> np.ndarray:
        """靶场帧 -> 输入神经元的驱动。

        frame: (H, W, 3) uint8, 或 (H, W) uint8 灰度
        return: (n_input, 2) float32
                [:, 0] =  excitatory drive (>=0), 照度/ON 通道
                [:, 1] =  inhibitory drive (>=0), OFF/抑制通道
        行顺序必须与 self.input_neuron_ids 严格对应。
        """
    input_neuron_ids: np.ndarray  # (n_input,) int64, 全局索引
```

设计要点(不强制,但需在 docstring 说明所选方案):
- **输入靶点是真实的 6,098 个感光细胞**(R1-R6 / R7 / R8)。因此编码器要做的是
  **把像素映射到小眼阵列的照度**,而不是凭空虚构一个视觉系统 —— 这是本方案
  比"手搓假眼睛"更强的关键。
- 生物启发基线:小眼阵列 → ON/OFF 通路 + 延迟抑制(类似 lamina 的侧抑制),产生**运动敏感**响应。
- 建议把主要驱动给 `R1-R6`(3,377 个,运动/明暗通路),色觉通道 `R7/R8` 可用于
  **把靶的颜色与背景区分开** —— 靶场设计已保证颜色对比明显。
- 禁止直接使用目标坐标/标签作为输入(那是作弊)。**仅允许使用像素**,这是"是否真的在看"的底线。
- `somaSide` 不可靠(6,098 中仅 36 个有标注),**不要**用它划分左右视野;
  如需左右分区请用 `assignedOlHex1/2` 或索引顺序。

### 2.2 `Connectome`(A 线产数据 / B 线产引擎)

```python
class Connectome:
    def __init__(self, path: str, cfg: BrainConfig) -> None: ...
    def step(self, in_exc: np.ndarray, in_inh: np.ndarray,
             t_ms: float, neuromod: np.ndarray | None = None) -> None:
        """推进一个仿真时间步(dt = cfg.dt_ms),保持内部状态。"""

    @property
    def rates(self) -> np.ndarray:
        """(N,) float32, Hz, 全局索引空间的瞬时放电率。"""

    @property
    def spikes(self) -> np.ndarray:
        """(N,) uint8/bool, 本步是否发放。"""

    def reset(self) -> None: ...
```

- 神经元模型:LIF(漏积分放电),`dt_ms` 默认 1.0,膜时间常数 `tau_m` 默认 20 ms。
- 权重:由突触计数构成,分**兴奋/抑制两张 CSR 矩阵**(依据 2.1 的抑制性判定)。
- **读出层独立于连接组**,不得把训练参数写回连接组权重(那是造假)。

### 2.3 `Readout`(B 线实现)

```python
class Readout:
    def act(self, brain: Connectome) -> np.ndarray:
        """返回 (2,) float32 鼠标增量 [dx, dy],已归一化到约 [-1, 1]。"""
```

- 输入只允许是 **DN 群**(+ 可选 MN 群)的放电率。
- 两种模式必须都支持:
  - `mode="fixed"`:手写固定映射(如左右 DN 对做差),**无学习**。
  - `mode="trained"`:线性层,用监督学习拟合"目标角误差 → 期望方向"。训练集与测试集**必须按随机种子分离**。

### 2.4 `Arena` 与 `Metrics`(C 线实现,`flyaim/arena/`)

```python
class Arena:
    def __init__(self, cfg: ArenaConfig, seed: int) -> None: ...
    def reset(self) -> np.ndarray:                      # -> frame
    def step(self, action: np.ndarray) -> StepResult:   # action = (2,) dx,dy in [-1,1]
        # StepResult: frame, hit(bool), target_dist(float), done(bool), info(dict)

class Metrics:
    def update(self, res: StepResult) -> None: ...
    def summary(self) -> dict:
        """必须包含:
        hits, shots, hit_rate, mean_target_dist_px, time_to_first_hit_ms,
        frames, fps_loop, acq_curve (list[float], 分箱命中率)
        """
```

**靶场约束:** 靶为随机散布的圆;准星以 `cfg.speed_px_per_action` 移动;开火判定为"准星在靶半径内"。必须暴露 `render()` 供可视化。

---

## 3. 对照实验契约(Phase 3,三方共用)

同一靶场、同一 `seed` 列表、同一 `frames` 预算下跑四路:

| 名称 | 定义 | 期望语义 |
|---|---|---|
| `fly` | 真连接组 + 读出 | 实验组 |
| `shuffle` | **连接组权重随机重排**(保持边数与权重分布,打乱拓扑) | 零模型 |
| `pid` | 目标角误差 → 比例控制器(理想化,允许直接读靶位) | 性能上限参照 |
| `random` | 均匀随机鼠标增量 | 下界 |

**判定规则(预先注册,不得事后修改):**
- 若 `fly` 的 `hit_rate` **不显著优于** `shuffle`(以 seed 间配对检验判定),则结论为
  **"该接线图对本任务无因果贡献"**,如实报告。
- `pid` 仅作刻度参照,**不参与**"果蝇是否有效"的判定。
- 采样数 ≥ 10 个种子。报告必须附效应量与置信区间,不能只报均值。

---

## 4. 产出物清单

```
flyaim/data/build/     neuron_index.parquet, manifest.json, *.npz(CSR)
flyaim/runs/<ts>/      metrics.json, config.json, acq_curve.png, 录屏
reports/RESULTS.md     三路对照结论
README.md              复现步骤
```

`manifest.json` 必须记录:神经元数、边数、各角色细胞数量、**是否发生视觉输入降级**、数据文件校验和。

---

## 5. 禁止事项

1. 禁止用靶位坐标作为 `Retina` 输入(仅像素)。
2. 禁止修改连接组权重来"提升成绩";学习只能发生在 `Readout`。
3. 禁止在对比不同条件时更换靶场 seed、帧预算或评分口径。
4. 禁止把 `pid` 的成绩当作"果蝇的成绩"对外表述。
5. 禁止在结果不利时删除或隐藏零模型数据。
