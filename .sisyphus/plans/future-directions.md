# DarkFireNet 后续探索方向

**生成时间**: 2026-03-20
**关联模型**: `DarkFireNet` (`model/model.py`)、`LightAwareConvGRU` (`model/submodules.py`)
**研究目标**: 暗光场景下的高效 Event-to-Video 重建

---

## 当前已实现

| 模块 | 位置 | 功能 |
|------|------|------|
| `LightAwareConvGRU` | `model/submodules.py:307` | 密度调制更新门 + SE channel attention |
| `DarkFireNet` | `model/model.py:285` | 使用 LightAwareConvGRU 的 FireNet 变体 |
| `config/dark_firenet.json` | `config/` | 对应训练配置 |

---

## 方向 A：稀疏卷积前置（替换 head 层）

### 原理

Event voxel grid 天然稀疏——大多数像素位置在任意时间窗口内没有事件，对应 voxel 值为零。标准 Dense Conv（`ConvLayer`）在这些零区域仍然执行完整的乘加运算，造成大量无效计算。Sparse Conv 只在非零位置计算，理论上可以将计算量降低到与事件密度成正比。

### 实现思路

1. 安装稀疏卷积库（二选一）：
   - [`torchsparse`](https://github.com/mit-han-lab/torchsparse)：MIT 出品，安装相对简单
   - [`MinkowskiEngine`](https://github.com/NVIDIA/MinkowskiEngine)：NVIDIA 出品，功能更全

2. 在 `DarkFireNet.__init__` 中替换 head：
   ```python
   # 原来
   self.head = ConvLayer(self.num_bins, base_num_channels, kernel_size, padding=padding)
   
   # 替换为（以 torchsparse 为例）
   import torchsparse.nn as spnn
   self.head = spnn.Conv3d(self.num_bins, base_num_channels, kernel_size=kernel_size)
   self.to_dense = spnn.ToDense()  # 稀疏→稠密，输出给后续 GRU
   ```

3. 在 `forward` 中转换数据格式：
   ```python
   def forward(self, x):
       # x: [B, num_bins, H, W] dense tensor
       x_sparse = to_sparse(x)          # 转换为稀疏张量
       x = self.head(x_sparse)          # 稀疏卷积
       x = self.to_dense(x)             # 转回稠密，后续 GRU 需要稠密输入
       x = self.G1(x, self._states[0])
       ...
   ```

### 代码切入点

- `DarkFireNet.__init__` 中的 `self.head = ConvLayer(...)`（`model/model.py:306`）
- `DarkFireNet.forward` 开头的 `x = self.head(x)`（`model/model.py:332`）
- 需要新增一个 `voxel_to_sparse()` 工具函数（可放在 `utils/` 或 `model/model_util.py`）

### 预期难点

| 难点 | 说明 |
|------|------|
| 稀疏→稠密转换开销 | `to_dense` 操作本身有开销，低密度时收益大，高密度时可能反而更慢 |
| 外部库依赖 | `torchsparse`/`MinkowskiEngine` 需要编译 CUDA 扩展，安装复杂 |
| 数据流改造 | voxel grid 需要转换为 (coords, feats) 格式，涉及数据加载器改动 |
| 批处理复杂度 | 稀疏张量的 batch 维度处理与稠密张量不同 |

### 推荐时机

**暂不实现**。工程复杂度高，建议先通过训练验证 `LightAwareConvGRU` 的效果，再考虑效率优化。如果模型效果好但推理速度成为瓶颈，再引入稀疏卷积。

---

## 方向 B：事件密度指导的自适应归一化（Illumination Conditioning）

### 原理

`LightAwareConvGRU` 已经在 GRU 内部利用了密度信息来调制更新门。但密度信息可以更系统地注入整个网络——类似于条件生成模型中的 FiLM（Feature-wise Linear Modulation）或 AdaIN（Adaptive Instance Normalization）。

核心思想：从输入 voxel 提取一个全局光照 condition 向量，用它对每层特征做仿射变换 `gamma * feat + beta`，让网络在不同光照条件下自适应调整特征分布。

### 实现思路

**轻量版（推荐先试）**：

```python
class DensityEncoder(nn.Module):
    """从 voxel 提取全局密度 condition 向量"""
    def __init__(self, num_bins, cond_dim=16):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)   # 全局平均池化
        self.fc = nn.Sequential(
            nn.Linear(num_bins, cond_dim),
            nn.ReLU(),
            nn.Linear(cond_dim, cond_dim)
        )
    
    def forward(self, x):
        # x: [B, num_bins, H, W]
        pooled = self.pool(x).view(x.size(0), -1)  # [B, num_bins]
        return self.fc(pooled)  # [B, cond_dim]

class FiLMLayer(nn.Module):
    """用 condition 向量对特征做仿射调制"""
    def __init__(self, feat_channels, cond_dim):
        super().__init__()
        self.gamma_fc = nn.Linear(cond_dim, feat_channels)
        self.beta_fc  = nn.Linear(cond_dim, feat_channels)
    
    def forward(self, feat, cond):
        # feat: [B, C, H, W], cond: [B, cond_dim]
        gamma = self.gamma_fc(cond).view(-1, feat.size(1), 1, 1)
        beta  = self.beta_fc(cond).view(-1, feat.size(1), 1, 1)
        return gamma * feat + beta
```

在 `DarkFireNet` 中使用：
```python
def __init__(self, ...):
    ...
    self.density_enc = DensityEncoder(num_bins, cond_dim=16)
    self.film1 = FiLMLayer(base_num_channels, cond_dim=16)
    self.film2 = FiLMLayer(base_num_channels, cond_dim=16)

def forward(self, x):
    cond = self.density_enc(x)          # 提取全局 condition
    x = self.head(x)
    x = self.G1(x, self._states[0])
    self._states[0] = x
    x = self.film1(x, cond)             # 注入 condition
    x = self.R1(x)
    x = self.G2(x, self._states[1])
    self._states[1] = x
    x = self.film2(x, cond)             # 注入 condition
    x = self.R2(x)
    return {'image': self.pred(x)}
```

### 代码切入点

- `DarkFireNet.__init__` 和 `forward`（`model/model.py:298-339`）
- 可新建 `model/conditioning.py` 存放 `DensityEncoder` 和 `FiLMLayer`
- 或直接在 `model/submodules.py` 末尾追加（保持现有结构）

### 预期难点

| 难点 | 说明 |
|------|------|
| 效果验证困难 | 需要暗光/亮光对比实验才能验证 conditioning 是否真正有效 |
| 与 LightAwareConvGRU 的关系 | GRU 内部已有密度调制，外部再加 FiLM 可能冗余，需要消融实验 |
| condition 信号设计 | 全局平均池化可能丢失空间信息，局部密度 map 更准确但参数更多 |

### 推荐时机

**部分推荐**。`LightAwareConvGRU` 中已经计算了 `density_modulation`，可以考虑将其传递到外层作为 condition 信号复用，避免重复计算。这是一个低成本的扩展方向。

---

## 方向 C：输入侧运动模糊（拖影）修正分支

### 原理

暗光场景下事件稀疏，相邻事件之间的时间间隔较大。当物体快速运动时，voxel grid 的各个 bin 之间会出现明显的位移，导致重建图像出现"拖影"（smearing）现象。这是 FireNet 系列模型的已知缺陷。

可以在输入侧加入一个轻量的去模糊/去拖影分支，在特征提取之前对 voxel 进行预处理。

### 实现方案

**方案 1（简单，推荐先试）**：在 `head` 前加一个 `ResidualBlock` 做特征域预处理

```python
def __init__(self, ...):
    ...
    # 在 head 前加入去拖影预处理（在 voxel 域操作）
    self.deblur = ResidualBlock(self.num_bins, self.num_bins)
    self.head = ConvLayer(self.num_bins, base_num_channels, ...)

def forward(self, x):
    x = self.deblur(x)   # 去拖影预处理
    x = self.head(x)
    ...
```

**方案 2（复杂，研究性）**：双分支结构

```
输入 voxel
    ├── 重建分支: head → G1 → R1 → G2 → R2 → pred → 重建图像
    └── 模糊估计分支: blur_head → blur_estimator → blur_kernel
                                                        ↓
                                              对重建图像做 deconvolution
```

### 代码切入点

- `DarkFireNet.__init__` 中在 `self.head` 之前新增 `self.deblur`（`model/model.py:306`）
- `DarkFireNet.forward` 开头新增 `x = self.deblur(x)`（`model/model.py:332`）
- `ResidualBlock` 已在 `model/submodules.py` 中实现，可直接复用

### 预期难点

| 难点 | 说明 |
|------|------|
| 缺少监督信号 | 没有 blur-sharp 配对的 event 数据集，难以直接监督去模糊分支 |
| 自监督设计 | 需要设计无监督/自监督损失（如时序一致性损失）来间接约束去模糊 |
| 与 GRU 状态的交互 | 去模糊后的 voxel 会改变 GRU 的输入分布，可能影响状态累积的稳定性 |
| 方案 2 的复杂度 | blur kernel 估计需要额外的网络分支和损失函数设计 |

### 推荐时机

**暂不实现**。方案 1 可以作为消融实验的一部分（加/不加 deblur block），但需要有对应的训练数据和评估指标才能验证效果。建议在有暗光数据集后再考虑。

---

## 优先级总结

| 方向 | 推荐优先级 | 理由 |
|------|-----------|------|
| B: Illumination Conditioning | ⭐⭐⭐ 高 | 代码改动小，与现有 LightAwareConvGRU 协同，可低成本验证 |
| C: 去拖影（方案 1） | ⭐⭐ 中 | 实现简单（一个 ResidualBlock），可作为消融实验 |
| A: 稀疏卷积 | ⭐ 低 | 工程复杂度高，建议等模型效果验证后再做效率优化 |
| C: 去拖影（方案 2） | ⭐ 低 | 需要专门数据集，研究成本高 |

---

## 下一步建议

1. **先训练 DarkFireNet**：用现有数据集训练，与 FireNet baseline 对比 SSIM/LPIPS 指标
2. **消融实验**：对比有/无 density modulation、有/无 SE attention 的效果
3. **再考虑扩展**：根据消融结果决定哪个方向最值得深入
