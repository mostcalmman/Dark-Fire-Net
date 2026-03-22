# DarkFireNet: 暗光场景 Event-to-Video 模型改造

## TL;DR

> **Quick Summary**: 在不修改任何现有代码的前提下，新增 `LightAwareConvGRU`（光照感知 GRU）和 `DarkFireNet` 模型类，替换 FireNet 中的标准 GRU 为密度调制版本，使模型在暗光（低事件密度）场景下减少状态遗忘，从而改善低光重建质量。
>
> **Deliverables**:
> - `model/submodules.py` 新增 `LightAwareConvGRU` 类
> - `model/model.py` 新增 `DarkFireNet` 类
> - `config/dark_firenet.json` 新的训练配置文件
> - `.sisyphus/plans/future-directions.md` 后续探索方向文档
>
> **Estimated Effort**: Short
> **Parallel Execution**: NO - sequential (三个模块有依赖关系)
> **Critical Path**: LightAwareConvGRU → DarkFireNet → Config → 验证

---

## Context

### Original Request
用户希望基于 FireNet 架构进行改造，目标是暗光场景下的高效 Event-to-Video。
改造方向：更换模块顺序、添加自定义模块、魔改 GRU（让更新门感知光照/事件密度）。
首要任务：实现 GRU 魔改，同时规划后续探索方向。

### Interview Summary
**Key Discussions**:
- **大目标**: 暗光场景高效 Event-to-Video，针对暗光做出优化，保持高效
- **GRU 魔改**: 根据事件密度调控更新门，暗光时减少遗忘；探索全局注意力机制（SE block）
- **密度特征**: 用 voxel grid 各 bin 的事件计数均值/标准差（从输入归一化），无需额外输入
- **安全性**: 零修改现有文件，完全新增
- **开始方式**: 先做 GRU 魔改，然后给出后续方向指导

**Research Findings**:
- `ConvGRU.forward()` 返回 `new_state`（tensor），非 tuple（LSTM 才是 tuple）
- `FireNet.forward()` 调用方式: `x = self.G1(x, self._states[0]); self._states[0] = x`
- `parse_config.py` 的 `init_obj` 用 `getattr(module, class_name)` 实例化模型
- `model/__init__.py` 是空文件，`train.py` 直接 `import model as module_arch`
- 凡是在 `model/model.py` 顶层定义的类，均可被 `getattr` 找到

### 关键代码事实（执行者必读）

```python
# ConvGRU 当前实现（submodules.py:238-278）
def forward(self, input_, prev_state):
    if prev_state is None:
        prev_state = torch.zeros(state_size, ...)
    stacked_inputs = torch.cat([input_, prev_state], dim=1)
    update = torch.sigmoid(self.update_gate(stacked_inputs))
    reset  = torch.sigmoid(self.reset_gate(stacked_inputs))
    out_inputs = torch.tanh(self.out_gate(torch.cat([input_, prev_state * reset], dim=1)))
    new_state  = prev_state * (1 - update) + out_inputs * update
    return new_state  # 返回 tensor，不是 tuple！

# FireNet 调用方式（model.py:270-282）
def forward(self, x):
    x = self.head(x)
    x = self.G1(x, self._states[0])   # G1 返回 tensor
    self._states[0] = x                # 直接赋值
    x = self.R1(x)
    x = self.G2(x, self._states[1])
    self._states[1] = x
    x = self.R2(x)
    return {'image': self.pred(x)}
```

---

## Work Objectives

### Core Objective
实现 `LightAwareConvGRU`（光照感知 GRU）并在 `DarkFireNet` 中替换标准 GRU，使模型在低事件密度（暗光）场景下自适应减少状态遗忘，同时记录后续探索方向。

### Concrete Deliverables
- `model/submodules.py` 末尾追加 `LightAwareConvGRU` 类（约 60 行）
- `model/model.py` 末尾追加 `DarkFireNet` 类（约 45 行）
- `config/dark_firenet.json`（完整可运行的训练配置）
- `.sisyphus/plans/future-directions.md`（后续探索方向记录）

### Definition of Done
- [ ] `python -c "from model.model import DarkFireNet; m = DarkFireNet(); print(m)"` 无报错
- [ ] `python -c "import torch; from model.model import DarkFireNet; m = DarkFireNet(); x = torch.randn(1,5,128,128); out = m(x); print(out['image'].shape)"` 输出 `torch.Size([1, 1, 128, 128])`
- [ ] `python -c "from model.model import FireNet; m = FireNet(); print(m)"` 仍正常（原有模型未受影响）
- [ ] `config/dark_firenet.json` 通过 JSON 语法检查

### Must Have
- `LightAwareConvGRU` 与 `ConvGRU` 接口完全兼容（相同的 `__init__` 参数和 `forward` 签名）
- 密度 map 从 `input_` 动态计算，无需额外输入参数
- `DarkFireNet` 的 `states` property 和 `reset_states()` 与 FireNet 完全一致
- 零修改 `FireNet`、`ConvGRU` 等现有代码

### Must NOT Have (Guardrails)
- 不得修改 `model/model.py` 中任何现有类（FireNet, E2VIDRecurrent, FlowNet 等）
- 不得修改 `model/submodules.py` 中任何现有类
- 不得修改 `parse_config.py`、`train.py`、`inference.py`
- 不得在 `LightAwareConvGRU.__init__` 增加现有 `ConvGRU` 没有的必填参数（否则现有调用代码会报错）
- SE block 如果加入，参数量不得超过原 GRU 的 10%

---

## Verification Strategy

### Test Decision
- **Infrastructure exists**: NO
- **Automated tests**: None（仓库无测试框架）
- **Framework**: N/A

### QA Policy
所有验证均通过 Bash 命令直接执行，Agent 运行并检查输出。

---

## Execution Strategy

### Parallel Execution Waves

```
Wave 1 (Sequential — 依赖关系明确):
  Task 1: LightAwareConvGRU — submodules.py 新增模块
  Task 2: DarkFireNet — model.py 新增模型类 (依赖 Task 1)
  Task 3: dark_firenet.json — 配置文件 (依赖 Task 2 的参数名)

Wave 2 (验证 + 文档, 可并行):
  Task 4: 端到端验证 (依赖 Task 1-3)
  Task 5: future-directions.md 文档 (独立)
```

---

## TODOs

- [x] 1. `model/submodules.py`: 末尾追加 `LightAwareConvGRU` 类，实现密度调制更新门

  **What to do**:
  - 在 `submodules.py` 文件**末尾**（304行之后）追加新类，不修改任何现有代码
  - 类签名与 `ConvGRU` 完全一致：`__init__(self, input_size, hidden_size, kernel_size)`
  - 内部保留全部三个门（reset/update/out），权重初始化方式与 `ConvGRU` 一致（`init.orthogonal_`）
  - 新增一个 `density_fc`：小型 `nn.Conv2d(1, 1, 1)` 用于学习密度到门控的映射（可选缩放）
  - 新增 SE（Squeeze-Excitation）block：`nn.AdaptiveAvgPool2d(1)` + 两个 `nn.Linear`，用于对 hidden state 做 channel attention（reduction ratio = 4）

  **LightAwareConvGRU 完整设计规格**:
  ```python
  class LightAwareConvGRU(nn.Module):
      def __init__(self, input_size, hidden_size, kernel_size):
          super().__init__()
          padding = kernel_size // 2
          self.input_size = input_size
          self.hidden_size = hidden_size

          # 标准 GRU 三个门（与 ConvGRU 完全一致）
          self.reset_gate  = nn.Conv2d(input_size + hidden_size, hidden_size, kernel_size, padding=padding)
          self.update_gate = nn.Conv2d(input_size + hidden_size, hidden_size, kernel_size, padding=padding)
          self.out_gate    = nn.Conv2d(input_size + hidden_size, hidden_size, kernel_size, padding=padding)

          # 密度调制：将归一化密度 map [B,1,H,W] 映射为 update gate 的调制系数
          self.density_gate = nn.Conv2d(1, 1, kernel_size=1)  # 1x1 conv，参数极少

          # SE block：对 hidden state 做 channel attention
          r = 4  # reduction ratio
          self.se_avg   = nn.AdaptiveAvgPool2d(1)
          self.se_fc1   = nn.Linear(hidden_size, hidden_size // r, bias=False)
          self.se_fc2   = nn.Linear(hidden_size // r, hidden_size, bias=False)

          # 初始化（与 ConvGRU 一致）
          init.orthogonal_(self.reset_gate.weight)
          init.orthogonal_(self.update_gate.weight)
          init.orthogonal_(self.out_gate.weight)
          init.constant_(self.reset_gate.bias, 0.)
          init.constant_(self.update_gate.bias, 0.)
          init.constant_(self.out_gate.bias, 0.)

      def forward(self, input_, prev_state):
          batch_size = input_.data.size()[0]
          spatial_size = input_.data.size()[2:]

          if prev_state is None:
              state_size = [batch_size, self.hidden_size] + list(spatial_size)
              prev_state = torch.zeros(state_size, dtype=input_.dtype).to(input_.device)

          # === 密度 map 计算（从输入自动提取，无需额外输入）===
          # [B, num_bins, H, W] → [B, 1, H, W]，归一化到 [0,1]
          density = input_.abs().sum(dim=1, keepdim=True)
          density = density / (density.amax(dim=[-2,-1], keepdim=True) + 1e-6)
          # 学习一个从密度到 update gate 调制系数的映射（sigmoid 保证在 0-1）
          density_modulation = torch.sigmoid(self.density_gate(density))  # [B,1,H,W]

          # === 标准 GRU 门控（与 ConvGRU 完全一致）===
          stacked = torch.cat([input_, prev_state], dim=1)
          update  = torch.sigmoid(self.update_gate(stacked))
          reset   = torch.sigmoid(self.reset_gate(stacked))
          out_inp = torch.tanh(self.out_gate(torch.cat([input_, prev_state * reset], dim=1)))

          # === 密度调制：暗光(density→0)时 update→0，更多保留 prev_state ===
          update = update * density_modulation  # element-wise，低密度时压制更新

          new_state = prev_state * (1 - update) + out_inp * update

          # === SE channel attention on new_state ===
          se = self.se_avg(new_state).view(batch_size, self.hidden_size)   # [B, C]
          se = torch.relu(self.se_fc1(se))                                  # [B, C/r]
          se = torch.sigmoid(self.se_fc2(se)).view(batch_size, self.hidden_size, 1, 1)  # [B, C, 1, 1]
          new_state = new_state * se

          return new_state  # 返回 tensor（与 ConvGRU 接口完全一致）
  ```

  **Must NOT do**:
  - 不得修改 `ConvGRU` 类的任何代码
  - 不得改变 `forward` 的返回类型（必须是 tensor，不是 tuple）
  - SE block 中不得使用 `bias=True` 的 Linear（减少参数）

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Wave 1, Step 1
  - **Blocks**: Task 2, Task 3
  - **Blocked By**: None（可立即开始）

  **References**:
  - `model/submodules.py:238-278` — `ConvGRU` 现有实现，照抄门控结构和初始化方式
  - `model/submodules.py:1-5` — 已有 import（`torch`, `nn`, `f`, `init`），不需要新增 import
  - 追加位置：第 304 行之后（`RecurrentResidualLayer` 类结束之后）

  **Acceptance Criteria**:

  QA Scenarios:

  ```
  Scenario: LightAwareConvGRU 可实例化并前向传播
    Tool: Bash
    Preconditions: 在 E:\research\HQF 目录下
    Steps:
      1. python -c "import torch; from model.submodules import LightAwareConvGRU; gru = LightAwareConvGRU(16, 16, 3); x = torch.randn(1,16,32,32); s = gru(x, None); print('shape:', s.shape, 'dtype:', s.dtype)"
    Expected Result: shape: torch.Size([1, 16, 32, 32]) dtype: torch.float32
    Evidence: terminal output

  Scenario: 低密度输入时 update gate 被压制（暗光行为验证）
    Tool: Bash
    Preconditions: 在 E:\research\HQF 目录下
    Steps:
      1. python -c "
  import torch
  from model.submodules import LightAwareConvGRU
  gru = LightAwareConvGRU(16, 16, 3)
  gru.eval()
  # 高密度输入
  x_bright = torch.ones(1,16,32,32) * 10.0
  s0 = torch.randn(1,16,32,32)
  state_bright = gru(x_bright, s0)
  # 低密度输入
  x_dark = torch.ones(1,16,32,32) * 0.001
  state_dark = gru(x_dark, s0)
  # 暗光时 state 应更接近 s0（遗忘更少）
  diff_bright = (state_bright - s0).abs().mean().item()
  diff_dark   = (state_dark   - s0).abs().mean().item()
  print(f'bright diff from prev: {diff_bright:.4f}')
  print(f'dark   diff from prev: {diff_dark:.4f}')
  print('OK' if diff_dark < diff_bright else 'FAIL: dark should preserve state more')
  "
    Expected Result: 最后一行输出 OK
    Evidence: terminal output

  Scenario: 与原 ConvGRU 接口兼容（原代码调用方式不变）
    Tool: Bash
    Steps:
      1. python -c "import torch; from model.submodules import ConvGRU; gru = ConvGRU(16,16,3); x = torch.randn(1,16,32,32); s = gru(x, None); print('ConvGRU still works:', s.shape)"
    Expected Result: ConvGRU still works: torch.Size([1, 16, 32, 32])
    Evidence: terminal output
  ```

  **Commit**: YES (与 Task 2、3 合并提交)

---

- [x] 2. `model/model.py`: 末尾追加 `DarkFireNet` 类

  **What to do**:
  - 在 `model.py` 文件**末尾**（第 282 行之后）追加新类
  - 在文件顶部 import 区域追加 `from .submodules import ResidualBlock, ConvGRU, ConvLayer, LightAwareConvGRU`（修改已有 import 行，只增加 `LightAwareConvGRU`）
  - `DarkFireNet` 结构与 FireNet 完全一致，仅将 `self.G1`、`self.G2` 替换为 `LightAwareConvGRU`

  **DarkFireNet 完整设计规格**:
  ```python
  class DarkFireNet(BaseModel):
      """
      Low-light optimized version of FireNet.
      Replaces standard ConvGRU with LightAwareConvGRU, which modulates the update gate
      based on event density (proxy for illumination). In dark scenes (low event density),
      the update gate is suppressed, causing the model to retain more temporal memory
      and reduce smearing artifacts.

      Additionally uses SE (Squeeze-Excitation) channel attention on GRU hidden states
      to adaptively weight feature channels.

      Architecture: head → LightAwareGRU1 → Res1 → LightAwareGRU2 → Res2 → pred
      """
      def __init__(self, num_bins=5, base_num_channels=16, kernel_size=3, unet_kwargs={}):
          super().__init__()
          if unet_kwargs:
              num_bins = unet_kwargs.get('num_bins', num_bins)
              base_num_channels = unet_kwargs.get('base_num_channels', base_num_channels)
              kernel_size = unet_kwargs.get('kernel_size', kernel_size)
          self.num_bins = num_bins
          padding = kernel_size // 2
          self.head = ConvLayer(self.num_bins, base_num_channels, kernel_size, padding=padding)
          self.G1   = LightAwareConvGRU(base_num_channels, base_num_channels, kernel_size)
          self.R1   = ResidualBlock(base_num_channels, base_num_channels)
          self.G2   = LightAwareConvGRU(base_num_channels, base_num_channels, kernel_size)
          self.R2   = ResidualBlock(base_num_channels, base_num_channels)
          self.pred = ConvLayer(base_num_channels, out_channels=1, kernel_size=1, activation=None)
          self.num_encoders = 0
          self.num_recurrent_units = 2
          self.reset_states()

      @property
      def states(self):
          return copy_states(self._states)

      @states.setter
      def states(self, states):
          self._states = states

      def reset_states(self):
          self._states = [None] * self.num_recurrent_units

      def forward(self, x):
          """
          :param x: N x num_bins x H x W event voxel tensor
          :return: dict with 'image': N x 1 x H x W reconstructed frame
          """
          x = self.head(x)
          x = self.G1(x, self._states[0])
          self._states[0] = x
          x = self.R1(x)
          x = self.G2(x, self._states[1])
          self._states[1] = x
          x = self.R2(x)
          return {'image': self.pred(x)}
  ```

  **Must NOT do**:
  - 不得修改 `FireNet` 类或任何其他现有类
  - import 修改只能在已有的 `from .submodules import ...` 那一行上追加 `LightAwareConvGRU`，不新增 import 行

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Wave 1, Step 2
  - **Blocks**: Task 3, Task 4
  - **Blocked By**: Task 1

  **References**:
  - `model/model.py:235-282` — `FireNet` 类，照抄整体结构，仅替换 GRU 类型
  - `model/model.py:11` — 现有 import 行：`from .submodules import ResidualBlock, ConvGRU, ConvLayer`，在此行追加 `, LightAwareConvGRU`
  - `model/model.py:17-24` — `copy_states` 函数，`DarkFireNet` 的 `states` property 使用此函数

  **Acceptance Criteria**:

  QA Scenarios:

  ```
  Scenario: DarkFireNet 可实例化、前向传播正常
    Tool: Bash
    Steps:
      1. python -c "import torch; from model.model import DarkFireNet; m = DarkFireNet(); x = torch.randn(1,5,128,128); out = m(x); print(out['image'].shape)"
    Expected Result: torch.Size([1, 1, 128, 128])
    Evidence: terminal output

  Scenario: 状态管理正确（reset → forward → states 更新）
    Tool: Bash
    Steps:
      1. python -c "
  import torch
  from model.model import DarkFireNet
  m = DarkFireNet()
  m.reset_states()
  print('after reset:', m.states)
  x = torch.randn(1,5,64,64)
  out = m(x)
  states_after = m.states
  print('states[0] is None:', states_after[0] is None)
  print('states[0] shape:', states_after[0].shape)
  "
    Expected Result: after reset: [None, None] | states[0] is None: False | states[0] shape: torch.Size([1, 16, 64, 64])
    Evidence: terminal output

  Scenario: 原 FireNet 未受影响
    Tool: Bash
    Steps:
      1. python -c "import torch; from model.model import FireNet; m = FireNet(); x = torch.randn(1,5,128,128); out = m(x); print('FireNet OK:', out['image'].shape)"
    Expected Result: FireNet OK: torch.Size([1, 1, 128, 128])
    Evidence: terminal output
  ```

  **Commit**: YES (与 Task 1、3 合并提交)

---

- [x] 3. `config/dark_firenet.json`: 创建 DarkFireNet 训练配置文件

  **What to do**:
  - 以 `config/reconstruction.json` 为模板，修改以下字段：
    - `"name"`: `"dark_firenet_reconstruction"`
    - `"arch"."type"`: `"DarkFireNet"`
    - `"arch"."args"`: 使用 DarkFireNet 的直接参数（非 unet_kwargs）
    - `"data_loader"."args"."data_file"`: 填写用户本地数据集路径占位符
    - `"trainer"."save_dir"`: 填写合理的输出路径占位符
  - 删除 reconstruction.json 中不适用的 unet_kwargs 字段

  **DarkFireNet 的 arch args 格式**（参照 FireNet，非 unet_kwargs）:
  ```json
  "arch": {
      "type": "DarkFireNet",
      "args": {
          "num_bins": 5,
          "base_num_channels": 16,
          "kernel_size": 3
      }
  }
  ```
  注意：`parse_config.py` 的 `init_obj` 会把 `args` 中的 key-value 展开为 `DarkFireNet(**args)`，所以参数名必须与 `__init__` 签名完全一致。

  **Must NOT do**:
  - 不得修改 `reconstruction.json`、`flow.json`、`config.json`

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Wave 1, Step 3
  - **Blocks**: Task 4
  - **Blocked By**: Task 2（需要确认参数名）

  **References**:
  - `config/reconstruction.json` — 模板，整体结构照抄
  - `model/model.py` Task 2 实现的 `DarkFireNet.__init__` 签名 — 确认 `args` 字段名

  **Acceptance Criteria**:

  QA Scenarios:

  ```
  Scenario: JSON 语法正确
    Tool: Bash
    Steps:
      1. python -c "import json; f=open('config/dark_firenet.json'); d=json.load(f); print('arch type:', d['arch']['type'])"
    Expected Result: arch type: DarkFireNet
    Evidence: terminal output

  Scenario: parse_config 可正常解析并实例化模型
    Tool: Bash
    Steps:
      1. python -c "
  import json, types
  import model as module_arch
  config_dict = json.load(open('config/dark_firenet.json'))
  arch_cfg = config_dict['arch']
  cls = getattr(module_arch, arch_cfg['type'])
  m = cls(**arch_cfg['args'])
  print('instantiated:', type(m).__name__)
  "
    Expected Result: instantiated: DarkFireNet
    Evidence: terminal output
  ```

  **Commit**: YES (与 Task 1、2 合并提交)

---

- [x] 4. 端到端验证（综合验证 Task 1-3 的集成正确性）

  **What to do**:
  - 运行所有 QA Scenarios，确认全部通过
  - 额外检查：连续多帧前向传播，确认 state 正确累积
  - 额外检查：梯度可以反向传播（`loss.backward()` 不报错）

  **Must NOT do**:
  - 不需要完整训练，只验证代码路径正确

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES（与 Task 5 并行）
  - **Parallel Group**: Wave 2
  - **Blocks**: F1, F2
  - **Blocked By**: Task 1, 2, 3

  **References**:
  - 上述 Task 1-3 的所有 QA Scenarios
  - `model/model.py` — 查看 `FireNet.forward` 确认调用方式

  **Acceptance Criteria**:

  QA Scenarios:

  ```
  Scenario: 多帧连续推理，state 正确累积
    Tool: Bash
    Steps:
      1. python -c "
  import torch
  from model.model import DarkFireNet
  m = DarkFireNet()
  m.reset_states()
  for i in range(5):
      x = torch.randn(1,5,64,64)
      out = m(x)
  print('5-frame inference OK, last output shape:', out['image'].shape)
  s0_shape = m.states[0].shape
  s1_shape = m.states[1].shape
  print('state[0] shape:', s0_shape)
  print('state[1] shape:', s1_shape)
  "
    Expected Result: 5-frame inference OK | state[0] shape: torch.Size([1, 16, 64, 64]) | state[1] shape: torch.Size([1, 16, 64, 64])
    Evidence: terminal output

  Scenario: 梯度可正常反向传播
    Tool: Bash
    Steps:
      1. python -c "
  import torch
  from model.model import DarkFireNet
  m = DarkFireNet()
  m.reset_states()
  x = torch.randn(1,5,64,64, requires_grad=False)
  out = m(x)
  loss = out['image'].mean()
  loss.backward()
  grads = [p.grad for p in m.parameters() if p.grad is not None]
  print(f'Backward OK: {len(grads)} param groups have gradients')
  "
    Expected Result: Backward OK: N param groups have gradients（N > 0）
    Evidence: terminal output
  ```

  **Commit**: NO（验证任务不提交）

---

- [x] 5. `.sisyphus/plans/future-directions.md`: 记录后续探索方向

  **What to do**:
  - 创建文档，记录以下三个探索方向的技术分析，每个方向包含：原理、实现思路、代码切入点、预期难点

  **文档结构**:

  **方向 A: 稀疏卷积前置（替换 head 层）**
  - 原理：Event voxel grid 天然稀疏，标准 Dense Conv 在零区域浪费计算。Sparse Conv 只在有事件的位置计算。
  - 实现：使用 `torchsparse` 或 `MinkowskiEngine` 库，将 `self.head = ConvLayer(...)` 替换为稀疏卷积，输出 dense feature 给后续 GRU
  - 切入点：`DarkFireNet.__init__` 中的 `self.head` 和 `forward` 开头的 `x = self.head(x)`
  - 预期难点：稀疏到稠密的转换开销；外部库依赖安装；voxel grid 转稀疏张量的数据流改造
  - 是否推荐立即实现：No — 工程复杂度高，建议先验证 GRU 改造效果

  **方向 B: 事件密度指导的自适应归一化（illumination conditioning）**
  - 原理：从输入 voxel 提取全局/局部密度统计量，作为 condition 信号注入各层（类似 FiLM 或 AdaIN）
  - 实现：在 `DarkFireNet` 中新增一个轻量 `DensityEncoder`（2-3 层 Conv），输出 `gamma`, `beta` 参数，对每层特征做 `gamma * feat + beta`
  - 切入点：`DarkFireNet.forward` 中每层之后插入 AdaIN；或在 `LightAwareConvGRU` 内部扩展
  - 预期难点：需要较多数据验证 condition 信号是否真正有用；需要设计归一化统计量
  - 是否推荐立即实现：Partial — 密度 map 已经在 `LightAwareConvGRU` 中计算，可考虑传递到外层使用

  **方向 C: 输入侧运动模糊（拖影）修正分支**
  - 原理：暗光场景事件稀疏，时间戳间隔大，voxel 各 bin 之间可能存在运动模糊。可在输入侧加一个轻量去模糊分支。
  - 实现方案 1（简单）：在 `head` 前加一个 `ResidualBlock` 做特征域去模糊
  - 实现方案 2（复杂）：双分支结构，一路正常重建，一路估计 blur kernel，做 deconvolution
  - 切入点：`DarkFireNet.__init__` 中在 `self.head` 前新增 `self.deblur` 模块
  - 预期难点：缺少 blur-sharp 配对的 event 数据集；难以监督训练
  - 是否推荐立即实现：No — 需要专门数据集支持

  **Recommended Agent Profile**:
  - **Category**: `writing`
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES（与 Task 4 并行）
  - **Parallel Group**: Wave 2
  - **Blocks**: None
  - **Blocked By**: None（可在 Task 1 完成后立即开始）

  **References**:
  - 本计划中的所有设计决策
  - `model/submodules.py` — 现有模块供参考
  - `model/model.py` — DarkFireNet 架构供参考

  **Acceptance Criteria**:

  QA Scenarios:

  ```
  Scenario: 文档存在且包含三个方向
    Tool: Bash
    Steps:
      1. python -c "
  content = open('.sisyphus/plans/future-directions.md').read()
  assert '方向 A' in content or 'Direction A' in content, 'missing A'
  assert '方向 B' in content or 'Direction B' in content, 'missing B'
  assert '方向 C' in content or 'Direction C' in content, 'missing C'
  print('Document structure OK')
  "
    Expected Result: Document structure OK
    Evidence: terminal output
  ```

  **Commit**: YES
  - Message: `docs: add future exploration directions for DarkFireNet`
  - Files: `.sisyphus/plans/future-directions.md`

---

## Final Verification Wave

- [x] F1. **功能验证** — `quick`
  运行 Task 4 中所有 QA Scenarios，确认新模型可实例化、可前向传播、原模型未受影响。

- [x] F2. **范围合规检查** — `quick`
  用 `git diff` 确认只新增了规定文件，没有修改任何现有代码行。

---

## Commit Strategy
- **Task 1-3**: `feat(model): add LightAwareConvGRU and DarkFireNet for low-light event reconstruction`
  - Files: `model/submodules.py`, `model/model.py`, `config/dark_firenet.json`

---

## Success Criteria

### Verification Commands
```bash
# 新模型可用
python -c "import torch; from model.model import DarkFireNet; m = DarkFireNet(); x = torch.randn(1,5,128,128); out = m(x); print(out['image'].shape)"
# Expected: torch.Size([1, 1, 128, 128])

# 原模型未受影响
python -c "import torch; from model.model import FireNet; m = FireNet(); x = torch.randn(1,5,128,128); out = m(x); print(out['image'].shape)"
# Expected: torch.Size([1, 1, 128, 128])

# 状态重置正常
python -c "from model.model import DarkFireNet; m = DarkFireNet(); m.reset_states(); print('states:', m.states)"
# Expected: states: [None, None]
```
