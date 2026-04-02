import torch
import torch.nn as nn
import torch.nn.functional as f
from torch.nn import init


class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, activation='relu', norm=None,
                 BN_momentum=0.1):
        super(ConvLayer, self).__init__()

        bias = False if norm == 'BN' else True
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        if activation is not None:
            self.activation = getattr(torch, activation)
        else:
            self.activation = None

        self.norm = norm
        if norm == 'BN':
            self.norm_layer = nn.BatchNorm2d(out_channels, momentum=BN_momentum)
        elif norm == 'IN':
            self.norm_layer = nn.InstanceNorm2d(out_channels, track_running_stats=True)

    def forward(self, x):
        out = self.conv2d(x)

        if self.norm in ['BN', 'IN']:
            out = self.norm_layer(out)

        if self.activation is not None:
            out = self.activation(out)

        return out


class TransposedConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, activation='relu', norm=None):
        super(TransposedConvLayer, self).__init__()

        bias = False if norm == 'BN' else True
        self.transposed_conv2d = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size, stride=2, padding=padding, output_padding=1, bias=bias)

        if activation is not None:
            self.activation = getattr(torch, activation)
        else:
            self.activation = None

        self.norm = norm
        if norm == 'BN':
            self.norm_layer = nn.BatchNorm2d(out_channels)
        elif norm == 'IN':
            self.norm_layer = nn.InstanceNorm2d(out_channels, track_running_stats=True)

    def forward(self, x):
        out = self.transposed_conv2d(x)

        if self.norm in ['BN', 'IN']:
            out = self.norm_layer(out)

        if self.activation is not None:
            out = self.activation(out)

        return out


class UpsampleConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, activation='relu', norm=None):
        super(UpsampleConvLayer, self).__init__()

        bias = False if norm == 'BN' else True
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)

        if activation is not None:
            self.activation = getattr(torch, activation)
        else:
            self.activation = None

        self.norm = norm
        if norm == 'BN':
            self.norm_layer = nn.BatchNorm2d(out_channels)
        elif norm == 'IN':
            self.norm_layer = nn.InstanceNorm2d(out_channels, track_running_stats=True)

    def forward(self, x):
        x_upsampled = f.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        out = self.conv2d(x_upsampled)

        if self.norm in ['BN', 'IN']:
            out = self.norm_layer(out)

        if self.activation is not None:
            out = self.activation(out)

        return out


class RecurrentConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0,
                 recurrent_block_type='convlstm', activation='relu', norm=None, BN_momentum=0.1):
        super(RecurrentConvLayer, self).__init__()

        assert(recurrent_block_type in ['convlstm', 'convgru'])
        self.recurrent_block_type = recurrent_block_type
        if self.recurrent_block_type == 'convlstm':
            RecurrentBlock = ConvLSTM
        else:
            RecurrentBlock = ConvGRU
        self.conv = ConvLayer(in_channels, out_channels, kernel_size, stride, padding, activation, norm,
                              BN_momentum=BN_momentum)
        self.recurrent_block = RecurrentBlock(input_size=out_channels, hidden_size=out_channels, kernel_size=3)

    def forward(self, x, prev_state):
        x = self.conv(x)
        state = self.recurrent_block(x, prev_state)
        x = state[0] if self.recurrent_block_type == 'convlstm' else state
        return x, state


class DownsampleRecurrentConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, recurrent_block_type='convlstm', padding=0, activation='relu'):
        super(DownsampleRecurrentConvLayer, self).__init__()

        self.activation = getattr(torch, activation)

        assert(recurrent_block_type in ['convlstm', 'convgru'])
        self.recurrent_block_type = recurrent_block_type
        if self.recurrent_block_type == 'convlstm':
            RecurrentBlock = ConvLSTM
        else:
            RecurrentBlock = ConvGRU
        self.recurrent_block = RecurrentBlock(input_size=in_channels, hidden_size=out_channels, kernel_size=kernel_size)

    def forward(self, x, prev_state):
        state = self.recurrent_block(x, prev_state)
        x = state[0] if self.recurrent_block_type == 'convlstm' else state
        x = f.interpolate(x, scale_factor=0.5, mode='bilinear', align_corners=False)
        return self.activation(x), state


# Residual block
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None, norm=None,
                 BN_momentum=0.1):
        super(ResidualBlock, self).__init__()
        bias = False if norm == 'BN' else True
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=bias)
        self.norm = norm
        if norm == 'BN':
            self.bn1 = nn.BatchNorm2d(out_channels, momentum=BN_momentum)
            self.bn2 = nn.BatchNorm2d(out_channels, momentum=BN_momentum)
        elif norm == 'IN':
            self.bn1 = nn.InstanceNorm2d(out_channels)
            self.bn2 = nn.InstanceNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=bias)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        if self.norm in ['BN', 'IN']:
            out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        if self.norm in ['BN', 'IN']:
            out = self.bn2(out)

        if self.downsample:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out


class ConvLSTM(nn.Module):
    """Adapted from: https://github.com/Atcold/pytorch-CortexNet/blob/master/model/ConvLSTMCell.py """

    def __init__(self, input_size, hidden_size, kernel_size):
        super(ConvLSTM, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        pad = kernel_size // 2

        # cache a tensor filled with zeros to avoid reallocating memory at each inference step if --no-recurrent is enabled
        self.zero_tensors = {}

        self.Gates = nn.Conv2d(input_size + hidden_size, 4 * hidden_size, kernel_size, padding=pad)

    def forward(self, input_, prev_state=None):

        # get batch and spatial sizes
        batch_size = input_.size()[0]
        spatial_size = input_.size()[2:]

        # generate empty prev_state, if None is provided
        if prev_state is None:

            # create the zero tensor if it has not been created already
            state_size = tuple([batch_size, self.hidden_size] + list(spatial_size))
            if state_size not in self.zero_tensors:
                # allocate a tensor with size `spatial_size`, filled with zero (if it has not been allocated already)
                self.zero_tensors[state_size] = (
                    torch.zeros(state_size, dtype=input_.dtype).to(input_.device),
                    torch.zeros(state_size, dtype=input_.dtype).to(input_.device)
                )

            prev_state = self.zero_tensors[tuple(state_size)]

        prev_hidden, prev_cell = prev_state

        # data size is [batch, channel, height, width]
        stacked_inputs = torch.cat((input_, prev_hidden), 1)
        gates = self.Gates(stacked_inputs)

        # chunk across channel dimension
        in_gate, remember_gate, out_gate, cell_gate = gates.chunk(4, 1)

        # apply sigmoid non linearity
        in_gate = torch.sigmoid(in_gate)
        remember_gate = torch.sigmoid(remember_gate)
        out_gate = torch.sigmoid(out_gate)

        # apply tanh non linearity
        cell_gate = torch.tanh(cell_gate)

        # compute current cell and hidden state
        cell = (remember_gate * prev_cell) + (in_gate * cell_gate)
        hidden = out_gate * torch.tanh(cell)

        return hidden, cell


class ConvGRU(nn.Module):
    """
    Generate a convolutional GRU cell
    Adapted from: https://github.com/jacobkimmel/pytorch_convgru/blob/master/convgru.py
    """

    def __init__(self, input_size, hidden_size, kernel_size):
        super().__init__()
        padding = kernel_size // 2
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.reset_gate = nn.Conv2d(input_size + hidden_size, hidden_size, kernel_size, padding=padding)
        self.update_gate = nn.Conv2d(input_size + hidden_size, hidden_size, kernel_size, padding=padding)
        self.out_gate = nn.Conv2d(input_size + hidden_size, hidden_size, kernel_size, padding=padding)

        init.orthogonal_(self.reset_gate.weight)
        init.orthogonal_(self.update_gate.weight)
        init.orthogonal_(self.out_gate.weight)
        init.constant_(self.reset_gate.bias, 0.)
        init.constant_(self.update_gate.bias, 0.)
        init.constant_(self.out_gate.bias, 0.)

    def forward(self, input_, prev_state):

        # get batch and spatial sizes
        batch_size = input_.size()[0]
        spatial_size = input_.size()[2:]

        # generate empty prev_state, if None is provided
        if prev_state is None:
            state_size = [batch_size, self.hidden_size] + list(spatial_size)
            prev_state = torch.zeros(state_size, dtype=input_.dtype).to(input_.device)

        # data size is [batch, channel, height, width]
        stacked_inputs = torch.cat([input_, prev_state], dim=1)
        update = torch.sigmoid(self.update_gate(stacked_inputs))
        reset = torch.sigmoid(self.reset_gate(stacked_inputs))
        out_inputs = torch.tanh(self.out_gate(torch.cat([input_, prev_state * reset], dim=1)))
        new_state = prev_state * (1 - update) + out_inputs * update

        return new_state


class RecurrentResidualLayer(nn.Module):
    def __init__(self, in_channels, out_channels,
                 recurrent_block_type='convlstm', norm=None, BN_momentum=0.1):
        super(RecurrentResidualLayer, self).__init__()

        assert(recurrent_block_type in ['convlstm', 'convgru'])
        self.recurrent_block_type = recurrent_block_type
        if self.recurrent_block_type == 'convlstm':
            RecurrentBlock = ConvLSTM
        else:
            RecurrentBlock = ConvGRU
        self.conv = ResidualBlock(in_channels=in_channels,
                                  out_channels=out_channels,
                                  norm=norm,
                                  BN_momentum=BN_momentum)
        self.recurrent_block = RecurrentBlock(input_size=out_channels,
                                              hidden_size=out_channels,
                                              kernel_size=3)

    def forward(self, x, prev_state):
        x = self.conv(x)
        state = self.recurrent_block(x, prev_state)
        x = state[0] if self.recurrent_block_type == 'convlstm' else state
        return x, state


class LightAwareConvGRU(nn.Module):
    """
    Light-aware Convolutional GRU for low-light event-to-video reconstruction.
    Modulates the update gate based on event density (proxy for illumination):
    - High density (bright scene): update gate operates normally
    - Low density (dark scene): update gate is suppressed → more temporal memory retained

    Interface is identical to ConvGRU: forward(input_, prev_state) -> new_state tensor.
    """

    def __init__(self, input_size, hidden_size, kernel_size):
        super().__init__()
        padding = kernel_size // 2
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Standard GRU gates (identical structure to ConvGRU)
        self.reset_gate  = nn.Conv2d(input_size + hidden_size, hidden_size, kernel_size, padding=padding)
        self.update_gate = nn.Conv2d(input_size + hidden_size, hidden_size, kernel_size, padding=padding)
        self.out_gate    = nn.Conv2d(input_size + hidden_size, hidden_size, kernel_size, padding=padding)

        # Density modulation: learns a mapping from normalized density map [B,1,H,W] to update gate scale
        # Initialize bias to +2.0 so sigmoid starts near 0.88, avoiding heavily suppressed updates
        self.density_gate = nn.Conv2d(1, 1, kernel_size=1)
        init.constant_(self.density_gate.bias, 2.0)

        # Weight initialization (identical to ConvGRU)
        init.orthogonal_(self.reset_gate.weight)
        init.orthogonal_(self.update_gate.weight)
        init.orthogonal_(self.out_gate.weight)
        init.constant_(self.reset_gate.bias, 0.)
        init.constant_(self.update_gate.bias, 0.)
        init.constant_(self.out_gate.bias, 0.)

    def forward(self, input_, prev_state):
        # get batch and spatial sizes
        batch_size = input_.size()[0]
        spatial_size = input_.size()[2:]

        # generate empty prev_state if None is provided
        if prev_state is None:
            state_size = [batch_size, self.hidden_size] + list(spatial_size)
            prev_state = torch.zeros(state_size, dtype=input_.dtype).to(input_.device)

        # === Density map: proxy for illumination ===
        # [B, num_bins, H, W] -> [B, 1, H, W], normalized to [0, 1]
        density = input_.abs().sum(dim=1, keepdim=True)
        density_max = density.view(batch_size, -1).max(dim=1)[0].view(batch_size, 1, 1, 1)
        density = density / (density_max + 1e-6)
        density_modulation = torch.sigmoid(self.density_gate(density))  # [B, 1, H, W]

        # === Standard GRU gating (identical to ConvGRU) ===
        stacked = torch.cat([input_, prev_state], dim=1)
        update  = torch.sigmoid(self.update_gate(stacked))
        reset   = torch.sigmoid(self.reset_gate(stacked))
        out_inp = torch.tanh(self.out_gate(torch.cat([input_, prev_state * reset], dim=1)))

        # === Density modulation: the ONLY difference from ConvGRU ===
        # Low density (dark) → small modulation → small update → preserve prev_state (more memory)
        # High density (bright) → large modulation → normal update
        update = update * density_modulation

        new_state = prev_state * (1 - update) + out_inp * update

        return new_state


class LAGConvGRU(nn.Module):
    """
    Original LAG (LSTM, ref/submodules.py):
        i_t = σ(...)                          # input gate
        f_t = σ(...)                          # forget gate
        α   = exp(σ(LAG_conv(x_t)))           # α ∈ (1, e) ≈ (1, 2.718)
        f_t = σ(f_t - α · i_t)               # LAG: couple forget gate with input gate
        c   = f_t · c_prev + i_t · g_t

    GRU adaptation:
        GRU update gate z ↔ LSTM input gate i (how much new info to accept)
        GRU (1 - z)       ↔ LSTM forget gate f (how much old state to keep)
        GRU: h = (1-z)·h_prev + z·candidate

        Faithful mapping of LAG to GRU:
          ref:  f_new = σ(f - α·i),  then c = f_new·c_prev + i·g
          GRU:  forget = (1 - z), so:
                forget_new = σ((1-z) - α·z)
                h = forget_new · h_prev + z · candidate

        Effect: bright scene → large z (update) → α·z large → forget drops → more new info
                dark scene  → small z (update) → α·z small → forget stays high → preserve memory
    """

    def __init__(self, input_size, hidden_size, kernel_size):
        super().__init__()
        padding = kernel_size // 2
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Standard GRU gates (identical structure to ConvGRU)
        self.reset_gate  = nn.Conv2d(input_size + hidden_size, hidden_size, kernel_size, padding=padding)
        self.update_gate = nn.Conv2d(input_size + hidden_size, hidden_size, kernel_size, padding=padding)
        self.out_gate    = nn.Conv2d(input_size + hidden_size, hidden_size, kernel_size, padding=padding)

        # LAG: learnable 1x1 conv on input features, identical to ref
        self.LAG_conv = nn.Conv2d(input_size, hidden_size, kernel_size=1, stride=1, padding=0, bias=False)

        # Weight initialization (identical to ConvGRU)
        init.orthogonal_(self.reset_gate.weight)
        init.orthogonal_(self.update_gate.weight)
        init.orthogonal_(self.out_gate.weight)
        init.constant_(self.reset_gate.bias, 0.)
        init.constant_(self.update_gate.bias, 0.)
        init.constant_(self.out_gate.bias, 0.)

    def forward(self, input_, prev_state):
        batch_size = input_.size()[0]
        spatial_size = input_.size()[2:]

        if prev_state is None:
            state_size = [batch_size, self.hidden_size] + list(spatial_size)
            prev_state = torch.zeros(state_size, dtype=input_.dtype).to(input_.device)

        # === Standard GRU gating ===
        stacked = torch.cat([input_, prev_state], dim=1)
        update = torch.sigmoid(self.update_gate(stacked))
        reset  = torch.sigmoid(self.reset_gate(stacked))
        out_inp = torch.tanh(self.out_gate(torch.cat([input_, prev_state * reset], dim=1)))

        # === LAG (Local Adaptation Gate) ===
        # Identical formula to ref: alpha = exp(sigmoid(LAG_conv(x)))
        alpha = torch.exp(torch.sigmoid(self.LAG_conv(input_)))  # α ∈ (1, e)
        # Ref LSTM: f_new = σ(f - α·i)
        # GRU equiv: forget_new = σ((1 - update) - α · update)
        forget_new = torch.sigmoid((1.0 - update) - alpha * update)

        new_state = forget_new * prev_state + update * out_inp

        return new_state


class NewLAGConvGRU(nn.Module):
    """
    详见模型主体
    """

    def __init__(self, input_size, hidden_size, kernel_size):
        super().__init__()
        padding = kernel_size // 2
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Standard GRU gates
        self.reset_gate  = nn.Conv2d(input_size + hidden_size, hidden_size, kernel_size, padding=padding)
        self.update_gate = nn.Conv2d(input_size + hidden_size, hidden_size, kernel_size, padding=padding)
        self.out_gate    = nn.Conv2d(input_size + hidden_size, hidden_size, kernel_size, padding=padding)

        # NewLAG, initilize weights and bias to 0 to correspond standard ConvGRU
        self.LAG_conv = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=True)
        init.constant_(self.LAG_conv.weight, 0.)
        init.constant_(self.LAG_conv.bias, 0.)

        # Weight initialization
        init.orthogonal_(self.reset_gate.weight)
        init.orthogonal_(self.update_gate.weight)
        init.orthogonal_(self.out_gate.weight)
        init.constant_(self.reset_gate.bias, 0.)
        init.constant_(self.update_gate.bias, 0.)
        init.constant_(self.out_gate.bias, 0.)

    def forward(self, input_, prev_state):
        batch_size = input_.size()[0]
        spatial_size = input_.size()[2:]

        if prev_state is None:
            state_size = [batch_size, self.hidden_size] + list(spatial_size)
            prev_state = torch.zeros(state_size, dtype=input_.dtype).to(input_.device)

        # === Standard GRU gating ===
        stacked = torch.cat([input_, prev_state], dim=1)
        update = torch.sigmoid(self.update_gate(stacked))
        reset  = torch.sigmoid(self.reset_gate(stacked))
        h_candidate = torch.tanh(self.out_gate(torch.cat([input_, prev_state * reset], dim=1)))

        # === NewLAG ===
        light_feat = torch.mean(input_, dim=1, keepdim=True)  # (B, 1, H, W)

        alpha = 2.0 * torch.sigmoid(self.LAG_conv(light_feat))  # (B, 1, H, W) α∈(0, 2)

        update_eff = torch.clamp(alpha * update, min=0.1, max=1.0) # broadcast (B,1,H,W) × (B,C,H,W), 至少要有10%的更新, 防止异常积累

        new_state = (1 - update_eff) * prev_state + update_eff * h_candidate

        return new_state


class NewLAGConvGRU2(nn.Module):
    """
    相比1, 把α调回了逐通道的, 采用深度可分离卷积, 参数增加不多, 效果或许更好
    把clamp下限改成了0.0, 不强制更新10%
    下一步(这里没实现)计划把prev_state也引入知道调控

    结论: 反而比1差
    """

    def __init__(self, input_size, hidden_size, kernel_size):
        super().__init__()
        padding = kernel_size // 2
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Standard GRU gates
        self.reset_gate  = nn.Conv2d(input_size + hidden_size, hidden_size, kernel_size, padding=padding)
        self.update_gate = nn.Conv2d(input_size + hidden_size, hidden_size, kernel_size, padding=padding)
        self.out_gate    = nn.Conv2d(input_size + hidden_size, hidden_size, kernel_size, padding=padding)

        # NewLAG, initilize weights and bias to 0 to correspond standard ConvGRU
        self.LAG_conv = nn.Conv2d(input_size, hidden_size, kernel_size=3, stride=1, padding=1, groups=input_size, bias=True) # change
        init.constant_(self.LAG_conv.weight, 0.)
        init.constant_(self.LAG_conv.bias, 0.)

        # Weight initialization
        init.orthogonal_(self.reset_gate.weight)
        init.orthogonal_(self.update_gate.weight)
        init.orthogonal_(self.out_gate.weight)
        init.constant_(self.reset_gate.bias, 0.)
        init.constant_(self.update_gate.bias, 0.)
        init.constant_(self.out_gate.bias, 0.)

    def forward(self, input_, prev_state):
        batch_size = input_.size()[0]
        spatial_size = input_.size()[2:]

        if prev_state is None:
            state_size = [batch_size, self.hidden_size] + list(spatial_size)
            prev_state = torch.zeros(state_size, dtype=input_.dtype).to(input_.device)

        # === Standard GRU gating ===
        stacked = torch.cat([input_, prev_state], dim=1)
        update = torch.sigmoid(self.update_gate(stacked))
        reset  = torch.sigmoid(self.reset_gate(stacked))
        h_candidate = torch.tanh(self.out_gate(torch.cat([input_, prev_state * reset], dim=1)))

        # === NewLAG ===
        # light_feat = torch.mean(input_, dim=1, keepdim=True)  # (B, 1, H, W)

        alpha = 2.0 * torch.sigmoid(self.LAG_conv(input_))  # (B, 16, H, W) α∈(0, 2) change

        update_eff = torch.clamp(alpha * update, min=0.0, max=1.0) # change

        new_state = (1 - update_eff) * prev_state + update_eff * h_candidate

        return new_state
    

class NewLAGConvGRU3(nn.Module):
    """
    只把clamp下限改成了0.05
    """

    def __init__(self, input_size, hidden_size, kernel_size):
        super().__init__()
        padding = kernel_size // 2
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Standard GRU gates
        self.reset_gate  = nn.Conv2d(input_size + hidden_size, hidden_size, kernel_size, padding=padding)
        self.update_gate = nn.Conv2d(input_size + hidden_size, hidden_size, kernel_size, padding=padding)
        self.out_gate    = nn.Conv2d(input_size + hidden_size, hidden_size, kernel_size, padding=padding)

        # NewLAG, initilize weights and bias to 0 to correspond standard ConvGRU
        self.LAG_conv = nn.Conv2d(input_size, hidden_size, kernel_size=3, stride=1, padding=1, groups=input_size, bias=True)
        init.constant_(self.LAG_conv.weight, 0.)
        init.constant_(self.LAG_conv.bias, 0.)

        # Weight initialization
        init.orthogonal_(self.reset_gate.weight)
        init.orthogonal_(self.update_gate.weight)
        init.orthogonal_(self.out_gate.weight)
        init.constant_(self.reset_gate.bias, 0.)
        init.constant_(self.update_gate.bias, 0.)
        init.constant_(self.out_gate.bias, 0.)

    def forward(self, input_, prev_state):
        batch_size = input_.size()[0]
        spatial_size = input_.size()[2:]

        if prev_state is None:
            state_size = [batch_size, self.hidden_size] + list(spatial_size)
            prev_state = torch.zeros(state_size, dtype=input_.dtype).to(input_.device)

        # === Standard GRU gating ===
        stacked = torch.cat([input_, prev_state], dim=1)
        update = torch.sigmoid(self.update_gate(stacked))
        reset  = torch.sigmoid(self.reset_gate(stacked))
        h_candidate = torch.tanh(self.out_gate(torch.cat([input_, prev_state * reset], dim=1)))

        # === NewLAG ===
        # light_feat = torch.mean(input_, dim=1, keepdim=True)  # (B, 1, H, W)

        alpha = 2.0 * torch.sigmoid(self.LAG_conv(input_))  # (B, 16, H, W) α∈(0, 2) change

        update_eff = torch.clamp(alpha * update, min=0.05, max=1.0) # change

        new_state = (1 - update_eff) * prev_state + update_eff * h_candidate

        return new_state