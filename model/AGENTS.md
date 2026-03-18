# MODEL AGENTS

## OVERVIEW
Core PyTorch neural network architectures for event-based optical flow and image reconstruction.

## STRUCTURE
- `base/`: Abstract `BaseModel` and foundational classes.
- `unet.py`: UNet-based architectures (Recurrent, WNet, UNetFlow, UNetRecurrent).
- `submodules.py`: Building blocks (ConvLayer, ResidualBlock, ConvLSTM, ConvGRU).
- `model.py`: High-level model wrappers (ColorNet, FlowNet, E2VIDRecurrent, EVFlowNet, FireNet).

## WHERE TO LOOK
| Component | Location | Role |
|-----------|----------|------|
| Base Logic | `base/base_model.py` | State management and saving/loading |
| Recurrent Blocks | `submodules.py` | LSTM/GRU implementations for event data |
| Flow Estimation | `model.py`, `unet.py` | `FlowNet`, `UNetFlow`, `WNet` |
| Reconstruction | `model.py`, `unet.py` | `E2VIDRecurrent`, `FireNet`, `UNetRecurrent` |
| Color Events | `model.py` | `ColorNet` for RGBW event splitting |

## CONVENTIONS
- **Stateful Models**: Recurrent models (LSTM/GRU) must implement `reset_states()` and `states` property.
- **Flexible Backends**: UNets support both `UpsampleConvLayer` (no artifacts) and `TransposedConvLayer` (fast).
- **Skip Connections**: Supports `sum` or `concat` skip types via `model_util.py`.

## ANTI-PATTERNS
- Do not add training or data-loading logic inside model classes.
- Avoid hardcoding input channels; use `num_bins` from configuration.
- Never bypass the `states` setter/getter in recurrent models when doing inference.