# PROJECT KNOWLEDGE BASE

**Generated:** 2026-03-18
**Commit:** N/A
**Branch:** main

## OVERVIEW
Python-based project for events contrast maximization. Involves model definition, trainers, data loaders, and scripts for evaluation/processing.

## STRUCTURE
```
./
├── base/                         # Base classes for models and trainers
├── config/                       # Configuration JSONs
├── data_loader/                  # Data loading pipelines
├── events_contrast_maximization/ # Core maximization algorithms & utilities
├── logger/                       # Logging and Tensorboard utilities
├── model/                        # Neural network architectures
├── scripts/                      # Evaluation and runner scripts
├── trainer/                      # Training loops
└── utils/                        # Shared utility functions
```

## WHERE TO LOOK
| Task | Location | Notes |
|------|----------|-------|
| Neural Network Models | `model/` | PyTorch architectures |
| Contrast Maximization | `events_contrast_maximization/` | Core logic |
| Helpers/Utils | `utils/` | General Python utilities |

## CODE MAP
| Symbol | Type | Location | Refs | Role |
|--------|------|----------|------|------|
| `BaseModel` | class | `base/` | - | Base architecture |
| `BaseTrainer` | class | `base/` | - | Base training loop |

## CONVENTIONS
- Standard Python data science structure.
- PyTorch used for models/trainers.
- Configurations driven by JSON files in `config/`.

## ANTI-PATTERNS (THIS PROJECT)
- Do not mix training logic in the `model/` definitions.
- Avoid placing custom maximization logic outside `events_contrast_maximization/`.
