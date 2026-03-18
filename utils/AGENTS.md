# UTILITIES KNOWLEDGE BASE

## OVERVIEW
General-purpose helper functions and classes for data processing, math, parsing, and system operations.

## WHERE TO LOOK
| Task | Location | Notes |
|------|----------|-------|
| Configuration Parsing | `parse_config.py` | Configuration loading logic |
| General Utils | `util.py` | Shared helper functions |
| Visualization/Logging | `visualization.py` | Helpers for tensorboard/logging |

## CONVENTIONS
- Keep functions pure where possible.
- Reusable logic that spans across multiple domains (models, data loaders, trainers) goes here.

## ANTI-PATTERNS (THIS PROJECT)
- Do not put domain-specific logic (like specific model architectures) in the `utils` folder.