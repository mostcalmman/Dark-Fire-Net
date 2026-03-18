# PROJECT KNOWLEDGE BASE (events_contrast_maximization)

**Generated:** 2026-03-18

## OVERVIEW
Core library for event-based contrast maximization, including HDF5 event processing, warping, and reward/objective function optimization.

## WHERE TO LOOK
| Task | Location | Notes |
|------|----------|-------|
| Event Loading/Conversion | `tools/` | HDF5, rosbag, and txt/zip packagers |
| Contrast Maximization | `utils/events_cmax.py` | Main entry point for optimization |
| Objective/Reward Functions | `utils/objectives.py` | Variance, RMS, SOS, SOA, and Zhu objectives |
| Warp Functions | `utils/warps.py` | Geometric transformations (e.g., linear velocity) |
| Event Processing Utils | `utils/event_utils.py` | Voxel grid generation and binary search for HDF5 |

## CODE MAP
| Symbol | Type | Location | Role |
|--------|------|----------|------|
| `objective_function` | class | `utils/objectives.py` | Base class for contrast reward functions |
| `warp_function` | class | `utils/warps.py` | Base class for event motion compensation |
| `read_h5_events` | function | `utils/event_utils.py` | Optimized HDF5 event reading |
| `events_to_voxel_torch` | function | `utils/event_utils.py` | PyTorch-accelerated voxel grid generation |

## CONVENTIONS
- Functions suffixed with `_torch` expect `torch.tensor` inputs; otherwise `np.array`.
- HDF5 is the preferred storage format for performance; avoids loading full sequences into RAM using binary search.
- New reward functions must inherit from `objective_function` and implement `evaluate_function` (and optionally `evaluate_gradient`).

## ANTI-PATTERNS
- Do not use raw rosbags for iterative processing (extremely slow); convert to HDF5 using `tools/rosbag_to_h5.py`.
- Avoid mixing NumPy and PyTorch logic in the same processing pipeline to prevent excessive device transfers.
- Do not implement custom warps directly in `events_cmax.py`; use `utils/warps.py`.
