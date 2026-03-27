# Modernize HQF - Learnings & Conventions

## Project Conventions
- PyTorch-based project for event camera processing
- Standard Python data science structure
- Configurations driven by JSON files in `config/`
- No logic changes allowed - only API modernization

## Critical Gotchas
- `np.int` → `np.int64` (NOT Python `int`)
- `torch.meshgrid` needs `indexing='ij'` to preserve old behavior
- LPIPS API: `lpips.LPIPS(net=net)` then `.to(device)` for GPU
- Some files run as scripts AND are importable - use try/except for relative imports
- `.data.size()` → `.size()` (but `.data.fill_()` is OK for parameter init)
- `torch.FloatTensor()` → `torch.tensor(..., dtype=torch.float32)`

## Files Modified (Expected)
- requirements.txt
- data_loader/dataset.py
- events_contrast_maximization/tools/txt_to_h5.py
- events_contrast_maximization/utils/objectives.py
- events_contrast_maximization/utils/events_cmax.py
- events_contrast_maximization/tools/event_packagers.py
- events_contrast_maximization/utils/warps.py
- events_contrast_maximization/tools/rosbag_to_h5.py
- events_contrast_maximization/__init__.py (create)
- events_contrast_maximization/utils/__init__.py (create)
- events_contrast_maximization/tools/__init__.py (create)
- utils/loss.py
- utils/data_augmentation.py
- model/submodules.py
- logger/visualization.py
- config/dark_firenet.json
- model/loss.py

## Task Dependencies
- Wave 1: Tasks 1-12 (all independent, max parallel)
- Wave 2: Task 13 (depends on Task 1 for lpips in requirements)
- Wave FINAL: F1-F4 (depends on all implementation tasks)
