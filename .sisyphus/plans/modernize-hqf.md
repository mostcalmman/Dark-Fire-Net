# Modernize HQF Codebase for CUDA 12.8 + RTX 5090 + Linux

## TL;DR

> **Quick Summary**: Update the HQF (event_cnn_minimal) PyTorch codebase from Python 3.7/CUDA 10.1 era to be compatible with Linux + RTX 5090 + CUDA 12.8 + Python 3.10+ + PyTorch 2.x + NumPy 1.24+. Replace local PerceptualSimilarity package with standard `lpips`. No logic changes.
> 
> **Deliverables**:
> - All deprecated API calls replaced with modern equivalents
> - PerceptualSimilarity → lpips migration complete
> - Python 2 remnants removed (implicit imports, metaclass syntax)
> - requirements.txt updated with all dependencies
> - Config files with Windows paths converted to POSIX
> 
> **Estimated Effort**: Medium
> **Parallel Execution**: YES - 3 waves
> **Critical Path**: Task 1 (requirements.txt) → Task 10 (LPIPS migration) → Final Verification

---

## Context

### Original Request
User wants to modernize the HQF codebase (originally Python 3.7 + CUDA 10.1) to run on Linux + RTX 5090 + CUDA 12.8. The local PerceptualSimilarity package should be replaced with the standard `lpips` pip package. No original logic should be modified.

### Interview Summary
**Key Discussions**:
- Target: Linux, RTX 5090, CUDA 12.8, Python 3.10+, PyTorch 2.x, NumPy 1.24+, SciPy 1.14+, pandas 2.2+
- Pure API modernization — zero logic changes
- No test infrastructure needed — Agent QA only
- opencode environment is Windows, target environment is Linux

**Research Findings**:
- `np.double` is NOT deprecated (alias for np.float64, kept in NumPy 2.0) — no change needed
- `.float()` on tensors still works in PyTorch 2.x — not a priority fix
- `np.dstack` still works in NumPy 2.0 — no change needed
- LPIPS API differs from PerceptualSimilarity: `models.PerceptualLoss(net, use_gpu)` → `lpips.LPIPS(net=net)` with `.to(device)` for GPU
- `rosbag_to_h5.py` depends on ROS (rosbag, rospy, cv_bridge) — optional tooling, not core

### Metis Review
**Identified Gaps** (addressed):
- torch.meshgrid needs explicit `indexing='ij'` to preserve old default behavior
- LPIPS `.forward()` call returns different shape — need to check `.mean()` behavior
- events_contrast_maximization files may be run as standalone scripts — relative imports must handle this
- `torch.FloatTensor()` creates CPU tensor — replacement must also be CPU to preserve behavior
- Config paths are user-configurable — converting backslashes is minimal but necessary

---

## Work Objectives

### Core Objective
Update all deprecated API calls to modern equivalents so the codebase runs without errors or deprecation warnings on Linux + CUDA 12.8 + Python 3.10+ + PyTorch 2.x + NumPy 1.24+ + SciPy 1.14+ + pandas 2.2+.

### Concrete Deliverables
- 16 modified Python files with updated APIs
- 1 updated requirements.txt
- 2 updated JSON config files
- Zero new files created

### Definition of Done
- [ ] `python -c "from model.loss import perceptual_loss"` succeeds (LPIPS migration works)
- [ ] `python -c "from data_loader.dataset import *"` succeeds (no np.int crash)
- [ ] `python -W error::DeprecationWarning -c "import torch; torch.meshgrid(torch.arange(3), torch.arange(3), indexing='ij')"` succeeds
- [ ] `python -c "from scipy.ndimage import gaussian_filter"` succeeds
- [ ] All JSON configs parse without error

### Must Have
- All `np.int` → `np.int64` replacements
- PerceptualSimilarity → lpips migration with identical interface behavior
- All Python 2 implicit relative imports → explicit relative imports
- `scipy.ndimage.filters` → `scipy.ndimage`
- `torch.meshgrid` with explicit `indexing='ij'`
- `requirements.txt` with all needed dependencies
- Windows backslash paths → forward slashes in configs

### Must NOT Have (Guardrails)
- **No logic changes**: Do not modify any algorithmic logic, loss calculations, model architectures, or data processing pipelines
- **No code reformatting**: Do not run black, autopep8, isort, or any formatter — keep original code style
- **No new features**: Do not add type hints, docstrings, error handling, or logging beyond what's minimally needed for API changes
- **No variable renaming**: Preserve all variable names, function signatures, and class names exactly
- **No new files**: Do not create test files, CI configs, documentation, or any new Python modules
- **No README updates**: Do not modify README.md or any documentation
- **No pathlib migration**: Only convert backslashes to forward slashes — do not introduce pathlib
- **No optimization**: Do not "improve" any code, even if tempting (e.g., don't replace DataParallel with DistributedDataParallel)
- **No .float() migration**: `.float()` still works in PyTorch 2.x, leave it alone
- **No np.double migration**: `np.double` is still valid, leave it alone

---

## Verification Strategy

> **ZERO HUMAN INTERVENTION** — ALL verification is agent-executed. No exceptions.

### Test Decision
- **Infrastructure exists**: NO
- **Automated tests**: None
- **Framework**: None
- **QA Policy**: Agent-executed QA scenarios only

### QA Policy
Every task MUST include agent-executed QA scenarios (see TODO template below).
Evidence saved to `.sisyphus/evidence/task-{N}-{scenario-slug}.{ext}`.

- **Library/Module**: Use Bash (python -c) — Import, call functions, compare output
- **Config files**: Use Bash (python -c) — Parse JSON, verify structure

---

## Execution Strategy

### Parallel Execution Waves

```
Wave 1 (Start Immediately — independent fixes, MAX PARALLEL):
├── Task 1: Update requirements.txt [quick]
├── Task 2: Fix NumPy deprecated type aliases [quick]
├── Task 3: Fix SciPy deprecated imports [quick]
├── Task 4: Fix pandas deprecated API [quick]
├── Task 5: Fix Python 2 metaclass syntax [quick]
├── Task 6: Fix Python 2 implicit relative imports [quick]
├── Task 7: Fix torch.meshgrid missing indexing parameter [quick]
├── Task 8: Fix torch.FloatTensor deprecation [quick]
├── Task 9: Fix .data.size() deprecation [quick]
├── Task 10: Fix in-place transpose operations [quick]
├── Task 11: Modernize TensorBoard import pattern [quick]
└── Task 12: Fix Windows paths in JSON configs [quick]

Wave 2 (After Task 1 — needs lpips in requirements):
└── Task 13: Migrate PerceptualSimilarity to lpips [deep]

Wave FINAL (After ALL tasks — 4 parallel reviews, then user okay):
├── Task F1: Plan compliance audit (oracle)
├── Task F2: Code quality review (unspecified-high)
├── Task F3: Real manual QA (unspecified-high)
└── Task F4: Scope fidelity check (deep)
-> Present results -> Get explicit user okay
```

### Dependency Matrix

| Task | Depends On | Blocks | Wave |
|------|-----------|--------|------|
| 1 | — | 13 | 1 |
| 2 | — | F1-F4 | 1 |
| 3 | — | F1-F4 | 1 |
| 4 | — | F1-F4 | 1 |
| 5 | — | F1-F4 | 1 |
| 6 | — | F1-F4 | 1 |
| 7 | — | F1-F4 | 1 |
| 8 | — | F1-F4 | 1 |
| 9 | — | F1-F4 | 1 |
| 10 | — | F1-F4 | 1 |
| 11 | — | F1-F4 | 1 |
| 12 | — | F1-F4 | 1 |
| 13 | 1 | F1-F4 | 2 |
| F1-F4 | 1-13 | — | FINAL |

### Agent Dispatch Summary

- **Wave 1**: **12** — T1-T12 → all `quick`
- **Wave 2**: **1** — T13 → `deep`
- **FINAL**: **4** — F1 → `oracle`, F2 → `unspecified-high`, F3 → `unspecified-high`, F4 → `deep`

---

## TODOs

- [x] 1. Update requirements.txt with all dependencies

  **What to do**:
  - Replace the current minimal `requirements.txt` with a properly versioned one that includes all dependencies needed for the modernized codebase
  - Required packages: `torch>=2.0.0`, `torchvision>=0.15.0`, `numpy>=1.24.0`, `scipy>=1.14.0`, `pandas>=2.0.0`, `matplotlib>=3.7.0`, `opencv-python>=4.8.0`, `tqdm>=4.65.0`, `h5py>=3.10.0`, `lpips>=0.1.4`, `tensorboard>=2.13.0`, `scikit-image>=0.21.0`
  - Use `>=` minimum version constraints (not exact pins)
  - Do NOT include ROS packages (rosbag, rospy, cv_bridge) — those are optional tools

  **Must NOT do**:
  - Do not create pyproject.toml, setup.py, or any other packaging files
  - Do not add comments or section headers in requirements.txt
  - Do not pin torch to a specific CUDA version (user handles CUDA via install command)

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Tasks 2-12)
  - **Blocks**: Task 13
  - **Blocked By**: None

  **References**:
  - `requirements.txt` — Current file has 7 packages with no version constraints: torch, torchvision, matplotlib, opencv-python, tqdm, pandas, h5py
  - `model/loss.py:4` — Uses PerceptualSimilarity (needs `lpips` package)
  - `events_contrast_maximization/utils/objectives.py:4` — Uses scipy.ndimage (needs `scipy`)
  - `logger/visualization.py:15` — Uses torch.utils.tensorboard (needs `tensorboard`)
  - `README.md` — Lists additional deps: scikit-image, thop

  **Acceptance Criteria**:

  **QA Scenarios (MANDATORY):**

  ```
  Scenario: requirements.txt contains all necessary packages
    Tool: Bash
    Preconditions: requirements.txt has been updated
    Steps:
      1. Run: python -c "with open('requirements.txt') as f: content = f.read(); assert 'lpips' in content; assert 'scipy' in content; assert 'tensorboard' in content; assert 'scikit-image' in content; assert 'numpy' in content; print('All deps present')"
    Expected Result: Prints "All deps present"
    Failure Indicators: AssertionError for missing package
    Evidence: .sisyphus/evidence/task-1-deps-check.txt

  Scenario: requirements.txt does NOT contain ROS packages
    Tool: Bash
    Preconditions: requirements.txt has been updated
    Steps:
      1. Run: python -c "with open('requirements.txt') as f: content = f.read(); assert 'rosbag' not in content; assert 'rospy' not in content; print('No ROS deps')"
    Expected Result: Prints "No ROS deps"
    Failure Indicators: AssertionError if ROS package found
    Evidence: .sisyphus/evidence/task-1-no-ros.txt
  ```

  **Commit**: YES
  - Message: `fix(deps): update requirements.txt for modern Python/CUDA stack`
  - Files: `requirements.txt`

- [x] 2. Fix NumPy deprecated type aliases

  **What to do**:
  - In `data_loader/dataset.py` line 139: Replace `np.int` with `np.int64` (2 occurrences on same line)
    - Change: `xs.astype(np.int), ys.astype(np.int)` → `xs.astype(np.int64), ys.astype(np.int64)`
  - In `events_contrast_maximization/tools/txt_to_h5.py` line 13: Replace `np.int` with `np.int64` (2 occurrences)
    - Change: `dtype={'width': np.int, 'height': np.int}` → `dtype={'width': np.int64, 'height': np.int64}`

  **Must NOT do**:
  - Do not touch `np.double` — it is NOT deprecated (alias for np.float64)
  - Do not touch `np.float32`, `np.float64`, `np.int16`, `np.uint8` — these are specific dtypes, not deprecated aliases
  - Do not touch `np.bool_` — the underscore version is correct
  - Do not change any surrounding code or logic
  - Use `np.int64` specifically, NOT Python `int` (to preserve array dtype behavior)

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Tasks 1, 3-12)
  - **Blocks**: F1-F4
  - **Blocked By**: None

  **References**:
  - `data_loader/dataset.py:139` — `xs.astype(np.int), ys.astype(np.int)` in `get_hot_event_mask` call
  - `events_contrast_maximization/tools/txt_to_h5.py:12-13` — `dtype={'width': np.int, 'height': np.int}` in `pd.read_csv()` call
  - NumPy 1.24 changelog: `np.int` was alias for Python `int`, removed because it was confusing. `np.int64` is the explicit replacement preserving behavior.

  **Acceptance Criteria**:

  **QA Scenarios (MANDATORY):**

  ```
  Scenario: No np.int alias remains in dataset.py
    Tool: Bash
    Preconditions: dataset.py has been modified
    Steps:
      1. Run: python -c "import re; content = open('data_loader/dataset.py').read(); matches = re.findall(r'np\.int\b', content); assert len(matches) == 0, f'Found {len(matches)} np.int occurrences'; print('dataset.py clean')"
    Expected Result: Prints "dataset.py clean"
    Failure Indicators: AssertionError with count of remaining np.int
    Evidence: .sisyphus/evidence/task-2-dataset-clean.txt

  Scenario: No np.int alias remains in txt_to_h5.py
    Tool: Bash
    Preconditions: txt_to_h5.py has been modified
    Steps:
      1. Run: python -c "import re; content = open('events_contrast_maximization/tools/txt_to_h5.py').read(); matches = re.findall(r'np\.int\b', content); assert len(matches) == 0, f'Found {len(matches)} np.int occurrences'; print('txt_to_h5.py clean')"
    Expected Result: Prints "txt_to_h5.py clean"
    Failure Indicators: AssertionError
    Evidence: .sisyphus/evidence/task-2-txt-to-h5-clean.txt
  ```

  **Commit**: YES
  - Message: `fix(compat): replace deprecated np.int with np.int64`
  - Files: `data_loader/dataset.py`, `events_contrast_maximization/tools/txt_to_h5.py`

- [x] 3. Fix SciPy deprecated imports

  **What to do**:
  - In `events_contrast_maximization/utils/objectives.py` line 4: Change `from scipy.ndimage.filters import gaussian_filter` → `from scipy.ndimage import gaussian_filter`
  - In `events_contrast_maximization/utils/events_cmax.py` line 6: Change `from scipy.ndimage.filters import gaussian_filter` → `from scipy.ndimage import gaussian_filter`

  **Must NOT do**:
  - Do not change any other scipy imports
  - Do not modify any code that uses gaussian_filter (the function signature is identical)

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Tasks 1-2, 4-12)
  - **Blocks**: F1-F4
  - **Blocked By**: None

  **References**:
  - `events_contrast_maximization/utils/objectives.py:4` — `from scipy.ndimage.filters import gaussian_filter`
  - `events_contrast_maximization/utils/events_cmax.py:6` — `from scipy.ndimage.filters import gaussian_filter`
  - SciPy 1.14 changelog: `scipy.ndimage.filters` submodule removed, functions moved to `scipy.ndimage` directly

  **Acceptance Criteria**:

  **QA Scenarios (MANDATORY):**

  ```
  Scenario: scipy.ndimage.filters no longer referenced
    Tool: Bash
    Preconditions: Both files modified
    Steps:
      1. Run: python -c "import os; files = ['events_contrast_maximization/utils/objectives.py', 'events_contrast_maximization/utils/events_cmax.py']; [print(f'{f}: clean') if 'scipy.ndimage.filters' not in open(f).read() else print(f'{f}: FAIL') for f in files]"
    Expected Result: Both files print "clean"
    Failure Indicators: Any file prints "FAIL"
    Evidence: .sisyphus/evidence/task-3-scipy-clean.txt
  ```

  **Commit**: YES
  - Message: `fix(compat): update scipy.ndimage.filters to scipy.ndimage`
  - Files: `events_contrast_maximization/utils/objectives.py`, `events_contrast_maximization/utils/events_cmax.py`

- [x] 4. Fix pandas deprecated delim_whitespace parameter

  **What to do**:
  - In `events_contrast_maximization/tools/txt_to_h5.py` line 12: Replace `delim_whitespace=True` with `sep=r'\s+'`
    - Change: `pd.read_csv(txt_path, delim_whitespace=True, header=None, ...)` → `pd.read_csv(txt_path, sep=r'\s+', header=None, ...)`

  **Must NOT do**:
  - Do not modify any other pd.read_csv parameters
  - Do not change the header, names, dtype, or nrows parameters

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Tasks 1-3, 5-12)
  - **Blocks**: F1-F4
  - **Blocked By**: None

  **References**:
  - `events_contrast_maximization/tools/txt_to_h5.py:12` — `pd.read_csv(txt_path, delim_whitespace=True, header=None, names=['width', 'height'], ...)`
  - pandas 2.2 changelog: `delim_whitespace` parameter deprecated, use `sep=r'\s+'` instead (identical behavior)

  **Acceptance Criteria**:

  **QA Scenarios (MANDATORY):**

  ```
  Scenario: delim_whitespace no longer in txt_to_h5.py
    Tool: Bash
    Preconditions: txt_to_h5.py has been modified
    Steps:
      1. Run: python -c "content = open('events_contrast_maximization/tools/txt_to_h5.py').read(); assert 'delim_whitespace' not in content; assert \"sep=r'\\\\s+'\" in content or 'sep=r\"\\\\s+\"' in content; print('pandas fix ok')"
    Expected Result: Prints "pandas fix ok"
    Failure Indicators: AssertionError
    Evidence: .sisyphus/evidence/task-4-pandas-clean.txt
  ```

  **Commit**: YES (groups with commit 2 — same file)
  - Message: `fix(compat): replace deprecated pandas delim_whitespace with sep`
  - Files: `events_contrast_maximization/tools/txt_to_h5.py`

- [x] 5. Fix Python 2 metaclass syntax

  **What to do**:
  - In `events_contrast_maximization/tools/event_packagers.py`:
    - Line 6: Change `class packager():` → `class packager(metaclass=ABCMeta):`
    - Line 8: Remove the line `__metaclass__ = ABCMeta`
    - Keep the `from abc import ABCMeta, abstractmethod` import on line 1 unchanged
  - Also clean up redundant `np.dtype()` wrappers on lines 44-47, 63, 70:
    - `np.dtype(np.int16)` → `np.int16`
    - `np.dtype(np.float64)` → `np.float64`
    - `np.dtype(np.bool_)` → `np.bool_`
    - `np.dtype(np.uint8)` → `np.uint8`
    - `np.dtype(np.float32)` → `np.float32`

  **Must NOT do**:
  - Do not change the class hierarchy or method signatures
  - Do not rename the class or add ABC as a base class (use metaclass= only)
  - Do not modify the hdf5_packager subclass

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Tasks 1-4, 6-12)
  - **Blocks**: F1-F4
  - **Blocked By**: None

  **References**:
  - `events_contrast_maximization/tools/event_packagers.py:1` — `from abc import ABCMeta, abstractmethod`
  - `events_contrast_maximization/tools/event_packagers.py:6-8` — `class packager():` with `__metaclass__ = ABCMeta` on line 8
  - `events_contrast_maximization/tools/event_packagers.py:44-47,63,70` — redundant `np.dtype()` wrappers
  - Python 3 migration: `__metaclass__` attribute is ignored in Python 3; must use `class X(metaclass=ABCMeta)` syntax

  **Acceptance Criteria**:

  **QA Scenarios (MANDATORY):**

  ```
  Scenario: metaclass syntax is Python 3 compatible
    Tool: Bash
    Preconditions: event_packagers.py has been modified
    Steps:
      1. Run: python -c "content = open('events_contrast_maximization/tools/event_packagers.py').read(); assert '__metaclass__' not in content; assert 'metaclass=ABCMeta' in content; print('metaclass fix ok')"
    Expected Result: Prints "metaclass fix ok"
    Failure Indicators: AssertionError
    Evidence: .sisyphus/evidence/task-5-metaclass-clean.txt

  Scenario: redundant np.dtype wrappers removed
    Tool: Bash
    Preconditions: event_packagers.py has been modified
    Steps:
      1. Run: python -c "content = open('events_contrast_maximization/tools/event_packagers.py').read(); assert 'np.dtype(' not in content; print('dtype wrappers removed')"
    Expected Result: Prints "dtype wrappers removed"
    Failure Indicators: AssertionError
    Evidence: .sisyphus/evidence/task-5-dtype-clean.txt
  ```

  **Commit**: YES
  - Message: `fix(compat): update Python 2 metaclass to Python 3 syntax`
  - Files: `events_contrast_maximization/tools/event_packagers.py`

- [x] 6. Fix Python 2 implicit relative imports in events_contrast_maximization

  **What to do**:
  - Convert all implicit relative imports to explicit relative imports (using `.` prefix) in these files:
  - `events_contrast_maximization/utils/warps.py` line 3: `from event_utils import *` → `from .event_utils import *`
  - `events_contrast_maximization/utils/objectives.py` line 3: `from event_utils import *` → `from .event_utils import *`
  - `events_contrast_maximization/utils/events_cmax.py` lines 8-10:
    - `from event_utils import *` → `from .event_utils import *`
    - `from objectives import *` → `from .objectives import *`
    - `from warps import *` → `from .warps import *`
  - `events_contrast_maximization/tools/txt_to_h5.py` line 7: `from event_packagers import *` → `from .event_packagers import *`
  - `events_contrast_maximization/tools/rosbag_to_h5.py` line 9: `from event_packagers import *` → `from .event_packagers import *`
  - Create `__init__.py` files if they don't exist in:
    - `events_contrast_maximization/__init__.py`
    - `events_contrast_maximization/utils/__init__.py`
    - `events_contrast_maximization/tools/__init__.py`
  - IMPORTANT: Some of these files (events_cmax.py, rosbag_to_h5.py, txt_to_h5.py) have `if __name__ == '__main__':` blocks that run them as scripts. Relative imports break when running as scripts. For these files, use a try/except pattern:
    ```python
    try:
        from .event_utils import *
    except ImportError:
        from event_utils import *
    ```
    This preserves both use-cases: importable as module AND runnable as script.

  **Must NOT do**:
  - Do not convert to absolute imports (would require package installation)
  - Do not reorganize import order
  - Do not modify any code below the import block
  - Do not create __init__.py with any content beyond empty (or minimal `# noqa` if needed)

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Tasks 1-5, 7-12)
  - **Blocks**: F1-F4
  - **Blocked By**: None

  **References**:
  - `events_contrast_maximization/utils/warps.py:3` — `from event_utils import *`
  - `events_contrast_maximization/utils/objectives.py:3` — `from event_utils import *`
  - `events_contrast_maximization/utils/events_cmax.py:8-10` — three implicit relative imports
  - `events_contrast_maximization/tools/txt_to_h5.py:7` — `from event_packagers import *`
  - `events_contrast_maximization/tools/rosbag_to_h5.py:9` — `from event_packagers import *`
  - `events_contrast_maximization/utils/events_cmax.py:147-167` — has `if __name__ == '__main__':` block
  - `events_contrast_maximization/tools/txt_to_h5.py:25-116` — has `if __name__ == '__main__':` block
  - `events_contrast_maximization/tools/rosbag_to_h5.py:12-180` — has `if __name__ == '__main__':` block
  - Python 3 migration: Implicit relative imports were removed in Python 3; PEP 328 requires explicit relative or absolute imports

  **Acceptance Criteria**:

  **QA Scenarios (MANDATORY):**

  ```
  Scenario: No implicit relative imports remain in events_contrast_maximization
    Tool: Bash
    Preconditions: All files modified, __init__.py files created
    Steps:
      1. Run: python -c "
         import re
         files = [
           'events_contrast_maximization/utils/warps.py',
           'events_contrast_maximization/utils/objectives.py',
           'events_contrast_maximization/utils/events_cmax.py',
           'events_contrast_maximization/tools/txt_to_h5.py',
           'events_contrast_maximization/tools/rosbag_to_h5.py'
         ]
         for f in files:
           content = open(f).read()
           # Check no bare 'from event_utils' without dot prefix (allow try/except fallback)
           lines = content.split('\n')
           for i, line in enumerate(lines):
             stripped = line.strip()
             if stripped.startswith('from event_utils import') or stripped.startswith('from objectives import') or stripped.startswith('from warps import') or stripped.startswith('from event_packagers import'):
               # Check if this is inside an except ImportError block (fallback is OK)
               if i > 0 and 'except' not in lines[i-1] and 'except' not in lines[i-2]:
                 print(f'FAIL: {f} line {i+1}: {stripped}')
                 exit(1)
         print('All imports modernized')
         "
    Expected Result: Prints "All imports modernized"
    Failure Indicators: FAIL with file and line info
    Evidence: .sisyphus/evidence/task-6-imports-clean.txt

  Scenario: __init__.py files exist
    Tool: Bash
    Preconditions: __init__.py files created
    Steps:
      1. Run: python -c "
         import os
         paths = [
           'events_contrast_maximization/__init__.py',
           'events_contrast_maximization/utils/__init__.py',
           'events_contrast_maximization/tools/__init__.py'
         ]
         for p in paths:
           assert os.path.exists(p), f'Missing: {p}'
         print('All __init__.py exist')
         "
    Expected Result: Prints "All __init__.py exist"
    Failure Indicators: AssertionError with missing path
    Evidence: .sisyphus/evidence/task-6-init-files.txt
  ```

  **Commit**: YES
  - Message: `fix(compat): convert implicit relative imports to explicit in events_contrast_maximization`
  - Files: `events_contrast_maximization/utils/warps.py`, `events_contrast_maximization/utils/objectives.py`, `events_contrast_maximization/utils/events_cmax.py`, `events_contrast_maximization/tools/txt_to_h5.py`, `events_contrast_maximization/tools/rosbag_to_h5.py`, `events_contrast_maximization/__init__.py`, `events_contrast_maximization/utils/__init__.py`, `events_contrast_maximization/tools/__init__.py`

- [x] 7. Fix torch.meshgrid missing indexing parameter

  **What to do**:
  - In `utils/loss.py`, add `indexing='ij'` to all 3 `torch.meshgrid` calls:
    - Line 22: `torch.meshgrid(torch.arange(t_width), torch.arange(t_height))` → `torch.meshgrid(torch.arange(t_width), torch.arange(t_height), indexing='ij')`
    - Line 79: same change
    - Line 121: same change
  - Use `indexing='ij'` (NOT `'xy'`) because this is the OLD default behavior that the code relies on. The code then does `xx.transpose_(0, 1)` which assumes ij-indexing.

  **Must NOT do**:
  - Do not use `indexing='xy'` — the existing code was written for the old default which is 'ij'
  - Do not change any code around the meshgrid calls

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Tasks 1-6, 8-12)
  - **Blocks**: F1-F4
  - **Blocked By**: None

  **References**:
  - `utils/loss.py:22` — `xx, yy = torch.meshgrid(torch.arange(t_width), torch.arange(t_height))  # xx, yy -> WxH`
  - `utils/loss.py:79` — same pattern
  - `utils/loss.py:121` — same pattern
  - PyTorch 2.0 changelog: `torch.meshgrid` default changed from 'ij' to requiring explicit indexing parameter

  **Acceptance Criteria**:

  **QA Scenarios (MANDATORY):**

  ```
  Scenario: All meshgrid calls have explicit indexing
    Tool: Bash
    Preconditions: utils/loss.py has been modified
    Steps:
      1. Run: python -c "
         import re
         content = open('utils/loss.py').read()
         meshgrid_calls = re.findall(r'torch\.meshgrid\([^)]+\)', content)
         for call in meshgrid_calls:
           assert 'indexing=' in call, f'Missing indexing in: {call}'
         print(f'All {len(meshgrid_calls)} meshgrid calls have indexing param')
         "
    Expected Result: Prints "All 3 meshgrid calls have indexing param"
    Failure Indicators: AssertionError with offending call
    Evidence: .sisyphus/evidence/task-7-meshgrid-clean.txt
  ```

  **Commit**: YES (groups with commit 10 — same file)
  - Message: `fix(compat): add indexing param to torch.meshgrid calls`
  - Files: `utils/loss.py`

- [x] 8. Fix torch.FloatTensor deprecation

  **What to do**:
  - In `utils/data_augmentation.py` line 287-289: Replace `torch.FloatTensor(...)` with `torch.tensor(..., dtype=torch.float32)`
    - The original code creates a 2D rotation matrix:
      ```python
      M_original_transformed = torch.FloatTensor([[cos(angle_rad), -sin(angle_rad), 0],
                                                    [sin(angle_rad), cos(angle_rad), 0]])
      ```
    - Replace with:
      ```python
      M_original_transformed = torch.tensor([[cos(angle_rad), -sin(angle_rad), 0],
                                              [sin(angle_rad), cos(angle_rad), 0]], dtype=torch.float32)
      ```

  **Must NOT do**:
  - Do not change the matrix values or shape
  - Do not add device placement (original creates CPU tensor, keep it CPU)
  - Do not modify the surrounding `torch.inverse` or `F.affine_grid` calls

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Tasks 1-7, 9-12)
  - **Blocks**: F1-F4
  - **Blocked By**: None

  **References**:
  - `utils/data_augmentation.py:287-289` — `torch.FloatTensor([[cos(angle_rad), -sin(angle_rad), 0], [sin(angle_rad), cos(angle_rad), 0]])`
  - PyTorch 2.x deprecation: `torch.FloatTensor` class constructors deprecated in favor of `torch.tensor()` with explicit dtype

  **Acceptance Criteria**:

  **QA Scenarios (MANDATORY):**

  ```
  Scenario: No torch.FloatTensor remains in data_augmentation.py
    Tool: Bash
    Preconditions: data_augmentation.py has been modified
    Steps:
      1. Run: python -c "content = open('utils/data_augmentation.py').read(); assert 'torch.FloatTensor' not in content; assert 'torch.tensor(' in content; print('FloatTensor fix ok')"
    Expected Result: Prints "FloatTensor fix ok"
    Failure Indicators: AssertionError
    Evidence: .sisyphus/evidence/task-8-floattensor-clean.txt
  ```

  **Commit**: YES
  - Message: `fix(compat): replace torch.FloatTensor with torch.tensor`
  - Files: `utils/data_augmentation.py`

- [x] 9. Fix .data.size() deprecation in submodules.py

  **What to do**:
  - In `model/submodules.py`, replace all `.data.size()` calls with `.size()` directly:
    - Line 197: `input_.data.size()[0]` → `input_.size()[0]`
    - Line 198: `input_.data.size()[2:]` → `input_.size()[2:]`
    - Line 263: `input_.data.size()[0]` → `input_.size()[0]`
    - Line 264: `input_.data.size()[2:]` → `input_.size()[2:]`
    - Line 349: `input_.data.size()[0]` → `input_.size()[0]`
    - Line 350: `input_.data.size()[2:]` → `input_.size()[2:]`
  - These are in ConvLSTM.forward(), ConvGRU.forward(), and LightAwareConvGRU.forward()
  - `.data` was used in PyTorch < 0.4 when tensors wrapped Variable; since PyTorch 0.4, tensors ARE variables, so `.data` is unnecessary

  **Must NOT do**:
  - Do not change `self.pred.conv2d.bias.data.fill_(0.5)` in `model/legacy.py:147` — that use of `.data` is correct (modifying parameter in-place during init)
  - Do not change any other code in submodules.py

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Tasks 1-8, 10-12)
  - **Blocks**: F1-F4
  - **Blocked By**: None

  **References**:
  - `model/submodules.py:197-198` — ConvLSTM.forward(): `batch_size = input_.data.size()[0]` and `spatial_size = input_.data.size()[2:]`
  - `model/submodules.py:263-264` — ConvGRU.forward(): same pattern
  - `model/submodules.py:349-350` — LightAwareConvGRU.forward(): same pattern
  - PyTorch 0.4 migration guide: Variable and Tensor merged; `.data` access is no longer needed for size queries

  **Acceptance Criteria**:

  **QA Scenarios (MANDATORY):**

  ```
  Scenario: No .data.size() remains in submodules.py
    Tool: Bash
    Preconditions: submodules.py has been modified
    Steps:
      1. Run: python -c "content = open('model/submodules.py').read(); assert '.data.size()' not in content; print('data.size() fix ok')"
    Expected Result: Prints "data.size() fix ok"
    Failure Indicators: AssertionError
    Evidence: .sisyphus/evidence/task-9-data-size-clean.txt

  Scenario: legacy.py .data.fill_() NOT modified
    Tool: Bash
    Preconditions: Check legacy.py is untouched
    Steps:
      1. Run: python -c "content = open('model/legacy.py').read(); assert '.data.fill_(' in content; print('legacy.py preserved')"
    Expected Result: Prints "legacy.py preserved"
    Failure Indicators: AssertionError if .data.fill_ was removed
    Evidence: .sisyphus/evidence/task-9-legacy-preserved.txt
  ```

  **Commit**: YES
  - Message: `fix(compat): replace .data.size() with .size() in submodules`
  - Files: `model/submodules.py`

- [x] 10. Fix in-place transpose operations

  **What to do**:
  - In `utils/loss.py`, replace in-place `transpose_()` with non-in-place `transpose()`:
    - Line 26: `xx.transpose_(0, 1)` → `xx = xx.transpose(0, 1)`
    - Line 27: `yy.transpose_(0, 1)` → `yy = yy.transpose(0, 1)`
    - Line 81: `xx.transpose_(0, 1)` → `xx = xx.transpose(0, 1)`
    - Line 82: `yy.transpose_(0, 1)` → `yy = yy.transpose(0, 1)`
  - Note: lines 121+ do NOT have `transpose_()` calls (different code path), so leave those alone

  **Must NOT do**:
  - Do not change any other code in the function
  - Do not change the meshgrid calls (that's Task 7)

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Tasks 1-9, 11-12)
  - **Blocks**: F1-F4
  - **Blocked By**: None

  **References**:
  - `utils/loss.py:26-27` — `xx.transpose_(0, 1)` and `yy.transpose_(0, 1)` in first function
  - `utils/loss.py:81-82` — same pattern in second function
  - PyTorch 2.x: In-place `transpose_()` may cause issues with autograd and is discouraged; non-in-place `transpose()` is preferred

  **Acceptance Criteria**:

  **QA Scenarios (MANDATORY):**

  ```
  Scenario: No in-place transpose_ remains in loss.py
    Tool: Bash
    Preconditions: loss.py has been modified
    Steps:
      1. Run: python -c "content = open('utils/loss.py').read(); assert '.transpose_(0, 1)' not in content; assert 'xx = xx.transpose(0, 1)' in content; print('transpose fix ok')"
    Expected Result: Prints "transpose fix ok"
    Failure Indicators: AssertionError
    Evidence: .sisyphus/evidence/task-10-transpose-clean.txt
  ```

  **Commit**: YES (groups with Task 7 — same file)
  - Message: `fix(compat): replace in-place transpose_ with transpose in loss.py`
  - Files: `utils/loss.py`

- [x] 11. Modernize TensorBoard import pattern

  **What to do**:
  - In `logger/visualization.py` lines 15-27: Simplify the import fallback to prioritize `torch.utils.tensorboard` and update the warning message
  - Current code tries both `torch.utils.tensorboard` and `tensorboardX` with a fallback message mentioning `tensorboardX`
  - Update the warning message (line 25-27) to remove the reference to tensorboardX installation:
    - Change: `"Please install TensorboardX with 'pip install tensorboardx', upgrade PyTorch to version >= 1.1 to use 'torch.utils.tensorboard' or turn off the option in the 'config.json' file."`
    - To: `"Please install tensorboard with 'pip install tensorboard' or turn off the option in the 'config.json' file."`
  - Keep the fallback mechanism itself (it still works and is harmless)

  **Must NOT do**:
  - Do not remove the fallback to tensorboardX (it's harmless and provides backward compat)
  - Do not change the SummaryWriter usage or the wrapper class logic
  - Do not add new imports at the top of the file

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Tasks 1-10, 12)
  - **Blocks**: F1-F4
  - **Blocked By**: None

  **References**:
  - `logger/visualization.py:15-27` — Current fallback pattern with outdated warning message
  - PyTorch 1.1+: `torch.utils.tensorboard.SummaryWriter` is the standard, no need for tensorboardX

  **Acceptance Criteria**:

  **QA Scenarios (MANDATORY):**

  ```
  Scenario: Warning message no longer recommends tensorboardX installation
    Tool: Bash
    Preconditions: visualization.py has been modified
    Steps:
      1. Run: python -c "content = open('logger/visualization.py').read(); assert 'pip install tensorboardx' not in content.lower(); assert 'pip install tensorboard' in content.lower(); print('tensorboard msg fix ok')"
    Expected Result: Prints "tensorboard msg fix ok"
    Failure Indicators: AssertionError
    Evidence: .sisyphus/evidence/task-11-tensorboard-clean.txt
  ```

  **Commit**: YES
  - Message: `fix(compat): update tensorboard import warning message`
  - Files: `logger/visualization.py`

- [x] 12. Fix Windows paths in JSON configs

  **What to do**:
  - In `config/dark_firenet.json`:
    - Line 15: `"data_file": "..\\dataset\\HQF\\"` → `"data_file": "../dataset/HQF/"`
    - Line 98: `"save_dir": ".\\checkpoints\\quick_test_HQF"` → `"save_dir": "./checkpoints/quick_test_HQF"`

  **Must NOT do**:
  - Do not change any other config values
  - Do not modify config.json, reconstruction.json, or flow.json (they use forward slashes or placeholder paths already)
  - Do not use pathlib or os.path.join
  - Do not change the structure of the JSON

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Tasks 1-11)
  - **Blocks**: F1-F4
  - **Blocked By**: None

  **References**:
  - `config/dark_firenet.json:15` — `"data_file": "..\\dataset\\HQF\\"` — Windows backslash path
  - `config/dark_firenet.json:98` — `"save_dir": ".\\checkpoints\\quick_test_HQF"` — Windows backslash path

  **Acceptance Criteria**:

  **QA Scenarios (MANDATORY):**

  ```
  Scenario: No backslash paths in dark_firenet.json
    Tool: Bash
    Preconditions: dark_firenet.json has been modified
    Steps:
      1. Run: python -c "import json; config = json.load(open('config/dark_firenet.json')); assert '\\\\' not in json.dumps(config); assert '../dataset/HQF/' in json.dumps(config); print('config paths fix ok')"
    Expected Result: Prints "config paths fix ok"
    Failure Indicators: AssertionError
    Evidence: .sisyphus/evidence/task-12-config-clean.txt

  Scenario: Config is still valid JSON
    Tool: Bash
    Preconditions: dark_firenet.json has been modified
    Steps:
      1. Run: python -c "import json; config = json.load(open('config/dark_firenet.json')); assert 'arch' in config; assert 'data_loader' in config; print('config structure ok')"
    Expected Result: Prints "config structure ok"
    Failure Indicators: json.JSONDecodeError or AssertionError
    Evidence: .sisyphus/evidence/task-12-config-valid.txt
  ```

  **Commit**: YES
  - Message: `fix(config): convert Windows backslash paths to POSIX forward slashes`
  - Files: `config/dark_firenet.json`

- [ ] 13. Migrate PerceptualSimilarity to standard lpips package

  **What to do**:
  - In `model/loss.py`, replace the PerceptualSimilarity dependency with standard `lpips`:
    - Line 4: `from PerceptualSimilarity import models` → `import lpips`
    - Line 95-101 (`perceptual_loss.__init__`): Replace the model initialization:
      ```python
      # OLD:
      self.model = models.PerceptualLoss(net=net, use_gpu=use_gpu)
      
      # NEW:
      self.model = lpips.LPIPS(net=net)
      if use_gpu:
          self.model = self.model.cuda()
      ```
    - Line 113 (`perceptual_loss.__call__`): The forward call needs adjustment. Old API: `self.model.forward(pred, target, normalize=normalize)` returns distance tensor. New lpips API: `self.model(pred, target, normalize=normalize)` also returns distance tensor. The `.forward()` call still works but using `self.model(pred, target, normalize=normalize)` is preferred. Both return the same shape. Keep using `.forward()` or change to direct call — both work.
  - Verify the lpips API compatibility:
    - `lpips.LPIPS(net='alex')` — equivalent to `models.PerceptualLoss(net='alex')`
    - `lpips.LPIPS(net='vgg')` — equivalent to `models.PerceptualLoss(net='vgg')`
    - The `normalize` parameter exists in both APIs
    - The output shape is identical: `[N, 1, 1, 1]` — the existing `.mean()` call on line 114 handles this correctly

  **Must NOT do**:
  - Do not change the `perceptual_loss` class interface (constructor signature, __call__ signature)
  - Do not change how the loss is weighted or normalized
  - Do not modify `combined_perceptual_loss`, `flow_perceptual_loss`, or any other loss class
  - Do not change the `use_gpu` parameter name (even though it's old-fashioned — keep for backward compat)
  - Do not add device auto-detection — preserve the explicit `use_gpu` flag behavior

  **Recommended Agent Profile**:
  - **Category**: `deep`
    - Reason: This is the most critical migration — API compatibility must be verified carefully
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Wave 2 (after Wave 1)
  - **Blocks**: F1-F4
  - **Blocked By**: Task 1 (lpips must be in requirements.txt)

  **References**:
  - `model/loss.py:4` — `from PerceptualSimilarity import models`
  - `model/loss.py:95-101` — `perceptual_loss.__init__()` — creates `models.PerceptualLoss(net=net, use_gpu=use_gpu)`
  - `model/loss.py:103-114` — `perceptual_loss.__call__()` — calls `self.model.forward(pred, target, normalize=normalize)` and returns `self.weight * dist.mean()`
  - `model/loss.py:8-13` — `combined_perceptual_loss` — wraps `perceptual_loss` (no changes needed here)
  - `model/loss.py:64-79` — `flow_perceptual_loss` — wraps `perceptual_loss` (no changes needed here)
  - `config/dark_firenet.json:80-83` — loss config uses `"net": "vgg"` — this is the network type used in training
  - lpips official docs: `lpips.LPIPS(net='alex'|'vgg'|'squeeze')`, call with `(img0, img1, normalize=True/False)`, returns `[N,1,1,1]` tensor

  **Acceptance Criteria**:

  **QA Scenarios (MANDATORY):**

  ```
  Scenario: LPIPS import and basic forward pass works
    Tool: Bash
    Preconditions: model/loss.py has been modified, lpips is installed
    Steps:
      1. Run: python -c "
         import torch
         import lpips
         # Test that lpips.LPIPS works like old PerceptualLoss
         model = lpips.LPIPS(net='alex')
         img1 = torch.randn(1, 3, 64, 64)
         img2 = torch.randn(1, 3, 64, 64)
         dist = model(img1, img2, normalize=True)
         assert dist.shape[0] == 1, f'Expected batch dim 1, got {dist.shape}'
         print(f'LPIPS forward ok, dist shape: {dist.shape}, value: {dist.mean().item():.4f}')
         "
    Expected Result: Prints LPIPS forward ok with shape and value
    Failure Indicators: ImportError, RuntimeError, or shape mismatch
    Evidence: .sisyphus/evidence/task-13-lpips-forward.txt

  Scenario: PerceptualSimilarity no longer referenced
    Tool: Bash
    Preconditions: model/loss.py has been modified
    Steps:
      1. Run: python -c "content = open('model/loss.py').read(); assert 'PerceptualSimilarity' not in content; assert 'import lpips' in content; print('migration clean')"
    Expected Result: Prints "migration clean"
    Failure Indicators: AssertionError
    Evidence: .sisyphus/evidence/task-13-no-perceptualsimilarity.txt

  Scenario: perceptual_loss class still works with use_gpu=False
    Tool: Bash
    Preconditions: model/loss.py has been modified
    Steps:
      1. Run: python -c "
         import sys; sys.path.insert(0, '.')
         from model.loss import perceptual_loss
         loss_fn = perceptual_loss(weight=1.0, net='alex', use_gpu=False)
         import torch
         pred = torch.randn(1, 3, 64, 64)
         target = torch.randn(1, 3, 64, 64)
         result = loss_fn(pred, target, normalize=True)
         assert isinstance(result, torch.Tensor), f'Expected Tensor, got {type(result)}'
         print(f'perceptual_loss class ok, result: {result.item():.4f}')
         "
    Expected Result: Prints perceptual_loss class ok with result value
    Failure Indicators: ImportError, AttributeError, RuntimeError
    Evidence: .sisyphus/evidence/task-13-perceptual-loss-class.txt
  ```

  **Commit**: YES
  - Message: `fix(deps): migrate PerceptualSimilarity to standard lpips package`
  - Files: `model/loss.py`

---

## Final Verification Wave

> 4 review agents run in PARALLEL. ALL must APPROVE. Present consolidated results to user and get explicit "okay" before completing.

- [ ] F1. **Plan Compliance Audit** — `oracle`
  Read the plan end-to-end. For each "Must Have": verify implementation exists (import module, check file content). For each "Must NOT Have": search codebase for forbidden patterns — reject with file:line if found. Check evidence files exist in .sisyphus/evidence/. Compare deliverables against plan.
  Output: `Must Have [N/N] | Must NOT Have [N/N] | Tasks [N/N] | VERDICT: APPROVE/REJECT`

- [ ] F2. **Code Quality Review** — `unspecified-high`
  Run `python -m py_compile` on every modified .py file. Grep for any remaining `np.int `, `scipy.ndimage.filters`, `from PerceptualSimilarity`, `__metaclass__`, `from event_utils import`, `from objectives import`, `from warps import`, `from event_packagers import`. Check for unused imports introduced by changes.
  Output: `Compile [PASS/FAIL] | Deprecated Patterns [N remaining] | VERDICT`

- [ ] F3. **Real Manual QA** — `unspecified-high`
  Start from clean state. Execute EVERY QA scenario from EVERY task — follow exact steps, capture evidence. Test cross-task integration: `python -c "from model.loss import perceptual_loss; from utils.loss import flow_loss; from data_loader.dataset import *"`. Save to `.sisyphus/evidence/final-qa/`.
  Output: `Scenarios [N/N pass] | Integration [N/N] | VERDICT`

- [ ] F4. **Scope Fidelity Check** — `deep`
  For each task: read "What to do", read actual diff (git diff). Verify 1:1 — everything in spec was built (no missing), nothing beyond spec was built (no creep). Check "Must NOT do" compliance: no logic changes, no code reformatting, no new files. Flag unaccounted changes.
  Output: `Tasks [N/N compliant] | Creep [CLEAN/N issues] | VERDICT`

---

## Commit Strategy

| Commit | Message | Files | Pre-commit Check |
|--------|---------|-------|-----------------|
| 1 | `fix(deps): update requirements.txt for modern Python/CUDA stack` | `requirements.txt` | `python -c "print('ok')"` |
| 2 | `fix(compat): replace deprecated np.int with np.int64` | `data_loader/dataset.py`, `events_contrast_maximization/tools/txt_to_h5.py` | `python -c "from data_loader.dataset import *"` |
| 3 | `fix(compat): update scipy.ndimage.filters to scipy.ndimage` | `events_contrast_maximization/utils/objectives.py`, `events_contrast_maximization/utils/events_cmax.py` | `python -c "from scipy.ndimage import gaussian_filter"` |
| 4 | `fix(compat): replace deprecated pandas delim_whitespace` | `events_contrast_maximization/tools/txt_to_h5.py` | `python -c "import pandas"` |
| 5 | `fix(compat): update Python 2 metaclass to Python 3 syntax` | `events_contrast_maximization/tools/event_packagers.py` | `python -c "from events_contrast_maximization.tools.event_packagers import packager"` |
| 6 | `fix(compat): convert implicit relative imports to explicit` | 6 files in `events_contrast_maximization/` | `python -m py_compile` on each |
| 7 | `fix(compat): add indexing param to torch.meshgrid calls` | `utils/loss.py` | `python -c "from utils.loss import *"` |
| 8 | `fix(compat): replace torch.FloatTensor with torch.tensor` | `utils/data_augmentation.py` | `python -c "from utils.data_augmentation import *"` |
| 9 | `fix(compat): replace .data.size() with .size()` | `model/submodules.py` | `python -c "from model.submodules import *"` |
| 10 | `fix(compat): replace in-place transpose_ with transpose` | `utils/loss.py` | `python -c "from utils.loss import *"` |
| 11 | `fix(compat): simplify tensorboard import to torch.utils.tensorboard` | `logger/visualization.py` | `python -c "from logger.visualization import *"` |
| 12 | `fix(config): convert Windows backslash paths to POSIX` | `config/dark_firenet.json` | `python -c "import json; json.load(open('config/dark_firenet.json'))"` |
| 13 | `fix(deps): migrate PerceptualSimilarity to standard lpips package` | `model/loss.py` | `python -c "import lpips"` |

---

## Success Criteria

### Verification Commands
```bash
python -c "from model.loss import perceptual_loss, combined_perceptual_loss, flow_perceptual_loss"  # Expected: no error
python -c "from data_loader.dataset import *"  # Expected: no error
python -c "from utils.loss import *"  # Expected: no error
python -c "from model.submodules import *"  # Expected: no error
python -c "import json; [json.load(open(f'config/{c}')) for c in ['dark_firenet.json','config.json','reconstruction.json','flow.json']]"  # Expected: no error
python -c "from scipy.ndimage import gaussian_filter; print('ok')"  # Expected: ok
python -c "import lpips; print('ok')"  # Expected: ok
```

### Final Checklist
- [ ] All "Must Have" present
- [ ] All "Must NOT Have" absent
- [ ] Zero deprecated API patterns remain in modified files
- [ ] requirements.txt includes all dependencies
- [ ] All configs parse as valid JSON
