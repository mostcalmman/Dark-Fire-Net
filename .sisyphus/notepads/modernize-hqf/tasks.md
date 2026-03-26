# Modernize HQF - Task Breakdown

## Wave 1: Independent Fixes (Max Parallel)

### Task 1: Update requirements.txt
- [ ] Replace current minimal requirements.txt with modern versioned dependencies
- [ ] Include: torch>=2.0.0, torchvision>=0.15.0, numpy>=1.24.0, scipy>=1.14.0, pandas>=2.0.0
- [ ] Include: matplotlib>=3.7.0, opencv-python>=4.8.0, tqdm>=4.65.0, h5py>=3.10.0
- [ ] Include: lpips>=0.1.4, tensorboard>=2.13.0, scikit-image>=0.21.0
- [ ] Verify NO ROS packages included

### Task 2: Fix NumPy deprecated type aliases
- [ ] data_loader/dataset.py line 139: np.int → np.int64 (2 occurrences)
- [ ] events_contrast_maximization/tools/txt_to_h5.py line 13: np.int → np.int64 (2 occurrences)
- [ ] Verify np.double, np.float32, np.float64 left untouched

### Task 3: Fix SciPy deprecated imports
- [ ] events_contrast_maximization/utils/objectives.py line 4: scipy.ndimage.filters → scipy.ndimage
- [ ] events_contrast_maximization/utils/events_cmax.py line 6: scipy.ndimage.filters → scipy.ndimage

### Task 4: Fix pandas deprecated delim_whitespace parameter
- [ ] events_contrast_maximization/tools/txt_to_h5.py line 12: delim_whitespace=True → sep=r'\s+'

### Task 5: Fix Python 2 metaclass syntax
- [ ] events_contrast_maximization/tools/event_packagers.py line 6: class packager() → class packager(metaclass=ABCMeta)
- [ ] events_contrast_maximization/tools/event_packagers.py line 8: Remove __metaclass__ = ABCMeta
- [ ] events_contrast_maximization/tools/event_packagers.py lines 44-47,63,70: Remove redundant np.dtype() wrappers

### Task 6: Fix Python 2 implicit relative imports
- [ ] events_contrast_maximization/utils/warps.py line 3: from event_utils → from .event_utils
- [ ] events_contrast_maximization/utils/objectives.py line 3: from event_utils → from .event_utils
- [ ] events_contrast_maximization/utils/events_cmax.py lines 8-10: Add dots to imports + try/except fallback
- [ ] events_contrast_maximization/tools/txt_to_h5.py line 7: from event_packagers → from .event_packagers + try/except
- [ ] events_contrast_maximization/tools/rosbag_to_h5.py line 9: from event_packagers → from .event_packagers + try/except
- [ ] Create events_contrast_maximization/__init__.py
- [ ] Create events_contrast_maximization/utils/__init__.py
- [ ] Create events_contrast_maximization/tools/__init__.py

### Task 7: Fix torch.meshgrid missing indexing parameter
- [ ] utils/loss.py line 22: Add indexing='ij' to meshgrid
- [ ] utils/loss.py line 79: Add indexing='ij' to meshgrid
- [ ] utils/loss.py line 121: Add indexing='ij' to meshgrid

### Task 8: Fix torch.FloatTensor deprecation
- [ ] utils/data_augmentation.py lines 287-289: torch.FloatTensor → torch.tensor(..., dtype=torch.float32)

### Task 9: Fix .data.size() deprecation
- [ ] model/submodules.py line 197: input_.data.size()[0] → input_.size()[0]
- [ ] model/submodules.py line 198: input_.data.size()[2:] → input_.size()[2:]
- [ ] model/submodules.py line 263: input_.data.size()[0] → input_.size()[0]
- [ ] model/submodules.py line 264: input_.data.size()[2:] → input_.size()[2:]
- [ ] model/submodules.py line 349: input_.data.size()[0] → input_.size()[0]
- [ ] model/submodules.py line 350: input_.data.size()[2:] → input_.size()[2:]
- [ ] Verify model/legacy.py .data.fill_() NOT modified

### Task 10: Fix in-place transpose operations
- [ ] utils/loss.py line 26: xx.transpose_(0, 1) → xx = xx.transpose(0, 1)
- [ ] utils/loss.py line 27: yy.transpose_(0, 1) → yy = yy.transpose(0, 1)
- [ ] utils/loss.py line 81: xx.transpose_(0, 1) → xx = xx.transpose(0, 1)
- [ ] utils/loss.py line 82: yy.transpose_(0, 1) → yy = yy.transpose(0, 1)

### Task 11: Modernize TensorBoard import pattern
- [ ] logger/visualization.py lines 25-27: Update warning message to remove tensorboardX reference

### Task 12: Fix Windows paths in JSON configs
- [ ] config/dark_firenet.json line 15: \\ → / in data_file path
- [ ] config/dark_firenet.json line 98: \\ → / in save_dir path

## Wave 2: Dependent Task

### Task 13: Migrate PerceptualSimilarity to standard lpips package
- [ ] model/loss.py line 4: from PerceptualSimilarity import models → import lpips
- [ ] model/loss.py lines 95-101: Replace models.PerceptualLoss with lpips.LPIPS
- [ ] model/loss.py line 113: Verify forward() call works with new API
- [ ] Verify perceptual_loss class interface unchanged
- [ ] Verify combined_perceptual_loss unchanged
- [ ] Verify flow_perceptual_loss unchanged

## Wave FINAL: Verification

### F1: Plan Compliance Audit
- [ ] Verify all Must Have items implemented
- [ ] Verify all Must NOT Have items absent
- [ ] Check evidence files exist

### F2: Code Quality Review
- [ ] Run python -m py_compile on all modified files
- [ ] Grep for remaining deprecated patterns
- [ ] Check for unused imports

### F3: Real Manual QA
- [ ] Execute all QA scenarios from all tasks
- [ ] Test cross-task integration

### F4: Scope Fidelity Check
- [ ] Verify 1:1 spec compliance for each task
- [ ] Check for scope creep
