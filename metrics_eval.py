"""
Metrics Evaluation Script
Calculates MSE, LPIPS, TC (Temporal Consistency), and SSIM between reconstructed images and GT.

Expected directory structure:
    results/
    ├── scene1/
    │   ├── rec/           # Reconstructed images (frame_*.png)
    │   ├── gt/            # Ground truth images (frame_*.png)
    ├── scene2/
    │   ├── rec/
    │   ├── gt/
    ├── scene1_flow.npz      # Flow file for scene1 (optional, required for TC)
    └── scene2_flow.npz
"""

import argparse
import os
import re
import json
from os.path import join
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings

import numpy as np
import cv2
import torch
import torch.nn.functional as F
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim_skimage

try:
    import lpips
    LPIPS_AVAILABLE = True
except ImportError:
    LPIPS_AVAILABLE = False
    warnings.warn("lpips not installed. LPIPS metric will be skipped. Install with: pip install lpips")

# Import temporal consistency loss
from utils.loss import temporal_consistency_loss


def load_image(path: str) -> np.ndarray:
    """Load image as grayscale float32 in [0, 1]."""
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not load image: {path}")
    return img.astype(np.float32) / 255.0


def load_flow_from_npz(path: str, frame_idx: int) -> Optional[np.ndarray]:
    """Load flow from .npz file for a specific frame index.
    
    Args:
        path: Path to the .npz file containing flow data
        frame_idx: Frame index to load flow for
    
    Returns:
        Flow array of shape [2, H, W] or None if not found
    """
    try:
        flow_data = np.load(path)
        if 'x_flow_dist' not in flow_data or 'y_flow_dist' not in flow_data:
            return None
        
        x_flow = flow_data['x_flow_dist']
        y_flow = flow_data['y_flow_dist']
        
        # Handle different array shapes
        # Expected: x_flow.shape = [N, H, W] where N is number of flow frames
        if x_flow.ndim == 3:
            if frame_idx < x_flow.shape[0]:
                flow = np.stack([x_flow[frame_idx], y_flow[frame_idx]], axis=0)
                return flow.astype(np.float32)
        elif x_flow.ndim == 2:
            # Single flow frame
            flow = np.stack([x_flow, y_flow], axis=0)
            return flow.astype(np.float32)
        
        return None
    except Exception as e:
        return None


def calculate_mse(img1: np.ndarray, img2: np.ndarray) -> float:
    """Calculate Mean Squared Error between two images."""
    return float(np.mean((img1 - img2) ** 2))


def calculate_ssim_skimage(img1: np.ndarray, img2: np.ndarray) -> float:
    """Calculate window-based SSIM between two images using skimage.
    
    This uses an 11x11 sliding window for local structure comparison,
    matching the standard implementation used in most papers.
    """
    # Use data_range=1.0 since images are in [0, 1]
    return float(ssim_skimage(img1, img2, data_range=1.0))


def calculate_ssim_global(img1: np.ndarray, img2: np.ndarray) -> float:
    """Calculate global SSIM between two images using mean over entire image.
    
    This matches the trainer.py implementation and computes SSIM using
    global statistics (mean/variance over the entire image) rather than
    a sliding window approach.
    """
    # Ensure images are float32 in [0, 1]
    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)
    
    # Global means
    mu1 = img1.mean()
    mu2 = img2.mean()
    
    # Centered images
    img1_centered = img1 - mu1
    img2_centered = img2 - mu2
    
    # Global variances and covariance
    var1 = (img1_centered ** 2).mean()
    var2 = (img2_centered ** 2).mean()
    cov = (img1_centered * img2_centered).mean()
    
    # SSIM constants
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    
    # SSIM formula
    numerator = (2.0 * mu1 * mu2 + c1) * (2.0 * cov + c2)
    denominator = (mu1 ** 2 + mu2 ** 2 + c1) * (var1 + var2 + c2)
    ssim = numerator / (denominator + 1e-8)
    
    return float(ssim)


def calculate_lpips(img1: np.ndarray, img2: np.ndarray, loss_fn: torch.nn.Module) -> float:
    """Calculate LPIPS between two images."""
    # Convert to torch tensors: [H, W] -> [1, 1, H, W]
    img1_t = torch.from_numpy(img1).unsqueeze(0).unsqueeze(0).float()
    img2_t = torch.from_numpy(img2).unsqueeze(0).unsqueeze(0).float()
    
    # Move to GPU if available
    if torch.cuda.is_available():
        img1_t = img1_t.cuda()
        img2_t = img2_t.cuda()
    
    # LPIPS expects input in [0, 1] with shape [N, C, H, W]
    # Duplicate grayscale to 3 channels
    img1_t = img1_t.repeat(1, 3, 1, 1)
    img2_t = img2_t.repeat(1, 3, 1, 1)
    
    with torch.no_grad():
        dist = loss_fn(img1_t, img2_t, normalize=True)
    
    return float(dist.item())


def calculate_tc_pair(
    rec0: np.ndarray, 
    rec1: np.ndarray,
    gt0: np.ndarray,
    gt1: np.ndarray,
    flow: np.ndarray
) -> float:
    """
    Calculate Temporal Consistency between two consecutive frames.
    
    Args:
        rec0, rec1: Reconstructed images at time t and t+1
        gt0, gt1: Ground truth images at time t and t+1
        flow: Flow from t+1 to t (displacement map) [2, H, W]
    
    Returns:
        TC loss value
    """
    # Convert to torch tensors [1, 1, H, W]
    rec0_t = torch.from_numpy(rec0).unsqueeze(0).unsqueeze(0).float()
    rec1_t = torch.from_numpy(rec1).unsqueeze(0).unsqueeze(0).float()
    gt0_t = torch.from_numpy(gt0).unsqueeze(0).unsqueeze(0).float()
    gt1_t = torch.from_numpy(gt1).unsqueeze(0).unsqueeze(0).float()
    flow_t = torch.from_numpy(flow).unsqueeze(0).float()  # [1, 2, H, W]
    
    if torch.cuda.is_available():
        rec0_t = rec0_t.cuda()
        rec1_t = rec1_t.cuda()
        gt0_t = gt0_t.cuda()
        gt1_t = gt1_t.cuda()
        flow_t = flow_t.cuda()
    
    # Calculate TC loss (i=1 for second frame, but we set i >= L0 manually)
    tc = temporal_consistency_loss(gt0_t, gt1_t, rec0_t, rec1_t, flow_t, alpha=50.0)
    
    return float(tc.item())  # type: ignore


def get_image_pairs(rec_dir: str, gt_dir: str) -> List[Tuple[str, str]]:
    """Get matching image pairs from rec and gt directories."""
    rec_files = sorted([f for f in os.listdir(rec_dir) if f.endswith('.png')])
    gt_files = set(os.listdir(gt_dir))
    
    pairs = []
    for rec_file in rec_files:
        if rec_file in gt_files:
            pairs.append((join(rec_dir, rec_file), join(gt_dir, rec_file)))
        else:
            print(f"Warning: No matching GT for {rec_file}")
    
    return pairs


def extract_frame_index(filename: str) -> int:
    """Extract frame index from filename like 'frame_0000000000.png'."""
    match = re.search(r'\d+', filename)
    if match:
        return int(match.group())
    return -1


def evaluate_directory(
    rec_dir: str, 
    gt_dir: str, 
    flow_file: Optional[str] = None,
    calc_lpips: bool = True,
    calc_tc: bool = False,
    save_error_maps: bool = False,
    error_map_dir: Optional[str] = None
) -> Dict[str, float]:
    """
    Evaluate metrics for a directory of images.
    
    Returns:
        Dictionary with mean metrics
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize LPIPS model (using VGG version)
    lpips_model = None
    if calc_lpips:
        if LPIPS_AVAILABLE and 'lpips' in globals():
            lpips_model = lpips.LPIPS(net='vgg').to(device)  # type: ignore
            lpips_model.eval()
        else:
            print("Warning: LPIPS not available, skipping LPIPS calculation")
    
    # Get image pairs
    pairs = get_image_pairs(rec_dir, gt_dir)
    if len(pairs) == 0:
        raise ValueError(f"No matching image pairs found in {rec_dir} and {gt_dir}")
    
    print(f"Found {len(pairs)} image pairs")
    
    # Storage for individual metrics
    mses = []
    ssims_skimage = []
    ssims_global = []
    lpips_vals = []
    tcs = []
    
    # Process each pair
    prev_rec = None
    prev_gt = None
    prev_idx = -1
    
    for i, (rec_path, gt_path) in enumerate(tqdm(pairs, desc="Calculating metrics")):
        # Load images
        rec_img = load_image(rec_path)
        gt_img = load_image(gt_path)
        
        # MSE
        mse = calculate_mse(rec_img, gt_img)
        mses.append(mse)
        
        # SSIM - both window-based (skimage) and global
        ssim_skimage_val = calculate_ssim_skimage(rec_img, gt_img)
        ssim_global_val = calculate_ssim_global(rec_img, gt_img)
        ssims_skimage.append(ssim_skimage_val)
        ssims_global.append(ssim_global_val)
        
        # LPIPS
        if lpips_model is not None:
            lpips_val = calculate_lpips(rec_img, gt_img, lpips_model)
            lpips_vals.append(lpips_val)
        
        # TC (requires consecutive frames and flow)
        if calc_tc and flow_file is not None and os.path.exists(flow_file):
            current_idx = extract_frame_index(os.path.basename(rec_path))
            
            # Check if this is consecutive to previous
            if prev_rec is not None and current_idx == prev_idx + 1:
                # Load flow from npz file for this frame
                flow = load_flow_from_npz(flow_file, current_idx)
                if flow is not None and prev_gt is not None:
                    tc = calculate_tc_pair(prev_rec, rec_img, prev_gt, gt_img, flow)
                    tcs.append(tc)
            
            prev_rec = rec_img
            prev_gt = gt_img
            prev_idx = current_idx
        
        # Save error map if requested
        if save_error_maps and error_map_dir is not None:
            error_map = np.abs(rec_img - gt_img)
            error_img = (error_map * 255).astype(np.uint8)
            applyColorMap = getattr(cv2, 'applyColorMap', None)
            if applyColorMap:
                error_colored = cv2.applyColorMap(error_img, cv2.COLORMAP_JET)
            else:
                error_colored = cv2.cvtColor(error_img, cv2.COLOR_GRAY2BGR)
            fname = os.path.basename(rec_path)
            cv2.imwrite(join(error_map_dir, fname), error_colored)
    
    # Calculate means
    # Naming convention matches trainer.py (without 'raw_' prefix):
    # - ssim: window-based SSIM (like skimage, primary metric)
    # - ssim_global: global mean-based SSIM (secondary metric)
    results = {
        'mse': float(np.mean(mses)) if mses else 0.0,
        'ssim': float(np.mean(ssims_skimage)) if ssims_skimage else 0.0,
        'ssim_global': float(np.mean(ssims_global)) if ssims_global else 0.0,
        'lpips': float(np.mean(lpips_vals)) if lpips_vals else 0.0,
        'tc': float(np.mean(tcs)) if tcs else 0.0,
        'num_frames': len(pairs),
        'num_tc_frames': len(tcs),
    }
    
    return results


def evaluate_batch(
    root_dir: str,
    calc_lpips: bool = True,
    calc_tc: bool = False,
    save_error_maps: bool = False
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate metrics for all subdirectories in root_dir.
    Each subdirectory should have rec/ and gt/ folders.
    
    If calc_tc is True, looks for {scene_name}_flow.npz in the parent directory of root_dir.
    
    Returns:
        Dictionary mapping scene name to metrics
    """
    all_results = {}
    
    # Check if root_dir directly contains rec/ and gt/
    if os.path.isdir(join(root_dir, 'rec')) and os.path.isdir(join(root_dir, 'gt')):
        print(f"Processing single scene: {root_dir}")
        if save_error_maps:
            error_map_dir = join(root_dir, 'error_maps')
            os.makedirs(error_map_dir, exist_ok=True)
        else:
            error_map_dir = None
        
        # Determine flow file path: {root_dir}_flow.npz in parent directory
        flow_file = None
        if calc_tc:
            root_dir_name = os.path.basename(os.path.normpath(root_dir))
            parent_dir = os.path.dirname(os.path.normpath(root_dir))
            flow_file = join(parent_dir, f"{root_dir_name}_flow.npz")
            if os.path.exists(flow_file):
                print(f"Using flow file: {flow_file}")
            else:
                print(f"Warning: Flow file not found: {flow_file}")
                flow_file = None
        
        results = evaluate_directory(
            join(root_dir, 'rec'),
            join(root_dir, 'gt'),
            flow_file=flow_file,
            calc_lpips=calc_lpips,
            calc_tc=calc_tc,
            save_error_maps=save_error_maps,
            error_map_dir=error_map_dir
        )
        all_results['scene'] = results
    else:
        # Process each subdirectory as a separate scene
        scenes = [d for d in os.listdir(root_dir) if os.path.isdir(join(root_dir, d))]
        scenes.sort()
        
        for scene in scenes:
            scene_path = join(root_dir, scene)
            rec_dir = join(scene_path, 'rec')
            gt_dir = join(scene_path, 'gt')
            
            if not os.path.isdir(rec_dir) or not os.path.isdir(gt_dir):
                print(f"Skipping {scene}: missing rec/ or gt/ directory")
                continue
            
            print(f"\n{'='*50}")
            print(f"Processing scene: {scene}")
            print(f"{'='*50}")
            
            if save_error_maps:
                error_map_dir = join(scene_path, 'error_maps')
                os.makedirs(error_map_dir, exist_ok=True)
            else:
                error_map_dir = None
            
            # Determine flow file path: {scene}_flow.npz in parent directory of root_dir
            flow_file = None
            if calc_tc:
                parent_dir = os.path.dirname(os.path.normpath(root_dir))
                flow_file = join(parent_dir, f"{scene}_flow.npz")
                if os.path.exists(flow_file):
                    print(f"Using flow file: {flow_file}")
                else:
                    print(f"Warning: Flow file not found: {flow_file}")
                    flow_file = None
            
            try:
                results = evaluate_directory(
                    rec_dir,
                    gt_dir,
                    flow_file=flow_file,
                    calc_lpips=calc_lpips,
                    calc_tc=calc_tc,
                    save_error_maps=save_error_maps,
                    error_map_dir=error_map_dir
                )
                all_results[scene] = results
            except Exception as e:
                print(f"Error processing {scene}: {e}")
                continue
    
    return all_results


def print_results(results: Dict[str, Dict[str, float]], output_file: Optional[str] = None):
    """Print and optionally save results."""
    lines = []
    
    # Header with both SSIM columns
    header = f"{'Scene':<20} {'MSE':<12} {'SSIM(win)':<12} {'SSIM(glb)':<12} {'LPIPS':<12} {'TC':<12} {'Frames':<10}"
    sep = "-" * len(header)
    
    lines.append(header)
    lines.append(sep)
    print(header)
    print(sep)
    
    # Calculate overall means
    all_mse = []
    all_ssim = []
    all_ssim_global = []
    all_lpips = []
    all_tc = []
    
    for scene, metrics in sorted(results.items()):
        mse = metrics['mse']
        ssim = metrics.get('ssim', 0.0)
        ssim_global = metrics.get('ssim_global', 0.0)
        lpips_val = metrics['lpips']
        tc = metrics['tc']
        n_frames = metrics['num_frames']
        n_tc = metrics['num_tc_frames']
        
        all_mse.append(mse)
        all_ssim.append(ssim)
        all_ssim_global.append(ssim_global)
        if lpips_val > 0:
            all_lpips.append(lpips_val)
        if tc > 0:
            all_tc.append(tc)
        
        tc_str = f"{tc:.6f}" if n_tc > 0 else "N/A"
        lpips_str = f"{lpips_val:.6f}" if lpips_val > 0 else "N/A"
        
        line = f"{scene:<20} {mse:<12.6f} {ssim:<12.6f} {ssim_global:<12.6f} {lpips_str:<12} {tc_str:<12} {n_frames:<10}"
        lines.append(line)
        print(line)
    
    lines.append(sep)
    print(sep)
    
    # Overall mean
    if len(results) > 1:
        mean_mse = np.mean(all_mse)
        mean_ssim = np.mean(all_ssim) if all_ssim else 0.0
        mean_ssim_global = np.mean(all_ssim_global) if all_ssim_global else 0.0
        mean_lpips = np.mean(all_lpips) if all_lpips else 0.0
        mean_tc = np.mean(all_tc) if all_tc else 0.0
        
        lpips_str = f"{mean_lpips:.6f}" if all_lpips else "N/A"
        tc_str = f"{mean_tc:.6f}" if all_tc else "N/A"
        
        line = f"{'MEAN':<20} {mean_mse:<12.6f} {mean_ssim:<12.6f} {mean_ssim_global:<12.6f} {lpips_str:<12} {tc_str:<12}"
        lines.append(line)
        print(line)
    
    # Save to file if requested
    if output_file:
        with open(output_file, 'w') as f:
            f.write('\n'.join(lines))
        print(f"\nResults saved to: {output_file}")
    
    # Also save as JSON
    if output_file:
        json_file = output_file.replace('.txt', '.json')
        if json_file == output_file:
            json_file = output_file + '.json'
        with open(json_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Detailed results saved to: {json_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate reconstruction metrics (MSE, LPIPS, TC, SSIM)'
    )
    parser.add_argument('--input_dir', required=True, type=str,
                        help='Directory containing reconstruction results. Should have subdirectories '
                             'with rec/ and gt/ folders, or directly contain rec/ and gt/.')
    parser.add_argument('--output_file', default='metrics_results.txt', type=str,
                        help='Output file to save results')
    parser.add_argument('--no_lpips', action='store_true',
                        help='Skip LPIPS calculation')
    parser.add_argument('--use_tc', action='store_true',
                        help='Enable Temporal Consistency (TC) calculation. '
                             'Looks for {scene_name}_flow.npz in the parent directory of input_dir.')
    parser.add_argument('--save_error_maps', action='store_true',
                        help='Save error visualization maps')
    parser.add_argument('--device', default='0', type=str,
                        help='GPU device ID')
    
    args = parser.parse_args()
    
    # Set GPU
    if args.device is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.device
    
    print(f"Input directory: {args.input_dir}")
    print(f"Output file: {args.output_file}")
    print(f"Calculate LPIPS: {not args.no_lpips}")
    print(f"Calculate TC: {args.use_tc}")
    print(f"Save error maps: {args.save_error_maps}")
    print()
    
    # Run evaluation
    results = evaluate_batch(
        args.input_dir,
        calc_lpips=not args.no_lpips,
        calc_tc=args.use_tc,
        save_error_maps=args.save_error_maps
    )
    
    if not results:
        print("No valid scenes found!")
        return
    
    # Print and save results
    print_results(results, args.output_file)


if __name__ == '__main__':
    main()

# Usage examples:
# Single scene:
#   python metrics_eval.py --input_dir results/scene1 --output_file results/scene1_metrics.txt
#
# Batch evaluation (multiple scenes):
#   python metrics_eval.py --input_dir results/ --output_file results/all_metrics.txt
#
# With error maps:
#   python metrics_eval.py --input_dir results/ --output_file results/metrics.txt --save_error_maps
#
# Enable TC calculation (requires {scene_name}_flow.npz in parent directory):
#   python metrics_eval.py --input_dir results/ --output_file results/metrics.txt --use_tc
