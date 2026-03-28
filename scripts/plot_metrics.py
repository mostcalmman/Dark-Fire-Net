import re
import argparse
from pathlib import Path
import matplotlib.pyplot as plt

def parse_log(log_path):
    """Parse info.log to extract epoch metrics."""
    epochs, loss, lpips, ssim, mse = [], [], [], [], []
    
    with open(log_path, 'r') as f:
        current_epoch = None
        for line in f:
            # Match epoch number
            epoch_match = re.search(r'epoch\s*:\s*(\d+)', line, re.IGNORECASE)
            if epoch_match:
                current_epoch = int(epoch_match.group(1))
            
            if current_epoch is None:
                continue
            
            # Match metrics (handle both 'loss' and 'val_loss')
            if re.search(r'\bloss\s*:', line, re.IGNORECASE):
                loss_match = re.search(r':\s*([\d.]+)', line)
                if loss_match and current_epoch not in epochs:
                    epochs.append(current_epoch)
                    loss.append(float(loss_match.group(1)))
            
            if 'raw_lpips' in line.lower():
                lpips_match = re.search(r':\s*([\d.]+)', line)
                if lpips_match:
                    lpips.append(float(lpips_match.group(1)))
            
            if 'raw_ssim' in line.lower():
                ssim_match = re.search(r':\s*([\d.]+)', line)
                if ssim_match:
                    ssim.append(float(ssim_match.group(1)))
            
            if 'raw_mse' in line.lower():
                mse_match = re.search(r':\s*([\d.]+)', line)
                if mse_match:
                    mse.append(float(mse_match.group(1)))
    
    # Ensure all lists have same length
    min_len = min(len(epochs), len(loss), len(lpips), len(ssim), len(mse))
    return epochs[:min_len], loss[:min_len], lpips[:min_len], ssim[:min_len], mse[:min_len]

def plot_metrics(epochs, loss, lpips, ssim, mse, save_path):
    """Plot metrics over epochs."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    axes[0, 0].plot(epochs, loss, 'b-', linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training Loss')
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].plot(epochs, lpips, 'r-', linewidth=2)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('LPIPS')
    axes[0, 1].set_title('LPIPS (Perceptual Distance)')
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 0].plot(epochs, ssim, 'g-', linewidth=2)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('SSIM')
    axes[1, 0].set_title('SSIM (Structural Similarity)')
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].plot(epochs, mse, 'm-', linewidth=2)
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('MSE')
    axes[1, 1].set_title('MSE (Mean Squared Error)')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved to {save_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot training metrics from log file')
    parser.add_argument('--log', type=str, required=True, help='Path to info.log file')
    parser.add_argument('--output', type=str, default=None, help='Output plot path (default: same dir as log)')
    args = parser.parse_args()
    
    log_path = Path(args.log)
    if not log_path.exists():
        raise FileNotFoundError(f"Log file not found: {log_path}")
    
    output_path = args.output if args.output else log_path.parent / 'metrics_plot.png'
    
    epochs, loss, lpips, ssim, mse = parse_log(log_path)
    
    if not epochs:
        print("No metrics found in log file. Check log format.")
    else:
        print(f"Found {len(epochs)} epochs of data")
        plot_metrics(epochs, loss, lpips, ssim, mse, output_path)
