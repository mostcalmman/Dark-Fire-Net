import re
import argparse
from pathlib import Path
import matplotlib.pyplot as plt

def parse_log(log_path):
    """Parse info.log to extract epoch metrics."""
    data = {}
    
    with open(log_path, 'r') as f:
        current_epoch = None
        epoch_data = {}
        
        for line in f:
            if 'epoch' in line and ':' in line:
                match = re.search(r'epoch\s*:\s*(\d+)', line)
                if match:
                    if current_epoch and epoch_data:
                        data[current_epoch] = epoch_data
                    current_epoch = int(match.group(1))
                    epoch_data = {}
            
            if current_epoch:
                if 'loss' in line and 'raw' not in line:
                    match = re.search(r'loss\s*:\s*([\d.]+)', line)
                    if match:
                        epoch_data['loss'] = float(match.group(1))
                elif 'raw_lpips' in line:
                    match = re.search(r'raw_lpips\s*:\s*([\d.]+)', line)
                    if match:
                        epoch_data['lpips'] = float(match.group(1))
                elif 'raw_ssim' in line:
                    match = re.search(r'raw_ssim\s*:\s*([\d.]+)', line)
                    if match:
                        epoch_data['ssim'] = float(match.group(1))
                elif 'raw_mse' in line:
                    match = re.search(r'raw_mse\s*:\s*([\d.]+)', line)
                    if match:
                        epoch_data['mse'] = float(match.group(1))
        
        if current_epoch and epoch_data:
            data[current_epoch] = epoch_data
    
    epochs = sorted(data.keys())
    loss = [data[e].get('loss', 0) for e in epochs]
    lpips = [data[e].get('lpips', 0) for e in epochs]
    ssim = [data[e].get('ssim', 0) for e in epochs]
    mse = [data[e].get('mse', 0) for e in epochs]
    
    return epochs, loss, lpips, ssim, mse

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
