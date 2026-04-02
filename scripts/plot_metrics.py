import argparse
import re
from pathlib import Path
import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


METRICS = ["loss", "raw_ssim", "raw_ssim_global", "raw_mse", "raw_lpips", "raw_tc"]
SUMMARY_METRICS = [
	("Overall Loss", "loss"),
	("    LPIPS   ", "raw_lpips"),
	("  SSIM(win) ", "raw_ssim"),
	("  SSIM(glb) ", "raw_ssim_global"),
	("     MSE    ", "raw_mse"),
	("     TC     ", "raw_tc"),
]
VAL_KEY_MAP = {
	"val_loss": "loss",
	"val_raw_ssim": "raw_ssim",
	"val_raw_ssim_global": "raw_ssim_global",
	"val_raw_mse": "raw_mse",
	"val_raw_lpips": "raw_lpips",
	"val_raw_tc": "raw_tc",
}
PLOT_TITLES = {
	"loss": "Loss",
	"raw_ssim": "SSIM (window)",
	"raw_ssim_global": "SSIM (global)",
	"raw_mse": "MSE",
	"raw_lpips": "LPIPS",
	"raw_tc": "TC",
}
EXTREMA_MODE = {
	"loss": "min",
	"raw_ssim": "max",
	"raw_ssim_global": "max",
	"raw_mse": "min",
	"raw_lpips": "min",
	"raw_tc": "min",
}


def parse_log(log_path: Path):
	train_records = {}
	val_records = {}
	current_epoch = None

	line_pattern = re.compile(
		r"INFO\s*-\s*(?P<key>[^:]+?)\s*:\s*(?P<value>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*$"
	)

	with log_path.open("r", encoding="utf-8", errors="ignore") as f:
		for line in f:
			m = line_pattern.search(line)
			if not m:
				continue

			key = m.group("key").strip()
			value = float(m.group("value"))

			if key == "epoch":
				current_epoch = int(value)
				train_records.setdefault(current_epoch, {})
				val_records.setdefault(current_epoch, {})
				continue

			if current_epoch is None:
				continue

			if key in METRICS:
				train_records[current_epoch][key] = value
			elif key in VAL_KEY_MAP:
				val_metric = VAL_KEY_MAP[key]
				val_records[current_epoch][val_metric] = value

	return train_records, val_records


def extract_series(records, metric):
	epochs = sorted(epoch for epoch, vals in records.items() if metric in vals)
	values = [records[epoch][metric] for epoch in epochs]
	return epochs, values


def annotate_extrema(ax, epochs, values, metric):
	if not epochs:
		return

	mode = EXTREMA_MODE[metric]
	if mode == "min":
		idx = min(range(len(values)), key=lambda i: values[i])
	else:
		idx = max(range(len(values)), key=lambda i: values[i])

	x = epochs[idx]
	y = values[idx]
	label = f"({x}, {y:.6f})"

	ax.scatter([x], [y], s=52, zorder=6, color="red", edgecolor="black", linewidth=0.6)
	ax.annotate(
		label,
		xy=(x, y),
		xytext=(6, 6),
		textcoords="offset points",
		fontsize=8,
		color="red",
		bbox={"boxstyle": "round,pad=0.2", "fc": "white", "ec": "red", "alpha": 0.6},
	)


def plot_all_metrics(train_records, val_records, out_dir: Path):
	available_metrics = []
	for metric in METRICS:
		train_epochs, _ = extract_series(train_records, metric)
		val_epochs, _ = extract_series(val_records, metric)
		if train_epochs or val_epochs:
			available_metrics.append(metric)

	if not available_metrics:
		return None

	fig = plt.figure(figsize=(14, 10), constrained_layout=True)
	gs = fig.add_gridspec(3, 2)

	# Create subplots - both SSIM metrics share one subplot
	axes = {
		"loss": fig.add_subplot(gs[0, :]),
		"ssim_combined": fig.add_subplot(gs[1, 0]),  # Combined SSIM subplot
		"raw_mse": fig.add_subplot(gs[1, 1]),
		"raw_lpips": fig.add_subplot(gs[2, 0]),
		"raw_tc": fig.add_subplot(gs[2, 1]),
	}

	# Plot regular metrics
	for metric in ["loss", "raw_mse", "raw_lpips", "raw_tc"]:
		if metric not in available_metrics:
			continue
		ax = axes[metric]
		train_epochs, train_values = extract_series(train_records, metric)
		val_epochs, val_values = extract_series(val_records, metric)

		if train_epochs:
			ax.plot(train_epochs, train_values, marker="o", linewidth=1.5, label="Train")
		if val_epochs:
			ax.plot(val_epochs, val_values, marker="s", linewidth=1.5, label="Val")

		annotate_extrema(ax, train_epochs, train_values, metric)
		annotate_extrema(ax, val_epochs, val_values, metric)

		ax.set_title(PLOT_TITLES[metric])
		ax.set_xlabel("Epoch")
		ax.set_ylabel(PLOT_TITLES[metric])
		ax.grid(True, alpha=0.3)
		if train_epochs or val_epochs:
			ax.legend()

	# Plot both SSIM metrics on the same subplot with different colors
	ax_ssim = axes["ssim_combined"]
	colors = plt.cm.tab10(np.linspace(0, 1, 10))

	# SSIM Window (skimage-style)
	train_epochs_win, train_values_win = extract_series(train_records, "raw_ssim")
	val_epochs_win, val_values_win = extract_series(val_records, "raw_ssim")
	if train_epochs_win:
		ax_ssim.plot(train_epochs_win, train_values_win, marker="o", linewidth=1.5,
					 color=colors[0], label="Window-Train")
	if val_epochs_win:
		ax_ssim.plot(val_epochs_win, val_values_win, marker="s", linewidth=1.5,
					 color=colors[0], linestyle='--', label="Window-Val")
	annotate_extrema(ax_ssim, train_epochs_win, train_values_win, "raw_ssim")
	annotate_extrema(ax_ssim, val_epochs_win, val_values_win, "raw_ssim")

	# SSIM Global (mean-based)
	train_epochs_glb, train_values_glb = extract_series(train_records, "raw_ssim_global")
	val_epochs_glb, val_values_glb = extract_series(val_records, "raw_ssim_global")
	if train_epochs_glb:
		ax_ssim.plot(train_epochs_glb, train_values_glb, marker="o", linewidth=1.5,
					 color=colors[1], label="Global-Train")
	if val_epochs_glb:
		ax_ssim.plot(val_epochs_glb, val_values_glb, marker="s", linewidth=1.5,
					 color=colors[1], linestyle='--', label="Global-Val")
	annotate_extrema(ax_ssim, train_epochs_glb, train_values_glb, "raw_ssim_global")
	annotate_extrema(ax_ssim, val_epochs_glb, val_values_glb, "raw_ssim_global")

	ax_ssim.set_title("SSIM (Window vs Global)")
	ax_ssim.set_xlabel("Epoch")
	ax_ssim.set_ylabel("SSIM")
	ax_ssim.grid(True, alpha=0.3)
	if train_epochs_win or val_epochs_win or train_epochs_glb or val_epochs_glb:
		ax_ssim.legend(ncol=2, fontsize=8)

	out_file = out_dir / "metrics_overview.png"
	fig.savefig(out_file, dpi=150)
	plt.close(fig)
	return out_file


def _format_table_value(value):
	if value is None:
		return "N/A"
	return f"{value:.4f}"


def _render_markdown_table(title, epoch, record):
	lines = [
		f"{title} (Epoch {epoch})",
		"|-----------------------|",
	]

	for metric_name, metric_key in SUMMARY_METRICS:
		value = _format_table_value(record.get(metric_key))
		lines.append(f"| {metric_name} | {value} |")

	return "\n".join(lines)


def write_last_epoch_summary(train_records, val_records, out_dir: Path):
	all_epochs = sorted(set(train_records.keys()) | set(val_records.keys()))
	if not all_epochs:
		return None

	last_epoch = all_epochs[-1]
	train_record = train_records.get(last_epoch, {})
	val_record = val_records.get(last_epoch, {})

	content = "\n\n".join(
		[
			_render_markdown_table("Train", last_epoch, train_record),
			_render_markdown_table("Validation", last_epoch, val_record),
		]
	)

	out_file = out_dir / "last_epoch_metrics.txt"
	out_file.write_text(content + "\n", encoding="utf-8")
	return out_file


def main():
	parser = argparse.ArgumentParser(
		description="Visualize train/val metrics from trainer log by epoch."
	)
	parser.add_argument(
		"--log",
		type=Path,
		required=True,
		help="Path to log file (default: info.log)",
	)
	parser.add_argument(
		"--outdir",
		type=Path,
		default=None,
		help="Output directory for figures (default: same directory as --log)",
	)
	args = parser.parse_args()

	if not args.log.exists():
		raise FileNotFoundError(f"Log file not found: {args.log}")

	if args.outdir is None:
		args.outdir = args.log.parent
	else:
		args.outdir.mkdir(parents=True, exist_ok=True)

	train_records, val_records = parse_log(args.log)
	out_file = plot_all_metrics(train_records, val_records, args.outdir)
	summary_file = write_last_epoch_summary(train_records, val_records, args.outdir)

	if out_file is None:
		raise RuntimeError("No target metrics were found in log.")
	if summary_file is None:
		raise RuntimeError("No epoch records were found in log.")

	print("Generated figure:")
	print(out_file)
	print("Generated summary:")
	print(summary_file)


if __name__ == "__main__":
	main()

# python plot_metrics.py --log path/to/info.log