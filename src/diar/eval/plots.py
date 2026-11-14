"""Figures for the README and the demo: a clean, minimal look (white background, light grid)."""

import csv

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import torch
from matplotlib.patches import Patch
from pyannote.core import Annotation

# Colour-blind-safe series colours, always used in this order.
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
EXTRA = "#8a8a84"  # groups beyond the 8th
INK, MUTED, GRID = "#1a1a1a", "#5c5c58", "#e6e6e3"
plt.rcParams.update({
    "font.family": ["Liberation Sans", "DejaVu Sans"],
    "font.size": 10,
    "axes.labelsize": 10.5,
    "axes.labelcolor": MUTED,
    "legend.title_fontsize": 10,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
})

# Row names of the results tables -> labels in the figures.
LABELS = {
    "B1": "B1  pyannote 3.1",
    "B2": "B2  DiariZen",
    "E1": "E1  from scratch",
    "E2": "E2  frozen WavLM",
    "E3": "E3  fine-tuned WavLM",
}


def palette(n: int) -> list[str]:
    return COLORS[:n] + [EXTRA] * max(0, n - len(COLORS))


def _style(ax, grid: str = "both") -> None:
    ax.set_facecolor("white")
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(colors=MUTED, length=0)
    ax.grid(axis=grid, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _legend(ax, title: str, **kwargs) -> None:
    legend = ax.legend(title=title, loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False, labelcolor=INK, **kwargs)
    legend.get_title().set_color(MUTED)
    legend._legend_box.align = "left"


def plot_timeline(annotations: dict[str, Annotation], title: str | None = None):
    """One row per system, one coloured bar per speech turn."""
    speakers = sorted({label for a in annotations.values() for label in a.labels()})
    colors = dict(zip(speakers, palette(len(speakers))))

    fig, ax = plt.subplots(figsize=(10, 1.3 + 0.65 * len(annotations)))
    _style(ax, grid="x")
    for row, annotation in enumerate(annotations.values()):
        for segment, _, speaker in annotation.itertracks(yield_label=True):
            ax.barh(row, segment.duration, left=segment.start, height=0.6, color=colors[speaker], alpha=0.85)
    ax.set_yticks(range(len(annotations)), list(annotations), color=INK)
    ax.invert_yaxis()
    ax.set_xlabel("Time (s)")
    if title:
        ax.set_title(title, loc="left", color=INK)
    _legend(ax, "Speaker", handles=[Patch(color=colors[s], label=s) for s in speakers])
    fig.tight_layout()
    return fig


def _rows(results_csv: str, split: str) -> dict[str, dict]:
    with open(results_csv) as f:
        return {r["experiment"]: r for r in csv.DictReader(f) if r["split"] == split}  # last run of each wins


def plot_results(results_csv: str, out_path: str, setting: str = "strict", split: str = "test", corpus: str = "AMI") -> None:
    """DER of every experiment, split into missed speech, false alarm and confusion."""
    rows = _rows(results_csv, split)
    names = list(rows)
    labels = [LABELS.get(n, n) for n in names]
    parts = [("missed", "Missed speech"), ("false_alarm", "False alarm"), ("confusion", "Speaker confusion")]

    fig, ax = plt.subplots(figsize=(8, 0.9 + 0.5 * len(names)))
    _style(ax, grid="x")
    left = [0.0] * len(names)
    for (key, label), color in zip(parts, palette(3)):
        values = [float(rows[n][f"{key}_{setting}"]) for n in names]
        ax.barh(labels, values, left=left, height=0.62, color=color, label=label, edgecolor="white", linewidth=1.5)
        left = [a + b for a, b in zip(left, values)]
    for i, n in enumerate(names):
        ax.text(left[i] + max(left) * 0.012, i, f"{float(rows[n][f'der_{setting}']):.1f}", va="center", color=INK)
    ax.invert_yaxis()
    ax.set_xlim(0, max(left) * 1.1)
    ax.tick_params(axis="y", labelcolor=INK)
    collar = ", 0.25 s collar" if setting == "collar" else ""
    ax.set_xlabel(f"DER on {corpus} {split} (%{collar})")
    _legend(ax, "Error type")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_jer(results_csv: str, out_path: str, split: str = "test", corpus: str = "AMI") -> None:
    """Jaccard error rate of every experiment (one number each)."""
    rows = _rows(results_csv, split)
    names = list(rows)
    values = [float(rows[n]["jer"]) for n in names]

    fig, ax = plt.subplots(figsize=(7, 0.9 + 0.5 * len(names)))
    _style(ax, grid="x")
    ax.barh([LABELS.get(n, n) for n in names], values, height=0.62, color=palette(1)[0])
    for i, v in enumerate(values):
        ax.text(v + max(values) * 0.012, i, f"{v:.1f}", va="center", color=INK)
    ax.invert_yaxis()
    ax.set_xlim(0, max(values) * 1.1)
    ax.tick_params(axis="y", labelcolor=INK)
    ax.set_xlabel(f"JER on {corpus} {split} (%)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_layer_weights(checkpoints: dict[str, str], out_path: str) -> None:
    """How much each WavLM layer contributes to the features (after softmax)."""
    fig, ax = plt.subplots(figsize=(8, 3.2))
    _style(ax, grid="y")
    width = 0.8 / len(checkpoints)
    for i, ((name, path), color) in enumerate(zip(checkpoints.items(), palette(len(checkpoints)))):
        weights = torch.softmax(torch.load(path, map_location="cpu")["model"]["layer_weights"], dim=0)
        x = [k + (i - (len(checkpoints) - 1) / 2) * width for k in range(len(weights))]
        ax.bar(x, weights.tolist(), width=width * 0.92, color=color, label=LABELS.get(name, name))
    ax.set_xticks(range(len(weights)), ["CNN"] + [str(k) for k in range(1, len(weights))])
    ax.set_xlabel("WavLM layer")
    ax.set_ylabel("Weight")
    _legend(ax, "Model")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_training(runs: dict[str, str], out_path: str) -> None:
    """Validation loss and chunk-level DER after every epoch, read from the TensorBoard logs."""
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    for (name, run_dir), color in zip(runs.items(), palette(len(runs))):
        log = EventAccumulator(f"{run_dir}/tb")
        log.Reload()
        for ax, tag, scale in zip(axes, ("val/loss", "val/der"), (1.0, 100.0)):
            points = log.Scalars(tag)
            ax.plot([p.step for p in points], [p.value * scale for p in points],
                    color=color, marker="o", markersize=4, linewidth=2, label=LABELS.get(name, name))
    for ax, title in zip(axes, ("Validation loss", "Validation DER on 10 s chunks (%)")):
        _style(ax)
        ax.set_title(title, loc="left", color=INK, fontsize=10.5)
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
        ax.set_xlabel("Epoch")
    _legend(axes[1], "Model")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
