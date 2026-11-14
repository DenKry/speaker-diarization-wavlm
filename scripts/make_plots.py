"""All figures of the project:
python scripts/make_plots.py --layers E2=runs/e2_frozen/best.pt E3=runs/e3_finetune/best.pt \
    --training E1=runs/e1_scratch E2=runs/e2_frozen E3=runs/e3_finetune --example test_010 540 600
"""

import argparse
from pathlib import Path

from diar.eval.plots import LABELS, plot_jer, plot_layer_weights, plot_results, plot_timeline, plot_training


def plot_example(meeting: str, start: float, end: float, systems: list[str], data: str, out_path: str) -> None:
    """Timeline of a short excerpt: the reference and the saved outputs (RTTM) of some systems."""
    from pyannote.core import Segment
    from pyannote.metrics.diarization import DiarizationErrorRate

    from diar.eval.der import read_annotation

    excerpt = Segment(start, end)
    reference = read_annotation(Path(data) / f"{meeting}.rttm")
    reference = reference.rename_labels(mapping={s: f"Speaker {i + 1}" for i, s in enumerate(sorted(reference.labels()))})
    rows = {"Reference": reference.crop(excerpt)}
    for name in systems:
        hypothesis = read_annotation(Path(f"results/rttm/{name}_test/{meeting}.rttm"))
        # give every system speaker the name of the reference speaker it matches best
        mapping = DiarizationErrorRate().optimal_mapping(reference, hypothesis)
        mapping = {s: mapping.get(s, f"Other {i + 1}") for i, s in enumerate(hypothesis.labels())}
        rows[LABELS.get(name, name)] = hypothesis.rename_labels(mapping=mapping).crop(excerpt)
    plot_timeline(rows).savefig(out_path, dpi=300)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="results/results.csv")
    parser.add_argument("--layers", nargs="*", default=[], help="NAME=checkpoint of WavLM models")
    parser.add_argument("--training", nargs="*", default=[], help="NAME=run folder, for the training curves")
    parser.add_argument("--example", nargs=3, metavar=("MEETING", "START", "END"), help="excerpt for the timeline figure")
    parser.add_argument("--example-systems", nargs="*", default=["B1", "E3"])
    parser.add_argument("--data", default="data/ami/test", help="folder with the reference RTTMs of the example")
    parser.add_argument("--out", default="results/figures")
    parser.add_argument("--corpus", default="AMI", help="corpus name for the axis labels")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if Path(args.results).exists():
        for setting in ("strict", "collar"):
            plot_results(args.results, str(out / f"der_{setting}.png"), setting, corpus=args.corpus)
        plot_jer(args.results, str(out / "jer.png"), corpus=args.corpus)
    if args.layers:
        plot_layer_weights(dict(item.split("=", 1) for item in args.layers), str(out / "layer_weights.png"))
    if args.training:
        plot_training(dict(item.split("=", 1) for item in args.training), str(out / "training.png"))
    if args.example:
        meeting, start, end = args.example
        plot_example(meeting, float(start), float(end), args.example_systems, args.data, str(out / "timeline_example.png"))
    print(f"Figures saved to {out}")
