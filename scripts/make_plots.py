"""Figures for the report:
python scripts/make_plots.py --layers E2=runs/e2_frozen/best.pt E3=runs/e3_finetune/best.pt \
    --training E1=runs/e1_scratch E2=runs/e2_frozen E3=runs/e3_finetune
"""

import argparse
from pathlib import Path

from diar.eval.plots import plot_jer, plot_layer_weights, plot_results, plot_training

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="results/results.csv")
    parser.add_argument("--layers", nargs="*", default=[], help="NAME=checkpoint of WavLM models")
    parser.add_argument("--training", nargs="*", default=[], help="NAME=run folder, for the training curves")
    parser.add_argument("--out", default="results/figures")
    parser.add_argument("--corpus", default="AMI", help="corpus name for the figure titles")
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
    print(f"Figures saved to {out}")
