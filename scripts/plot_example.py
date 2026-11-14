"""Timeline of a short excerpt: the reference, the baseline B1 and our model E3 (from saved RTTMs):
python scripts/plot_example.py --meeting test_010 --start 540 --end 600
"""

import argparse
from pathlib import Path

from pyannote.core import Segment
from pyannote.metrics.diarization import DiarizationErrorRate

from diar.eval.der import read_annotation
from diar.eval.plots import LABELS, plot_timeline

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--meeting", default="test_010")
    parser.add_argument("--start", type=float, default=540.0)
    parser.add_argument("--end", type=float, default=600.0)
    parser.add_argument("--systems", nargs="*", default=["B1", "E3"])
    parser.add_argument("--data", default="data/ami/test")
    parser.add_argument("--out", default="results/figures/timeline_example.png")
    args = parser.parse_args()

    excerpt = Segment(args.start, args.end)
    reference = read_annotation(Path(args.data) / f"{args.meeting}.rttm")
    reference = reference.rename_labels(mapping={s: f"Speaker {i + 1}" for i, s in enumerate(sorted(reference.labels()))})
    rows = {"Reference": reference.crop(excerpt)}
    for name in args.systems:
        hypothesis = read_annotation(Path(f"results/rttm/{name}_test/{args.meeting}.rttm"))
        # give every system speaker the name of the reference speaker it matches best
        mapping = DiarizationErrorRate().optimal_mapping(reference, hypothesis)
        mapping = {s: mapping.get(s, f"Other {i + 1}") for i, s in enumerate(hypothesis.labels())}
        rows[LABELS.get(name, name)] = hypothesis.rename_labels(mapping=mapping).crop(excerpt)

    fig = plot_timeline(rows)
    fig.savefig(args.out, dpi=300)
    print(f"Figure saved to {args.out}")
