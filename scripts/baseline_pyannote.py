"""B1, the pyannote baseline:  python scripts/baseline_pyannote.py"""

import argparse

from diar.eval.der import evaluate
from diar.inference.pipeline import build_pipeline

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/ami")
    parser.add_argument("--split", default="test")
    parser.add_argument("--name", default="B1")
    parser.add_argument("--results", default="results/results.csv", help="table to add the row to")
    parser.add_argument("--out", help="folder for the RTTMs (default: results/rttm/<name>_<split>)")
    args = parser.parse_args()
    out = args.out or f"results/rttm/{args.name}_{args.split}"
    evaluate(build_pipeline(None), f"{args.data}/{args.split}", out, args.name, args.results)
