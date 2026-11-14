"""DER of a trained model:  python scripts/evaluate.py --checkpoint runs/e2_frozen/best.pt --name E2"""

import argparse

from diar.eval.der import evaluate
from diar.inference.pipeline import build_pipeline

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--name", required=True, help="row name in results/results.csv, e.g. E2")
    parser.add_argument("--data", default="data/ami")
    parser.add_argument("--split", default="test")
    parser.add_argument("--results", default="results/results.csv", help="table to add the row to")
    parser.add_argument("--out", help="folder for the RTTMs (default: results/rttm/<name>_<split>)")
    parser.add_argument("--resume", action="store_true", help="reuse the RTTMs of an interrupted run")
    args = parser.parse_args()
    out = args.out or f"results/rttm/{args.name}_{args.split}"
    evaluate(build_pipeline(args.checkpoint), f"{args.data}/{args.split}", out, args.name, args.results, args.resume)
