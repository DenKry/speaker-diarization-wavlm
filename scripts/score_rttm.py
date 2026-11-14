"""DER of RTTMs written by an external system:
python scripts/score_rttm.py --rttm results/rttm/B2_test --name B2
"""

import argparse

from diar.eval.der import evaluate_rttms

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rttm", required=True, help="folder with one RTTM per meeting")
    parser.add_argument("--name", required=True)
    parser.add_argument("--data", default="data/ami")
    parser.add_argument("--split", default="test")
    parser.add_argument("--results", default="results/results.csv", help="table to add the row to")
    parser.add_argument("--exclude", nargs="*", default=[], help="recordings to leave out, e.g. test_014")
    args = parser.parse_args()
    evaluate_rttms(args.rttm, f"{args.data}/{args.split}", args.name, args.results, tuple(args.exclude))
