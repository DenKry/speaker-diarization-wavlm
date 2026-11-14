"""Train a model:  python scripts/train.py --config configs/e1_scratch.yaml [--resume]"""

import argparse

from diar.train.trainer import load_config, train

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", action="store_true", help="continue from <out_dir>/last.pt")
    args = parser.parse_args()
    train(load_config(args.config), resume=args.resume)
