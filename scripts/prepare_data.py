"""Prepare AMI:  python scripts/prepare_data.py [--max-meetings 3]

Another corpus, only for testing (VoxConverse, first 30 recordings of its test set):
python scripts/prepare_data.py --dataset diarizers-community/voxconverse --config default \
    --out data/voxconverse --splits test --max-meetings 30
"""

import argparse

from diar.data.prepare_ami import prepare_split

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/ami")
    parser.add_argument("--dataset", default="diarizers-community/ami")
    parser.add_argument("--config", default="ihm", help="ihm (headset) or sdm (distant microphone)")
    parser.add_argument("--splits", nargs="+", default=["train", "validation", "test"])
    parser.add_argument("--max-meetings", type=int, default=None, help="only the first N meetings per split")
    args = parser.parse_args()
    for split in args.splits:
        prepare_split(split, args.out, args.config, args.max_meetings, args.dataset)
