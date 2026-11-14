"""DiariZen (WavLM-large, BUT Speech@FIT) as a second ready-made system.

It needs its own environment (Python 3.10, PyTorch 2.1.1), so it only writes RTTMs;
the DER is computed afterwards in the main environment:

    .venv-diarizen/bin/python scripts/baseline_diarizen.py
    python scripts/score_rttm.py --rttm results/rttm/B2_test --name B2
"""

import argparse
from pathlib import Path

from diarizen.pipelines.inference import DiariZenPipeline

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="BUT-FIT/diarizen-wavlm-large-s80-md-v2")
    parser.add_argument("--data", default="data/ami")
    parser.add_argument("--split", default="test")
    parser.add_argument("--name", default="B2")
    parser.add_argument("--batch-size", type=int, default=8, help="the model's own 32 does not fit in 6 GB")
    args = parser.parse_args()

    out_dir = f"results/rttm/{args.name}_{args.split}"
    pipeline = DiariZenPipeline.from_pretrained(args.model, rttm_out_dir=out_dir)
    pipeline.segmentation_batch_size = args.batch_size
    pipeline.embedding_batch_size = args.batch_size
    for audio in sorted(Path(args.data, args.split).glob("*.flac")):
        if Path(out_dir, audio.stem + ".rttm").exists():  # done in an earlier, interrupted run
            continue
        print(audio.stem, flush=True)
        pipeline(str(audio), sess_name=audio.stem)
    print(f"RTTMs saved to {out_dir}")
