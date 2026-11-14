"""Turn a trained WavLM checkpoint into a Hugging Face model folder (and optionally upload it):
python scripts/export_hf.py --checkpoint runs/e3_finetune/best.pt --out hf_export [--push USER/NAME]

The folder contains the weights, the config, the model code from hf/ and the model card,
so that anyone can load it with AutoModel.from_pretrained(..., trust_remote_code=True).
"""

import argparse
import shutil
import sys
from pathlib import Path

import torch
from transformers import WavLMConfig

HF_DIR = Path(__file__).resolve().parent.parent / "hf"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="runs/e3_finetune/best.pt")
    parser.add_argument("--out", default="hf_export")
    parser.add_argument("--push", help="repository on the Hugging Face Hub, e.g. user/model-name")
    parser.add_argument("--private", action="store_true")
    args = parser.parse_args()

    sys.path.insert(0, str(HF_DIR.parent))
    from hf.configuration_wavlm_diarization import WavLMDiarizationConfig
    from hf.modeling_wavlm_diarization import WavLMDiarizationModel

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    cfg = ckpt["config"]["model"]
    wavlm_config = WavLMConfig.from_pretrained(cfg["name"], mask_time_prob=0.0, layerdrop=0.0)
    config = WavLMDiarizationConfig(
        wavlm_config=wavlm_config.to_dict(), hidden=cfg["hidden"], layers=cfg["layers"], dropout=cfg["dropout"],
    )
    model = WavLMDiarizationModel(config)
    model.load_state_dict(ckpt["model"])  # strict: every weight must match

    # auto_map lets AutoModel find the classes in the code files of the repository
    WavLMDiarizationConfig.register_for_auto_class()
    WavLMDiarizationModel.register_for_auto_class("AutoModel")
    out = Path(args.out)
    model.save_pretrained(out)
    for name in ("configuration_wavlm_diarization.py", "modeling_wavlm_diarization.py", "README.md"):
        shutil.copy(HF_DIR / name, out / name)
    print(f"Model folder written to {out}")

    if args.push:
        from huggingface_hub import HfApi

        api = HfApi()
        api.create_repo(args.push, private=args.private, exist_ok=True)
        api.upload_folder(folder_path=str(out), repo_id=args.push)
        print(f"Uploaded to https://huggingface.co/{args.push}")
