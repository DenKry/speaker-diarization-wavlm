from torch import nn


def build_model(cfg: dict) -> nn.Module:
    """Create a model from the `model:` section of a config."""
    head_args = {k: cfg[k] for k in ("hidden", "layers", "dropout") if k in cfg}
    head_args["kind"] = cfg.get("head", "bilstm")
    if cfg["type"] == "wavlm":
        from diar.model.wavlm_seg import WavLMSeg

        return WavLMSeg(cfg.get("name", "microsoft/wavlm-base-plus"), freeze=cfg["freeze"], **head_args)
    if cfg["type"] == "logmel":
        from diar.model.logmel_seg import LogMelSeg

        return LogMelSeg(cfg.get("channels", 256), **head_args)
    raise ValueError(f"Unknown model type: {cfg['type']}")
