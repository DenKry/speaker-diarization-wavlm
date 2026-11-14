"""Training loop: AMP, warmup + cosine schedule, validation, checkpoints, resume."""

import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from diar.data.dataset import ChunkDataset
from diar.model import build_model
from diar.train.loss import frame_errors, pit_powerset_loss


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def hub_upload(out_dir: Path, repo: str) -> None:
    """Copy the run folder to a private Hugging Face repo (survives a Colab disconnect)."""
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo, private=True, exist_ok=True)
    api.upload_folder(folder_path=str(out_dir), repo_id=repo, path_in_repo=out_dir.name)


def hub_download(out_dir: Path, repo: str) -> None:
    from huggingface_hub import snapshot_download

    try:
        snapshot_download(repo, allow_patterns=f"{out_dir.name}/*", local_dir=str(out_dir.parent))
    except Exception as e:  # no repo yet = nothing to resume from
        print(f"Nothing downloaded from {repo}: {e}")


@torch.no_grad()
def validate(model, loader, device, amp: bool) -> dict[str, float]:
    """Validation loss and chunk-level DER (with its three components)."""
    model.eval()
    loss_sum, batches = 0.0, 0
    counts = {"missed": 0.0, "false_alarm": 0.0, "confusion": 0.0, "total": 0.0}
    for wav, target in loader:
        wav, target = wav.to(device), target.to(device)
        with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
            logits = model(wav)
        loss, best_perm = pit_powerset_loss(logits, target)
        loss_sum += loss.item()
        batches += 1
        for k, v in frame_errors(logits, target, best_perm).items():
            counts[k] += v
    total = max(counts["total"], 1.0)
    return {
        "loss": loss_sum / max(batches, 1),
        "der": (counts["missed"] + counts["false_alarm"] + counts["confusion"]) / total,
        "missed": counts["missed"] / total,
        "false_alarm": counts["false_alarm"] / total,
        "confusion": counts["confusion"] / total,
    }


def train(cfg: dict, resume: bool = False) -> None:
    tcfg, dcfg = cfg["train"], cfg["data"]
    seed = cfg.get("seed", 42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = bool(tcfg.get("amp", True)) and device.type == "cuda"
    out_dir = Path(cfg["out_dir"])
    hub_repo = cfg.get("hub_repo")
    if resume and hub_repo and not (out_dir / "last.pt").exists():
        hub_download(out_dir, hub_repo)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "config.yaml", "w") as f:
        yaml.safe_dump(cfg, f)

    # ---- data
    batch_size, steps_per_epoch = tcfg["batch_size"], tcfg["steps_per_epoch"]
    train_set = ChunkDataset(
        Path(dcfg["dir"]) / "train", dcfg["chunk_sec"], random_chunks=True,
        epoch_size=batch_size * steps_per_epoch, fraction=dcfg.get("fraction", 1.0),
        augment=dcfg.get("augment", False),
    )
    val_set = ChunkDataset(
        Path(dcfg["dir"]) / "validation", dcfg["chunk_sec"], random_chunks=False,
        max_chunks=tcfg.get("val_max_chunks"),
    )
    workers = dcfg.get("num_workers", 2)
    train_loader = DataLoader(train_set, batch_size, num_workers=workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size, num_workers=workers, pin_memory=True)
    print(f"train: {len(train_set.meetings)} meetings | validation: {len(val_set)} chunks | device: {device}")

    # ---- model
    model = build_model(cfg["model"]).to(device)
    if tcfg.get("init_from") and not (resume and (out_dir / "last.pt").exists()):
        if hub_repo and not Path(tcfg["init_from"]).exists():
            hub_download(Path(tcfg["init_from"]).parent, hub_repo)
        state = torch.load(tcfg["init_from"], map_location="cpu")["model"]
        model.load_state_dict(state)
        print(f"Weights loaded from {tcfg['init_from']}")

    # ---- optimizer: WavLM gets its own (small) learning rate
    wavlm_params = [p for n, p in model.named_parameters() if n.startswith("wavlm.") and p.requires_grad]
    other_params = [p for n, p in model.named_parameters() if not n.startswith("wavlm.") and p.requires_grad]
    groups = [{"params": other_params, "lr": float(tcfg["lr"])}]
    if wavlm_params:
        groups.append({"params": wavlm_params, "lr": float(tcfg["wavlm_lr"])})
    optimizer = torch.optim.AdamW(groups, weight_decay=float(tcfg.get("weight_decay", 0.01)))
    trainable = sum(p.numel() for p in wavlm_params + other_params)
    print(f"trainable parameters: {trainable / 1e6:.2f}M")

    total_steps = tcfg["epochs"] * steps_per_epoch
    warmup_steps = int(tcfg.get("warmup", 0.1) * total_steps)

    def lr_factor(step: int) -> float:  # linear warmup, then cosine decay to zero
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        return 0.5 * (1 + math.cos(math.pi * (step - warmup_steps) / max(1, total_steps - warmup_steps)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
    scaler = torch.amp.GradScaler(device.type, enabled=amp)

    # ---- resume
    start_epoch, best_der = 0, float("inf")
    if resume and (out_dir / "last.pt").exists():
        ckpt = torch.load(out_dir / "last.pt", map_location="cpu")
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        scaler.load_state_dict(ckpt["scaler"])
        start_epoch, best_der = ckpt["epoch"] + 1, ckpt["best_der"]
        print(f"Resumed from epoch {start_epoch} (best validation DER so far: {best_der:.4f})")

    writer = SummaryWriter(str(out_dir / "tb"))
    params = wavlm_params + other_params
    for epoch in range(start_epoch, tcfg["epochs"]):
        model.train()
        t0, running = time.time(), 0.0
        for step, (wav, target) in enumerate(train_loader, start=1):
            wav, target = wav.to(device, non_blocking=True), target.to(device, non_blocking=True)
            with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
                logits = model(wav)
            loss, _ = pit_powerset_loss(logits, target)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(params, float(tcfg.get("grad_clip", 5.0)))
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            running += loss.item()
            if step % 50 == 0:
                global_step = epoch * steps_per_epoch + step
                speed = step / (time.time() - t0)
                print(f"epoch {epoch + 1} step {step}/{steps_per_epoch} loss {running / 50:.4f} {speed:.2f} it/s", flush=True)
                writer.add_scalar("train/loss", running / 50, global_step)
                writer.add_scalar("train/lr", scheduler.get_last_lr()[0], global_step)
                running = 0.0

        val = validate(model, val_loader, device, amp)
        for k, v in val.items():
            writer.add_scalar(f"val/{k}", v, epoch + 1)
        print(
            f"== epoch {epoch + 1}: val loss {val['loss']:.4f} | DER {val['der']:.2%} "
            f"(missed {val['missed']:.2%}, false alarm {val['false_alarm']:.2%}, confusion {val['confusion']:.2%}) "
            f"| {(time.time() - t0) / 60:.1f} min",
            flush=True,
        )

        if val["der"] < best_der:
            best_der = val["der"]
            torch.save({"model": model.state_dict(), "config": cfg, "epoch": epoch, "val": val}, out_dir / "best.pt")
            print(f"   new best model saved (DER {best_der:.2%})")
        torch.save(
            {
                "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
                "epoch": epoch, "best_der": best_der,
            },
            out_dir / "last.pt",
        )
        writer.flush()
        if hub_repo:
            hub_upload(out_dir, hub_repo)
    writer.close()
    print(f"Done. Best validation DER: {best_der:.2%} -> {out_dir / 'best.pt'}")
