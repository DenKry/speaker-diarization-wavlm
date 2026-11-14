"""Powerset cross-entropy with speaker permutations (Plaquet & Bredin, 2023).

Up to 3 speakers per chunk, at most 2 at the same time -> 7 classes per frame:
    {}, {1}, {2}, {3}, {1,2}, {1,3}, {2,3}
The class order is the same as in pyannote, so the model fits its pipeline.
"""

import itertools

import torch
import torch.nn.functional as F

# class index -> which of the 3 speakers are active
POWERSET = torch.tensor(
    [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1]], dtype=torch.float32
)
# speakers as a bit mask (spk1 + 2*spk2 + 4*spk3) -> class index.
# All three speaking at once has no class: those frames are ignored in the loss.
IGNORE = -100
MASK_TO_CLASS = torch.tensor([0, 1, 2, 4, 3, 5, 6, IGNORE])
PERMUTATIONS = list(itertools.permutations(range(3)))


def multilabel_to_powerset(target: torch.Tensor) -> torch.Tensor:
    """(..., 3) zeros/ones -> (...) class indices."""
    mask = (target[..., 0] + 2 * target[..., 1] + 4 * target[..., 2]).long()
    return MASK_TO_CLASS.to(target.device)[mask]


def powerset_to_multilabel(classes: torch.Tensor) -> torch.Tensor:
    """(...) class indices -> (..., 3) zeros/ones."""
    return POWERSET.to(classes.device)[classes]


def pit_powerset_loss(logits: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Cross-entropy under the best speaker permutation of each chunk.

    The speaker order in the target is arbitrary, so we try all 6 orderings
    of the 3 speakers and keep the one with the lowest loss.

    logits: (batch, frames, 7), target: (batch, frames, 3).
    Returns the mean loss and the best permutation index per chunk.
    """
    logits = logits.float().transpose(1, 2)  # (batch, 7, frames)
    losses = []
    for perm in PERMUTATIONS:
        classes = multilabel_to_powerset(target[..., list(perm)])
        ce = F.cross_entropy(logits, classes, ignore_index=IGNORE, reduction="none")  # (batch, frames)
        valid = (classes != IGNORE).sum(dim=1).clamp(min=1)
        losses.append(ce.sum(dim=1) / valid)
    best, best_perm = torch.stack(losses).min(dim=0)
    return best.mean(), best_perm


def frame_errors(logits: torch.Tensor, target: torch.Tensor, best_perm: torch.Tensor) -> dict[str, float]:
    """Frame counts for a chunk-level DER: missed, false alarm, confusion, total speech."""
    perms = torch.tensor(PERMUTATIONS, device=target.device)[best_perm]  # (batch, 3)
    ref = torch.gather(target, 2, perms[:, None, :].expand(-1, target.shape[1], -1))
    hyp = powerset_to_multilabel(logits.argmax(dim=-1))
    n_ref, n_hyp = ref.sum(-1), hyp.sum(-1)
    correct = (ref * hyp).sum(-1)
    return {
        "missed": (n_ref - n_hyp).clamp(min=0).sum().item(),
        "false_alarm": (n_hyp - n_ref).clamp(min=0).sum().item(),
        "confusion": (torch.minimum(n_ref, n_hyp) - correct).sum().item(),
        "total": n_ref.sum().item(),
    }
