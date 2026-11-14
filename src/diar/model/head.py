"""Classification head shared by all our models."""

import torch
from torch import nn

NUM_CLASSES = 7  # powerset classes, see train/loss.py


class Head(nn.Module):
    """Features (batch, frames, in_dim) -> logits (batch, frames, 7)."""

    def __init__(self, in_dim: int, kind: str = "bilstm", hidden: int = 256, layers: int = 2, dropout: float = 0.1):
        super().__init__()
        if kind == "bilstm":
            self.lstm = nn.LSTM(
                in_dim, hidden, num_layers=layers, bidirectional=True, batch_first=True,
                dropout=dropout if layers > 1 else 0.0,
            )
            self.out = nn.Linear(2 * hidden, NUM_CLASSES)
        elif kind == "linear":
            self.lstm = None
            self.out = nn.Linear(in_dim, NUM_CLASSES)
        else:
            raise ValueError(f"Unknown head: {kind}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.lstm is not None:
            x, _ = self.lstm(x)
        return self.out(x)
