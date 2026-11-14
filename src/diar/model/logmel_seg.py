"""E1: small model trained from scratch (log-mel spectrogram + CNN + head)."""

import torch
import torchaudio
from torch import nn

from diar.model.head import Head


class LogMelSeg(nn.Module):
    def __init__(self, channels: int = 256, **head_args):
        super().__init__()
        # 25 ms window, 20 ms hop, no padding: the same frames as WavLM.
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=16000, n_fft=400, hop_length=320, n_mels=80, center=False
        )
        blocks, in_ch = [], 80
        for _ in range(3):
            blocks += [nn.Conv1d(in_ch, channels, kernel_size=5, padding=2), nn.BatchNorm1d(channels), nn.ReLU()]
            in_ch = channels
        self.cnn = nn.Sequential(*blocks)
        self.head = Head(channels, **head_args)

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        """wav (batch, samples) -> logits (batch, frames, 7)."""
        with torch.autocast(wav.device.type, enabled=False):  # the spectrogram needs full precision
            x = torch.log(self.mel(wav.float()) + 1e-6)  # (batch, 80, frames)
            x = x - x.mean(dim=-1, keepdim=True)  # remove the channel/loudness offset
        x = self.cnn(x).transpose(1, 2)  # (batch, frames, channels)
        return self.head(x)
