"""E2 / E3: pre-trained WavLM + weighted sum of its layers + head."""

import torch
from torch import nn
from transformers import WavLMModel

from diar.model.head import Head


class WavLMSeg(nn.Module):
    def __init__(self, name: str = "microsoft/wavlm-base-plus", freeze: bool = True, **head_args):
        super().__init__()
        # No SpecAugment masking and no layer dropping: we want plain features.
        self.wavlm = WavLMModel.from_pretrained(name, mask_time_prob=0.0, layerdrop=0.0)
        self.freeze = freeze
        if freeze:
            self.wavlm.requires_grad_(False)  # E2: only the head is trained
        else:
            self.wavlm.freeze_feature_encoder()  # E3: the CNN encoder always stays frozen
        num_layers = self.wavlm.config.num_hidden_layers + 1  # CNN output + 12 transformer layers
        self.layer_weights = nn.Parameter(torch.zeros(num_layers))
        self.head = Head(self.wavlm.config.hidden_size, **head_args)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze:
            self.wavlm.eval()  # a frozen WavLM must not use dropout
        return self

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        """wav (batch, samples) -> logits (batch, frames, 7)."""
        with torch.set_grad_enabled(not self.freeze and torch.is_grad_enabled()):
            layers = self.wavlm(wav, output_hidden_states=True).hidden_states
        weights = torch.softmax(self.layer_weights, dim=0)
        features = sum(w * layer for w, layer in zip(weights, layers))
        return self.head(features)
