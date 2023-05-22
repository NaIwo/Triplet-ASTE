from typing import List, Optional

import torch
from torch import Tensor
from torch.nn import Sequential
from torch import nn


def sequential_blocks(
        neurons: List,
        device: Optional[torch.device],
        blocks: Optional[Sequential] = None,
        is_last: bool = True
) -> Sequential:
    if blocks is None:
        blocks = Sequential()

    idx: int
    for idx in range(len(neurons[:-1 - int(is_last)])):
        blocks.append(
            Sequential(
                torch.nn.LayerNorm(neurons[idx]),
                torch.nn.Linear(neurons[idx], neurons[idx + 1]),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.1)
            )
        )
    if is_last:
        blocks.append(
            torch.nn.Linear(neurons[-2], neurons[-1])
        )

    return blocks.to(device)


def scale_scores(scores: Tensor) -> Tensor:
    return torch.clamp((scores + 1.) / 2., min=0., max=1.)


class TransformerModel(nn.Module):
    def __init__(self, input_dim: int, model_dim: int, attention_heads: int, num_layers: int, output_dim: int, device):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Linear(input_dim, model_dim)
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(model_dim, attention_heads), num_layers
        )
        self.output_layer = sequential_blocks([model_dim, model_dim // 2, output_dim], device, is_last=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(x)
        embedded = embedded.permute(1, 0, 2)
        encoded = self.transformer_encoder(embedded)
        encoded = encoded.permute(1, 0, 2)
        output = self.output_layer(encoded)
        return output
