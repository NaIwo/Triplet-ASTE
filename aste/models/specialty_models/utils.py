from typing import List, Optional

import torch
from torch import Tensor
from torch.nn import Sequential


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
                # torch.nn.LayerNorm(neurons[idx]),
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
