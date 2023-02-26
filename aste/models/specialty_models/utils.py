from typing import List, Optional, Dict

import torch
from torch.nn import Sequential


def sequential_blocks(neurons: List, config: Dict, blocks: Optional[Sequential] = None) -> Sequential:
    if blocks is None:
        blocks = Sequential()

    idx: int
    for idx in range(len(neurons[:-1])):
        blocks.append(
            Sequential(
                torch.nn.Linear(neurons[idx], neurons[idx + 1]),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.1)
            )
        )

    return blocks.to(config['general-training']['device'])
