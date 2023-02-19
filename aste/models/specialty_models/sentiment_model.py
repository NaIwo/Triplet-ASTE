from typing import Dict, List, Optional

import torch
from torch.nn import Sequential

from .. import BaseModel
from ..outputs import SentimentModelOutput
from ...dataset.domain.const import ASTELabels


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


class EmbeddingsExtenderModel(BaseModel):
    def __init__(self, input_dim: int, config: Dict, model_name: str = 'Sentiment extender model'):
        super(EmbeddingsExtenderModel, self).__init__(model_name=model_name, config=config)
        self.common_model = sequential_blocks([input_dim, input_dim], self.config)
        neurons: List = [
            input_dim,
            input_dim // 2,
            input_dim // 4,
            input_dim // 2,
            input_dim
        ]
        self.models = [
            sequential_blocks(neurons, self.config) for _ in range(len(self.config['dataset']['polarities']))
        ]

    def forward(self, data: torch.Tensor) -> SentimentModelOutput:
        data = self.common_model(data)
        return SentimentModelOutput(sentiment_features={
            el.value: self.models[idx](data)
            for idx, el in enumerate(ASTELabels)
            if el.name in self.config['dataset']['polarities']
        })
