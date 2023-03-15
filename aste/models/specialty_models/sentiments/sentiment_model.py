from typing import Dict, List

import torch

from ....models.base_model import BaseModel
from ..utils import sequential_blocks
from ...outputs import SentimentModelOutput
from ....dataset.domain.const import ASTELabels


class EmbeddingsExtenderModel(BaseModel):
    def __init__(self, input_dim: int, config: Dict, model_name: str = 'Sentiment extender model'):
        super(EmbeddingsExtenderModel, self).__init__(model_name=model_name, config=config)
        self.common_model = sequential_blocks([input_dim, input_dim], self.config)
        neurons: List = [
            input_dim,
            input_dim // 2,
            input_dim // 4,
            input_dim // 4,
            input_dim // 2,
            input_dim
        ]
        self.models = {
            p: sequential_blocks(neurons, self.config) for p in self.config['dataset']['polarities']
        }

    def forward(self, data: torch.Tensor) -> SentimentModelOutput:
        data = self.common_model(data)
        return SentimentModelOutput(sentiment_features={
            ASTELabels[p]: self.models[p](data)
            for p in self.config['dataset']['polarities']
        })