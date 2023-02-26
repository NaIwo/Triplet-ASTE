from typing import Dict

from torch import Tensor

from ...outputs import BaseModelOutput


class SentimentModelOutput(BaseModelOutput):
    def __init__(self, sentiment_features: Dict[int, Tensor]):
        super().__init__()
        self.sentiment_features: Dict[int, Tensor] = sentiment_features

    def __getitem__(self, item: int) -> Tensor:
        return self.sentiment_features[item]
