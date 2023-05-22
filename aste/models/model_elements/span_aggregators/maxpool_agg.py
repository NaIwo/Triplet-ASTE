from typing import List, Dict

import torch
from torch import Tensor

from .base_agg import BaseAggregator


class MaxPoolAggregator(BaseAggregator):
    def __init__(self, input_dim: int, config: Dict, model_name: str = 'Max Pool Aggregator', *args, **kwargs):
        self._out_dim: int = input_dim
        BaseAggregator.__init__(self, input_dim=self._out_dim, model_name=model_name, config=config)

    @property
    def output_dim(self):
        return self._out_dim

    def _get_agg_sentence_embeddings(self, sentence_embeddings: Tensor, sentence_spans: Tensor) -> Tensor:
        sentence_agg_embeddings: List = list()
        span: Tensor
        for span in sentence_spans:
            span_emb: Tensor = sentence_embeddings[span[0]:span[1] + 1]
            agg_emb: Tensor = torch.max(span_emb, dim=0)[0]
            sentence_agg_embeddings.append(agg_emb)
        return torch.stack(sentence_agg_embeddings, dim=0)
