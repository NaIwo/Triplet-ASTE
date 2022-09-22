from abc import abstractmethod
from typing import List

import torch
from torch.nn.utils.rnn import pad_sequence

from ASTE.utils import config


class AggResult:
    def __init__(self, agg_embeddings: torch.Tensor, lengths: List[int]):
        self.agg_embeddings: torch.Tensor = agg_embeddings
        self.lengths: torch.Tensor = torch.tensor(lengths, device=config['general']['device'])


class BaseAggregator:
    def __init__(self, input_dim: int, model_name: str = 'base aggregator', *args, **kwargs):
        self.model_name: str = model_name
        self.input_dim: int = input_dim

    def aggregate(self, embeddings: torch.Tensor, spans: List[torch.Tensor]) -> AggResult:
        agg_embeddings: List = list()
        lengths: List = list()
        sentence_embeddings: torch.Tensor
        sentence_spans: torch.Tensor
        for sentence_embeddings, sentence_spans in zip(embeddings, spans):
            sentence_agg_embeddings: List = self._get_agg_sentence_embeddings(sentence_embeddings, sentence_spans)
            agg_embeddings.append(torch.stack(sentence_agg_embeddings, dim=0))
            lengths.append(agg_embeddings[-1].shape[0])
        return AggResult(agg_embeddings=self.pad_sequence(agg_embeddings), lengths=lengths)

    def _get_agg_sentence_embeddings(self, sentence_embeddings: torch.Tensor, sentence_spans: torch.Tensor) -> List:
        raise NotImplementedError

    @property
    @abstractmethod
    def output_dim(self):
        raise NotImplementedError

    @staticmethod
    def pad_sequence(agg_embeddings: List[torch.Tensor]) -> torch.Tensor:
        return pad_sequence(agg_embeddings, padding_value=0, batch_first=True)
