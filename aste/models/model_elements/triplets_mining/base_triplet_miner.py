from typing import TypeVar

import torch
from torch import Tensor

from ASTE.aste.models.model_elements.span_aggregators import AggResult
from ASTE.utils import config

BTM = TypeVar('BTM', bound='BaseTripletsMiner')


class BaseTripletsMiner:
    def __init__(self, distance_matrix: Tensor, aspect_mask: Tensor, opinion_mask: Tensor, other_mask: Tensor):
        self.distance_matrix: Tensor = distance_matrix
        self.aspect_mask: Tensor = aspect_mask
        self.opinion_mask: Tensor = opinion_mask
        self.other_mask: Tensor = other_mask

    @classmethod
    def from_prediction(cls, seq_pred: Tensor, embeddings: AggResult) -> BTM:
        raise NotImplementedError

    @staticmethod
    def construct_distance_matrix(agg_results: AggResult) -> Tensor:
        raise NotImplementedError

    @staticmethod
    def construct_valid_elements_mask(seq_classes: Tensor, target: int) -> Tensor:
        return torch.where(seq_classes == target, 1, 0)

    @staticmethod
    def construct_embeddings_matrix(data: Tensor) -> Tensor:
        max_len: int = int(data.shape[1])
        return data.unsqueeze(1).expand(-1, max_len, -1, -1)

    @staticmethod
    def construct_matrix_embeddings_mask(dist: Tensor, agg_results: AggResult) -> Tensor:
        upper_diagonal_mask: Tensor = BaseTripletsMiner.construct_upper_diagonal_mask(dist)
        len_mask: Tensor = BaseTripletsMiner.construct_lengths_mask(agg_results)
        return upper_diagonal_mask * len_mask

    @staticmethod
    def construct_upper_diagonal_mask(dist):
        upper_diagonal_mask: Tensor = torch.ones(dist.shape[:3]).triu().to(config['general']['device'])
        return upper_diagonal_mask.unsqueeze(-1)

    @staticmethod
    def construct_lengths_mask(agg_results: AggResult) -> Tensor:
        max_len: int = max(agg_results.lengths).item()
        len_mask: Tensor = torch.arange(max_len).expand(len(agg_results.lengths), max_len).to(
            config['general']['device'])
        len_mask = (len_mask < agg_results.lengths.unsqueeze(1)).type(torch.int8)
        return len_mask.unsqueeze(1).expand(-1, max_len, -1).unsqueeze(-1)
