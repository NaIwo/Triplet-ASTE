from typing import TypeVar

import torch
from torch import Tensor

from ASTE.aste.models.model_elements.span_aggregators import AggResult
from ASTE.dataset.domain.const import SpanLabels
from .base_triplet_miner import BaseTripletsMiner

MTM = TypeVar('MTM', bound='ManhattanTripletsMiner')


class ManhattanTripletsMiner(BaseTripletsMiner):
    def __init__(self, distance_matrix: Tensor, aspect_mask: Tensor, opinion_mask: Tensor, other_mask: Tensor):
        super().__init__(distance_matrix, aspect_mask, opinion_mask, other_mask)

    @classmethod
    def from_prediction(cls, seq_pred: Tensor, agg_results: AggResult) -> MTM:
        seq_classes: Tensor = torch.argmax(seq_pred, dim=-1)
        return cls(distance_matrix=cls.construct_distance_matrix(agg_results),
                   aspect_mask=cls.construct_valid_elements_mask(seq_classes, SpanLabels.ASPECT),
                   opinion_mask=cls.construct_valid_elements_mask(seq_classes, SpanLabels.OPINION),
                   other_mask=cls.construct_valid_elements_mask(seq_classes, SpanLabels.NOT_RELEVANT))

    @staticmethod
    def construct_distance_matrix(agg_results: AggResult) -> Tensor:
        data_matrix: Tensor = ManhattanTripletsMiner.construct_embeddings_matrix(agg_results.agg_embeddings)
        data_matrix_t: Tensor = torch.transpose(data_matrix, 1, 2)

        dist: Tensor = torch.abs(data_matrix - data_matrix_t)
        mask: Tensor = ManhattanTripletsMiner.construct_matrix_embeddings_mask(dist, agg_results)

        return mask * dist
