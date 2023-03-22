from typing import Dict, List

import torch
from torch import Tensor

from .base_triplets_extractor import BaseTripletExtractorModel
from ...outputs import (
    SpanCreatorOutput
)
from ..utils import sequential_blocks
from ...utils.triplet_utils import (
    create_embedding_mask_matrix,
    expand_aspect_and_opinion
)


def scale_scores(scores: Tensor) -> Tensor:
    return torch.clamp((scores + 1.) / 2., min=0., max=1.)


class MetricTripletExtractorModel(BaseTripletExtractorModel):
    def __init__(self, input_dim: int,
                 config: Dict,
                 model_name: str = 'Metric Triplet Extractor Model',
                 *args, **kwargs
                 ):
        super(MetricTripletExtractorModel, self).__init__(model_name=model_name, config=config)

        neurons: List = [
            input_dim,
            input_dim // 2,
            input_dim // 2,
            input_dim
        ]
        self.aspect_net = sequential_blocks(neurons=neurons, device=self.device)
        self.opinion_net = sequential_blocks(neurons=neurons, device=self.device)

        self.similarity_metric = torch.nn.CosineSimilarity(dim=-1)

    def _forward_embeddings(self, data_input: SpanCreatorOutput) -> Tensor:
        aspects = self.aspect_net(data_input.aspects_agg_emb)
        opinions = self.opinion_net(data_input.opinions_agg_emb)

        aspects, opinions = expand_aspect_and_opinion(aspects, opinions)

        matrix: Tensor = self.similarity_metric(aspects, opinions)
        matrix = scale_scores(matrix)
        mask: Tensor = create_embedding_mask_matrix(data_input)
        return matrix * mask

