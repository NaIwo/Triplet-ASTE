from typing import Dict, List

import torch
from torch import Tensor
from torch.nn import Sequential

from .base_triplets_extractor import BaseTripletExtractorModel
from ..utils import sequential_blocks
from ...outputs import (
    SpanCreatorOutput
)
from ...utils.triplet_utils import (
    create_embeddings_matrix_by_concat_tensors,
    create_embedding_mask_matrix
)


class NeuralTripletExtractorModel(BaseTripletExtractorModel):
    def __init__(self, input_dim: int,
                 config: Dict,
                 model_name: str = 'Neural Triplet Extractor Model',
                 *args, **kwargs
                 ):
        super(NeuralTripletExtractorModel, self).__init__(model_name=model_name, config=config)

        input_dimension: int = input_dim * 2

        neurons: List = [input_dim, input_dim // 2, input_dim // 2, input_dim]
        self.aspect_net = sequential_blocks(neurons=neurons, device=self.device)
        self.opinion_net = sequential_blocks(neurons=neurons, device=self.device)

        neurons: List = [input_dimension, input_dimension // 2, input_dimension // 4, input_dimension // 8, 1]
        self.similarity: Sequential = sequential_blocks(neurons, self.device)
        self.similarity.append(torch.nn.Sigmoid())

    def _forward_embeddings(self, data_input: SpanCreatorOutput) -> Tensor:
        aspects = self.aspect_net(data_input.aspects_agg_emb)
        opinions = self.opinion_net(data_input.opinions_agg_emb)

        matrix: Tensor = create_embeddings_matrix_by_concat_tensors(aspects, opinions)
        mask: Tensor = create_embedding_mask_matrix(data_input)

        matrix = self.similarity(matrix)

        return matrix.squeeze(-1) * mask
