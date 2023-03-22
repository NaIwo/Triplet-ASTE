from typing import Dict, List

import torch
import torch.nn.functional as F
from torch import Tensor

from .base_triplets_extractor import BaseTripletExtractorModel
from ..utils import sequential_blocks
from ...outputs import (
    SpanCreatorOutput
)
from ...utils.triplet_utils import (
    create_embedding_mask_matrix,
    expand_aspect_and_opinion
)


class AttentionTripletExtractorModel(BaseTripletExtractorModel):
    def __init__(self, input_dim: int,
                 config: Dict,
                 model_name: str = 'Metric Triplet Extractor Model',
                 *args, **kwargs
                 ):
        super(AttentionTripletExtractorModel, self).__init__(model_name=model_name, config=config)

        neurons: List = [input_dim, input_dim // 2, input_dim // 2, input_dim]
        self.query = sequential_blocks(neurons=neurons, device=self.device)
        self.key = sequential_blocks(neurons=neurons, device=self.device)
        neurons: List = [input_dim, input_dim // 2, input_dim // 4, 1]
        self.value = sequential_blocks(neurons=neurons, device=self.device)

    def _forward_embeddings(self, data_input: SpanCreatorOutput) -> Tensor:
        queries = self.query(data_input.aspects_agg_emb)
        keys = self.key(data_input.opinions_agg_emb)
        values = self.value(data_input.opinions_agg_emb)

        matrix: Tensor = torch.matmul(queries, keys.transpose(-1, -2)) / torch.sqrt(
            torch.tensor(keys.shape[-1], dtype=torch.float32))
        matrix = F.softmax(matrix)
        matrix = torch.matmul(matrix, values)
        matrix = F.sigmoid(matrix)

        mask: Tensor = create_embedding_mask_matrix(data_input)
        return matrix * mask
