from typing import List

import torch
from torch import Tensor

from ....dataset.domain.span import Span
from ....dataset.reader import Sentence
from ....models.outputs import SpanCreatorOutput, SpanInformationOutput, SpanPredictionsOutput


def create_embeddings_matrix_by_concat(data: SpanCreatorOutput) -> Tensor:
    aspects: Tensor = data.aspects_agg_emb.unsqueeze(2)
    opinions: Tensor = data.opinions_agg_emb.unsqueeze(1)

    aspects = aspects.expand(-1, -1, data.opinions_agg_emb.shape[1], -1)
    opinions = opinions.expand(-1, data.aspects_agg_emb.shape[1], -1, -1)

    return torch.cat([aspects, opinions], dim=-1)


def create_mask_matrix_for_training(data: SpanCreatorOutput) -> Tensor:
    aspects: Tensor = data.predicted_spans.get_aspect_span_labels().unsqueeze(2)
    opinions: Tensor = data.predicted_spans.get_opinion_span_labels().unsqueeze(1)

    aspects = aspects.expand(-1, -1, data.predicted_spans.get_opinion_span_labels().shape[-1])
    opinions = opinions.expand(-1, data.predicted_spans.get_aspect_span_labels().shape[-1], -1)

    return aspects & opinions


def create_mask_matrix_for_validation(data: SpanCreatorOutput) -> Tensor:
    final_mask: Tensor = create_mask_matrix_for_training(data)

    ps: SpanPredictionsOutput = data.predicted_spans

    sample_aspects: SpanInformationOutput
    sample_opinions: SpanInformationOutput
    for sample_aspects, sample_opinions, mask in zip(ps.aspects, ps.opinions, final_mask):
        temp_mask: Tensor = torch.zeros_like(mask).bool()

        a_idx: Tensor = sample_aspects.mapping_indexes
        o_idx: Tensor = sample_opinions.mapping_indexes
        a_idx = a_idx[a_idx >= 0].repeat(sample_opinions.repeated)
        o_idx = o_idx[o_idx >= 0]
        temp_mask[a_idx, o_idx] = True
        mask &= temp_mask

    return final_mask
