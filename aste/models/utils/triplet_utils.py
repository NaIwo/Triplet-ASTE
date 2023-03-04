from typing import Tuple, Optional, List

import torch
from torch import Tensor

from .const import CreatedSpanCodes, TripletDimensions
from ..outputs import SpanInformationOutput, SpanPredictionsOutput, SpanCreatorOutput


def create_embeddings_matrix_by_concat(data: SpanCreatorOutput) -> Tensor:
    aspects, opinions = _expand_aspect_and_opinion(data.aspects_agg_emb, data.opinions_agg_emb)
    return torch.cat([aspects, opinions], dim=-1)


def create_embedding_mask_matrix(data: SpanCreatorOutput) -> Tensor:
    relevant_elements: Tensor = _create_bool_mask(data, diff_from=CreatedSpanCodes.NOT_RELEVANT)

    return relevant_elements


def get_true_predicted_mask(data: SpanCreatorOutput) -> Tensor:
    true_elements = _create_bool_mask(data, equals_to=CreatedSpanCodes.PREDICTED_TRUE)

    return true_elements


def create_mask_matrix_for_loss(data: SpanCreatorOutput) -> Tensor:
    true_elements: Tensor = _create_bool_mask(data, equals_to=CreatedSpanCodes.ADDED_TRUE)
    true_elements |= _create_bool_mask(data, equals_to=CreatedSpanCodes.PREDICTED_TRUE)

    return _create_final_mask(data, true_elements)


def create_mask_matrix_for_prediction(data: SpanCreatorOutput) -> Tensor:
    predicted_elements: Tensor = _create_bool_mask(data, equals_to=CreatedSpanCodes.PREDICTED_TRUE)
    predicted_elements |= _create_bool_mask(data, equals_to=CreatedSpanCodes.PREDICTED_FALSE)

    return predicted_elements


def _create_bool_mask(data: SpanCreatorOutput, *, diff_from: Optional[int] = None,
                      equals_to: Optional[int] = None) -> Tensor:
    aspects, opinions = _expand_aspect_and_opinion(
        data.predicted_spans.get_aspect_span_creation_info(),
        data.predicted_spans.get_opinion_span_creation_info()
    )

    if (diff_from is not None) and (equals_to is None):
        aspects = aspects != diff_from
        opinions = opinions != diff_from
    elif (equals_to is not None) and (diff_from is None):
        aspects = aspects == equals_to
        opinions = opinions == equals_to
    else:
        raise ValueError('Exactly one of diff_from or equals_to must be specified')

    return aspects & opinions


def _expand_aspect_and_opinion(aspect: Tensor, opinion: Tensor) -> Tuple[Tensor, Tensor]:
    aspects: Tensor = aspect.unsqueeze(TripletDimensions.ASPECT)
    opinions: Tensor = opinion.unsqueeze(TripletDimensions.OPINION)

    aspect_shape: List = [-1, -1, -1, -1]
    aspect_shape[TripletDimensions.ASPECT] = opinion.shape[1]

    opinion_shape: List = [-1, -1, -1, -1]
    opinion_shape[TripletDimensions.OPINION] = aspect.shape[1]

    aspects = aspects.expand(aspect_shape[:len(aspects.shape)])
    opinions = opinions.expand(opinion_shape[:len(opinions.shape)])

    return aspects, opinions


def _create_final_mask(data: SpanCreatorOutput, final_mask: Tensor) -> Tensor:
    ps: SpanPredictionsOutput = data.predicted_spans
    sample_aspects: SpanInformationOutput
    sample_opinions: SpanInformationOutput
    for sample_aspects, sample_opinions, mask in zip(ps.aspects, ps.opinions, final_mask):
        temp_mask: Tensor = torch.zeros_like(mask).bool()

        a_idx: Tensor = sample_aspects.mapping_indexes
        o_idx: Tensor = sample_opinions.mapping_indexes
        a_idx = a_idx[a_idx >= 0].repeat(sample_opinions.repeated)
        o_idx = o_idx[o_idx >= 0]
        if TripletDimensions.ASPECT == 2:
            temp_mask[a_idx, o_idx] = True
        else:
            temp_mask[o_idx, a_idx] = True
        mask &= temp_mask
    return final_mask
