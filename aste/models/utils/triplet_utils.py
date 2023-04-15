from typing import Tuple, Optional, List

import torch
from torch import Tensor

from .const import CreatedSpanCodes, TripletDimensions
from ..outputs import SpanInformationOutput, SpanCreatorOutput


def create_embeddings_matrix_by_concat(data: SpanCreatorOutput) -> Tensor:
    aspects, opinions = expand_aspect_and_opinion(data.aspects_agg_emb, data.opinions_agg_emb)
    return torch.cat([aspects, opinions], dim=-1)


def create_embeddings_matrix_by_concat_tensors(aspects: Tensor, opinions: Tensor) -> Tensor:
    aspects, opinions = expand_aspect_and_opinion(aspects, opinions)
    return torch.cat([aspects, opinions], dim=-1)


def create_embedding_mask_matrix(data: SpanCreatorOutput) -> Tensor:
    diff_from = torch.tensor([CreatedSpanCodes.NOT_RELEVANT]).to(data.aspects_agg_emb)
    relevant_elements: Tensor = _create_bool_mask(data, diff_from=diff_from)

    return relevant_elements


def get_true_predicted_mask(data: SpanCreatorOutput) -> Tensor:
    equals_to = torch.tensor([CreatedSpanCodes.PREDICTED_TRUE]).to(data.aspects_agg_emb)
    true_elements = _create_bool_mask(data, equals_to=equals_to)

    return _create_final_mask(data, true_elements)


def create_mask_matrix_for_loss(data: SpanCreatorOutput) -> Tensor:
    equals_to = torch.tensor([CreatedSpanCodes.ADDED_TRUE, CreatedSpanCodes.PREDICTED_TRUE]).to(data.aspects_agg_emb)
    true_elements: Tensor = _create_bool_mask(data, equals_to=equals_to)

    return _create_final_mask(data, true_elements)


def create_mask_matrix_for_prediction(data: SpanCreatorOutput) -> Tensor:
    equals_to = torch.tensor([CreatedSpanCodes.PREDICTED_FALSE, CreatedSpanCodes.PREDICTED_TRUE])
    predicted_elements: Tensor = _create_bool_mask(data, equals_to=equals_to.to(data.aspects_agg_emb))

    return predicted_elements


def _create_bool_mask(data: SpanCreatorOutput, *, diff_from: Optional[Tensor] = None,
                      equals_to: Optional[Tensor] = None) -> Tensor:
    aspects, opinions = expand_aspect_and_opinion(
        data.get_aspect_span_creation_info(),
        data.get_opinion_span_creation_info()
    )

    if (diff_from is not None) and (equals_to is None):
        aspects = ~torch.isin(aspects, diff_from)
        opinions = ~torch.isin(opinions, diff_from)
    elif (equals_to is not None) and (diff_from is None):
        aspects = torch.isin(aspects, equals_to)
        opinions = torch.isin(opinions, equals_to)
    else:
        raise ValueError('Exactly one of diff_from or equals_to must be specified')

    return aspects & opinions


def expand_aspect_and_opinion(aspect: Tensor, opinion: Tensor) -> Tuple[Tensor, Tensor]:
    aspects: Tensor = aspect.unsqueeze(TripletDimensions.OPINION)
    opinions: Tensor = opinion.unsqueeze(TripletDimensions.ASPECT)

    aspect_shape: List = [-1, -1, -1, -1]
    aspect_shape[TripletDimensions.OPINION] = opinion.shape[1]

    opinion_shape: List = [-1, -1, -1, -1]
    opinion_shape[TripletDimensions.ASPECT] = aspect.shape[1]

    aspects = aspects.expand(aspect_shape[:len(aspects.shape)])
    opinions = opinions.expand(opinion_shape[:len(opinions.shape)])

    return aspects, opinions


def _create_final_mask(data: SpanCreatorOutput, final_mask: Tensor) -> Tensor:
    sample_aspects: SpanInformationOutput
    sample_opinions: SpanInformationOutput
    for sample_aspects, sample_opinions, mask in zip(data.aspects, data.opinions, final_mask):
        temp_mask: Tensor = torch.zeros_like(mask).bool()

        a_idx, o_idx = _get_a_o_indexes(sample_aspects, sample_opinions)
        if TripletDimensions.ASPECT == 1:
            temp_mask[..., a_idx, o_idx] = True
        else:
            temp_mask[..., o_idx, a_idx] = True
        mask &= temp_mask
    return final_mask


def create_sentiment_matrix(data: SpanCreatorOutput) -> Tensor:
    matrix = create_embedding_mask_matrix(data).to(torch.int)
    sample_aspects: SpanInformationOutput
    sample_opinions: SpanInformationOutput
    for sample_aspects, sample_opinions, mt in zip(data.aspects, data.opinions, matrix):
        a_idx, o_idx = _get_a_o_indexes(sample_aspects, sample_opinions)
        sentiments = sample_opinions.sentiments
        mt -= 1
        sentiments = sentiments[sentiments > 0]
        if TripletDimensions.ASPECT == 1:
            mt[..., a_idx, o_idx] = sentiments.to(torch.int)
        else:
            mt[..., o_idx, a_idx] = sentiments.to(torch.int)
    return matrix


def _get_a_o_indexes(
    sample_aspects: SpanInformationOutput,
    sample_opinions: SpanInformationOutput
) -> Tuple[Tensor, Tensor]:

    a_idx: Tensor = sample_aspects.mapping_indexes
    o_idx: Tensor = sample_opinions.mapping_indexes
    num_repeated: int = sample_opinions.repeated if sample_opinions.repeated is not None else 1
    a_idx = a_idx[a_idx >= 0].repeat(num_repeated)
    o_idx = o_idx[o_idx >= 0]
    return a_idx, o_idx
