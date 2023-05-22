from typing import Dict, List, Tuple

import torch
from torch import Tensor

from .base_triplets_extractor import BaseTripletExtractorModel
from ..utils import scale_scores
from ..utils import sequential_blocks
from ...outputs import (
    SpanInformationOutput,
    SpanCreatorOutput,
    SampleTripletOutput
)
from ...utils.triplet_utils import (
    create_embeddings_matrix_by_concat_tensors
)
from ...utils.triplet_utils import (
    create_sentiment_matrix
)
from ...utils.triplet_utils import (
    expand_aspect_and_opinion
)


class BaseNonSentimentTripletExtractorModel(BaseTripletExtractorModel):
    def __init__(self, input_dim: int,
                 config: Dict,
                 model_name: str = 'base No Sentiment Triplet Extractor Model',
                 *args, **kwargs
                 ):
        super(BaseNonSentimentTripletExtractorModel, self).__init__(input_dim=input_dim,
                                                                    model_name=model_name,
                                                                    config=config, *args, **kwargs)

    def similarity(self, aspects: Tensor, opinions: Tensor) -> Tensor:
        raise NotImplementedError

    def get_triplets_from_matrix(self, matrix: Tensor, data_input: SpanCreatorOutput) -> List[SampleTripletOutput]:
        triplets: List = list()

        sample_idx: int
        sample: Tensor
        sample_aspects: SpanInformationOutput
        sample_opinions: SpanInformationOutput
        zip_ = zip(matrix, data_input.aspects, data_input.opinions)
        for sample_idx, (sample, sample_aspects, sample_opinions) in enumerate(zip_):
            significant: Tensor = self.threshold_data(sample).nonzero()

            a_ranges: Tensor = sample_aspects.span_range[significant[:, 0]]
            o_ranges: Tensor = sample_opinions.span_range[significant[:, 1]]
            a_emb: Tensor = data_input.aspects_agg_emb[sample_idx:sample_idx + 1, significant[:, 0]]
            o_emb: Tensor = data_input.opinions_agg_emb[sample_idx:sample_idx + 1, significant[:, 1]]

            span_creation_info = sample_opinions.span_creation_info[significant[:, 1]]

            sentiments = create_sentiment_matrix(data_input)
            sentiments = sentiments[sample_idx:sample_idx + 1, significant[:, 0], significant[:, 1]]

            features: Tensor = create_embeddings_matrix_by_concat_tensors(
                data_input.aspects_agg_emb[sample_idx:sample_idx + 1],
                data_input.opinions_agg_emb[sample_idx:sample_idx + 1]
            )
            features = features[:, significant[:, 0], significant[:, 1]]
            similarities = matrix[sample_idx: sample_idx + 1, significant[:, 0], significant[:, 1]]

            triplets.append(
                SampleTripletOutput(
                    aspect_ranges=a_ranges,
                    opinion_ranges=o_ranges,
                    sentence_emb=data_input.sentence_emb[sample_idx],
                    aspect_emb=a_emb.squeeze(dim=0),
                    opinion_emb=o_emb.squeeze(dim=0),
                    true_sentiments=sentiments.squeeze(dim=0),
                    sentence=sample_opinions.sentence,
                    similarities=similarities.squeeze(dim=0),
                    span_creation_info=span_creation_info,
                    features=features.squeeze(dim=0)
                )
            )

        return triplets


class NonSentimentMetricTripletExtractorModel(BaseNonSentimentTripletExtractorModel):
    def __init__(self, input_dim: int,
                 config: Dict,
                 model_name: str = 'Metric No Sentiment Triplet Extractor Model',
                 *args, **kwargs
                 ):
        super(NonSentimentMetricTripletExtractorModel, self).__init__(
            input_dim=input_dim, model_name=model_name, config=config
        )

        neurons: List = [
            input_dim,
            input_dim // 2,
            input_dim // 4,
            input_dim // 2
        ]
        self.aspect_net = sequential_blocks(neurons, device=self.device, is_last=True)
        self.opinion_net = sequential_blocks(neurons, device=self.device, is_last=True)

        self.similarity_metric = torch.nn.CosineSimilarity(dim=-1)

    def _forward_embeddings(self, data_input: SpanCreatorOutput) -> Tuple[Tensor, Tensor]:
        aspects = self.aspect_net(data_input.aspects_agg_emb)
        opinions = self.opinion_net(data_input.opinions_agg_emb)
        return aspects, opinions

    def similarity(self, aspects: Tensor, opinions: Tensor) -> Tensor:
        return torch.bmm(aspects, opinions.transpose(1, 2))

    def normalized_similarity(self, aspects: Tensor, opinions: Tensor) -> Tensor:
        aspects, opinions = expand_aspect_and_opinion(aspects, opinions)

        similarities = self.similarity_metric(aspects, opinions)
        return scale_scores(similarities)


class NonSentimentNeuralTripletExtractorModel(BaseNonSentimentTripletExtractorModel):
    def __init__(self, input_dim: int,
                 config: Dict,
                 model_name: str = 'Neural No Sentiment Triplet Extractor Model',
                 *args, **kwargs
                 ):
        super(NonSentimentNeuralTripletExtractorModel, self).__init__(
            input_dim=input_dim, model_name=model_name, config=config
        )

        input_dimension: int = input_dim * 2

        neurons: List = [input_dim, input_dim]
        self.aspect_net = sequential_blocks(neurons=neurons, device=self.device, is_last=False)
        self.opinion_net = sequential_blocks(neurons=neurons, device=self.device, is_last=False)

        neurons: List = [input_dimension, input_dimension // 2, input_dimension // 4, input_dimension // 8, 1]
        self.similarity_net = sequential_blocks(neurons, self.device)
        self.similarity_metric = torch.nn.Sigmoid()

    def _forward_embeddings(self, data_input: SpanCreatorOutput) -> Tuple[Tensor, Tensor]:
        matrix: Tensor = create_embeddings_matrix_by_concat_tensors(data_input.aspects_agg_emb, data_input.opinions_agg_emb)
        matrix = self.similarity_net(matrix)
        return matrix, matrix

    def similarity(self, aspects: Tensor, opinions: Tensor) -> Tensor:
        return aspects.squeeze(-1)

    def normalized_similarity(self, aspects: Tensor, opinions: Tensor) -> Tensor:
        similarities = self.similarity_metric(aspects)
        return similarities.squeeze(-1)


class NonSentimentAttentionTripletExtractorModel(BaseNonSentimentTripletExtractorModel):
    def __init__(self, input_dim: int,
                 config: Dict,
                 model_name: str = 'Metric No Sentiment Attention Extractor Model',
                 *args, **kwargs
                 ):
        super(NonSentimentAttentionTripletExtractorModel, self).__init__(
            input_dim=input_dim, model_name=model_name, config=config
        )

        neurons: List = [input_dim, input_dim // 2, input_dim // 4]
        self.aspect_net = sequential_blocks(neurons=neurons, device=self.device, is_last=True)
        self.opinion_net = sequential_blocks(neurons=neurons, device=self.device, is_last=True)
        self.keys = sequential_blocks(neurons=neurons, device=self.device, is_last=True)

        self.aspect_att = torch.nn.MultiheadAttention(input_dim // 4, 4, dropout=0.1, batch_first=True)
        self.opinion_att = torch.nn.MultiheadAttention(input_dim // 4, 4, dropout=0.1, batch_first=True)

        self.similarity_metric = torch.nn.CosineSimilarity(dim=-1)

    def _forward_embeddings(self, data_input: SpanCreatorOutput) -> Tuple[Tensor, Tensor]:
        aspects = data_input.aspects_agg_emb
        opinions = data_input.opinions_agg_emb

        o_keys = self.keys(opinions)
        a_keys = self.keys(aspects)
        aspects = self.aspect_net(aspects)
        opinions = self.opinion_net(opinions)

        aspects = self.aspect_att(aspects, opinions, o_keys)[0]
        opinions = self.opinion_att(opinions, aspects, a_keys)[0]

        return aspects, opinions

    def similarity(self, aspects: Tensor, opinions: Tensor) -> Tensor:
        return torch.bmm(aspects, opinions.transpose(1, 2))

    def normalized_similarity(self, aspects: Tensor, opinions: Tensor) -> Tensor:
        aspects, opinions = expand_aspect_and_opinion(aspects, opinions)

        similarities = self.similarity_metric(aspects, opinions)
        return scale_scores(similarities)
