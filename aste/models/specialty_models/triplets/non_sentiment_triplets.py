from typing import Dict, List

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
    expand_aspect_and_opinion,
    create_embeddings_matrix_by_concat_tensors,
    create_sentiment_matrix
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

    def _forward_embeddings(self, data_input: SpanCreatorOutput) -> Tensor:
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

        self.aspect_net = sequential_blocks([input_dim, input_dim], device=self.device, is_last=False)
        self.opinion_net = sequential_blocks([input_dim, input_dim], device=self.device, is_last=False)

        neurons: List = [
            input_dim,
            input_dim // 2,
            input_dim // 4,
            input_dim // 2
        ]
        self.span_net = sequential_blocks(neurons=neurons, device=self.device)

        self.similarity_metric = torch.nn.CosineSimilarity(dim=-1)

    def _forward_embeddings(self, data_input: SpanCreatorOutput) -> Tensor:
        aspects = self.aspect_net(data_input.aspects_agg_emb)
        opinions = self.opinion_net(data_input.opinions_agg_emb)

        aspects = self.span_net(aspects)
        opinions = self.span_net(opinions)

        aspects, opinions = expand_aspect_and_opinion(aspects, opinions)

        matrix: Tensor = self.similarity_metric(aspects, opinions)
        matrix = scale_scores(matrix)

        return matrix