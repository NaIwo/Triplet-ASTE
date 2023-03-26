from typing import Dict, List

import torch
from torch import Tensor
from torch.nn import Sequential
import torch.nn.functional as F

from .base_triplets_extractor import BaseTripletExtractorModel
from ..utils import scale_scores
from ..utils import sequential_blocks
from ...outputs import (
    SpanInformationOutput,
    SpanCreatorOutput,
    SampleTripletOutput,
    TripletModelOutput
)
from ...utils.triplet_utils import (
    create_embeddings_matrix_by_concat_tensors
)
from ...utils.triplet_utils import (
    expand_aspect_and_opinion
)
from ....models.outputs import (
    ModelLoss
)


class BaseSentimentTripletExtractorModel(BaseTripletExtractorModel):
    def __init__(self,
                 input_dim: int,
                 config: Dict,
                 model_name: str = 'Base Sentiment Triplet Extractor Model',
                 *args, **kwargs
                 ):
        super(BaseSentimentTripletExtractorModel, self).__init__(
            input_dim=input_dim, model_name=model_name, config=config)

    def _forward_embeddings(self, data_input: SpanCreatorOutput) -> Tensor:
        raise NotImplementedError

    def get_triplets_from_matrix(self, matrix: Tensor, data_input: SpanCreatorOutput) -> List[SampleTripletOutput]:
        triplets: List = list()

        sample: Tensor
        sample_aspects: SpanInformationOutput
        sample_opinions: SpanInformationOutput
        for sample, sample_aspects, sample_opinions in zip(matrix, data_input.aspects, data_input.opinions):
            significant: Tensor = self.threshold_data(sample).nonzero()
            a_ranges: Tensor = sample_aspects.span_range[significant[:, 0]]
            o_ranges: Tensor = sample_opinions.span_range[significant[:, 1]]
            sentiments: Tensor = sample_opinions.sentiments[significant[:, 1]]
            triplets.append(
                SampleTripletOutput(
                    aspect_ranges=a_ranges,
                    opinion_ranges=o_ranges,
                    pred_sentiments=sentiments,
                    sentence=sample_opinions.sentence
                )
            )

        return triplets


class MetricTripletExtractorModel(BaseSentimentTripletExtractorModel):
    def __init__(self, input_dim: int,
                 config: Dict,
                 model_name: str = 'Metric Triplet Extractor Model',
                 *args, **kwargs
                 ):
        super(MetricTripletExtractorModel, self).__init__(input_dim=input_dim, model_name=model_name, config=config)

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

        return matrix


class NeuralTripletExtractorModel(BaseSentimentTripletExtractorModel):
    def __init__(self, input_dim: int,
                 config: Dict,
                 model_name: str = 'Neural Triplet Extractor Model',
                 *args, **kwargs
                 ):
        super(NeuralTripletExtractorModel, self).__init__(input_dim=input_dim, model_name=model_name, config=config)

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

        matrix = self.similarity(matrix)

        return matrix.squeeze(-1)


class NeuralCrossEntropyExtractorModel(NeuralTripletExtractorModel):
    def __init__(self, input_dim: int,
                 config: Dict,
                 model_name: str = 'Neural Cross Entropy Extractor Model',
                 *args, **kwargs
                 ):
        super(NeuralCrossEntropyExtractorModel, self).__init__(input_dim=input_dim, model_name=model_name,
                                                               config=config)

        self.loss = torch.nn.CrossEntropyLoss(ignore_index=-1)

    def get_loss(self, model_out: TripletModelOutput) -> ModelLoss:
        preds = model_out.features.view(-1).unsqueeze(-1)
        preds = torch.cat([torch.ones_like(preds) - preds, preds], dim=-1)

        labels = model_out.loss_mask.view(-1).to(torch.long)
        labels = torch.where(model_out.pad_mask.view(-1), labels, torch.tensor(-1).to(self.device))

        loss = self.loss(preds, labels)
        full_loss = ModelLoss(
            config=self.config,
            losses={
                'triplet_extractor_loss': loss * self.config['model']['triplet-extractor'][
                    'loss-weight'] * self.trainable,
            }
        )
        return full_loss


class AttentionTripletExtractorModel(BaseSentimentTripletExtractorModel):
    def __init__(self, input_dim: int,
                 config: Dict,
                 model_name: str = 'Attention Triplet Extractor Model',
                 *args, **kwargs
                 ):
        super(AttentionTripletExtractorModel, self).__init__(input_dim=input_dim, model_name=model_name, config=config)

        neurons: List = [input_dim, input_dim // 2, input_dim // 2, input_dim]
        self.query = sequential_blocks(neurons=neurons, device=self.device)
        self.key = sequential_blocks(neurons=neurons, device=self.device)
        neurons: List = [input_dim, input_dim // 2, input_dim // 4, 1]
        self.value = sequential_blocks(neurons=neurons, device=self.device)

        self.softmax = torch.nn.Softmax(dim=-1)

    def _forward_embeddings(self, data_input: SpanCreatorOutput) -> Tensor:
        queries = self.query(data_input.aspects_agg_emb)
        keys = self.key(data_input.opinions_agg_emb)
        values = self.value(data_input.opinions_agg_emb)

        matrix: Tensor = torch.matmul(queries, keys.transpose(-1, -2)) / torch.sqrt(
            torch.tensor(keys.shape[-1], dtype=torch.float32))
        matrix = self.softmax(matrix)
        matrix = torch.matmul(matrix, values)
        matrix = F.sigmoid(matrix)

        return matrix
