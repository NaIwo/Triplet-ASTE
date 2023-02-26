from typing import Dict, Callable, List

import torch
from torch import Tensor
from torch.nn import Sequential

from .triplet_utils import (
    create_embeddings_matrix_by_concat,
    create_mask_matrix_for_loss,
    create_mask_matrix_for_prediction,
    create_embedding_mask_matrix
)
from ..utils import sequential_blocks
from ASTE.aste.models.full_models.triplet_model import BaseModel
from ....dataset.domain import ASTELabels
from ....models.outputs import (
    ModelLoss,
    ModelMetric
)
from .triplet_outputs import SampleTripletOutput, TripletModelOutput
from ..spans.span_outputs import SpanInformationOutput, SpanPredictionsOutput, SpanCreatorOutput
from ....tools.metrics import Metric, get_selected_metrics


class TripletExtractorModel(BaseModel):
    def __init__(self, input_dim: int, config: Dict, model_name: str = 'Triplet Extractor Model'):
        super(TripletExtractorModel, self).__init__(model_name=model_name, config=config)

        metrics = get_selected_metrics(for_spans=True)
        self.final_metrics: Metric = Metric(name='Final predictions', metrics=metrics).to(
            self.config['general-training']['device'])

        input_dimension: int = input_dim * 2

        neurons: List = [input_dimension, input_dimension // 2, input_dimension // 4, input_dimension // 8, 1]
        self.similarity: Sequential = sequential_blocks(neurons, self.config)
        self.similarity.append(torch.nn.Sigmoid())

    def forward(self, data_input: SpanCreatorOutput) -> TripletModelOutput:
        matrix: Tensor = self._forward_embeddings(data_input)

        loss_mask: Tensor = create_mask_matrix_for_loss(data_input)
        prediction_mask: Tensor = create_mask_matrix_for_prediction(data_input)

        triplets: List[SampleTripletOutput] = self.get_triplets_from_matrix(matrix * prediction_mask, data_input)
        return TripletModelOutput(
            batch=data_input.batch,
            triplets=triplets,
            similarities=matrix,
            loss_mask=loss_mask,
            prediction_mask=prediction_mask
        )

    def _forward_embeddings(self, data_input: SpanCreatorOutput) -> Tensor:
        matrix: Tensor = create_embeddings_matrix_by_concat(data_input)
        mask: Tensor = create_embedding_mask_matrix(data_input)
        matrix = self.similarity(matrix)
        return matrix.squeeze(-1) * mask

    def get_triplets_from_matrix(self, matrix: Tensor, data_input: SpanCreatorOutput) -> List[SampleTripletOutput]:
        thr: float = self.config['model']['triplet-extractor']['threshold']

        triplets: List = list()

        ps: SpanPredictionsOutput = data_input.predicted_spans

        sample: Tensor
        sample_aspects: SpanInformationOutput
        sample_opinions: SpanInformationOutput
        for sample, sample_aspects, sample_opinions in zip(matrix, ps.aspects, ps.opinions):
            significant: Tensor = (sample >= thr).nonzero()
            a_ranges: Tensor = sample_aspects.span_range[significant[:, 0]]
            o_ranges: Tensor = sample_opinions.span_range[significant[:, 1]]
            sentiments: Tensor = sample_opinions.sentiments[significant[:, 1]]
            triplets.append(
                SampleTripletOutput(
                    aspect_ranges=a_ranges,
                    opinion_ranges=o_ranges,
                    sentiments=sentiments,
                    sentence=sample_opinions.sentence
                )
            )

        return triplets

    def get_loss(self, model_out: TripletModelOutput) -> ModelLoss:

        return ModelLoss(triplet_extractor_loss=0, config=self.config)

    def update_metrics(self, model_out: TripletModelOutput) -> None:
        true_labels: torch.Tensor = self.construct_matrix_labels(model_out.batch, tuple(model_out.predicted_spans))
        true_triplets: torch.Tensor = self.get_triplets_from_matrix(true_labels).to(
            self.config['general-training']['device'])
        total_correct_count: int = self.get_total_correct_triplets_count(model_out.batch)
        predicted_labels: torch.Tensor = torch.argmax(model_out.triplet_results, dim=-1)
        predicted_labels = torch.where(true_labels == ASTELabels.NOT_RELEVANT, true_labels, predicted_labels)
        predicted_triplets: torch.Tensor = self.get_triplets_from_matrix(predicted_labels).to(
            self.config['general-training']['device'])

        self.final_metrics(predicted_triplets, true_triplets, full_target_count=total_correct_count)

    def get_metrics(self) -> ModelMetric:
        metrics: Dict = self.final_metrics.compute()
        return ModelMetric(triplet_metric=metrics)

    def reset_metrics(self) -> None:
        self.final_metrics.reset()
