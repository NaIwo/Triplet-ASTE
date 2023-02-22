from typing import Dict, Callable

import torch
from torch import Tensor

from .triplet_utils import (
    create_embeddings_matrix_by_concat,
    create_mask_matrix_for_training,
    create_mask_matrix_for_validation
)
from ...models import BaseModel
from ....dataset.domain import ASTELabels
from ....models.outputs import ModelOutput, ModelLoss, ModelMetric, TripletModelOutput, SpanCreatorOutput
from ....tools.metrics import Metric, get_selected_metrics


class TripletExtractorModel(BaseModel):
    def __init__(self, input_dim: int, config: Dict, model_name: str = 'Triplet Extractor Model'):
        super(TripletExtractorModel, self).__init__(model_name=model_name, config=config)

        metrics = get_selected_metrics(for_spans=True)
        self.final_metrics: Metric = Metric(name='Final predictions', metrics=metrics).to(
            self.config['general-training']['device'])

        self.create_embeddings_matrix: Callable = create_embeddings_matrix_by_concat
        input_dimension: int = input_dim * 2

    def forward(self, data_input: SpanCreatorOutput) -> TripletModelOutput:
        matrix: Tensor = self.create_embeddings_matrix(data_input)
        mask_matrix: Tensor = self.create_mask_matrix(data_input)
        a = 1

    def create_mask_matrix(self, data: SpanCreatorOutput) -> Tensor:
        if self.training:
            mask_matrix: Tensor = create_mask_matrix_for_training(data)
        else:
            mask_matrix: Tensor = create_mask_matrix_for_validation(data)
        return mask_matrix

    def get_loss(self, model_out: ModelOutput) -> ModelLoss:
        return ModelLoss(triplet_extractor_loss=0, config=self.config)

    def update_metrics(self, model_out: ModelOutput) -> None:
        true_labels: torch.Tensor = self.construct_matrix_labels(model_out.batch, tuple(model_out.predicted_spans))
        true_triplets: torch.Tensor = self.get_triplets_from_matrix(true_labels).to(
            self.config['general-training']['device'])
        total_correct_count: int = self.get_total_correct_triplets_count(model_out.batch)
        predicted_labels: torch.Tensor = torch.argmax(model_out.triplet_results, dim=-1)
        predicted_labels = torch.where(true_labels == ASTELabels.NOT_RELEVANT, true_labels, predicted_labels)
        predicted_triplets: torch.Tensor = self.get_triplets_from_matrix(predicted_labels).to(
            self.config['general-training']['device'])

        self.independent_metrics(predicted_labels.view([-1]), true_labels.view([-1]))
        self.final_metrics(predicted_triplets, true_triplets, full_target_count=total_correct_count)

    def get_metrics(self) -> ModelMetric:
        metrics: Dict = self.final_metrics.compute()
        return ModelMetric(triplet_metric=metrics)

    def reset_metrics(self) -> None:
        self.final_metrics.reset()
