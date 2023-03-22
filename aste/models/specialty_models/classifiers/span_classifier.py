from typing import Dict, List

import torch
from torch import Tensor

from ..utils import sequential_blocks
from ...base_model import BaseModel
from ...outputs import (
    ClassificationModelOutput,
    SpanCreatorOutput
)
from ...utils.const import CreatedSpanCodes
from ....models.outputs import (
    ModelLoss,
    ModelMetric
)
from ....tools.metrics import get_selected_metrics, CustomMetricCollection


class SpanClassifierModel(BaseModel):
    def __init__(self, input_dim: int, config: Dict, model_name: str = 'Span Classifier Model', *args, **kwargs):
        super(SpanClassifierModel, self).__init__(model_name=model_name, config=config)

        metrics = get_selected_metrics(
            dist_sync_on_step=True,
            ignore_index=CreatedSpanCodes.NOT_RELEVANT.value
        )
        self.metrics = CustomMetricCollection(
            name='Span classifier',
            ignore_index=CreatedSpanCodes.NOT_RELEVANT.value,
            metrics=metrics
        )

        self.loss = torch.nn.CrossEntropyLoss(ignore_index=CreatedSpanCodes.NOT_RELEVANT.value)

        neurons: List = [
            input_dim,
            input_dim // 2,
            input_dim // 2,
            input_dim
        ]
        self.aspect_net = sequential_blocks(neurons=neurons, is_last=False, device=self.device)
        self.opinion_net = sequential_blocks(neurons=neurons, is_last=False, device=self.device)

        neurons: List = [
            input_dim,
            input_dim // 4,
            1
        ]
        self.prediction_net = sequential_blocks(neurons=neurons, device=self.device)
        self.prediction_net.append(
            torch.nn.Sigmoid()
        )

    def forward(self, data_input: SpanCreatorOutput) -> ClassificationModelOutput:
        aspects = self.aspect_net(data_input.aspects_agg_emb)
        opinions = self.opinion_net(data_input.opinions_agg_emb)

        aspect_predictions = self.prediction_net(aspects)
        opinion_predictions = self.prediction_net(opinions)

        aspect_labels = data_input.get_aspect_span_creation_info()
        aspect_labels = self.get_labels_for_task(aspect_labels)

        opinion_labels = data_input.get_opinion_span_creation_info()
        opinion_labels = self.get_labels_for_task(opinion_labels)

        return ClassificationModelOutput(
            batch=data_input.batch,
            aspect_features=aspects,
            opinion_features=opinions,
            aspect_predictions=aspect_predictions.squeeze(dim=-1),
            opinion_predictions=opinion_predictions.squeeze(dim=-1),
            aspect_labels=aspect_labels,
            opinion_labels=opinion_labels
        )

    @staticmethod
    def get_labels_for_task(labels: Tensor) -> Tensor:
        labels = labels.clone()
        con = (labels == CreatedSpanCodes.ADDED_TRUE) | (labels == CreatedSpanCodes.PREDICTED_TRUE)
        labels = torch.where(con, 1., labels)
        con = (labels == CreatedSpanCodes.ADDED_FALSE) | (labels == CreatedSpanCodes.PREDICTED_FALSE)
        labels = torch.where(con, 0., labels)
        return labels

    def get_loss(self, model_out: ClassificationModelOutput) -> ModelLoss:
        loss = self.loss(model_out.aspect_predictions, model_out.aspect_labels)
        loss += self.loss(model_out.opinion_predictions, model_out.opinion_labels)

        full_loss = ModelLoss(
            config=self.config,
            losses={
                'spans_classifier_loss': loss * self.config['model']['span-classifier'][
                    'loss-weight'] * self.trainable,
            }
        )
        return full_loss

    def update_metrics(self, model_out: ClassificationModelOutput) -> None:
        self.metrics.update((model_out.aspect_predictions > 0.5).view([-1]), model_out.aspect_labels.view([-1]))
        self.metrics.update((model_out.opinion_predictions > 0.5).view([-1]), model_out.opinion_labels.view([-1]))

    def get_metrics(self) -> ModelMetric:
        return ModelMetric(
            metrics={
                'span_classifier_metric': self.metrics.compute(),
            }
        )

    def reset_metrics(self) -> None:
        self.metrics.reset()
