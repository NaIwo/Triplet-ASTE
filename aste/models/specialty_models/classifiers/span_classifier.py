from typing import Dict, List

import torch

from .classifier_utils import get_labels_for_task
from ..utils import sequential_blocks
from ...base_model import BaseModel
from ...outputs import (
    ClassificationModelOutput,
    SpanCreatorOutput
)
from ...utils.const import CreatedSpanCodes
from ....losses.dice_loss import DiceLoss
from ....losses.focal_loss import FocalLoss
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

        self.loss = DiceLoss(ignore_index=CreatedSpanCodes.NOT_RELEVANT.value, alpha=0.7, with_logits=False)

        neurons: List = [
            input_dim,
            input_dim // 2,
            input_dim // 4,
            input_dim // 8,
            1
        ]
        self.aspect_net = sequential_blocks(neurons=neurons, is_last=True, device=self.device)
        self.opinion_net = sequential_blocks(neurons=neurons, is_last=True, device=self.device)
        self.aspect_net.append(torch.nn.Sigmoid())
        self.opinion_net.append(torch.nn.Sigmoid())

    def forward(self, data_input: SpanCreatorOutput) -> ClassificationModelOutput:
        aspect_predictions = self.aspect_net(data_input.aspects_agg_emb)
        opinion_predictions = self.opinion_net(data_input.opinions_agg_emb)

        aspect_labels = data_input.get_aspect_span_creation_info()
        aspect_labels = get_labels_for_task(aspect_labels)

        opinion_labels = data_input.get_opinion_span_creation_info()
        opinion_labels = get_labels_for_task(opinion_labels)

        return ClassificationModelOutput(
            batch=data_input.batch,
            aspect_predictions=aspect_predictions,
            opinion_predictions=opinion_predictions,
            aspect_labels=aspect_labels,
            opinion_labels=opinion_labels
        )

    def get_loss(self, model_out: ClassificationModelOutput) -> ModelLoss:
        a_preds = torch.cat([
            1 - model_out.aspect_predictions,
            model_out.aspect_predictions
        ], dim=-1).to(model_out.aspect_labels.device).flatten(0, 1)
        o_preds = torch.cat([
            1 - model_out.opinion_predictions,
            model_out.opinion_predictions
        ], dim=-1).to(model_out.opinion_labels.device).flatten(0, 1)

        preds = torch.cat([a_preds, o_preds], dim=0)
        true = torch.cat([model_out.aspect_labels.flatten(0, 1), model_out.opinion_labels.flatten(0, 1)], dim=0)

        loss = self.loss(preds, true)

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
