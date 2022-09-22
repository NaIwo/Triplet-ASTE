from functools import lru_cache
from typing import List, Union, Optional

import torch
from torch.nn.utils.rnn import pad_sequence

from ASTE.aste.losses import DiceLoss
from ASTE.aste.models import ModelOutput, ModelLoss, ModelMetric, BaseModel
from ASTE.aste.tools.metrics import Metric, get_selected_metrics
from ASTE.dataset.domain.const import SpanLabels
from ASTE.utils import config


class Classifier(BaseModel):
    def __init__(self, input_dim: int, model_name: str = 'Proper Span Classifier Model'):
        super(Classifier, self).__init__(model_name=model_name)
        self._ignore_index: int = -1
        self.classifier_loss = DiceLoss(ignore_index=self._ignore_index,
                                        alpha=config['model']['classifier']['dice-loss-alpha'])

        metrics: List = get_selected_metrics(multiclass=True, num_classes=3)
        self.metrics: Metric = Metric(name='Span Classifier Metrics', metrics=metrics,
                                      ignore_index=self._ignore_index).to(config['general']['device'])

        self.dropout = torch.nn.Dropout(0.1)
        self.linear_layer_1 = torch.nn.Linear(input_dim, 300)
        self.linear_layer_2 = torch.nn.Linear(300, 100)
        self.final_layer = torch.nn.Linear(100, 3)
        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        layer: torch.nn.Linear
        for layer in [self.linear_layer_1, self.linear_layer_2]:
            data = layer(data)
            data = torch.relu(data)
            data = self.dropout(data)
        data = self.final_layer(data)
        return self.softmax(data)

    def get_loss(self, model_out: ModelOutput) -> ModelLoss:
        true_labels: torch.Tensor = self.get_labels(model_out, pad_with=self._ignore_index)
        loss: torch.Tensor = self.classifier_loss(
            model_out.span_classifier_output.view([-1, model_out.span_classifier_output.shape[-1]]),
            true_labels.view([-1]))
        return ModelLoss(span_classifier_loss=loss)

    def update_metrics(self, model_out: ModelOutput) -> None:
        true_labels: torch.Tensor = self.get_labels(model_out, pad_with=self._ignore_index)
        self.metrics(model_out.span_classifier_output.view([-1]), true_labels.view([-1]))

    @staticmethod
    @lru_cache(maxsize=None)
    def get_labels(model_out: ModelOutput, pad_with: Optional[int] = None) -> Union[List, torch.Tensor]:
        results: List = list()

        spans_idx: int
        spans: torch.Tensor
        for spans_idx in range(len(model_out.predicted_spans)):
            labels: torch.Tensor = Classifier.create_labels(model_out, spans_idx)
            results.append(labels)
        if pad_with is not None:
            results: torch.Tensor = pad_sequence(results, padding_value=pad_with, batch_first=True)
        return results

    @staticmethod
    @lru_cache(maxsize=None)
    def create_labels(model_out: ModelOutput, spans_idx: int) -> torch.Tensor:
        aspects: torch.Tensor = model_out.batch.aspect_spans[spans_idx]
        opinions: torch.Tensor = model_out.batch.opinion_spans[spans_idx]

        labels: List = list()
        span: torch.Tensor
        for span in model_out.predicted_spans[spans_idx]:
            if span in aspects:
                labels.append(SpanLabels.ASPECT)
            elif span in opinions:
                labels.append(SpanLabels.OPINION)
            else:
                labels.append(SpanLabels.NOT_RELEVANT)

        return torch.tensor(labels, device=config['general']['device'])

    def get_metrics(self) -> ModelMetric:
        return ModelMetric(span_classifier_metric=self.metrics.compute())

    def reset_metrics(self) -> None:
        self.metrics.reset()
