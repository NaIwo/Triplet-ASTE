from functools import lru_cache
from typing import List, Union, Optional

import torch
from aste.configs import config
from torch.nn.utils.rnn import pad_sequence

from .. import ModelOutput, ModelLoss, ModelMetric, BaseModel
from ...losses import DiceLoss
from ...tools.metrics import Metric, get_selected_metrics


class Selector(BaseModel):
    def __init__(self, input_dim: int, model_name: str = 'Proper Span Selector Model'):
        super(Selector, self).__init__(model_name=model_name)
        self._ignore_index: int = -1
        self.selector_loss = DiceLoss(ignore_index=self._ignore_index,
                                      alpha=config['model']['selector']['dice-loss-alpha'])

        metrics: List = get_selected_metrics()
        self.metrics: Metric = Metric(name='Span Selector Metrics', metrics=metrics,
                                      ignore_index=self._ignore_index).to(config['general-training']['device'])

        self.dropout = torch.nn.Dropout(0.1)
        self.linear_layer_1 = torch.nn.Linear(input_dim, 300)
        self.linear_layer_2 = torch.nn.Linear(300, 100)
        self.final_layer = torch.nn.Linear(100, 1)
        self.sigmoid = torch.nn.Sigmoid()
        self.sigmoid_multiplication: float = 1.

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        layer: torch.nn.Linear
        for layer in [self.linear_layer_1, self.linear_layer_2]:
            data = layer(data)
            data = torch.relu(data)
            data = self.dropout(data)
        data = self.final_layer(data) * self.sigmoid_multiplication
        return self.sigmoid(data)

    def get_loss(self, model_out: ModelOutput) -> ModelLoss:
        true_labels: torch.Tensor = self.get_labels(model_out, pad_with=self._ignore_index)
        loss: torch.Tensor = self.selector_loss(
            model_out.span_selector_output.view([-1, model_out.span_selector_output.shape[-1]]), true_labels.view([-1]))
        return ModelLoss(span_selector_loss=loss)

    def update_metrics(self, model_out: ModelOutput) -> None:
        true_labels: torch.Tensor = self.get_labels(model_out, pad_with=self._ignore_index)
        self.metrics(model_out.span_selector_output.view([-1]), true_labels.view([-1]))

    @staticmethod
    @lru_cache(maxsize=None)
    def get_labels(model_out: ModelOutput, pad_with: Optional[int] = None) -> Union[List, torch.Tensor]:
        results: List = list()

        spans_idx: int
        spans: torch.Tensor
        for spans_idx, spans in enumerate(model_out.predicted_spans):
            aspects: torch.Tensor = model_out.batch.aspect_spans[spans_idx]
            opinions: torch.Tensor = model_out.batch.opinion_spans[spans_idx]
            true_spans: torch.Tensor = torch.cat([aspects, opinions], dim=0)
            labels: torch.Tensor = torch.tensor([1 if span in true_spans else 0 for span in spans],
                                                device=config['general-training']['device'])
            results.append(labels)
        if pad_with is not None:
            results: torch.Tensor = pad_sequence(results, padding_value=pad_with, batch_first=True)
        return results

    def get_metrics(self) -> ModelMetric:
        return ModelMetric(span_selector_metric=self.metrics.compute())

    def reset_metrics(self) -> None:
        self.metrics.reset()
