from typing import Optional, Dict, List

import torch
from aste.utils import ignore_index
from torch import Tensor
from torchmetrics import FBetaScore, Accuracy, Precision, Recall, F1Score
from torchmetrics import Metric as TorchMetric
from torchmetrics import MetricCollection


class Metric(MetricCollection):
    def __init__(self, name: str, ignore_index: Optional[int] = None, *args, **kwargs):
        self.ignore_index = ignore_index
        self.name: str = name
        super().__init__(*args, **kwargs)

    @ignore_index
    def forward(self, preds, target, *args, **kwargs):
        super(Metric, self).forward(preds, target, *args, **kwargs)

    def compute(self):
        computed: Dict = super(Metric, self).compute()
        for metric_name, score in computed.items():
            computed[metric_name] = float(score)
        return computed


class SpanMetric(TorchMetric):
    def __init__(self, dist_sync_on_step: bool = False):
        TorchMetric.__init__(self, dist_sync_on_step=dist_sync_on_step)

        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total_predicted", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total_target", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: int, target_in_stage: int, full_target_count: Optional[int] = None) -> None:
        self.correct += target_in_stage
        self.total_predicted += preds
        self.total_target += full_target_count

    def compute(self) -> float:
        raise NotImplemented

    @staticmethod
    def safe_div(dividend: float, divider: float) -> float:
        return dividend / divider if divider != 0. else 0.


class SpanPrecision(SpanMetric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

    def compute(self) -> float:
        return self.safe_div(self.correct, self.total_predicted)


class SpanRecall(SpanMetric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

    def compute(self) -> float:
        return self.safe_div(self.correct, self.total_target)


class SpanF1(SpanMetric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

    def compute(self) -> float:
        precision: float = self.safe_div(self.correct, self.total_predicted)
        recall: float = self.safe_div(self.correct, self.total_target)

        return self.safe_div(2 * (precision * recall), (precision + recall))


def get_selected_metrics(num_classes: int = 1, task: str = 'binary', for_spans: bool = False) -> List:
    if for_spans:
        return [
            SpanPrecision(),
            SpanRecall(),
            SpanF1()
        ]
    else:
        return [
            Precision(num_classes=num_classes, task=task),
            Recall(num_classes=num_classes, task=task),
            Accuracy(num_classes=num_classes, task=task),
            FBetaScore(num_classes=num_classes, task=task, beta=0.5),
            F1Score(num_classes=num_classes, task=task)
        ]
