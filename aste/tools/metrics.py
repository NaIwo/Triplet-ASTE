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
    def forward(self, *args, **kwargs):
        super(Metric, self).forward(*args, **kwargs)

    def compute(self):
        computed: Dict = super(Metric, self).compute()
        for metric_name, score in computed.items():
            computed[metric_name] = float(score)
        return computed


class SpanMetric(TorchMetric):
    def __init__(self, dist_sync_on_step: bool = False):
        TorchMetric.__init__(self, dist_sync_on_step=dist_sync_on_step)

        self.add_state("tp", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("tp_fp", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("tp_fn", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, tp: int, tp_fp: int, tp_fn: Optional[int] = None) -> None:
        self.tp += tp
        self.tp_fp += tp_fp
        self.tp_fn += tp_fn

    def compute(self) -> float:
        raise NotImplemented

    @staticmethod
    def safe_div(dividend: float, divider: float) -> float:
        return dividend / divider if divider != 0. else 0.


class SpanPrecision(SpanMetric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

    def compute(self) -> float:
        return self.safe_div(self.tp, self.tp_fp)


class SpanRecall(SpanMetric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

    def compute(self) -> float:
        return self.safe_div(self.tp, self.tp_fn)


class SpanF1(SpanMetric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

    def compute(self) -> float:
        precision: float = self.safe_div(self.tp, self.tp_fp)
        recall: float = self.safe_div(self.tp, self.tp_fn)

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
