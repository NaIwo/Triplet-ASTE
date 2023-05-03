import json
import os
from typing import TypeVar, Optional, Dict, Tuple

from torch import Tensor
from torchmetrics import MetricCollection

from .outputs import FinalTriplets
from ...tools.metrics import get_selected_metrics

MM = TypeVar('MM', bound='ModelMetric')


class ModelMetric:
    NAME: str = 'Metrics'

    def __init__(self, *, metrics: Optional[Dict[str, Dict[str, Tensor]]] = None):
        self.metrics: Dict = metrics if metrics is not None else {}

    def update(self, metric: MM) -> MM:
        self.metrics.update(metric.metrics)
        return self

    def to_json(self, path: str) -> None:
        os.makedirs(path[:path.rfind(os.sep)], exist_ok=True)
        with open(path, 'a') as f:
            json.dump(self.metrics, f)

    def __repr__(self) -> str:
        return self.__str__()

    def __str__(self):
        return str(self.metrics)

    def __iter__(self):
        for metrics in self.metrics:
            yield metrics

    def __getitem__(self, item):
        return self.metrics[item]

    def metrics_with_prefix(self, prefix: str) -> Tuple:
        name: str
        score: Tensor
        for name, score in self.metrics.items():
            for k, v in score.items():
                yield f'{prefix}__{name}_{k}', v


class FinalMetric:
    def __init__(self):
        metrics = get_selected_metrics(for_spans=True, dist_sync_on_step=True)
        self.final_metrics: MetricCollection = MetricCollection(metrics=metrics)

    def update_metrics(self, model_out: FinalTriplets) -> None:
        tp = 0
        tp_fp = 0
        tp_fn = 0
        for true, pred in zip(model_out.true_triplets, model_out.pred_triplets):
            tp_fn += len(true)
            tp_fp += len(pred)
            tp += len(set(true).intersection(set(pred)))
        self.final_metrics.update(tp=tp, tp_fp=tp_fp, tp_fn=tp_fn)

    def get_metrics(self) -> ModelMetric:
        return ModelMetric(
            metrics={
                'final_metric': self.final_metrics.compute(),
            }
        )

    def reset_metrics(self) -> None:
        self.final_metrics.reset()
