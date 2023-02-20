import json
import os
from typing import TypeVar, Optional, Dict

from torch import Tensor

MM = TypeVar('MM', bound='ModelMetric')


class ModelMetric:
    NAME: str = 'Metrics'

    def __init__(self, *, span_creator_metric: Optional[Dict] = None, triplet_metric: Optional[Dict] = None):
        self.span_creator_metric: Optional[Dict] = span_creator_metric
        self.triplet_metric: Optional[Dict] = triplet_metric

    @classmethod
    def from_instances(cls, *, span_creator_metric: MM, triplet_metric: MM) -> MM:
        return cls(
            span_creator_metric=span_creator_metric.span_creator_metric,
            triplet_metric=triplet_metric.triplet_metric
        )

    @property
    def _all_metrics(self) -> Dict:
        return {
            'span_creator_metrics': self.span_creator_metric,
            'triplet_metric': self.triplet_metric
        }

    def to_json(self, path: str) -> None:
        os.makedirs(path[:path.rfind(os.sep)], exist_ok=True)
        with open(path, 'a') as f:
            json.dump(self._all_metrics, f)

    def __repr__(self) -> str:
        return self.__str__()

    def __str__(self):
        return str(self._all_metrics)

    def __iter__(self):
        for metrics in self._all_metrics:
            yield metrics

    def metrics(self, prefix: str) -> Dict:
        name: str
        score: Tensor
        return {f'{prefix}__{name}': score for name, score in self._all_metrics.items()}
