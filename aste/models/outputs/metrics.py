import json
import os
from typing import TypeVar, Optional, Dict, Tuple

from torch import Tensor

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

    def metrics_with_prefix(self, prefix: str) -> Tuple:
        name: str
        score: Tensor
        for name, score in self.metrics.items():
            for k, v in score.items():
                yield f'{prefix}__{name}_{k}', v
