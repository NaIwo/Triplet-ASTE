from typing import TypeVar, Union

SC = TypeVar('SC', bound='StatsCounter')


class StatsCounter:
    def __init__(self, numerator: float = 0.0, *, denominator: float = 0.0):
        self.numerator: float = numerator
        self.denominator: float = denominator
        self.tp: float = 0.0
        self.fp: float = 0.0
        self.fn: float = 0.0

    def update_tp_fp_fn(self, true_positive: float, false_positive: float, false_negative: float):
        self.tp = self.tp + true_positive
        self.fp = self.fp + false_positive
        self.fn = self.fn + false_negative

    def __radd__(self, other: 'StatsCounter'):
        return self.__add__(other)

    def __add__(self, other: 'StatsCounter'):
        result = StatsCounter(numerator=self.numerator + other.numerator,
                              denominator=self.denominator + other.denominator)
        result.tp = self.tp + other.tp
        result.fp = self.fp + other.fp
        result.fn = self.fn + other.fn
        return result

    def __repr__(self):
        return round(self.numerator / self.denominator, 4) if self.denominator else int(self.numerator)

    def __str__(self):
        return str(round(self.numerator / self.denominator, 4)) if self.denominator else str(int(self.numerator))

    def number(self) -> Union[int, float]:
        return round(self.numerator / self.denominator, 4) if self.denominator else int(self.numerator)

    def count(self) -> int:
        return int(self.denominator)

    def precision(self) -> float:
        if self.tp + self.fp == 0:
            return 0.0
        return self.tp / (self.tp + self.fp)

    def recall(self) -> float:
        if self.tp + self.fn == 0:
            return 0.0
        return self.tp / (self.tp + self.fn)

    def f1(self) -> float:
        precision = self.precision()
        recall = self.recall()
        if precision + recall == 0:
            return 0.0
        return 2 * (precision * recall) / (precision + recall)
