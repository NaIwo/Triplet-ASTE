from typing import List, Callable

import torch
from torch import Tensor

from ...utils.const import CreatedSpanCodes
from ....dataset.domain import ASTELabels
from ....dataset.reader import Batch


class SpanInformationManager:
    def __init__(self):
        self.span_ranges: List[List[int, int]] = list()
        self.span_creation_info: List[int] = list()
        self.sentiments: List[int] = list()
        self.mapping_indexes: List[int] = list()

    def add_true_information(self, sample: Batch, source: str, code: int = CreatedSpanCodes.ADDED_TRUE) -> None:
        true_spans: Tensor = getattr(sample, f'{source.lower()}_spans')[0]
        mask = (true_spans != torch.tensor([-1, -1]).to(true_spans)).all(dim=1)
        true_spans = true_spans[mask]
        if true_spans.nelement() == 0:
            return

        true_spans: Tensor
        unique_idx: Tensor
        true_spans, mapping_idx = torch.unique(true_spans, return_inverse=True, dim=0)
        self.span_ranges += true_spans.tolist()
        self.mapping_indexes += mapping_idx.tolist()
        self.span_creation_info += [code] * true_spans.shape[0]
        self.sentiments += sample.sentiments[0][mask].tolist()

    def extend_span_ranges(self, sample: Batch, extend_ranges: List[List[int]], max_number: int = 1000) -> None:
        before: Callable = sample.sentence_obj[0].get_index_before_encoding
        after: Callable = sample.sentence_obj[0].get_index_after_encoding

        span_ranges = self.span_ranges[:]

        begin: int
        end: int
        for begin, end in span_ranges:
            if len(self.span_ranges) >= max_number:
                return
            for shift_l, shift_r in extend_ranges:
                self.add_predicted_information(
                    after(before(begin) + shift_l),
                    after(before(end) + shift_r)
                )

    def add_span_manager(self, span_manager: 'SpanInformationManager') -> None:
        for (b_idx, e_idx) in span_manager.span_ranges:
            self.add_predicted_information(b_idx, e_idx)

    def add_predicted_information(self, b_idx: int, end_idx: int) -> None:
        span_range: List = [b_idx, end_idx]
        if span_range in self.span_ranges:
            index: int = self.span_ranges.index(span_range)
            if self.span_creation_info[index] == CreatedSpanCodes.ADDED_TRUE:
                self.span_creation_info[index] = CreatedSpanCodes.PREDICTED_TRUE
        elif (end_idx >= b_idx) and b_idx >= 0:
            self.span_ranges += [span_range]
            self.span_creation_info += [CreatedSpanCodes.PREDICTED_FALSE]
            self.sentiments += [ASTELabels.NOT_PAIR]
            self.mapping_indexes += [-1]
