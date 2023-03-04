from typing import List

from torch import Tensor

from ...dataset.domain import Sentence, Span


def construct_predicted_spans(span_range: Tensor, sentence: Sentence) -> List[Span]:
    spans: List[Span] = list()
    for s_range in span_range:
        s_range = [
            sentence.get_index_before_encoding(s_range[0].item()),
            sentence.get_index_before_encoding(s_range[1].item())
        ]
        spans.append(Span.from_range(s_range, sentence.sentence))
    return spans
