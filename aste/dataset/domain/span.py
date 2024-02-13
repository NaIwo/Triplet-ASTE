from typing import List, TypeVar

S = TypeVar('S', bound='Span')


class Span:
    def __init__(self, start_idx: int, end_idx: int, words: List[str]):
        self.start_idx: int = start_idx
        self.end_idx: int = end_idx
        self.span_words: List[str] = words

    @classmethod
    def from_range(cls, span_range: List[int], sentence: str) -> S:
        if len(span_range) == 1:
            span_range.append(span_range[0])
        words: List[str] = sentence.split()[span_range[0]:span_range[-1] + 1]
        return Span(start_idx=span_range[0], end_idx=span_range[-1], words=words)

    def __str__(self) -> str:
        return str({
            'start idx': self.start_idx,
            'end idx': self.end_idx,
            'span words': self.span_words
        })

    def __repr__(self):
        return self.__str__()

    def __hash__(self):
        return hash(self.__repr__())

    def __eq__(self, other) -> bool:
        return (self.start_idx == other.start_idx) and (self.end_idx == other.end_idx) and (
                self.span_words == other.span_words)

    def __lt__(self, other):
        if self.start_idx != other.start_idx:
            return self.start_idx < other.start_idx
        else:
            return self.end_idx < other.end_idx

    def __gt__(self, other):
        if self.start_idx != other.start_idx:
            return self.start_idx > other.start_idx
        else:
            return self.end_idx > other.end_idx

    def __bool__(self) -> bool:
        return (self.start_idx != -1) and (self.end_idx != -1) and (self.span_words != [])

    def intersect(self, other) -> S:
        start_idx: int = max(self.start_idx, other.start_idx)
        end_idx: int = min(self.end_idx, other.end_idx)
        if end_idx < start_idx:
            return Span(start_idx=-1, end_idx=-1, words=[])
        else:
            span_words: List = self._get_intersected_words(other, start_idx, end_idx)

            return Span(start_idx=start_idx, end_idx=end_idx, words=span_words)

    def _get_intersected_words(self, other, start_idx, end_idx) -> List:
        span_words: List
        if start_idx == self.start_idx:
            if end_idx == self.end_idx:
                span_words = self.span_words[:]
            else:
                span_words = self.span_words[:-(self.end_idx - end_idx)]
        else:
            if end_idx == other.end_idx:
                span_words = other.span_words[:]
            else:
                span_words = other.span_words[:-(other.end_idx - end_idx)]
        return span_words
