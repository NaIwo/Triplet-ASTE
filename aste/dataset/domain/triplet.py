from typing import TypeVar, Tuple, Optional

from .span import Span
from .const import ASTELabels

T = TypeVar('T', bound='Triplet')


class Triplet:
    def __init__(self, aspect_span: Span, opinion_span: Span, sentiment: str):
        self.aspect_span: Span = aspect_span
        self.opinion_span: Span = opinion_span
        self.sentiment: str = sentiment
        self.sentiment_code: int = ASTELabels[self.sentiment]

    @classmethod
    def from_triplet_info(cls, triplet_info: Tuple, sentence: str) -> T:
        return Triplet(
            aspect_span=Span.from_range(triplet_info[0], sentence),
            opinion_span=Span.from_range(triplet_info[1], sentence),
            sentiment=triplet_info[2]
        )

    def __hash__(self):
        return hash(self.__str__())

    def __str__(self) -> str:
        return str({
            'aspect span': self.aspect_span,
            'opinion span': self.opinion_span,
            'sentiment': self.sentiment
        })

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other) -> bool:
        return (self.aspect_span == other.aspect_span) and (self.opinion_span == other.opinion_span) and (
                self.sentiment == other.sentiment)

    def __lt__(self, other):
        if self.aspect_span != other.aspect_span:
            return self.aspect_span < other.aspect_span
        else:
            return self.opinion_span < other.opinion_span

    def __gt__(self, other):
        if self.aspect_span != other.aspect_span:
            return self.aspect_span > other.aspect_span
        else:
            return self.opinion_span > other.opinion_span

    def __bool__(self) -> bool:
        return bool(self.aspect_span) and bool(self.opinion_span)

    def intersect(self, other) -> Optional[T]:
        return Triplet(
            aspect_span=self.aspect_span.intersect(other.aspect_span),
            opinion_span=self.opinion_span.intersect(other.opinion_span),
            sentiment='NOT_PAIR' if self.sentiment != other.sentiment else self.sentiment
        )
