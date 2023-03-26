from enum import IntEnum


class SpanCode(IntEnum):
    NOT_RELEVANT: int = -1
    BEGIN_OPINION: int = 4
    INSIDE_OPINION: int = 3
    BEGIN_ASPECT: int = 2
    INSIDE_ASPECT: int = 1
    NOT_SPLIT: int = 0


class ASTELabels(IntEnum):
    NEU: int = 3
    POS: int = 2
    NEG: int = 1
    NOT_PAIR: int = 0
    NOT_RELEVANT: int = -1
