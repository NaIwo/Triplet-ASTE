from enum import IntEnum


class CreatedSpanCodes(IntEnum):
    NOT_RELEVANT: int = -1

    ADDED_FALSE: int = 0
    ADDED_TRUE: int = 1

    PREDICTED_FALSE: int = 2
    PREDICTED_TRUE: int = 3


class TripletDimensions(IntEnum):
    OPINION: int = 1
    ASPECT: int = 2
