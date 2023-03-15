from enum import IntEnum


class CreatedSpanCodes(IntEnum):
    NOT_RELEVANT = -1

    ADDED_FALSE = 0
    ADDED_TRUE = 1

    PREDICTED_FALSE = 2
    PREDICTED_TRUE = 3


class TripletDimensions(IntEnum):
    OPINION = 2
    ASPECT = 1
