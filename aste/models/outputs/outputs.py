from typing import TypeVar, Optional

from torch import Tensor

from ...dataset.reader import Batch

MO = TypeVar('MO', bound='BaseModelOutput')


class BaseModelOutput:
    NAME: str = 'Outputs'

    def __init__(self, batch: Optional[Batch] = None, features: Optional[Tensor] = None):
        self.batch: Optional[Batch] = batch
        self.features: Optional[Tensor] = features

    def __repr__(self) -> str:
        return self.__str__()

    def release_memory(self) -> MO:
        self.batch = None
        return self
