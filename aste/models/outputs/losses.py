import json
import os
from typing import TypeVar, Optional, Dict

import torch
from torch import Tensor

ML = TypeVar('ML', bound='ModelLoss')


class ModelLoss:
    NAME: str = 'Losses'

    def __init__(
            self, *,
            losses: Optional[Dict[str, Tensor]] = None,
            config: Dict,
    ):

        self.losses: Dict = losses if losses is not None else {}
        self.config: Dict = config

        self.raise_if_nan()

    def raise_if_nan(self) -> None:
        loss_name: str
        loss: Tensor
        for loss_name, loss in self.losses.items():
            if torch.isnan(loss):
                raise ValueError(f"Loss {loss_name} is NaN.")

    def update(self, loss: ML) -> None:
        self.losses.update(loss.losses)

    def to_device(self) -> ML:
        for loss_name, loss in self.losses.items():
            self.losses[loss_name] = loss.to(self.config['general-training']['device'])

        return self

    def backward(self) -> None:
        self.full_loss.backward()

    def items(self) -> ML:
        self.detach()
        return self

    def detach(self) -> None:
        for loss_name, loss in self.losses.items():
            self.losses[loss_name] = loss.detach()

    @property
    def full_loss(self) -> Tensor:
        full_loss: Optional[Tensor] = None
        for loss_name, loss in self.losses.items():
            if full_loss is None:
                full_loss = loss.clone()
            else:
                full_loss += loss.clone()

        return full_loss

    @property
    def _loss_dict(self) -> Dict:
        return {name: float(loss.item()) for name, loss in self.losses.items()}

    def to_json(self, path: str) -> None:
        os.makedirs(path[:path.rfind(os.sep)], exist_ok=True)
        with open(path, 'a') as f:
            json.dump(self._loss_dict, f)

    def __iter__(self):
        for element in self._loss_dict.items():
            yield element

    def __repr__(self) -> str:
        return self.__str__()

    def __str__(self):
        return str({name: round(value, 5) for name, value in self._loss_dict.items()})

    @property
    def logs(self) -> Dict:
        return self._loss_dict
