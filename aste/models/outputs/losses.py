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
            config: Optional[Dict],
            span_creator_loss: Optional[Tensor] = None,
            triplet_extractor_loss: Optional[Tensor] = None
    ):

        ZERO: Tensor = torch.tensor(0., device=config['general-training']['device'])
        self.span_creator_loss: Tensor = span_creator_loss if span_creator_loss is not None else ZERO
        self.triplet_extractor_loss: Tensor = triplet_extractor_loss if triplet_extractor_loss is not None else ZERO
        self.config: Dict = config

        if self.config['model']['weighted-loss']:
            self._include_weights()

    @classmethod
    def from_instances(
            cls, *,
            span_creator_loss: ML,
            triplet_extractor_loss: ML,
            config: Dict
    ) -> ML:
        return cls(
            span_creator_loss=span_creator_loss.span_creator_loss,
            triplet_extractor_loss=triplet_extractor_loss.triplet_extractor_loss,
            config=config
        )

    def to_device(self) -> ML:
        self.span_creator_loss = self.span_creator_loss.to(self.config['general-training']['device'])
        self.triplet_extractor_loss = self.triplet_extractor_loss.to(self.config['general-training']['device'])

        return self

    def _include_weights(self) -> None:
        self.span_creator_loss *= self.config['model']['span-creator']['loss-weight']
        self.triplet_extractor_loss *= self.config['model']['triplet-extractor']['loss-weight']

    def backward(self) -> None:
        self.full_loss.backward()

    def items(self) -> ML:
        self.detach()
        return self

    def detach(self) -> None:
        self.span_creator_loss = self.span_creator_loss.detach()
        self.triplet_extractor_loss = self.triplet_extractor_loss.detach()

    @property
    def full_loss(self) -> Tensor:
        return self.span_creator_loss + self.triplet_extractor_loss

    @property
    def _loss_dict(self) -> Dict:
        return {
            'span_creator_loss': float(self.span_creator_loss),
            'triplet_extractor_loss': float(self.triplet_extractor_loss),
            'full_loss': float(self.full_loss)
        }

    def to_json(self, path: str) -> None:
        os.makedirs(path[:path.rfind(os.sep)], exist_ok=True)
        with open(path, 'a') as f:
            json.dump(self._loss_dict, f)

    def __radd__(self, other: ML) -> ML:
        return self.__add__(other)

    def __add__(self, other: ML) -> ML:
        return ModelLoss(
            span_creator_loss=self.span_creator_loss + other.span_creator_loss,
            triplet_extractor_loss=self.triplet_extractor_loss + other.triplet_extractor_loss,
            config=self.config
        )

    def __truediv__(self, other: float) -> ML:
        return ModelLoss(
            span_creator_loss=torch.Tensor(self.span_creator_loss / other),
            triplet_extractor_loss=torch.Tensor(self.triplet_extractor_loss / other),
            config=self.config
        ).to_device()

    def __rmul__(self, other: float) -> ML:
        return self.__mul__(other)

    def __mul__(self, other: float) -> ML:
        return ModelLoss(
            span_creator_loss=torch.Tensor(self.span_creator_loss * other),
            triplet_extractor_loss=torch.Tensor(self.triplet_extractor_loss * other),
            config=self.config
        ).to_device()

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
