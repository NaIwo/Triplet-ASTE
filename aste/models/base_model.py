import logging
from typing import List, Dict, Any, Union, Optional

import pytorch_lightning as pl
import torch
from aste.configs import config
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch import Tensor

from . import ModelOutput, ModelLoss, ModelMetric


class BaseModel(pl.LightningModule):
    def __init__(self, model_name: str, *args, **kwargs):
        super().__init__()
        self.model_name = model_name
        self.performed_epochs: int = 0
        self.warmup: bool = False
        self.trainable: bool = True

    def configure_optimizers(self):
        return torch.optim.SGD(self.get_params_and_lr(), lr=1e-4)

    def forward(self, *args, **kwargs) -> Union[ModelOutput, Tensor]:
        raise NotImplementedError

    def training_step(self, *args, **kwargs) -> STEP_OUTPUT:
        raise NotImplemented

    def validation_step(self, *args, **kwargs) -> Optional[STEP_OUTPUT]:
        raise NotImplemented

    def test_step(self, *args, **kwargs) -> Optional[STEP_OUTPUT]:
        raise NotImplemented

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        return super(BaseModel, self).predict_step(batch, batch_idx, dataloader_idx)

    def get_loss(self, model_out: ModelOutput) -> ModelLoss:
        raise NotImplemented

    def update_metrics(self, model_out: ModelOutput) -> None:
        raise NotImplemented

    def get_metrics_and_reset(self) -> ModelMetric:
        metrics: ModelMetric = self.get_metrics()
        self.reset_metrics()
        return metrics

    def get_metrics(self) -> ModelMetric:
        raise NotImplemented

    def reset_metrics(self) -> None:
        raise NotImplemented

    def update_trainable_parameters(self) -> None:
        self.performed_epochs += 1

    def freeze(self) -> None:
        logging.info(f"Model '{self.model_name}' freeze.")
        self.trainable = False
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze(self) -> None:
        logging.info(f"Model '{self.model_name}' unfreeze.")
        self.trainable = True
        for param in self.parameters():
            param.requires_grad = True

    def get_params_and_lr(self) -> List[Dict]:
        return [{
            "param": self.parameters(), 'lr': config['model']['learning-rate']
        }]
