import logging
from functools import singledispatchmethod
from typing import List, Dict, Union, Optional, Any

import pytorch_lightning as pl
import torch
import yaml
from aste.configs import base_config
from aste.dataset.domain import Sentence
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from .outputs import ModelLoss, ModelMetric, BaseModelOutput
from ..dataset.reader import Batch


class BaseModel(pl.LightningModule):
    def __init__(self, model_name: str, config: Dict = base_config, *args, **kwargs):
        super().__init__()
        self.model_name = model_name
        self.config: Dict = config
        self.performed_epochs: int = 0
        self.warmup: bool = False
        self.trainable: bool = True

    def forward(self, *args, **kwargs) -> Union[BaseModelOutput, Tensor]:
        raise NotImplementedError

    def training_step(self, batch: Batch, batch_idx: int, *args, **kwargs) -> STEP_OUTPUT:
        model_out: BaseModelOutput = self.forward(batch)
        loss: ModelLoss = self.get_loss(model_out)
        self.log("train_loss", loss.full_loss, on_epoch=True, prog_bar=True, logger=True, sync_dist=True,
                 batch_size=self.config['general-training']['batch-size'])
        return loss.full_loss

    def validation_step(self, batch: Batch, batch_idx: int, *args, **kwargs) -> Optional[STEP_OUTPUT]:
        model_out: BaseModelOutput = self.forward(batch)
        self.update_metrics(model_out)
        loss: ModelLoss = self.get_loss(model_out)

        self.log("val_loss", loss.full_loss, on_epoch=True, prog_bar=True, logger=True, sync_dist=True,
                 batch_size=self.config['general-training']['batch-size'])
        return loss.full_loss

    def validation_epoch_end(self, *args, **kwargs) -> None:
        metrics: ModelMetric = self.get_metrics_and_reset()
        if self.logger is not None:
            self.logger.log_metrics(metrics.metrics(prefix='val'))

    def test_step(self, batch: Batch, batch_idx: int, *args, **kwargs) -> Optional[STEP_OUTPUT]:
        model_out: BaseModelOutput = self.forward(batch)
        self.update_metrics(model_out)
        loss: ModelLoss = self.get_loss(model_out)

        self.log("test_loss", loss.full_loss, on_epoch=True, prog_bar=True, logger=True, sync_dist=True,
                 batch_size=self.config['general-training']['batch-size'])
        return loss.full_loss

    def test_epoch_end(self, *args, **kwargs) -> None:
        metrics: ModelMetric = self.get_metrics_and_reset()
        if self.logger is not None:
            self.logger.log_metrics(metrics.metrics(prefix='test'))

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        return super(BaseModel, self).predict_step(batch, batch_idx, dataloader_idx)

    def get_loss(self, model_out: BaseModelOutput) -> ModelLoss:
        raise NotImplemented

    def update_metrics(self, model_out: BaseModelOutput) -> None:
        raise NotImplemented

    def get_metrics_and_reset(self) -> ModelMetric:
        metrics: ModelMetric = self.get_metrics()
        self.reset_metrics()
        return metrics

    def get_metrics(self) -> ModelMetric:
        raise NotImplemented

    def reset_metrics(self) -> None:
        raise NotImplemented

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

    def configure_optimizers(self):
        return torch.optim.SGD(self.get_params_and_lr(), lr=1e-4)

    def get_params_and_lr(self) -> List[Dict]:
        return [{
            "param": self.parameters(), 'lr': self.config['model']['learning-rate']
        }]

    @staticmethod
    def pprint_metrics(metrics: ModelMetric) -> None:
        logging.info(f'\n{ModelMetric.NAME}\n'
                     f'{yaml.dump(metrics.__dict__, sort_keys=False, default_flow_style=False)}')

    @singledispatchmethod
    def predict(self, sample: Union[Batch, Sentence]) -> Union[BaseModelOutput, List[BaseModelOutput]]:
        raise ValueError(f'Cannot make a prediction on the passed input data type: {type(sample)}')

    @predict.register
    @torch.no_grad()
    def predict_dataset(self, sample: DataLoader) -> List[BaseModelOutput]:
        out: List[BaseModelOutput] = list()
        batch: Batch
        for batch in (tqdm(sample, desc=f'Model is running...')):
            model_out: BaseModelOutput = self.predict_batch(batch)
            out.append(model_out)
        return out

    @predict.register
    @torch.no_grad()
    def predict_batch(self, sample: Batch) -> BaseModelOutput:
        self.eval()
        out: BaseModelOutput = self.forward(sample)
        return out

    @predict.register
    @torch.no_grad()
    def predict_sentence(self, sample: Sentence) -> BaseModelOutput:
        sample = Batch.from_sentence(sample).to_device(self.config)
        self.eval()
        out: BaseModelOutput = self.forward(sample)
        return out
