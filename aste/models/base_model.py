import logging
from functools import singledispatchmethod
from typing import List, Dict, Union, Optional, Any

import pytorch_lightning as pl
import torch
import yaml
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from .outputs.losses import ModelLoss
from .outputs.metrics import ModelMetric
from .outputs.outputs import BaseModelOutput
from ..configs import base_config
from ..dataset.domain import Sentence
from ..dataset.reader import Batch


class BaseModel(pl.LightningModule):
    def __init__(self, model_name: str, config: Optional[Dict] = None, *args, **kwargs):
        super().__init__()
        if config is None:
            config: Dict = base_config

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

        self.log_loss(loss, prefix='train', on_epoch=True, on_step=False)
        return loss.full_loss

    def validation_step(self, batch: Batch, batch_idx: int, *args, **kwargs) -> Optional[STEP_OUTPUT]:
        model_out: BaseModelOutput = self.forward(batch)
        self.update_metrics(model_out)
        loss: ModelLoss = self.get_loss(model_out)

        self.log_loss(loss, prefix='val', on_epoch=True, on_step=False)
        return loss.full_loss

    def on_validation_epoch_end(self, *args, **kwargs) -> None:
        metrics: ModelMetric = self.get_metrics_and_reset()
        self.log_metrics(metrics, prefix='val')

    def test_step(self, batch: Batch, batch_idx: int, *args, **kwargs) -> Optional[STEP_OUTPUT]:
        model_out: BaseModelOutput = self.forward(batch)
        self.update_metrics(model_out)
        loss: ModelLoss = self.get_loss(model_out)

        self.log_loss(loss=loss, prefix='test', on_epoch=True, on_step=False)
        return loss.full_loss

    def on_test_epoch_end(self, *args, **kwargs) -> None:
        metrics: ModelMetric = self.get_metrics_and_reset()
        self.log_metrics(metrics, prefix='test')

    def log_loss(self, loss: ModelLoss, prefix: str = 'train', on_epoch: bool = True, on_step: bool = False) -> None:
        self.log(f"{prefix}_loss", loss.full_loss, on_epoch=on_epoch, prog_bar=True, on_step=on_step,
                 logger=True, sync_dist=True, batch_size=self.config['general-training']['batch-size'])

    def log_metrics(self, metrics: ModelMetric, prefix: str = 'train') -> None:
        for metric_name, metric_values in metrics.metrics_with_prefix(prefix=prefix):
            self.log(metric_name, metric_values, on_epoch=True, prog_bar=False,
                     logger=True, sync_dist=True, batch_size=self.config['general-training']['batch-size'])

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
        sample = Batch.from_sentence(sample)
        self.eval()
        out: BaseModelOutput = self.forward(sample)
        return out
