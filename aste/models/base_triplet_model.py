from typing import Dict, Optional, Any

import torch
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch import Tensor

from .base_model import BaseModel
from .model_elements.embeddings import BaseEmbedding, TransformerWithAggregation
from .outputs import (
    ModelLoss,
    ModelMetric,
    ModelOutput
)
from ..dataset.reader import Batch


class BaseTripletModel(BaseModel):
    def __init__(self, model_name='Base Triplet Model', config: Optional[Dict] = None, *args, **kwargs):
        super(BaseTripletModel, self).__init__(model_name, config=config)

        self.emb_layer: BaseEmbedding = TransformerWithAggregation(config=config)

        self.model_with_losses: Dict = dict()
        self.model_with_metrics: Dict = dict()

    def forward(self, batch: Batch) -> ModelOutput:
        raise NotImplementedError

    def get_loss(self, model_out: ModelOutput) -> ModelLoss:
        full_loss = ModelLoss(config=self.config)

        for model, output in self.model_with_losses.items():
            full_loss.update(model.get_loss(getattr(model_out, output)))

        return full_loss

    def update_metrics(self, model_out: ModelOutput) -> None:
        for model, output in self.model_with_metrics.items():
            model.update_metrics(getattr(model_out, output))

    def get_metrics(self) -> ModelMetric:
        metrics = ModelMetric()
        for model in self.model_with_metrics.keys():
            metrics.update(model.get_metrics())

        return metrics

    def reset_metrics(self) -> None:
        for model in self.model_with_metrics.keys():
            model.reset_metrics()

    def validation_step(self, batch: Batch, batch_idx: int, *args, **kwargs) -> Optional[STEP_OUTPUT]:
        model_out: ModelOutput = self.forward(batch)
        self.update_metrics(model_out)
        loss: ModelLoss = self.get_loss(model_out)

        mt = model_out.triplet_results
        similarity = float((mt.features * mt.loss_mask).sum() / mt.loss_mask.sum())
        non_similarity = float((mt.features * ((~mt.loss_mask) & mt.true_predicted_mask)).sum() / ((
                (~mt.loss_mask) & mt.true_predicted_mask).sum() + 1e-6))

        self.log('val_similarity', similarity, on_epoch=True, on_step=False, prog_bar=False,
                 batch_size=self.config['general-training']['batch-size'], logger=True, sync_dist=True)
        self.log('val_non-similarity', non_similarity, on_epoch=True, on_step=False, prog_bar=False,
                 batch_size=self.config['general-training']['batch-size'], logger=True, sync_dist=True)

        self.log_loss(loss, prefix='val', on_epoch=True, on_step=False)

        return loss.full_loss

    def log_loss(self, loss: ModelLoss, prefix: str = 'train', on_epoch: bool = True, on_step: bool = False) -> None:
        self.log(f"{prefix}_loss", loss.full_loss, on_epoch=on_epoch, prog_bar=True, on_step=on_step,
                 logger=True, sync_dist=True, batch_size=self.config['general-training']['batch-size'])

        for loss_name, loss in loss.losses.items():
            self.log(f"{prefix}_loss_{loss_name}", loss, on_epoch=on_epoch, on_step=on_step,
                     prog_bar=True, logger=True, sync_dist=True,
                     batch_size=self.config['general-training']['batch-size']
                     )

    def configure_optimizers(self):
        return torch.optim.AdamW(self.get_params_and_lr(), lr=1e-5)

    def get_params_and_lr(self) -> Any:
        return self.parameters()

    @staticmethod
    def _count_intersection(true_spans: Tensor, predicted_spans: Tensor) -> int:
        predicted_spans = predicted_spans.unique(dim=0)
        all_spans: Tensor = torch.cat([true_spans, predicted_spans], dim=0)
        uniques, counts = torch.unique(all_spans, return_counts=True, dim=0)
        return uniques[counts > 1].shape[0]
