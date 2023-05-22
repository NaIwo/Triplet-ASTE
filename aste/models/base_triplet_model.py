from collections import defaultdict
from typing import Dict, Optional, Any

import torch
import wandb
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

        mt = model_out.triplet_output
        loss_mask = mt.loss_mask  # & mt.true_predicted_mask
        reverse_loss_mask = (~mt.loss_mask) & mt.pad_mask  # & mt.true_predicted_mask
        sim = mt.normalized_similarities

        similarity = float((sim * loss_mask).sum() / (loss_mask.sum() + 1e-6))
        non_similarity = float((sim * reverse_loss_mask).sum() / (reverse_loss_mask.sum() + 1e-6))

        self.log('val_similarity', similarity, on_epoch=False, on_step=True, prog_bar=False,
                 batch_size=self.config['general-training']['batch-size'], logger=True, sync_dist=True)
        self.log('val_non-similarity', non_similarity, on_epoch=False, on_step=True, prog_bar=False,
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

    def triplet_precision_recall_different_thresholds(self, dataset):
        self.eval()
        results = defaultdict(list)
        if isinstance(self.config['model']['triplet-extractor']['threshold'], int):
            range_ = torch.arange(0, 20, 1, dtype=torch.int)
        else:
            range_ = torch.arange(0., 1., 0.05, dtype=torch.float)

        for threshold in range_:
            self.triplets_extractor.config['model']['triplet-extractor']['threshold'] = threshold.item()
            self.reset_metrics()
            for batch in dataset:
                batch.to_device(self.device)
                model_out = self.forward(batch)
                self.update_metrics(model_out)
            metrics = self.get_metrics()['triplet_extractor_metric']
            recall = metrics['SpanRecall'].item()
            precision = metrics['SpanPrecision'].item()
            results['recall'].append(recall)
            results['precision'].append(precision)
            self.reset_metrics()
        wandb.log(
            {
                'Precision_Recall_vs_Threshold':
                wandb.plot.line_series(
                    xs=range_.tolist(),
                    ys=[results['precision'], results['recall']],
                    keys=['Precision', 'Recall'],
                    title='Precision Recall vs Threshold',
                    xname='Threshold',
                )
            }
        )
