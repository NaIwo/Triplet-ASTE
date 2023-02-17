import logging
from functools import singledispatchmethod
from typing import List, Dict, Union, Optional, Any

import torch
import yaml
from aste.configs import base_config
from aste.dataset.domain import Sentence
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from . import ModelOutput, ModelLoss, ModelMetric, BaseModel
from .model_elements.embeddings import BaseEmbedding, TransformerWithAggregation
from .model_elements.span_aggregators import (
    BaseAggregator,
    EndPointAggregator
)
from .specialty_models import SpanCreatorModel, TripletExtractorModel, Selector
from ..dataset.reader import Batch


class TransformerBasedModel(BaseModel):
    def __init__(self, model_name='Transformer Based Model', config: Dict = base_config, *args, **kwargs):

        super(TransformerBasedModel, self).__init__(model_name, config=config)
        self.emb_layer: BaseEmbedding = TransformerWithAggregation(config=config)
        self.span_creator: BaseModel = SpanCreatorModel(input_dim=self.emb_layer.embedding_dim, config=config)
        self.aggregator: BaseAggregator = EndPointAggregator(input_dim=self.emb_layer.embedding_dim, config=config)
        self.span_selector: BaseModel = Selector(input_dim=self.aggregator.output_dim, config=config)
        self.triplets_extractor: BaseModel = TripletExtractorModel(input_dim=self.aggregator.output_dim, config=config)

        epochs: List = [2, 4, 100]

        self.training_scheduler: Dict = {
            range(0, epochs[0]): {
                'freeze': [self.triplets_extractor],
                'unfreeze': [self.span_creator, self.span_selector, self.emb_layer]
            },
            range(epochs[0], epochs[1]): {
                'freeze': [self.span_creator, self.span_selector, self.emb_layer],
                'unfreeze': [self.triplets_extractor]
            },
            range(epochs[1], epochs[2]): {
                'freeze': [],
                'unfreeze': [self.span_creator, self.span_selector, self.triplets_extractor, self.emb_layer],
            }
        }

    def forward(self, batch: Batch) -> Union[ModelOutput, Tensor]:
        emb_span_creator: Tensor = self.emb_layer(batch)

        span_creator_output: Tensor = self.span_creator(emb_span_creator)
        predicted_spans: List[Tensor] = self.span_creator.get_spans(span_creator_output, batch)

        agg_emb: Tensor = self.aggregator.aggregate(emb_span_creator, predicted_spans)

        span_selector_output: Tensor = self.span_selector(agg_emb)
        triplet_input: Tensor = span_selector_output[..., :] * agg_emb

        triplet_results: Tensor = self.triplets_extractor(triplet_input)

        return ModelOutput(
            batch=batch,
            span_creator_output=span_creator_output,
            predicted_spans=predicted_spans,
            span_selector_output=span_selector_output,
            triplet_results=triplet_results
            )

    def get_loss(self, model_out: ModelOutput) -> ModelLoss:
        return ModelLoss.from_instances(
            span_creator_loss=self.span_creator.get_loss(model_out) * self.span_creator.trainable,
            triplet_extractor_loss=self.triplets_extractor.get_loss(model_out) * self.triplets_extractor.trainable,
            span_selector_loss=self.span_selector.get_loss(model_out) * self.span_selector.trainable,
            config=self.config
        )

    def update_metrics(self, model_out: ModelOutput) -> None:
        self.span_creator.update_metrics(model_out)
        self.triplets_extractor.update_metrics(model_out)
        self.span_selector.update_metrics(model_out)

    def get_metrics(self) -> ModelMetric:
        return ModelMetric.from_instances(
            span_creator_metric=self.span_creator.get_metrics(),
            triplet_metric=self.triplets_extractor.get_metrics(),
            span_selector_metric=self.span_selector.get_metrics()
        )

    def reset_metrics(self) -> None:
        self.span_creator.reset_metrics()
        self.triplets_extractor.reset_metrics()
        self.span_selector.reset_metrics()

    def on_train_epoch_start(self) -> None:
        self.update_trainable_parameters()

    def training_step(self, batch: Batch, batch_idx: int, *args, **kwargs) -> STEP_OUTPUT:
        model_out: ModelOutput = self.forward(batch)
        loss: ModelLoss = self.get_loss(model_out)
        self.log("train_loss", loss.full_loss, on_epoch=True, prog_bar=True, logger=True, sync_dist=True,
                 batch_size=self.config['general-training']['batch-size'])
        return loss.full_loss

    def validation_step(self, batch: Batch, batch_idx: int, *args, **kwargs) -> Optional[STEP_OUTPUT]:
        model_out: ModelOutput = self.forward(batch)
        self.update_metrics(model_out)
        loss: ModelLoss = self.get_loss(model_out)

        self.log("val_loss", loss.full_loss, on_epoch=True, prog_bar=True, logger=True, sync_dist=True,
                 batch_size=self.config['general-training']['batch-size'])
        self.log("val_loss_span_creator_loss", loss.span_creator_loss, on_epoch=True, prog_bar=False, logger=True,
                 sync_dist=True,
                 batch_size=self.config['general-training']['batch-size'])
        self.log("val_loss_span_selector_loss", loss.span_selector_loss, on_epoch=True, prog_bar=False, logger=True,
                 sync_dist=True, batch_size=self.config['general-training']['batch-size'])
        self.log("val_loss_triplet_extractor_loss", loss.triplet_extractor_loss, on_epoch=True, prog_bar=False,
                 logger=True, sync_dist=True, batch_size=self.config['general-training']['batch-size'])
        return loss.full_loss

    def validation_epoch_end(self, *args, **kwargs) -> None:
        metrics: ModelMetric = self.get_metrics_and_reset()
        if self.logger is not None:
            self.logger.log_metrics(metrics.metrics(prefix='val'))

    def test_step(self, batch: Batch, batch_idx: int, *args, **kwargs) -> Optional[STEP_OUTPUT]:
        model_out: ModelOutput = self.forward(batch)
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

    def configure_optimizers(self):
        return torch.optim.Adam(self.get_params_and_lr(), lr=1e-5)

    def get_params_and_lr(self) -> List[Dict]:
        return [
            {'params': self.span_creator.parameters(), 'lr': self.config['model']['learning-rate']},
            {'params': self.aggregator.get_parameters(), 'lr': self.config['model']['learning-rate']},
            {'params': self.span_selector.parameters(), 'lr': self.config['model']['learning-rate']},
            {'params': self.triplets_extractor.parameters(), 'lr': self.config['model']['learning-rate']},
            {'params': self.emb_layer.parameters(), 'lr': self.config['model']['transformer']['learning-rate']}
        ]

    def update_trainable_parameters(self) -> None:
        model: BaseModel
        scheduler_idx: int
        for scheduler_idx, (keys, values) in enumerate(self.training_scheduler.items()):
            if self.performed_epochs in keys:
                [model.unfreeze() for model in values['unfreeze']]
                [model.freeze() for model in values['freeze']]
                self.warmup = scheduler_idx < 2
                break

        if self.performed_epochs >= list(self.training_scheduler.keys())[-1][0]:
            self.span_selector.sigmoid_multiplication = self.config['model']['selector']['sigmoid-multiplication']
        self.performed_epochs += 1

    @torch.no_grad()
    def check_coverage_detected_spans(self, data: DataLoader) -> Dict:
        num_predicted: int = 0
        num_correct_predicted: int = 0
        true_num: int = 0
        for batch in tqdm(data):
            sample: Batch
            for sample in batch:
                model_output: ModelOutput = self(sample)
                true_spans: Tensor = torch.cat([sample.aspect_spans[0], sample.opinion_spans[0]], dim=0).unique(
                    dim=0)
                num_correct_predicted += self._count_intersection(true_spans, model_output.predicted_spans[0])
                num_predicted += model_output.predicted_spans[0].unique(dim=0).shape[0]
                true_num += true_spans.shape[0] - int(-1 in true_spans)
        ratio: float = num_correct_predicted / true_num
        logging.info(
            f'Coverage of isolated spans: {ratio}. Extracted spans: {num_predicted}. Total correct spans: {true_num}')
        return {
            'Ratio': ratio,
            'Extracted spans': num_predicted,
            'Total correct spans': true_num
        }

    @staticmethod
    def pprint_metrics(metrics: ModelMetric) -> None:
        logging.info(f'\n{ModelMetric.NAME}\n'
                     f'{yaml.dump(metrics.__dict__, sort_keys=False, default_flow_style=False)}')

    @staticmethod
    def _count_intersection(true_spans: Tensor, predicted_spans: Tensor) -> int:
        predicted_spans = predicted_spans.unique(dim=0)
        all_spans: Tensor = torch.cat([true_spans, predicted_spans], dim=0)
        uniques, counts = torch.unique(all_spans, return_counts=True, dim=0)
        return uniques[counts > 1].shape[0]

    @singledispatchmethod
    def predict(self, sample: Union[Batch, Sentence]) -> Union[ModelOutput, List[ModelOutput]]:
        raise NotImplementedError(f'Cannot make a prediction on the passed input data type: {type(sample)}')

    @predict.register
    @torch.no_grad()
    def predict_dataset(self, sample: DataLoader) -> List[ModelOutput]:
        out: List[ModelOutput] = list()
        batch: Batch
        for batch in (tqdm(sample, desc=f'Model is running...')):
            model_out: ModelOutput = self.predict_batch(batch)
            out.append(model_out)
        return out

    @predict.register
    @torch.no_grad()
    def predict_batch(self, sample: Batch) -> ModelOutput:
        self.eval()
        out: ModelOutput = self.forward(sample)
        return out

    @predict.register
    @torch.no_grad()
    def predict_sentence(self, sample: Sentence) -> ModelOutput:
        sample = Batch.from_sentence(sample).to_device(self.config)
        self.eval()
        out: ModelOutput = self.forward(sample)
        return out
