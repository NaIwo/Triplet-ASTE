from typing import List, Dict, Optional

import torch
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch import Tensor

from .base_model import BaseModel
from .model_elements.embeddings import BaseEmbedding, TransformerWithAggregation
from .model_elements.span_aggregators import (
    BaseAggregator,
    EndPointAggregator
)
from .outputs import (
    ModelLoss,
    ModelMetric,
    BaseModelOutput,
    SpanCreatorOutput,
    TripletModelOutput,
    SentimentModelOutput, ModelOutput
)
from .specialty_models import SpanCreatorModel, TripletExtractorModel, EmbeddingsExtenderModel
from ..dataset.reader import Batch


class TripletModel(BaseModel):
    def __init__(self, model_name='Transformer Based Model', config: Optional[Dict] = None, *args, **kwargs):
        super(TripletModel, self).__init__(model_name, config=config)

        self.emb_layer: BaseEmbedding = TransformerWithAggregation(config=config)
        self.span_creator: BaseModel = SpanCreatorModel(input_dim=self.emb_layer.embedding_dim, config=config)
        self.aggregator: BaseAggregator = EndPointAggregator(input_dim=self.emb_layer.embedding_dim, config=config)
        self.sentiment_extender: BaseModel = EmbeddingsExtenderModel(input_dim=self.aggregator.output_dim,
                                                                     config=config)
        self.triplets_extractor: BaseModel = TripletExtractorModel(input_dim=self.aggregator.output_dim, config=config)

    def forward(self, batch: Batch) -> ModelOutput:
        emb_output: BaseModelOutput = self.emb_layer(batch)
        span_creator_output: SpanCreatorOutput = self.span_creator(emb_output)

        span_creator_output.aspects_agg_emb = self.aggregator.aggregate(
            emb_output.features,
            span_creator_output.get_aspect_span_predictions()
        )
        span_creator_output.opinions_agg_emb = self.aggregator.aggregate(
            emb_output.features,
            span_creator_output.get_opinion_span_predictions()
        )

        extended_opinions: SentimentModelOutput = self.sentiment_extender(span_creator_output.opinions_agg_emb)
        span_creator_output = span_creator_output.extend_opinions_with_sentiments(extended_opinions)

        triplet_output: TripletModelOutput = self.triplets_extractor(span_creator_output)

        return ModelOutput(
            batch=batch,
            span_creator_output=span_creator_output,
            triplet_results=triplet_output,
        )

    def get_loss(self, model_out: ModelOutput) -> ModelLoss:
        s_loss = self.span_creator.get_loss(model_out.span_creator_output) * self.span_creator.trainable
        t_loss = self.triplets_extractor.get_loss(model_out.triplet_results) * self.triplets_extractor.trainable
        return ModelLoss.from_instances(
            span_creator_loss=s_loss,
            triplet_extractor_loss=t_loss,
            config=self.config
        )

    def update_metrics(self, model_out: ModelOutput) -> None:
        self.span_creator.update_metrics(model_out.span_creator_output)
        self.triplets_extractor.update_metrics(model_out.triplet_results)

    def get_metrics(self) -> ModelMetric:
        return ModelMetric.from_instances(
            span_creator_metric=self.span_creator.get_metrics(),
            triplet_metric=self.triplets_extractor.get_metrics(),
        )

    def reset_metrics(self) -> None:
        self.span_creator.reset_metrics()
        self.triplets_extractor.reset_metrics()

    def validation_step(self, batch: Batch, batch_idx: int, *args, **kwargs) -> Optional[STEP_OUTPUT]:
        model_out: ModelOutput = self.forward(batch)
        self.update_metrics(model_out)
        loss: ModelLoss = self.get_loss(model_out)

        mt = model_out.triplet_results
        similarity = float((mt.features * mt.loss_mask).sum() / mt.loss_mask.sum())

        self.log('val_similarity', similarity, on_epoch=True, on_step=True, prog_bar=False,
                 batch_size=self.config['general-training']['batch-size'], logger=True, sync_dist=True)

        self.log_loss(loss, prefix='val', on_epoch=True, on_step=False)

        return loss.full_loss

    def log_loss(self, loss: ModelLoss, prefix: str = 'train', on_epoch: bool = True, on_step: bool = False) -> None:
        self.log(f"{prefix}_loss", loss.full_loss, on_epoch=on_epoch, prog_bar=True, on_step=on_step,
                 logger=True, sync_dist=True, batch_size=self.config['general-training']['batch-size'])
        self.log(f"{prefix}_loss_span_creator_loss", loss.span_creator_loss, on_epoch=on_epoch, on_step=on_step,
                 prog_bar=True, logger=True, sync_dist=True, batch_size=self.config['general-training']['batch-size'])
        self.log(f"{prefix}_loss_triplet_extractor_loss", loss.triplet_extractor_loss, on_epoch=on_epoch,
                 on_step=on_step, prog_bar=True, logger=True, sync_dist=True,
                 batch_size=self.config['general-training']['batch-size']
                 )

    def configure_optimizers(self):
        return torch.optim.Adam(self.get_params_and_lr(), lr=1e-5)

    def get_params_and_lr(self) -> List[Dict]:
        return [
            {'params': self.emb_layer.parameters(), 'lr': self.config['model']['transformer']['learning-rate']},
            {'params': self.span_creator.parameters(), 'lr': self.config['model']['learning-rate']},
            {'params': self.aggregator.get_parameters(), 'lr': self.config['model']['learning-rate']},
            {'params': self.sentiment_extender.parameters(), 'lr': self.config['model']['learning-rate']},
            {'params': self.triplets_extractor.parameters(), 'lr': self.config['model']['learning-rate']},
        ]

    @staticmethod
    def _count_intersection(true_spans: Tensor, predicted_spans: Tensor) -> int:
        predicted_spans = predicted_spans.unique(dim=0)
        all_spans: Tensor = torch.cat([true_spans, predicted_spans], dim=0)
        uniques, counts = torch.unique(all_spans, return_counts=True, dim=0)
        return uniques[counts > 1].shape[0]
