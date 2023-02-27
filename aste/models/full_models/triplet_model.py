import logging
from typing import List, Dict, Optional

import torch
from aste.configs import base_config
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..base_model import BaseModel
from .triplet_model_outputs import ModelOutput
from ..model_elements.embeddings import BaseEmbedding, TransformerWithAggregation
from ..model_elements.span_aggregators import (
    BaseAggregator,
    EndPointAggregator
)
from ..outputs import (
    ModelLoss,
    ModelMetric,
    BaseModelOutput
)
from ..specialty_models import SpanCreatorModel, TripletExtractorModel, EmbeddingsExtenderModel
from ..specialty_models import SpanCreatorOutput, TripletModelOutput, SentimentModelOutput
from ...dataset.reader import Batch


class TripletModel(BaseModel):
    def __init__(self, model_name='Transformer Based Model', config: Dict = base_config, *args, **kwargs):
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
            span_creator_output.predicted_spans.get_aspect_span_predictions()
        )
        span_creator_output.opinions_agg_emb = self.aggregator.aggregate(
            emb_output.features,
            span_creator_output.predicted_spans.get_opinion_span_predictions()
        )

        extended_opinions: SentimentModelOutput = self.sentiment_extender(span_creator_output.opinions_agg_emb)
        span_creator_output = span_creator_output.extend_opinion_embeddings(extended_opinions)

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

        self.log("val_loss", loss.full_loss, on_epoch=True, prog_bar=True, logger=True, sync_dist=True,
                 batch_size=self.config['general-training']['batch-size'])
        self.log("val_loss_span_creator_loss", loss.span_creator_loss, on_epoch=True, prog_bar=False, logger=True,
                 sync_dist=True, batch_size=self.config['general-training']['batch-size'])
        self.log("val_loss_triplet_extractor_loss", loss.triplet_extractor_loss, on_epoch=True, prog_bar=False,
                 logger=True, sync_dist=True, batch_size=self.config['general-training']['batch-size'])
        return loss.full_loss

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
