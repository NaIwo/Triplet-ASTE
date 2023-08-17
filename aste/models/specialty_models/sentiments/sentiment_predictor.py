from typing import Dict, List

import torch
from torchmetrics import MetricCollection

from ..utils import sequential_blocks, TransformerModel
from ...base_model import BaseModel
from ...outputs import (
    TripletModelOutput
)
from ....losses import DiceLoss, FocalLoss
from ....dataset.domain.const import ASTELabels
from ....models.outputs import (
    ModelLoss,
    ModelMetric
)
from ....tools.metrics import get_selected_metrics


class SentimentPredictor(BaseModel):
    def __init__(self, input_dim: int, config: Dict, model_name: str = 'Sentiment predictor model'):
        super(SentimentPredictor, self).__init__(model_name=model_name, config=config)

        self.n_polarities = len(self.config['dataset']['polarities']) + 1

        metrics = get_selected_metrics(
            for_spans=False,
            dist_sync_on_step=True,
            task='multiclass',
            num_classes=self.n_polarities
        )
        self.final_metrics: MetricCollection = MetricCollection(metrics=metrics)

        metrics = get_selected_metrics(
            for_spans=False,
            dist_sync_on_step=True,
        )
        self.negative_metrics: MetricCollection = MetricCollection(metrics=metrics)

        self.loss = FocalLoss(alpha=1., gamma=3.)

        input_dim = 3 * input_dim + 32

        self.predictor = TransformerModel(
            input_dim=input_dim,
            model_dim=input_dim // 4,
            attention_heads=4,
            num_layers=2,
            output_dim=self.n_polarities,
            device=self.device
        ).to(self.device)
        self.distance = sequential_blocks([1, 16, 32], self.device, is_last=False)

    def forward(self, data: TripletModelOutput) -> TripletModelOutput:
        out = data.copy()

        for triplet in out.triplets:
            features = triplet.features
            cls = triplet.sentence_emb[0].repeat(features.shape[0], 1)
            # if triplet.similarities.size() != torch.Size([0]):
            #     similarities = triplet.similarities.unsqueeze(-1).repeat(1, cls.size(-1))
            #     cls *= similarities
            distance = self.distance(
                torch.abs(triplet.opinion_ranges[:, 0:1] - triplet.aspect_ranges[:, 0:1]).float()
            )
            features = torch.cat([features, cls, distance], dim=1)
            if features.shape[0] != 0:
                features = features.unsqueeze(0)
                scores = self.predictor(features)
                scores = scores.squeeze(0)
            else:
                scores = torch.zeros(0, self.n_polarities).to(self.device)
            triplet.features = scores
            sentiments = torch.argmax(scores, dim=-1, keepdim=True)
            triplet.pred_sentiments = sentiments
            triplet.construct_triplets(self.config['model']['remove-intersected'])

        return out

    def get_loss(self, model_out: TripletModelOutput) -> ModelLoss:
        if model_out.get_true_sentiments().shape[0] == 0:
            loss = torch.tensor(0., device=self.device)
        else:
            loss = self.loss(model_out.get_features(), model_out.get_true_sentiments().long())

        full_loss = ModelLoss(
            config=self.config,
            losses={
                'sentiment_predictor_loss': loss * self.config['model']['sentiment-predictor'][
                    'loss-weight'] * self.trainable,
            }
        )
        return full_loss

    def update_metrics(self, model_out: TripletModelOutput) -> None:
        if model_out.get_true_sentiments().shape[0] == 0:
            return
        self.final_metrics.update(
            model_out.get_true_sentiments().view(-1),
            model_out.get_predicted_sentiments().view(-1)
        )
        self.negative_metrics.update(
            model_out.get_true_sentiments().view(-1) > 0,
            model_out.get_predicted_sentiments().view(-1) > 0
        )

    def get_metrics(self) -> ModelMetric:
        return ModelMetric(
            metrics={
                'sentiment_predictor_metrics': self.final_metrics.compute(),
                'sentiment_predictor_negative_metrics': self.negative_metrics.compute()
            }
        )

    def reset_metrics(self) -> None:
        self.final_metrics.reset()
        self.negative_metrics.reset()