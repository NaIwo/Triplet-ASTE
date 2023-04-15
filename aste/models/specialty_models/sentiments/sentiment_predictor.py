from typing import Dict, List

import torch
from torchmetrics import MetricCollection

from ..utils import sequential_blocks
from ...base_model import BaseModel
from ...outputs import (
    TripletModelOutput
)
from ....losses.dice_loss import DiceLoss
from ....dataset.domain.const import ASTELabels
from ....models.outputs import (
    ModelLoss,
    ModelMetric
)
from ....tools.metrics import get_selected_metrics


class SentimentPredictor(BaseModel):
    def __init__(self, input_dim: int, config: Dict, model_name: str = 'Sentiment predictor model'):
        super(SentimentPredictor, self).__init__(model_name=model_name, config=config)

        n_polarities = len(self.config['dataset']['polarities']) + 1

        metrics = get_selected_metrics(
            for_spans=False,
            dist_sync_on_step=True,
            task='multiclass',
            num_classes=n_polarities
        )
        self.final_metrics: MetricCollection = MetricCollection(metrics=metrics)
        self.loss = DiceLoss(ignore_index=ASTELabels.NOT_RELEVANT.value, alpha=0.7, with_logits=False)

        neurons: List = [
            input_dim,
            input_dim // 2,
            input_dim // 4,
            input_dim // 8,
            n_polarities
        ]
        self.predictor = sequential_blocks(neurons, self.device, is_last=True)
        self.predictor.append(torch.nn.Softmax(dim=-1))

    def forward(self, data: TripletModelOutput) -> TripletModelOutput:
        out = data.copy()

        for triplet in out.triplets:
            features = triplet.features
            # if triplet.similarities.size() != torch.Size([0]):
            #     similarities = triplet.similarities.unsqueeze(-1).repeat(1, features.size(-1))
            #     features *= similarities
            scores = self.predictor(features)
            triplet.features = scores
            sentiments = torch.argmax(scores, dim=-1, keepdim=True)
            triplet.pred_sentiments = sentiments
            triplet.construct_triplets()

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

    def get_metrics(self) -> ModelMetric:
        return ModelMetric(
            metrics={
                'sentiment_predictor_metrics': self.final_metrics.compute(),
            }
        )

    def reset_metrics(self) -> None:
        self.final_metrics.reset()
