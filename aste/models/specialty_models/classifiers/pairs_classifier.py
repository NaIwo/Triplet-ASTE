from typing import Dict, List

import torch

from ..utils import sequential_blocks
from ...base_model import BaseModel
from ...outputs import (
    TripletModelOutput
)
from ...utils.const import CreatedSpanCodes
from ....losses.dice_loss import DiceLoss
from ....models.outputs import (
    ModelLoss,
    ModelMetric
)
from ....dataset.domain import ASTELabels
from ....tools.metrics import get_selected_metrics, CustomMetricCollection


class PairClassifierModel(BaseModel):
    def __init__(self, input_dim: int, config: Dict, model_name: str = 'Pairs Classifier Model', *args, **kwargs):
        super(PairClassifierModel, self).__init__(model_name=model_name, config=config)

        metrics = get_selected_metrics(
            dist_sync_on_step=True,
            ignore_index=CreatedSpanCodes.NOT_RELEVANT.value
        )
        self.final_metrics = CustomMetricCollection(
            name='Pairs classifier',
            ignore_index=CreatedSpanCodes.NOT_RELEVANT.value,
            metrics=metrics
        )

        self.loss = torch.nn.BCELoss(reduction='mean')

        neurons: List = [
            input_dim,
            input_dim // 2,
            input_dim // 4,
            input_dim // 8,
            1
        ]
        self.predictor = sequential_blocks(neurons=neurons, is_last=True, device=self.device)
        self.predictor.append(torch.nn.Sigmoid())

    def forward(self, triplets: TripletModelOutput) -> TripletModelOutput:
        out = triplets.copy()

        for triplet in out.triplets:
            features = triplet.features
            if triplet.similarities.size() != torch.Size([0]):
                features *= triplet.similarities.unsqueeze(-1).repeat(1, features.size(-1))
            scores = self.predictor(features)
            triplet.features = scores
            sentiments = (scores <= 0.5).squeeze()
            triplet.pred_sentiments = torch.where(sentiments, ASTELabels.NOT_PAIR, triplet.pred_sentiments)
            triplet.construct_triplets()

        return out

    def get_loss(self, model_out: TripletModelOutput) -> ModelLoss:
        if model_out.get_true_sentiments().shape[0] == 0:
            loss = torch.tensor(0., device=self.device)
        else:
            features = model_out.get_features()
            # features = torch.cat([1 - features, features], dim=-1)
            loss = self.loss(features.view(-1), (model_out.get_span_creation_info() >= 0).view(-1).float())

        full_loss = ModelLoss(
            config=self.config,
            losses={
                'pair_classifier_loss': loss * self.config['model']['pair-classifier'][
                    'loss-weight'] * self.trainable,
            }
        )
        return full_loss

    def update_metrics(self, model_out: TripletModelOutput) -> None:
        if model_out.get_span_creation_info().shape[0] == 0:
            return
        self.final_metrics.update(
            model_out.get_span_creation_info().view(-1) >= 0,
            model_out.get_features().view(-1) > 0.5
        )

    def get_metrics(self) -> ModelMetric:
        return ModelMetric(
            metrics={
                'pairs_classifier_metric': self.final_metrics.compute(),
            }
        )

    def reset_metrics(self) -> None:
        self.final_metrics.reset()
