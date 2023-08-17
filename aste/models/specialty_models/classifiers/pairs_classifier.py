from typing import Dict, List

import torch

from ..utils import sequential_blocks, TransformerModel
from ...base_model import BaseModel
from ...outputs import (
    TripletModelOutput
)
from ...utils.const import CreatedSpanCodes
from ....losses.dice_loss import DiceLoss
from ....losses.focal_loss import FocalLoss
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

        self.loss = DiceLoss(alpha=0.7, with_logits=True)

        input_dim = 3 * input_dim + 32

        self.predictor = TransformerModel(
            input_dim=input_dim,
            model_dim=input_dim // 4,
            attention_heads=4,
            num_layers=2,
            output_dim=1,
            device=self.device
        ).to(self.device)
        self.distance = sequential_blocks([1, 16, 32], self.device, is_last=False)

    def forward(self, triplets: TripletModelOutput) -> TripletModelOutput:
        out = triplets.copy()

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
                scores = torch.zeros(0, 1).to(self.device)
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
            loss = self.loss(features, (model_out.get_span_creation_info() >= 0).view(-1).float())

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
