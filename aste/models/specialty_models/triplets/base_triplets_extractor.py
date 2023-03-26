from typing import Dict, List

import torch
from torch import Tensor
from torchmetrics import MetricCollection

from ...base_model import BaseModel
from ...outputs import (
    SpanInformationOutput,
    SpanCreatorOutput,
    SampleTripletOutput,
    TripletModelOutput
)
from ...utils.const import TripletDimensions
from ...utils.triplet_utils import (
    create_mask_matrix_for_loss,
    create_mask_matrix_for_prediction,
    get_true_predicted_mask,
    create_embedding_mask_matrix
)
from ....models.outputs import (
    ModelLoss,
    ModelMetric
)
from ....tools.metrics import get_selected_metrics


class BaseTripletExtractorModel(BaseModel):
    def __init__(self, config: Dict, model_name: str = 'Base Triplet Extractor Model', *args, **kwargs):
        super(BaseTripletExtractorModel, self).__init__(model_name=model_name, config=config)

        metrics = get_selected_metrics(for_spans=True, dist_sync_on_step=True)
        self.final_metrics: MetricCollection = MetricCollection(metrics=metrics)

    def forward(self, data_input: SpanCreatorOutput) -> TripletModelOutput:
        matrix: Tensor = self._forward_embeddings(data_input)
        pad_mask: Tensor = create_embedding_mask_matrix(data_input)
        matrix = matrix * pad_mask

        loss_mask: Tensor = create_mask_matrix_for_loss(data_input)
        prediction_mask: Tensor = create_mask_matrix_for_prediction(data_input)

        triplets: List[SampleTripletOutput] = self.get_triplets_from_matrix(matrix * prediction_mask, data_input)

        true_predicted_mask = get_true_predicted_mask(data_input)

        return TripletModelOutput(
            batch=data_input.batch,
            triplets=triplets,
            similarities=matrix,
            loss_mask=loss_mask,
            true_predicted_mask=true_predicted_mask,
            pad_mask=pad_mask
        )

    def _forward_embeddings(self, data_input: SpanCreatorOutput) -> Tensor:
        raise NotImplementedError

    def get_triplets_from_matrix(self, matrix: Tensor, data_input: SpanCreatorOutput) -> List[SampleTripletOutput]:
        raise NotImplementedError

    def get_loss(self, model_out: TripletModelOutput) -> ModelLoss:
        reverse_loss_mask = (~model_out.loss_mask) * model_out.pad_mask

        sim: Tensor = torch.exp(model_out.features)

        numerator: Tensor = sim * model_out.loss_mask

        negatives: Tensor = sim * reverse_loss_mask
        denominator: Tensor = torch.sum(negatives, dim=TripletDimensions.OPINION, keepdim=True)
        denominator = numerator + denominator
        denominator += 1e-8

        loss: Tensor = numerator / denominator
        loss = torch.sum(loss, dim=[1, 2]) / torch.sum(model_out.loss_mask, dim=[1, 2])
        loss = -torch.log(loss)
        loss = torch.sum(loss) / self.config['general-training']['batch-size']

        full_loss = ModelLoss(
            config=self.config,
            losses={
                'triplet_extractor_loss': loss * self.config['model']['triplet-extractor'][
                    'loss-weight'] * self.trainable,
            }
        )
        return full_loss

    def update_metrics(self, model_out: TripletModelOutput) -> None:
        tp_fn: int = model_out.loss_mask.sum().item()
        tp_fp: int = model_out.number_of_triplets()

        triplets: Tensor = self.threshold_data(model_out.features)
        tp: int = (triplets & model_out.true_predicted_mask).sum().item()

        self.final_metrics.update(tp=tp, tp_fp=tp_fp, tp_fn=tp_fn)

    def threshold_data(self, data: Tensor) -> Tensor:
        return data > self.config['model']['triplet-extractor']['threshold']

    def get_metrics(self) -> ModelMetric:
        return ModelMetric(
            metrics={
                'triplet_extractor_metric': self.final_metrics.compute(),
            }
        )

    def reset_metrics(self) -> None:
        self.final_metrics.reset()
