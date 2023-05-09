from typing import Dict, List, Tuple

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
from ...utils.utils import create_random_tensor_with_one_true_per_row as random_row_mask
from ...utils.triplet_utils import (
    create_mask_matrix_for_loss,
    create_mask_matrix_for_prediction,
    get_true_predicted_mask,
    create_embedding_mask_matrix,
)
from ....models.outputs import (
    ModelLoss,
    ModelMetric
)
from ....tools.metrics import get_selected_metrics


class BaseTripletExtractorModel(BaseModel):
    def __init__(self, config: Dict, input_dim: int, model_name: str = 'Base Triplet Extractor Model', *args, **kwargs):
        super(BaseTripletExtractorModel, self).__init__(model_name=model_name, config=config)

        self.input_dim: int = input_dim
        metrics = get_selected_metrics(for_spans=True, dist_sync_on_step=True)
        self.final_metrics: MetricCollection = MetricCollection(metrics=metrics)

    def forward(self, data_input: SpanCreatorOutput) -> TripletModelOutput:
        data_input = data_input.copy()
        aspects, opinions = self._forward_embeddings(data_input)
        data_input.aspects_agg_emb = aspects
        data_input.opinions_agg_emb = opinions
        pad_mask: Tensor = create_embedding_mask_matrix(data_input)

        norm_sim: Tensor = self.normalized_similarity(aspects, opinions) * pad_mask
        sim: Tensor = self.similarity(aspects, opinions) * pad_mask

        loss_mask: Tensor = create_mask_matrix_for_loss(data_input)
        prediction_mask: Tensor = create_mask_matrix_for_prediction(data_input)

        triplets: List[SampleTripletOutput] = self.get_triplets_from_matrix(norm_sim * prediction_mask, data_input)

        true_predicted_mask = get_true_predicted_mask(data_input)

        return TripletModelOutput(
            batch=data_input.batch,
            triplets=triplets,
            similarities=sim,
            normalized_similarities=norm_sim,
            loss_mask=loss_mask,
            true_predicted_mask=true_predicted_mask,
            prediction_mask=prediction_mask,
            pad_mask=pad_mask
        )

    @property
    def output_dim(self):
        return self.input_dim * 2

    def _forward_embeddings(self, data_input: SpanCreatorOutput) -> Tuple[Tensor, Tensor]:
        aspects = self.aspect_net(data_input.aspects_agg_emb)
        opinions = self.opinion_net(data_input.opinions_agg_emb)
        return aspects, opinions

    def normalized_similarity(self, aspects: Tensor, opinions: Tensor) -> Tensor:
        return self.similarity(aspects, opinions)

    def similarity(self, aspects: Tensor, opinions: Tensor) -> Tensor:
        raise NotImplementedError

    def get_triplets_from_matrix(self, matrix: Tensor, data_input: SpanCreatorOutput) -> List[SampleTripletOutput]:
        raise NotImplementedError

    def get_loss(self, model_out: TripletModelOutput) -> ModelLoss:
        denominator = 0
        loss = torch.tensor(0.0).to(self.device)

        if self.config['model']['triplet-extractor']['aspect-to-opinion']:
            denominator += 1
            loss += self._get_loss(model_out, dim=TripletDimensions.OPINION)

        if self.config['model']['triplet-extractor']['opinion-to-aspect']:
            denominator += 1
            loss += self._get_loss(model_out, dim=TripletDimensions.ASPECT)

        if denominator == 0:
            return ModelLoss(config=self.config, losses={})

        loss = loss / denominator
        full_loss = ModelLoss(
            config=self.config,
            losses={
                'triplet_extractor_loss': loss * self.config['model']['triplet-extractor'][
                    'loss-weight'] * self.trainable,
            }
        )

        return full_loss

    def _get_loss(self, model_out: TripletModelOutput, dim: int) -> Tensor:
        mask = model_out.loss_mask
        phrase_mask = torch.isin(mask.int(), 1).any(dim=dim, keepdim=True)
        sample_mask = random_row_mask(model_out.pad_mask, dim, self.device)
        mask = torch.where(phrase_mask, mask, sample_mask)
        mask = mask & model_out.pad_mask

        reverse_loss_mask = (~model_out.loss_mask) * model_out.pad_mask

        sim: Tensor = model_out.similarities
        sim = torch.exp(sim / self.config['model']['triplet-extractor']['temperature'])
        weight = self.config['model']['triplet-extractor']['negative-weight']
        sim_numerator = torch.where(phrase_mask, sim, torch.full_like(sim, weight).to(self.device))

        numerator: Tensor = sim_numerator * mask

        negatives: Tensor = sim * reverse_loss_mask
        k = self.config['model']['triplet-extractor']['num-negatives']
        k = min(k, negatives.shape[dim])
        negatives = torch.topk(negatives, k=k, dim=dim, sorted=False).values

        denominator: Tensor = torch.sum(negatives, dim=dim, keepdim=True)
        denominator = numerator + denominator
        denominator += 1e-8

        loss: Tensor = numerator / denominator
        phrase_normalizer = torch.sum(mask, dim=dim)
        loss = torch.sum(loss, dim=dim) / (phrase_normalizer + 1e-8)
        loss = torch.sum(loss, dim=-1) / ((phrase_normalizer > 0).sum().float() + 1e-8)
        loss = -torch.log(loss)
        loss = torch.sum(loss) / self.config['general-training']['batch-size']

        if loss.isinf():
            loss = torch.tensor(0.).to(self.device)

        return loss

    def update_metrics(self, model_out: TripletModelOutput) -> None:
        tp_fn: int = model_out.loss_mask.sum().item()
        # tp_fp: int = model_out.prediction_mask.sum().item()

        triplets: Tensor = self.threshold_data(model_out.normalized_similarities)
        tp_fp: int = (triplets & model_out.prediction_mask).sum().item()
        tp: int = (triplets & model_out.true_predicted_mask).sum().item()

        self.final_metrics.update(tp=tp, tp_fp=tp_fp, tp_fn=tp_fn)

    def threshold_data(self, data: Tensor) -> Tensor:
        thr = self.config['model']['triplet-extractor']['threshold']
        if isinstance(thr, float):
            return data > thr
        else:
            mask = data > 0.0
            thr = min(thr, data.shape[-1])
            indices = torch.topk(data, thr, dim=-1).indices
            mask = torch.zeros_like(data).scatter_(-1, indices, 1) * mask
            return mask * data > 0.0

    def get_metrics(self) -> ModelMetric:
        return ModelMetric(
            metrics={
                'triplet_extractor_metric': self.final_metrics.compute(),
            }
        )

    def reset_metrics(self) -> None:
        self.final_metrics.reset()
