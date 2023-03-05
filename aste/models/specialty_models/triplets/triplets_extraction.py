from typing import Dict, List

import torch
from torch import Tensor
from torch.nn import Sequential

from ...utils.triplet_utils import (
    create_embeddings_matrix_by_concat,
    create_mask_matrix_for_loss,
    create_mask_matrix_for_prediction,
    create_embedding_mask_matrix,
    get_true_predicted_mask
)
from ...utils.const import TripletDimensions
from ..utils import sequential_blocks
from ...base_model import BaseModel
from ...outputs import (
    SpanInformationOutput,
    SpanCreatorOutput,
    SampleTripletOutput,
    TripletModelOutput
)
from ....models.outputs import (
    ModelLoss,
    ModelMetric
)
from ....tools.metrics import Metric, get_selected_metrics


class TripletExtractorModel(BaseModel):
    def __init__(self, input_dim: int, config: Dict, model_name: str = 'Triplet Extractor Model'):
        super(TripletExtractorModel, self).__init__(model_name=model_name, config=config)

        metrics = get_selected_metrics(for_spans=True)
        self.final_metrics: Metric = Metric(name='Final predictions', metrics=metrics).to(
            self.config['general-training']['device'])

        input_dimension: int = input_dim * 2

        neurons: List = [input_dimension, input_dimension // 2, input_dimension // 4, input_dimension // 8, 1]
        self.similarity: Sequential = sequential_blocks(neurons, self.config)
        self.similarity.append(torch.nn.Sigmoid())

    def forward(self, data_input: SpanCreatorOutput) -> TripletModelOutput:
        matrix: Tensor = self._forward_embeddings(data_input)

        loss_mask: Tensor = create_mask_matrix_for_loss(data_input)
        prediction_mask: Tensor = create_mask_matrix_for_prediction(data_input)

        triplets: List[SampleTripletOutput] = self.get_triplets_from_matrix(matrix * prediction_mask, data_input)

        true_predicted_mask = get_true_predicted_mask(data_input)

        return TripletModelOutput(
            batch=data_input.batch,
            triplets=triplets,
            similarities=matrix,
            loss_mask=loss_mask,
            true_predicted_mask=true_predicted_mask
        )

    def _forward_embeddings(self, data_input: SpanCreatorOutput) -> Tensor:
        matrix: Tensor = create_embeddings_matrix_by_concat(data_input)
        mask: Tensor = create_embedding_mask_matrix(data_input)
        matrix = self.similarity(matrix)
        return matrix.squeeze(-1) * mask

    def get_triplets_from_matrix(self, matrix: Tensor, data_input: SpanCreatorOutput) -> List[SampleTripletOutput]:
        triplets: List = list()

        sample: Tensor
        sample_aspects: SpanInformationOutput
        sample_opinions: SpanInformationOutput
        for sample, sample_aspects, sample_opinions in zip(matrix, data_input.aspects, data_input.opinions):
            significant: Tensor = self.threshold_data(sample).nonzero()
            a_ranges: Tensor = sample_aspects.span_range[significant[:, 0]]
            o_ranges: Tensor = sample_opinions.span_range[significant[:, 1]]
            sentiments: Tensor = sample_opinions.sentiments[significant[:, 1]]
            triplets.append(
                SampleTripletOutput(
                    aspect_ranges=a_ranges,
                    opinion_ranges=o_ranges,
                    sentiments=sentiments,
                    sentence=sample_opinions.sentence
                )
            )

        return triplets

    def get_loss(self, model_out: TripletModelOutput) -> ModelLoss:
        sim: Tensor = torch.exp(model_out.features)

        numerator: Tensor = sim * model_out.loss_mask

        negatives: Tensor = sim * (~model_out.loss_mask)
        denominator: Tensor = torch.sum(negatives, dim=TripletDimensions.ASPECT, keepdim=True)
        denominator = numerator + denominator

        loss: Tensor = numerator / denominator
        loss = torch.sum(loss, dim=[1, 2]) / torch.sum(model_out.loss_mask, dim=[1, 2])
        loss = -torch.log(loss)
        loss = torch.sum(loss) / self.config['general-training']['batch-size']
        return ModelLoss(triplet_extractor_loss=loss, config=self.config)

    def update_metrics(self, model_out: TripletModelOutput) -> None:
        tp_fn: int = model_out.loss_mask.sum().item()
        tp_fp: int = model_out.number_of_triplets()

        triplets: Tensor = self.threshold_data(model_out.features)
        tp: int = (triplets & model_out.true_predicted_mask).sum().item()

        self.final_metrics(tp=tp, tp_fp=tp_fp, tp_fn=tp_fn)

    def threshold_data(self, data: Tensor) -> Tensor:
        return data >= self.config['model']['triplet-extractor']['threshold']

    def get_metrics(self) -> ModelMetric:
        metrics: Dict = self.final_metrics.compute()
        return ModelMetric(triplet_metric=metrics)

    def reset_metrics(self) -> None:
        self.final_metrics.reset()
