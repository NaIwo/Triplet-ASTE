import logging
from typing import List, Dict, Type

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from ASTE.aste.models import ModelOutput, ModelLoss, ModelMetric, BaseModel
from ASTE.aste.models.model_elements.embeddings import Bert, BaseEmbedding
from ASTE.aste.models.model_elements.span_aggregators import (BaseAggregator,
                                                              AggResult,
                                                              EndPointAggregator)
from ASTE.aste.models.model_elements.triplets_mining import ManhattanTripletsMiner, BaseTripletsMiner
from ASTE.dataset.reader import Batch
from ASTE.utils import config
from .specialty_models import SpanCreatorModel, TripletExtractorModel, Classifier


class BertBaseModel(BaseModel):
    def __init__(self, model_name='Bert Base Model', *args, **kwargs):

        super(BertBaseModel, self).__init__(model_name)
        self.emb_layer: BaseEmbedding = Bert()
        self.span_creator: BaseModel = SpanCreatorModel(input_dim=self.emb_layer.embedding_dim)
        self.aggregator: BaseAggregator = EndPointAggregator(input_dim=self.emb_layer.embedding_dim)
        self.span_classifier: BaseModel = Classifier(input_dim=self.aggregator.output_dim)
        self.triplets_miner: Type[BaseTripletsMiner] = ManhattanTripletsMiner
        self.triplets_extractor: BaseModel = TripletExtractorModel(input_dim=self.aggregator.output_dim)

        epochs: List = [3, 5, config['model']['total-epochs']]

        self.training_scheduler: Dict = {
            range(0, epochs[0]): {
                'freeze': [self.triplets_extractor],
                'unfreeze': [self.span_creator, self.span_classifier, self.emb_layer]
            },
            range(epochs[0], epochs[1]): {
                'freeze': [self.span_creator, self.span_classifier, self.emb_layer],
                'unfreeze': [self.triplets_extractor]
            },
            range(epochs[1], epochs[2]): {
                'freeze': [],
                'unfreeze': [self.span_creator, self.span_classifier, self.triplets_extractor, self.emb_layer],
            }
        }

    def forward(self, batch: Batch) -> ModelOutput:
        emb_span_creator: torch.Tensor = self.emb_layer(batch.sentence, batch.mask)

        span_creator_output: torch.Tensor = self.span_creator(emb_span_creator, batch.mask)
        predicted_spans: List[torch.Tensor] = self.span_creator.get_spans(span_creator_output, batch)

        agg_results: AggResult = self.aggregator.aggregate(emb_span_creator, predicted_spans)

        span_classifier_output: torch.Tensor = self.span_classifier(agg_results.agg_embeddings)
        triplet_input: torch.Tensor = span_classifier_output[..., 0:1] * agg_results.agg_embeddings

        m = self.triplets_miner.from_prediction(span_classifier_output, agg_results)

        triplet_results: torch.Tensor = self.triplets_extractor(triplet_input)

        return ModelOutput(batch=batch,
                           span_creator_output=span_creator_output,
                           predicted_spans=predicted_spans,
                           span_classifier_output=span_classifier_output,
                           triplet_results=triplet_results)

    def get_loss(self, model_out: ModelOutput) -> ModelLoss:
        return ModelLoss.from_instances(
            span_creator_loss=self.span_creator.get_loss(model_out) * self.span_creator.trainable,
            triplet_extractor_loss=self.triplets_extractor.get_loss(
                model_out) * self.triplets_extractor.trainable,
            span_classifier_loss=self.span_classifier.get_loss(
                model_out) * self.span_classifier.trainable)

    def update_metrics(self, model_out: ModelOutput) -> None:
        self.span_creator.update_metrics(model_out)
        self.triplets_extractor.update_metrics(model_out)
        self.span_classifier.update_metrics(model_out)

    def get_metrics(self) -> ModelMetric:
        return ModelMetric.from_instances(span_creator_metric=self.span_creator.get_metrics(),
                                          triplet_metric=self.triplets_extractor.get_metrics(),
                                          span_classifier_metric=self.span_classifier.get_metrics())

    def reset_metrics(self) -> None:
        self.span_creator.reset_metrics()
        self.triplets_extractor.reset_metrics()
        self.span_classifier.reset_metrics()

    def get_params_and_lr(self) -> List[Dict]:
        return [
            {'params': self.span_creator.parameters(), 'lr': config['model']['learning-rate']},
            {'params': self.aggregator.parameters(), 'lr': config['model']['learning-rate']},
            {'params': self.span_classifier.parameters(), 'lr': config['model']['learning-rate']},
            {'params': self.triplets_extractor.parameters(), 'lr': config['model']['learning-rate']},
            {'params': self.emb_layer.parameters(), 'lr': config['model']['bert']['learning-rate']}
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
                true_spans: torch.Tensor = torch.cat([sample.aspect_spans[0], sample.opinion_spans[0]], dim=0).unique(
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
    def _count_intersection(true_spans: torch.Tensor, predicted_spans: torch.Tensor) -> int:
        predicted_spans = predicted_spans.unique(dim=0)
        all_spans: torch.Tensor = torch.cat([true_spans, predicted_spans], dim=0)
        uniques, counts = torch.unique(all_spans, return_counts=True, dim=0)
        return uniques[counts > 1].shape[0]
