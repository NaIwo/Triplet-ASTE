from typing import List, Optional, Dict, Tuple

import torch
from torch.nn import Sequential
from torch import Tensor

from torchmetrics import MetricCollection
from .spans_manager import SpanInformationManager
from ...outputs.outputs import SpanInformationOutput, SpanCreatorOutput
from ...utils.const import CreatedSpanCodes
from ....dataset.domain import SpanCode
from ....dataset.reader import Batch
from ....models import BaseModel
from ....models.outputs import (
    ModelLoss,
    ModelMetric,
    BaseModelOutput
)
from ....models.specialty_models.spans.crf import CRF
from ....tools.metrics import get_selected_metrics
from ..utils import sequential_blocks


class SpanCreatorModel(BaseModel):
    def __init__(
            self,
            input_dim: int,
            config: Dict,
            model_name: str = 'Span Creator Model',
            extend_ranges: Optional[List[int]] = None
    ):
        super(SpanCreatorModel, self).__init__(model_name, config=config)

        self.metrics: MetricCollection = MetricCollection(
            metrics=get_selected_metrics(for_spans=True, dist_sync_on_step=True)
        ).to(self.config['general-training']['device'])

        self.extend_ranges: Optional[List[int]] = extend_ranges
        if extend_ranges is None:
            self.extend_ranges: List[int] = []

        self.input_dim: int = input_dim

        neurons: List = [input_dim, input_dim // 2, input_dim // 8, 5]
        self.span_creator: Sequential = sequential_blocks(neurons, self.config)
        self.crf = CRF(num_tags=5, batch_first=True)

    def forward(self, data_input: BaseModelOutput) -> SpanCreatorOutput:
        features: Tensor = self.get_features(data_input.features)
        aspects, opinions = self.get_spans(features, data_input.batch)
        return SpanCreatorOutput(
            batch=data_input.batch,
            features=features,
            aspects=aspects,
            opinions=opinions,
        )

    def get_features(self, data: Tensor) -> Tensor:
        return self.span_creator(data)

    def get_spans(self, data: Tensor, batch: Batch) -> Tuple[List[SpanInformationOutput], List[SpanInformationOutput]]:
        aspect_results: List[SpanInformationOutput] = list()
        opinion_results: List[SpanInformationOutput] = list()
        best_paths: List[List[int]] = self.crf.decode(data, mask=batch.emb_mask[:, :data.shape[1], ...])

        for best_path, sample in zip(best_paths, batch):
            best_path = torch.tensor(best_path).to(self.config['general-training']['device'])
            offset: int = sample.sentence_obj[0].encoder.offset
            best_path[:offset] = SpanCode.NOT_SPLIT
            best_path[sum(sample.emb_mask[0]) - offset:] = SpanCode.NOT_SPLIT
            aspect_results.append(
                self.get_spans_information_from_sequence(best_path, sample, 'ASPECT')
            )
            opinion_results.append(
                self.get_spans_information_from_sequence(best_path, sample, 'OPINION')
            )

        return aspect_results, opinion_results

    def get_spans_information_from_sequence(self, seq: Tensor, sample: Batch, source: str) -> SpanInformationOutput:
        seq = self._replace_not_split(seq, source)
        begins = self._get_begin_indices(seq, sample, source)

        span_manager = SpanInformationManager()

        code = CreatedSpanCodes.ADDED_TRUE if self.config['model']['add-true-spans'] else CreatedSpanCodes.NOT_RELEVANT
        span_manager.add_true_information(sample, source, code)

        idx: int
        b_idx: int
        for idx, b_idx in enumerate(begins[:-1]):
            end_idx: int = begins[idx + 1] - 1
            end_idx = self._get_end_idx(seq, b_idx, end_idx)
            span_manager.add_predicted_information(b_idx, end_idx)

        if not span_manager.span_ranges:
            span_manager.add_predicted_information(0, len(seq) - 1)
        else:
            span_manager.extend_span_ranges(sample, self.extend_ranges)

        return SpanInformationOutput.from_span_manager(
            span_manager,
            sample.sentence_obj[0]
        ).to_device(self.config['general-training']['device'])

    @staticmethod
    def _replace_not_split(seq: Tensor, source: str) -> Tensor:
        condition = (seq != SpanCode[f'BEGIN_{source}']) & \
                    (seq != SpanCode[f'INSIDE_{source}'])
        seq = torch.where(condition, SpanCode.NOT_SPLIT, seq)
        return seq

    def _get_begin_indices(self, seq: Tensor, sample: Batch, source: str) -> List[int]:
        begins = torch.where(seq == SpanCode[f'BEGIN_{source}'])[0]
        end = sum(sample.emb_mask[0]) - (2 * sample.sentence_obj[0].encoder.offset)
        end = torch.tensor([end], device=self.config['general-training']['device'])
        begins = torch.cat((begins, end))
        begins = [sample.sentence_obj[0].agree_index(idx) for idx in begins]
        begins[-1] += 1
        return begins

    @staticmethod
    def _get_end_idx(seq: Tensor, b_idx: int, end_idx: int) -> int:
        s: Tensor = seq[b_idx:end_idx]
        if SpanCode.NOT_SPLIT in s:
            end_idx = int(torch.where(s == SpanCode.NOT_SPLIT)[0][0])
            end_idx += b_idx - 1
        return end_idx

    def get_loss(self, model_out: SpanCreatorOutput) -> ModelLoss:
        loss = -self.crf(
            model_out.features,
            model_out.batch.chunk_label,
            model_out.batch.emb_mask,
            reduction='token_mean'
        )
        return ModelLoss(
            span_creator_loss=loss,
            config=self.config
        )

    def update_metrics(self, model_out: SpanCreatorOutput) -> None:
        b: Batch = model_out.batch
        pred: SpanCreatorOutput
        for pred, aspect, opinion in zip(model_out, b.aspect_spans, b.opinion_spans):
            tp = pred.aspects[0].get_number_of_predicted_true_elements()
            tp += pred.opinions[0].get_number_of_predicted_true_elements()

            tp_fp = pred.aspects[0].get_number_of_predicted_elements(with_repeated=False)
            tp_fp += pred.opinions[0].get_number_of_predicted_elements(with_repeated=False)

            true: Tensor = torch.cat([aspect, opinion], dim=0).unique(dim=0)
            tp_fn: int = true.shape[0] - int(-1 in true)

            self.metrics.update(tp=tp, tp_fp=tp_fp, tp_fn=tp_fn)

    def get_metrics(self) -> ModelMetric:
        return ModelMetric(
            span_creator_metric=self.metrics.compute()
        )

    def reset_metrics(self) -> None:
        self.metrics.reset()
