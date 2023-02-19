from typing import List, Optional, Callable, Dict

import numpy as np
import torch
from torch import Tensor

from ....dataset.domain import SpanCode, ASTELabels
from ....dataset.reader import Batch
from ....models import BaseModel
from ....models.outputs import (
    ModelLoss,
    ModelMetric,
    SpanCreatorOutput,
    BaseModelOutput,
    ModelOutput,
    SpanPredictionsOutput,
    SpanInformationOutput
)
from ....models.specialty_models.spans.crf import CRF
from ....tools.metrics import Metric, get_selected_metrics


class SpanCreatorModel(BaseModel):
    def __init__(
            self,
            input_dim: int,
            config: Dict,
            model_name: str = 'Span Creator Model',
            extend_spans: Optional[List[int]] = None
    ):
        super(SpanCreatorModel, self).__init__(model_name, config=config)

        self.metrics: Metric = Metric(
            name='Span Creator',
            metrics=get_selected_metrics(for_spans=True)
        ).to(self.config['general-training']['device'])

        self.extend_spans: Optional[List[int]] = extend_spans
        if extend_spans is None:
            self.extend_spans: List[int] = []

        self.input_dim: int = input_dim

        self.crf = CRF(num_tags=5, batch_first=True)
        self.linear_layer = torch.nn.Linear(input_dim, input_dim // 2)
        self.final_layer = torch.nn.Linear(input_dim // 2, 5)

    def forward(self, data_input: BaseModelOutput) -> SpanCreatorOutput:
        features: Tensor = self.get_features(data_input.features)
        predicted_spans: SpanPredictionsOutput = self.get_spans(features, data_input.batch)
        return SpanCreatorOutput(
            batch=data_input.batch,
            features=features,
            predicted_spans=predicted_spans
        )

    def get_features(self, data: Tensor) -> Tensor:
        out = self.linear_layer(data)
        return self.final_layer(out)

    def get_spans(self, data: Tensor, batch: Batch) -> SpanPredictionsOutput:
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

        return SpanPredictionsOutput(
            aspects=aspect_results,
            opinions=opinion_results
        )

    def get_spans_information_from_sequence(self, seq: Tensor, sample: Batch, source: str) -> SpanInformationOutput:
        seq = self._replace_not_split(seq, source)
        begins = self._get_begin_indices(seq, sample, source)

        span_ranges: List[List[int, int]] = list()
        labels: List[int] = list()
        sentiments: List[int] = list()

        if self.training:
            self._add_true_information(span_ranges, labels, sentiments, sample, source)

        idx: int
        b_idx: int
        for idx, b_idx in enumerate(begins[:-1]):
            end_idx: int = begins[idx + 1] - 1
            end_idx = self._get_end_idx(seq, b_idx, end_idx)
            if (end_idx >= b_idx) and ([b_idx, end_idx] not in span_ranges):
                span_ranges.append([b_idx, end_idx])
                self._add_labels_and_sentiments(labels, sentiments, 1)

        if not span_ranges:
            span_ranges.append([0, len(seq) - 1])
            self._add_labels_and_sentiments(labels, sentiments, 1)
        else:
            self.extend_span_ranges(span_ranges, sample)
            added_count: int = len(span_ranges) - len(sentiments)
            self._add_labels_and_sentiments(labels, sentiments, added_count)

        return SpanInformationOutput(
            span_range=torch.tensor(span_ranges).to(self.config['general-training']['device']),
            labels=torch.tensor(labels).to(self.config['general-training']['device']),
            sentiments=torch.tensor(sentiments).to(self.config['general-training']['device'])
        )

    @staticmethod
    def _replace_not_split(seq: Tensor, source: str) -> Tensor:
        condition = torch.tensor(seq != SpanCode[f'BEGIN_{source}']) & \
                    torch.tensor(seq != SpanCode[f'INSIDE_{source}'])
        seq = torch.where(condition, SpanCode.NOT_SPLIT, seq)
        return seq

    def _get_begin_indices(self, seq: Tensor, sample: Batch, source: str) -> List[int]:
        begins = torch.where(torch.tensor(seq == SpanCode[f'BEGIN_{source}']))[0]
        end = sum(sample.emb_mask[0]) - (2 * sample.sentence_obj[0].encoder.offset)
        end = torch.tensor([end], device=self.config['general-training']['device'])
        begins = torch.cat((begins, end))
        begins = [sample.sentence_obj[0].agree_index(idx) for idx in begins]
        begins[-1] += 1
        return begins

    @staticmethod
    def _add_true_information(span_ranges: List, labels: List, sentiments: List, sample: Batch, source: str) -> None:
        true_spans: List = getattr(sample, f'{source.lower()}_spans')[0].tolist()
        true_spans: np.ndarray
        unique_idx: np.ndarray
        true_spans, unique_idx = np.unique(true_spans, return_index=True, axis=0)
        true_sentiments: List = sample.sentiments[0][unique_idx].tolist()
        span_ranges += true_spans.tolist()
        labels += [True] * unique_idx.shape[0]
        sentiments += true_sentiments

    @staticmethod
    def _get_end_idx(seq: Tensor, b_idx: int, end_idx: int) -> int:
        s: Tensor = seq[b_idx:end_idx]
        if SpanCode.NOT_SPLIT in s:
            end_idx = int(torch.where(torch.tensor(s == SpanCode.NOT_SPLIT))[0][0])
            end_idx += b_idx - 1
        return end_idx

    def _add_labels_and_sentiments(self, labels: List, sentiments: List, count: int) -> None:
        labels += [not self.training] * count
        sentiments += [ASTELabels.NOT_RELEVANT] * count

    def extend_span_ranges(self, span_ranges: List, sample: Batch) -> None:
        before: Callable = sample.sentence_obj[0].get_index_before_encoding
        after: Callable = sample.sentence_obj[0].get_index_after_encoding

        begin: int
        end: int
        for begin, end in span_ranges:
            temp: List = [[after(before(begin) + shift), end] for shift in self.extend_spans]
            self._add_correct_extended(span_ranges, temp)

            temp: List = [[begin, after(before(end) + shift)] for shift in self.extend_spans]
            self._add_correct_extended(span_ranges, temp)

    @staticmethod
    def _add_correct_extended(results: List, extended: List) -> List:
        begin: int
        end: int
        for begin, end in extended:
            if -1 != end >= begin != -1 and [begin, end] not in results:
                results.append([begin, end])

        return results

    def get_loss(self, model_out: ModelOutput) -> ModelLoss:
        loss = -self.crf(
            model_out.span_creator_output.features,
            model_out.batch.chunk_label,
            model_out.batch.emb_mask,
            reduction='token_mean'
        )
        return ModelLoss(
            span_creator_loss=loss,
            config=self.config
        )

    def update_metrics(self, model_out: ModelOutput) -> None:
        predicted = model_out.span_creator_output.all_predicted_spans
        b: Batch = model_out.batch
        for pred, aspect, opinion in zip(predicted, b.aspect_spans, b.opinion_spans):
            true: Tensor = torch.cat([aspect, opinion], dim=0).unique(dim=0)
            true_count: int = true.shape[0] - int(-1 in true)
            self.metrics(pred, true, true_count)

    def get_metrics(self) -> ModelMetric:
        return ModelMetric(
            span_creator_metric=self.metrics.compute()
        )

    def reset_metrics(self) -> None:
        self.metrics.reset()
