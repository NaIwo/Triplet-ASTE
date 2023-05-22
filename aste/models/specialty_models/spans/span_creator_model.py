from typing import List, Optional, Dict, Tuple
from ast import literal_eval

import torch
from torch import Tensor
from torchmetrics import MetricCollection

from .spans_manager import SpanInformationManager
from ..utils import sequential_blocks
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


class SpanCreatorModel(BaseModel):
    def __init__(
            self,
            input_dim: int,
            config: Dict,
            model_name: str = 'Span Creator Model',
            extend_ranges: Optional[List[List[int]]] = None
    ):
        super(SpanCreatorModel, self).__init__(model_name, config=config)

        self.metrics: MetricCollection = MetricCollection(
            metrics=get_selected_metrics(for_spans=True, dist_sync_on_step=True)
        )

        self.input_dim: int = input_dim

        self.crf = CRF(num_tags=5, batch_first=True).to(self.device)
        neurons: List = [input_dim, input_dim // 2, input_dim // 4, input_dim // 8]
        self.linear_layer = sequential_blocks(neurons, self.device, is_last=False)
        self.final_layer = torch.nn.Linear(input_dim // 8, 5)

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
        out = self.linear_layer(data)
        return self.final_layer(out)

    def get_spans(self, data: Tensor, batch: Batch) -> Tuple[List[SpanInformationOutput], List[SpanInformationOutput]]:
        aspect_results: List[SpanInformationOutput] = list()
        opinion_results: List[SpanInformationOutput] = list()
        best_paths: List[List[int]] = self.crf.decode(data, mask=batch.emb_mask[:, :data.shape[1], ...])

        for best_path, sample in zip(best_paths, batch):
            best_path = torch.tensor(best_path).to(sample.emb_mask)
            offset: int = sample.sentence_obj[0].encoder.offset
            best_path[:offset] = SpanCode.NOT_SPLIT
            best_path[sum(sample.emb_mask[0]) - offset:] = SpanCode.NOT_SPLIT

            aspects = self.get_spans_information_from_sequence(best_path, sample, 'ASPECT')
            opinions = self.get_spans_information_from_sequence(best_path, sample, 'OPINION')

            if self.config['model']['span-creator']['add-aspects-to-opinions']:
                opinions.add_span_manager(aspects)
            if self.config['model']['span-creator']['add-opinions-to-aspects']:
                aspects.add_span_manager(opinions)

            if self.config['model']['span-creator']['all-opinion-spans-window'] > 0:
                opinions = self.get_all_spans_with_max_window(sample, 'OPINION')
            if self.config['model']['span-creator']['all-aspect-spans-window'] > 0:
                aspects = self.get_all_spans_with_max_window(sample, 'ASPECT')

            aspect_results.append(
                SpanInformationOutput.from_span_manager(aspects, sample.sentence_obj[0]).to_device(data.device)
            )
            opinion_results.append(
                SpanInformationOutput.from_span_manager(opinions, sample.sentence_obj[0]).to_device(data.device)
            )

        return aspect_results, opinion_results

    def get_spans_information_from_sequence(self, seq: Tensor, sample: Batch, source: str) -> SpanInformationManager:
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
        elif self.config['model']['span-creator'][f'extend-{source.lower()}-span-ranges'] is not None:
            max_number = self.config['model']['span-creator']['max-number-of-spans']
            extend_ranges = self.config['model']['span-creator'][f'extend-{source.lower()}-span-ranges']
            span_manager.extend_span_ranges(sample, extend_ranges, max_number=max_number)

        return span_manager

    def get_all_spans_with_max_window(self, sample: Batch, source: str) -> SpanInformationManager:
        span_manager = SpanInformationManager()
        code = CreatedSpanCodes.ADDED_TRUE if self.config['model']['add-true-spans'] else CreatedSpanCodes.NOT_RELEVANT
        span_manager.add_true_information(sample, source, code)

        offset: int = sample.sentence_obj[0].encoder.offset
        start_idx: int
        for start_idx in range(offset, sample.sub_words_mask.shape[1] - offset):
            if sample.sub_words_mask[:, start_idx] == 0:
                continue
            window: int = 1
            window_size: int = self.config['model']['span-creator']['all-opinion-spans-window']
            while (window <= window_size) and (start_idx + window < sample.sub_words_mask.shape[1]):
                if sample.sub_words_mask[:, start_idx + window] == 0:
                    window_size += 1
                else:
                    span_manager.add_predicted_information(start_idx, start_idx + window - 1)
                window += 1
        return span_manager

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
            reduction='mean'
        )

        full_loss = ModelLoss(
            config=self.config,
            losses={
                'span_creator_loss': loss * self.config['model']['span-creator']['loss-weight'] * self.trainable,
            }
        )
        return full_loss

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
            metrics={
                'span_creator_metric': self.metrics.compute()
            }
        )

    def reset_metrics(self) -> None:
        self.metrics.reset()
