from typing import Optional, List, TypeVar

import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence

from .spans_manager import SpanInformationManager
from ..sentiments.sentiment_outputs import SentimentModelOutput
from ...outputs import BaseModelOutput
from ASTE.aste.models.outputs.utils import construct_predicted_spans
from ....dataset.domain import Sentence, Span, ASTELabels, CreatedSpanCodes
from ....dataset.reader import Batch

SIO = TypeVar('SIO', bound='SpanInformationOutput')


class SpanInformationOutput(BaseModelOutput):
    def __init__(
            self,
            span_range: Tensor,
            span_creation_info: Tensor,
            sentiments: Tensor,
            mapping_indexes: Tensor,
            sentence: Sentence,
            repeated: Optional[int] = None
    ):
        super().__init__()
        self.span_range: Tensor = span_range
        self.span_creation_info: Tensor = span_creation_info
        self.sentiments: Tensor = sentiments
        self.mapping_indexes: Tensor = mapping_indexes
        self.sentence: Sentence = sentence
        self.spans: List[Span] = construct_predicted_spans(span_range, sentence)
        self.repeated: Optional[int] = repeated

    @classmethod
    def from_span_manager(cls, span_manager: SpanInformationManager, sentence: Sentence) -> SIO:
        return SpanInformationOutput(
            span_range=torch.tensor(span_manager.span_ranges),
            span_creation_info=torch.tensor(span_manager.span_creation_info),
            sentiments=torch.tensor(span_manager.sentiments),
            mapping_indexes=torch.tensor(span_manager.mapping_indexes),
            sentence=sentence
        )

    def to_device(self, device: str):
        self.span_range = self.span_range.to(device)
        self.span_creation_info = self.span_creation_info.to(device)
        self.sentiments = self.sentiments.to(device)
        self.mapping_indexes = self.mapping_indexes.to(device)

        return self

    def repeat(self, n: int):
        return SpanInformationOutput(
            span_range=self.span_range.repeat(n, 1).squeeze(dim=0),
            span_creation_info=self.span_creation_info.repeat(n),
            sentiments=self.sentiments.repeat(n),
            mapping_indexes=self._get_repeated_mapping_indexes(n),
            sentence=self.sentence,
            repeated=n
        )

    def _get_repeated_mapping_indexes(self, n: int) -> Tensor:
        repeated_indexes = self.mapping_indexes.repeat(n)

        mask = (self.mapping_indexes >= 0).repeat(n)
        indexes = torch.full_like(self.mapping_indexes, self.mapping_indexes.shape[0]).repeat(n).to(self.sentiments)
        skip = torch.arange(n).unsqueeze(-1).repeat(1, self.mapping_indexes.shape[0]).flatten().to(self.sentiments)
        repeated_indexes += (indexes * skip) * mask

        return repeated_indexes


class SpanPredictionsOutput(BaseModelOutput):
    def __init__(self, aspects: List[SpanInformationOutput], opinions: List[SpanInformationOutput]):
        super().__init__()
        self.aspects: List[SpanInformationOutput] = aspects
        self.opinions: List[SpanInformationOutput] = opinions

    def get_aspect_span_predictions(self) -> List[Tensor]:
        return self._get(self.aspects, 'span_range')

    def get_opinion_span_predictions(self) -> List[Tensor]:
        return self._get(self.opinions, 'span_range')

    def get_aspect_span_creation_info(self) -> Tensor:
        return self._pad(self._get(self.aspects, 'span_creation_info'), CreatedSpanCodes.NOT_RELEVANT)

    def get_opinion_span_creation_info(self) -> Tensor:
        return self._pad(self._get(self.opinions, 'span_creation_info'), CreatedSpanCodes.NOT_RELEVANT)

    def get_aspect_span_sentiments(self) -> Tensor:
        return self._pad(self._get(self.aspects, 'sentiments'), ASTELabels.NOT_RELEVANT)

    def get_opinion_span_sentiments(self) -> Tensor:
        return self._pad(self._get(self.opinions, 'sentiments'), ASTELabels.NOT_RELEVANT)

    @staticmethod
    def _get(source: List[SpanInformationOutput], attr: str) -> List[Tensor]:
        return [getattr(span, attr) for span in source]

    @staticmethod
    def _pad(data: List[Tensor], pad_value: int) -> Tensor:
        return pad_sequence(data, padding_value=pad_value, batch_first=True)


class SpanCreatorOutput(BaseModelOutput):
    def __init__(self,
                 batch: Batch, features: Tensor,
                 predicted_spans: SpanPredictionsOutput,
                 aspects_agg_emb: Optional[Tensor] = None,
                 opinions_agg_emb: Optional[Tensor] = None
                 ):
        super().__init__(batch=batch, features=features)
        self.predicted_spans: SpanPredictionsOutput = predicted_spans
        self.aspects_agg_emb: Optional[Tensor] = aspects_agg_emb
        self.opinions_agg_emb: Optional[Tensor] = opinions_agg_emb

    @property
    def all_predicted_spans(self) -> List[Tensor]:
        return [
            torch.cat([a, o], dim=0) for a, o in zip(
                self.predicted_spans.get_aspect_span_predictions(),
                self.predicted_spans.get_opinion_span_predictions()
            )
        ]

    def extend_opinion_embeddings(self, data: SentimentModelOutput):
        opinions_agg_emb = torch.cat(list(data.sentiment_features.values()), dim=1).to(self.aspects_agg_emb)
        predicted_spans = self.predicted_spans
        predicted_spans.opinions = self._extend_opinion_information(self.predicted_spans.opinions, data)
        return SpanCreatorOutput(
            batch=self.batch,
            features=self.features,
            predicted_spans=predicted_spans,
            aspects_agg_emb=self.aspects_agg_emb,
            opinions_agg_emb=opinions_agg_emb
        )

    @staticmethod
    def _extend_opinion_information(
            opinions: List[SpanInformationOutput],
            data: SentimentModelOutput
    ) -> List[SpanInformationOutput]:
        results: List = list()

        keys = data.sentiment_features.keys()
        opinion: SpanInformationOutput
        for opinion in opinions:
            num_elements: int = opinion.span_creation_info.shape[0]
            repeated_opinion: SpanInformationOutput = opinion.repeat(len(keys))
            for key_idx, key in enumerate(keys):
                indexes = range(key_idx * num_elements, num_elements * (key_idx + 1))

                condition = (repeated_opinion.sentiments[indexes] != key)

                c_added = repeated_opinion.span_creation_info[indexes] == CreatedSpanCodes.ADDED_TRUE
                c_added |= repeated_opinion.span_creation_info[indexes] == CreatedSpanCodes.ADDED_FALSE

                c_pred = repeated_opinion.span_creation_info[indexes] == CreatedSpanCodes.PREDICTED_TRUE
                c_pred |= repeated_opinion.span_creation_info[indexes] == CreatedSpanCodes.PREDICTED_FALSE

                repeated_opinion.span_creation_info[indexes] = torch.where(
                    condition & c_added,
                    CreatedSpanCodes.ADDED_FALSE,
                    repeated_opinion.span_creation_info[indexes]
                )

                repeated_opinion.span_creation_info[indexes] = torch.where(
                    condition & c_pred,
                    CreatedSpanCodes.PREDICTED_FALSE,
                    repeated_opinion.span_creation_info[indexes]
                )

                repeated_opinion.sentiments[indexes] = key
            results.append(repeated_opinion)
        return results
