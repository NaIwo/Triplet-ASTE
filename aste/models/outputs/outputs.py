from typing import List, TypeVar, Optional, Dict

import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence

from ...dataset.domain import ASTELabels, Sentence, Span
from ...dataset.reader import Batch

MO = TypeVar('MO', bound='BaseModelOutput')


class BaseModelOutput:
    NAME: str = 'Outputs'

    def __init__(self, batch: Optional[Batch] = None, features: Optional[Tensor] = None):
        self.batch: Optional[Batch] = batch
        self.features: Optional[Tensor] = features

    def __repr__(self) -> str:
        return self.__str__()

    def release_memory(self) -> MO:
        self.batch = None
        return self


class SentimentModelOutput(BaseModelOutput):
    def __init__(self, sentiment_features: Dict[int, Tensor]):
        super().__init__()
        self.sentiment_features: Dict[int, Tensor] = sentiment_features

    def __getitem__(self, item: int) -> Tensor:
        return self.sentiment_features[item]


class SpanInformationOutput(BaseModelOutput):
    def __init__(
            self,
            span_range: Tensor,
            labels: Tensor,
            sentiments: Tensor,
            mapping_indexes: Tensor,
            sentence: Sentence,
            repeated: Optional[int] = None
    ):
        super().__init__()
        self.span_range: Tensor = span_range
        self.labels: Tensor = labels
        self.sentiments: Tensor = sentiments
        self.mapping_indexes: Tensor = mapping_indexes
        self.sentence: Sentence = sentence
        self.spans: List[Span] = self._construct_predicted_spans()
        self.repeated: Optional[int] = repeated

    def _construct_predicted_spans(self) -> List[Span]:
        spans: List[Span] = list()
        for s_range in self.span_range:
            s_range = [
                self.sentence.get_index_before_encoding(s_range[0]),
                self.sentence.get_index_before_encoding(s_range[1])
            ]
            spans.append(Span.from_range(s_range, self.sentence.sentence))
        return spans

    def repeat(self, n: int):
        return SpanInformationOutput(
            span_range=self.span_range.repeat(n, 1).squeeze(dim=0),
            labels=self.labels.repeat(n),
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

    def get_aspect_span_labels(self) -> Tensor:
        return self._pad(self._get(self.aspects, 'labels'), 0)

    def get_opinion_span_labels(self) -> Tensor:
        return self._pad(self._get(self.opinions, 'labels'), 0)

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
            num_elements: int = opinion.labels.shape[0]
            repeated_opinion: SpanInformationOutput = opinion.repeat(len(keys))
            for key_idx, key in enumerate(keys):
                indexes = range(key_idx * num_elements, num_elements * (key_idx + 1))
                repeated_opinion.labels[indexes] = (repeated_opinion.sentiments[indexes] == key)
                labels: Tensor = repeated_opinion.labels[indexes]
                repeated_opinion.sentiments[indexes] = torch.where(labels, key, ASTELabels.NOT_RELEVANT)
            results.append(repeated_opinion)
        return results


class TripletModelOutput(BaseModelOutput):
    def __init__(self):
        super().__init__()


class ModelOutput(BaseModelOutput):

    def __init__(
            self,
            batch: Batch,
            span_creator_output: SpanCreatorOutput,
            triplet_results: TripletModelOutput
    ):
        super().__init__(batch=batch)

        self.span_creator_output: SpanCreatorOutput = span_creator_output.release_memory()
        self.triplet_results: TripletModelOutput = triplet_results

    # def __str__(self):
    #     return str(self.result)

    # @classmethod
    # def save_list_of_outputs(cls, results: List[MO], save_path: str) -> None:
    #     res: MO
    #     [res.save(save_path) for res in results]
    #
    # def save(self, save_path: str) -> None:
    #     os.makedirs(save_path[:save_path.rfind(os.sep)], exist_ok=True)
    #     triplets: List = self._triplet_results_for_save()
    #     idx: int
    #     triplet: List
    #     with open(save_path, 'a') as f:
    #         for idx, triplet in enumerate(triplets):
    #             line: str = f'{self.batch.sentence_obj[idx].sentence}{self.batch.sentence_obj[idx].SEP}{str(triplet)}\n'
    #             f.write(line)
    #
    # @property
    # def result(self) -> Dict:
    #     triplets: List = self._triplet_results_for_save()
    #     results: Dict = dict()
    #
    #     idx: int
    #     triplet: List
    #     for idx, triplet in enumerate(triplets):
    #         results[self.batch.sentence_obj[idx].sentence] = triplet
    #
    #     return results
    #
    # @lru_cache(maxsize=None)
    # def _triplet_results_for_save(self) -> List:
    #     triplet_results: List = [[] for _ in range(len(self.batch))]
    #     triplets: Tensor = self._get_triplets_from_matrix()
    #
    #     for triplet in triplets:
    #         sent_idx: int = int(triplet[0].cpu())
    #         spans: Tensor = self.predicted_spans[sent_idx]
    #
    #         aspect_span: List = spans[triplet[1]].cpu().numpy().tolist()
    #         aspect_span = self.get_result_spans(aspect_span, sent_idx)
    #
    #         opinion_span: List = spans[triplet[2]].cpu().numpy().tolist()
    #         opinion_span = self.get_result_spans(opinion_span, sent_idx)
    #
    #         result: Tuple = (aspect_span, opinion_span, ASTELabels(int(triplet[-1].cpu())).name)
    #         triplet_results[sent_idx].append(result)
    #
    #     return triplet_results
    #
    # def get_result_spans(self, curr_span: List, sent_idx: int) -> List:
    #     curr_span = [
    #         self.batch.sentence_obj[sent_idx].get_index_before_encoding(curr_span[0]),
    #         self.batch.sentence_obj[sent_idx].get_index_before_encoding(curr_span[1])
    #     ]
    #     return [curr_span[0]] if curr_span[0] == curr_span[1] else [curr_span[0], curr_span[1]]
