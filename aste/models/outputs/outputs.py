from typing import List, TypeVar, Optional, Dict

import torch
from torch import Tensor

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
    def __init__(self, span_range: Tensor, labels: Tensor, sentiments: Tensor):
        super().__init__()
        self.span_range: Tensor = span_range
        self.labels: Tensor = labels
        self.sentiments: Tensor = sentiments


class SpanPredictionsOutput(BaseModelOutput):
    def __init__(self, aspects: List[SpanInformationOutput], opinions: List[SpanInformationOutput]):
        super().__init__()
        self.aspects: List[SpanInformationOutput] = aspects
        self.opinions: List[SpanInformationOutput] = opinions

    def get_aspect_span_predictions(self) -> List[Tensor]:
        return self._get(self.aspects, 'span_range')

    def get_opinion_span_predictions(self) -> List[Tensor]:
        return self._get(self.opinions, 'span_range')

    def get_aspect_span_labels(self) -> List[Tensor]:
        return self._get(self.aspects, 'labels')

    def get_opinion_span_labels(self) -> List[Tensor]:
        return self._get(self.opinions, 'labels')

    def get_aspect_span_sentiments(self) -> List[Tensor]:
        return self._get(self.aspects, 'sentiments')

    def get_opinion_span_sentiments(self) -> List[Tensor]:
        return self._get(self.opinions, 'sentiments')

    @staticmethod
    def _get(source: List[SpanInformationOutput], attr: str) -> List[Tensor]:
        return [getattr(span, attr) for span in source]


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
        opinions_agg_emb = torch.tensor(data.sentiment_features.values()).to(data.sentiment_features)
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
        key: int
        opinion: SpanInformationOutput
        for opinion, key in zip(opinions, keys):
            op = [opinion] * len(keys)
            results.append(op)
        return results


class ModelOutput(BaseModelOutput):

    def __init__(
            self,
            batch: Batch,
            span_creator_output: SpanCreatorOutput,
            span_selector_output: Tensor,
            triplet_results: Tensor
    ):
        super().__init__(batch=batch)

        self.span_creator_output: SpanCreatorOutput = span_creator_output.release_memory()
        self.span_selector_output: Tensor = span_selector_output
        self.triplet_results: Tensor = triplet_results

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
