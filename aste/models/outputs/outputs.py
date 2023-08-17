from typing import TypeVar, Optional, Dict, List

import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence

from ..utils.const import CreatedSpanCodes
from ..utils.utils import construct_predicted_spans
from ...dataset.domain import Sentence, Span, ASTELabels, Triplet
from ...dataset.reader import Batch

MO = TypeVar('MO', bound='BaseModelOutput')
SIO = TypeVar('SIO', bound='SpanInformationOutput')
SIM = TypeVar('SIM', bound='SpanInformationManager')


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

    def get_number_of_predicted_elements(self, with_repeated: bool = True) -> int:
        condition = self.span_creation_info == CreatedSpanCodes.PREDICTED_TRUE
        condition |= self.span_creation_info == CreatedSpanCodes.PREDICTED_FALSE
        num = torch.sum(condition).item()
        if (not with_repeated) and (self.repeated is not None):
            num = num // self.repeated
        return num

    def get_number_of_predicted_true_elements(self) -> int:
        condition = self.span_creation_info == CreatedSpanCodes.PREDICTED_TRUE
        return torch.sum(condition).item()

    @classmethod
    def from_span_manager(cls, span_manager: SIM, sentence: Sentence) -> SIO:
        return SpanInformationOutput(
            span_range=torch.tensor(span_manager.span_ranges),
            span_creation_info=torch.tensor(span_manager.span_creation_info),
            sentiments=torch.tensor(span_manager.sentiments),
            mapping_indexes=torch.tensor(span_manager.mapping_indexes),
            sentence=sentence
        )

    def to_device(self, device: torch.device) -> SIO:
        self.span_range = self.span_range.to(device)
        self.span_creation_info = self.span_creation_info.to(device)
        self.sentiments = self.sentiments.to(device)
        self.mapping_indexes = self.mapping_indexes.to(device)

        return self

    def repeat(self, n: int, sentiment_collapse_keys: Optional[List[int]] = None) -> SIO:
        return SpanInformationOutput(
            span_range=self.span_range.repeat(n, 1).squeeze(dim=0),
            span_creation_info=self.span_creation_info.repeat(n),
            sentiments=self._get_repeated_sentiments(n, sentiment_collapse_keys),
            mapping_indexes=self._get_repeated_mapping_indexes(n),
            sentence=self.sentence,
            repeated=n
        )

    def _get_repeated_sentiments(self, n: int, sentiment_collapse_keys: Optional[List[int]] = None) -> Tensor:
        if sentiment_collapse_keys is None:
            sentiments = self.sentiments.repeat(n)
        else:
            num_elements = self.span_creation_info.shape[0]
            sentiments = torch.full_like(self.span_creation_info, ASTELabels.NOT_PAIR).repeat(n)
            for p_idx, p in enumerate(sentiment_collapse_keys):
                indexes = range(p_idx * num_elements, num_elements * (p_idx + 1))
                temp = self.sentiments == p
                temp = self.mapping_indexes[temp]
                sent = sentiments[indexes]
                sent[temp] = p
                sentiments[indexes] = sent
        return sentiments

    def _get_repeated_mapping_indexes(self, n: int) -> Tensor:
        repeated_indexes = self.mapping_indexes.repeat(n)

        mask = (self.mapping_indexes >= 0).repeat(n)
        indexes = torch.full_like(self.mapping_indexes, self.span_range.shape[0]).repeat(n).to(self.sentiments)
        skip = torch.arange(n).unsqueeze(-1).repeat(1, self.mapping_indexes.shape[0]).flatten().to(self.sentiments)
        repeated_indexes += (indexes * skip) * mask

        return repeated_indexes


class SpanCreatorOutput(BaseModelOutput):
    def __init__(self,
                 batch: Batch, features: Tensor,
                 aspects: List[SpanInformationOutput],
                 opinions: List[SpanInformationOutput],
                 aspects_agg_emb: Optional[Tensor] = None,
                 opinions_agg_emb: Optional[Tensor] = None,
                 sentence_emb: Optional[Tensor] = None
                 ):
        super().__init__(batch=batch, features=features)
        self.aspects: List[SpanInformationOutput] = aspects
        self.opinions: List[SpanInformationOutput] = opinions
        self.aspects_agg_emb: Optional[Tensor] = aspects_agg_emb
        self.opinions_agg_emb: Optional[Tensor] = opinions_agg_emb
        self.sentence_emb: Optional[Tensor] = sentence_emb

    def __iter__(self):
        self.num: int = -1
        return self

    def __next__(self):
        self.num += 1
        if self.num >= len(self.aspects):
            raise StopIteration
        return SpanCreatorOutput(
            batch=self.batch[self.num] if self.batch is not None else None,
            features=self.features[self.num] if self.features is not None else None,
            aspects_agg_emb=self.aspects_agg_emb[self.num] if self.aspects_agg_emb is not None else None,
            opinions_agg_emb=self.opinions_agg_emb[self.num] if self.opinions_agg_emb is not None else None,
            sentence_emb=self.sentence_emb[self.num] if self.sentence_emb is not None else None,
            aspects=[self.aspects[self.num]],
            opinions=[self.opinions[self.num]]
        )

    def copy(self):
        return SpanCreatorOutput(
            batch=self.batch,
            features=self.features.clone() if self.features is not None else None,
            aspects=self.aspects[:],
            opinions=self.opinions[:],
            aspects_agg_emb=self.aspects_agg_emb.clone() if self.aspects_agg_emb is not None else None,
            opinions_agg_emb=self.opinions_agg_emb.clone() if self.aspects_agg_emb is not None else None,
            sentence_emb=self.sentence_emb.clone() if self.sentence_emb is not None else None,
        )

    def get_number_of_predicted_spans(self, with_repeated: bool = True) -> int:
        num = sum([s.get_number_of_predicted_elements(with_repeated) for s in self.aspects + self.opinions])
        return num

    def get_aspect_span_predictions(self, with_repeated: bool = True) -> List[Tensor]:
        spans = self._get(self.aspects, 'span_range')
        if (self.aspects[0].repeated is not None) and (not with_repeated):
            r = self.aspects[0].repeated
            spans = [s[:s.shape[0] // r] for s in spans]
        return spans

    def get_opinion_span_predictions(self, with_repeated: bool = True) -> List[Tensor]:
        spans = self._get(self.opinions, 'span_range')
        if (self.opinions[0].repeated is not None) and (not with_repeated):
            r = self.opinions[0].repeated
            spans = [s[:s.shape[0] // r] for s in spans]
        return spans

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

    def all_predicted_spans(self, with_repeated: bool = True) -> List[Tensor]:
        return [
            torch.cat([a, o], dim=0) for a, o in zip(
                self.get_aspect_span_predictions(with_repeated),
                self.get_opinion_span_predictions(with_repeated)
            )
        ]

    def extend_opinions_with_sentiments(self, data: SentimentModelOutput):
        keys = sorted(data.sentiment_features.keys())
        opinions = self.opinions
        opinions_agg_emb = self._get_extended_opinion_emb(data, keys)
        self._extend_opinion_information(opinions, keys)
        return SpanCreatorOutput(
            batch=self.batch,
            features=self.features,
            aspects_agg_emb=self.aspects_agg_emb,
            opinions_agg_emb=opinions_agg_emb,
            sentence_emb=self.sentence_emb,
            opinions=opinions,
            aspects=self.aspects,
        )

    @staticmethod
    def _get_extended_opinion_emb(data: SentimentModelOutput, keys: List) -> Tensor:
        results: List = []
        for key in keys:
            results.append(data.sentiment_features[key])
        return torch.cat(results, dim=1)

    @staticmethod
    def _extend_opinion_information(opinions: List[SpanInformationOutput], keys: List) -> None:
        opinion: SpanInformationOutput
        for opinion_idx, opinion in enumerate(opinions):
            num_elements: int = opinion.span_creation_info.shape[0]
            repeated_opinion: SpanInformationOutput = opinion.repeat(len(keys), sentiment_collapse_keys=keys)

            for key_idx, key in enumerate(keys):
                indexes = range(key_idx * num_elements, num_elements * (key_idx + 1))
                condition = (repeated_opinion.sentiments[indexes] != key)
                SpanCreatorOutput._update_repeated_span_creation_info(condition, indexes, repeated_opinion)
                repeated_opinion.sentiments[indexes] = key

            opinions[opinion_idx] = repeated_opinion

    @staticmethod
    def _update_repeated_span_creation_info(condition: Tensor, indexes: range,
                                            repeated_opinion: SpanInformationOutput) -> None:
        c_added = repeated_opinion.span_creation_info[indexes] == CreatedSpanCodes.ADDED_TRUE
        c_added |= repeated_opinion.span_creation_info[indexes] == CreatedSpanCodes.ADDED_FALSE
        repeated_opinion.span_creation_info[indexes] = torch.where(
            condition & c_added,
            CreatedSpanCodes.ADDED_FALSE,
            repeated_opinion.span_creation_info[indexes]
        )
        c_pred = repeated_opinion.span_creation_info[indexes] == CreatedSpanCodes.PREDICTED_TRUE
        c_pred |= repeated_opinion.span_creation_info[indexes] == CreatedSpanCodes.PREDICTED_FALSE
        repeated_opinion.span_creation_info[indexes] = torch.where(
            condition & c_pred,
            CreatedSpanCodes.PREDICTED_FALSE,
            repeated_opinion.span_creation_info[indexes]
        )


class SampleTripletOutput(BaseModelOutput):
    def __init__(
            self,
            aspect_ranges: Tensor,
            opinion_ranges: Tensor,
            sentence: Sentence,
            pred_sentiments: Optional[Tensor] = None,
            true_sentiments: Optional[Tensor] = None,
            aspect_emb: Optional[Tensor] = None,
            opinion_emb: Optional[Tensor] = None,
            sentence_emb: Optional[Tensor] = None,
            similarities: Optional[Tensor] = None,
            span_creation_info: Optional[Tensor] = None,
            features: Optional[Tensor] = None,
            batch: Optional[Dict[str, Tensor]] = None,
    ):
        super().__init__(batch, features)
        self.sentence: Sentence = sentence
        self.triplets: List[Triplet] = list()

        self.sentence_emb: Optional[Tensor] = sentence_emb
        self.aspect_emb: Optional[Tensor] = aspect_emb
        self.opinion_emb: Optional[Tensor] = opinion_emb
        self.aspect_ranges = aspect_ranges
        self.opinion_ranges = opinion_ranges

        if pred_sentiments is None:
            pred_sentiments = torch.full(
                aspect_ranges.shape[0:1],
                ASTELabels.NOT_RELEVANT.value
            ).to(aspect_ranges.device)
        if true_sentiments is None:
            true_sentiments = torch.full(
                aspect_ranges.shape[0:1],
                ASTELabels.NOT_RELEVANT.value
            ).to(aspect_ranges.device)

        self.pred_sentiments = pred_sentiments
        self.true_sentiments = true_sentiments

        self.similarities = similarities
        self.span_creation_info = span_creation_info

        self.construct_triplets()

    def construct_triplets(self, remove_intersected: bool = False):
        highest_similarity = {}
        self.triplets: List[Triplet] = list()
        triplet_idx = 0
        zip_ = zip(self.aspect_ranges, self.opinion_ranges, self.pred_sentiments)
        for idx, (aspect_range, opinion_range, s) in enumerate(zip_):
            if s == ASTELabels.NOT_RELEVANT.value or s == ASTELabels.NOT_PAIR.value:
                continue

            key = (aspect_range[0].item(), aspect_range[1].item(), opinion_range[0].item(), opinion_range[1].item())
            if (key in highest_similarity.keys()) and (self.similarities[idx] <= highest_similarity[key]['score']):
                continue

            triplet: Triplet = Triplet(
                aspect_span=construct_predicted_spans(aspect_range.unsqueeze(0), self.sentence)[0],
                opinion_span=construct_predicted_spans(opinion_range.unsqueeze(0), self.sentence)[0],
                sentiment=ASTELabels(int(s)).name
            )

            if key in highest_similarity.keys():
                self.triplets[highest_similarity[key]['idx']] = triplet
            else:
                self.triplets.append(triplet)
                highest_similarity[key] = {'score': self.similarities[idx], 'idx': triplet_idx}
                triplet_idx += 1

        if remove_intersected:
            self.triplets = self.filter_triplets(self.triplets, highest_similarity=highest_similarity)

    @staticmethod
    def filter_triplets(triplets: List[Triplet], highest_similarity: Dict) -> List[Triplet]:
        filtered_triplets = []
        sorted_indices = sorted(highest_similarity, key=lambda key: highest_similarity[key]['score'], reverse=True)

        for key in sorted_indices:
            current_triplet = triplets[highest_similarity[key]['idx']]
            is_intersected = False

            for filtered_triplet in filtered_triplets:
                t_intersect = current_triplet.intersect(filtered_triplet)

                if t_intersect.aspect_span.start_idx != -1 and t_intersect.opinion_span.start_idx != -1:
                    is_intersected = True
                    break

            if not is_intersected:
                filtered_triplets.append(current_triplet)

        return filtered_triplets

    def __repr__(self):
        return ' || '.join([str(t) for t in self.triplets])

    def copy(self):
        return SampleTripletOutput(
            aspect_ranges=self.aspect_ranges.clone(),
            opinion_ranges=self.opinion_ranges.clone(),
            sentence=self.sentence,
            pred_sentiments=self.pred_sentiments.clone() if self.pred_sentiments is not None else None,
            true_sentiments=self.true_sentiments.clone() if self.true_sentiments is not None else None,
            sentence_emb=self.sentence_emb.clone() if self.sentence_emb is not None else None,
            aspect_emb=self.aspect_emb.clone() if self.aspect_emb is not None else None,
            opinion_emb=self.opinion_emb.clone() if self.opinion_emb is not None else None,
            similarities=self.similarities.clone() if self.similarities is not None else None,
            span_creation_info=self.span_creation_info.clone() if self.span_creation_info is not None else None,
            features=self.features.clone() if self.features is not None else None,
            batch=self.batch
        )


class TripletModelOutput(BaseModelOutput):
    def __init__(
            self,
            batch: Batch,
            triplets: List[SampleTripletOutput],
            aspect_features: Tensor,
            opinion_features: Tensor,
            similarities: Tensor,
            normalized_similarities: Tensor,
            true_predicted_mask: Tensor,
            prediction_mask: Tensor,
            loss_mask: Tensor,
            pad_mask: Tensor,
    ):
        super().__init__(batch=batch)
        self.triplets: List[SampleTripletOutput] = triplets
        self.aspect_features: Tensor = aspect_features
        self.opinion_features: Tensor = opinion_features
        self.similarities: Tensor = similarities
        self.normalized_similarities: Tensor = normalized_similarities

        self.true_predicted_mask: Tensor = true_predicted_mask
        self.prediction_mask: Tensor = prediction_mask
        self.loss_mask: Tensor = loss_mask
        self.pad_mask: Tensor = pad_mask

    def number_of_triplets(self) -> int:
        return sum(len(sample.triplets) for sample in self.triplets)

    def number_of_significant_similarities(self) -> int:
        return sum([s.similarities.shape[0] for s in self.triplets])

    def get_predicted_sentiments(self) -> Tensor:
        return torch.cat([sample.pred_sentiments for sample in self.triplets])

    def get_span_creation_info(self) -> Tensor:
        return torch.cat([sample.span_creation_info for sample in self.triplets])

    def get_true_sentiments(self) -> Tensor:
        return torch.cat([sample.true_sentiments for sample in self.triplets])

    def get_features(self) -> Tensor:
        return torch.cat([sample.features for sample in self.triplets])

    def get_triplets(self) -> List[List[Triplet]]:
        return [sample.triplets for sample in self.triplets]

    def copy(self):
        return TripletModelOutput(
            batch=self.batch,
            triplets=self.copy_triplets(),
            aspect_features=self.aspect_features.clone(),
            opinion_features=self.opinion_features.clone(),
            similarities=self.similarities.clone(),
            normalized_similarities=self.normalized_similarities.clone(),
            true_predicted_mask=self.true_predicted_mask.clone(),
            prediction_mask=self.prediction_mask.clone(),
            loss_mask=self.loss_mask.clone(),
            pad_mask=self.pad_mask.clone(),
        )

    def copy_triplets(self):
        return [sample.copy() for sample in self.triplets]


class ClassificationModelOutput(BaseModelOutput):
    def __init__(
            self,
            batch: Batch,
            aspect_predictions: Tensor,
            opinion_predictions: Tensor,
            aspect_labels: Tensor,
            opinion_labels: Tensor,
    ):
        super().__init__(batch=batch)
        self.aspect_predictions: Tensor = aspect_predictions
        self.opinion_predictions: Tensor = opinion_predictions
        self.aspect_labels: Tensor = aspect_labels
        self.opinion_labels: Tensor = opinion_labels


class ModelOutput(BaseModelOutput):

    def __init__(
            self,
            batch: Batch,
            **kwargs
    ):
        super().__init__(batch=batch)
        self.outputs = kwargs

    def __getattr__(self, item):
        return self.outputs[item]


class FinalTriplets(BaseModelOutput):
    def __init__(
            self,
            batch: Batch,
            pred_triplets: List[List[Triplet]]
    ):
        super().__init__(batch=batch)

        self.pred_triplets: List[List[Triplet]] = pred_triplets

        self.true_triplets = []
        for sample in batch:
            self.true_triplets.append(sample.sentence_obj[0].triplets)

    def __repr__(self):
        return ' || '.join([str(t) for t in self.pred_triplets])

    def to_string(self) -> str:
        output = []

        def indices_string(start_idx, end_idx):
            return ', '.join(map(str, list(range(start_idx, end_idx + 1))))

        for batch_idx, batch_triplets in enumerate(self.pred_triplets):
            sentence = self.batch.sentence_obj[batch_idx].sentence

            triplets_list = []
            for triplet in batch_triplets:
                aspect_indices = indices_string(triplet.aspect_span.start_idx, triplet.aspect_span.end_idx)
                opinion_indices = indices_string(triplet.opinion_span.start_idx, triplet.opinion_span.end_idx)
                sentiment = triplet.sentiment
                triplets_list.append(f"([{aspect_indices}], [{opinion_indices}], '{sentiment}')")

            triplets_str = ', '.join(triplets_list)
            output.append(f"{sentence}{self.batch.sentence_obj[0].SEP}[{triplets_str}]")
        return '\n'.join(output)
