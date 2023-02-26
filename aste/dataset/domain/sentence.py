from ast import literal_eval
from typing import List, Tuple

from .triplet import Triplet
from ..encoders import BaseEncoder, TransformerEncoder


class Sentence:
    SEP: str = '#### #### ####'

    def __init__(self, raw_sentence: str,
                 encoder: BaseEncoder = TransformerEncoder(),
                 include_sub_words_info_in_mask: bool = True):
        self.encoder: BaseEncoder = encoder
        self.raw_line: str = raw_sentence
        splitted_sentence: List = raw_sentence.strip().split(Sentence.SEP)
        self.sentence: str = splitted_sentence[0]
        self.triplets: List[Triplet] = []
        # If data with labels
        if len(splitted_sentence) == 2:
            triplets_info: List[Tuple] = literal_eval(splitted_sentence[1])
            self.triplets = [Triplet.from_triplet_info(triplet_info, self.sentence) for triplet_info in triplets_info]

        self.encoded_sentence: List[int] = self.encoder.encode(sentence=self.sentence)
        self.encoded_words_in_sentence: List = self.encoder.encode_word_by_word(sentence=self.sentence)

        self.include_sub_words_info_in_mask: bool = include_sub_words_info_in_mask

        self._sub_words_lengths: List[int] = list()
        self._true_sub_words_lengths: List[int] = list()
        self._sub_words_mask: List[int] = list()
        self._true_sub_words_mask: List[int] = list()

        self._fill_sub_words_information()

        self.sentence_length: int = len(self.sentence.split())
        self.encoded_sentence_length: int = len(self.encoded_sentence)
        self.emb_sentence_length: int = len(self._sub_words_mask)

    def _fill_sub_words_information(self):
        word: List[int]
        for word in self.encoded_words_in_sentence:
            len_sub_word: int = len(word) - 1

            self._sub_words_lengths.append(len_sub_word * int(self.include_sub_words_info_in_mask))
            self._true_sub_words_lengths.append(len_sub_word)

            self._sub_words_mask += [1] + ([0] * (len_sub_word * int(self.include_sub_words_info_in_mask)))
            self._true_sub_words_mask += [1] + [0] * len_sub_word

        offset: List[int] = [0] * self.encoder.offset
        self._sub_words_mask = offset + self._sub_words_mask + offset
        self._true_sub_words_mask = offset + self._true_sub_words_mask + offset

    def get_sub_words_mask(self, force_true_mask: bool = False):
        return self._true_sub_words_mask if force_true_mask else self._sub_words_mask

    def get_sentiments(self) -> List[int]:
        return [triplet.sentiment_code for triplet in self.triplets]

    def get_aspect_spans(self) -> List[Tuple[int, int]]:
        return self._get_selected_spans('aspect_span')

    def get_opinion_spans(self) -> List[Tuple[int, int]]:
        return self._get_selected_spans('opinion_span')

    def _get_selected_spans(self, span_source: str) -> List[Tuple[int, int]]:
        assert span_source in ('aspect_span', 'opinion_span'), f'Invalid span source: {span_source}!'
        spans: List = list()
        triplet: Triplet
        for triplet in self.triplets:
            # +1)-1 -> If end index is the same as start_idx and word is constructed from sub-tokens
            # end index is shifted by number equals to this sub-words count.
            span: Tuple = (self.get_index_after_encoding(getattr(triplet, span_source).start_idx),
                           self.get_index_after_encoding(getattr(triplet, span_source).end_idx + 1) - 1)
            spans.append(span)
        return spans

    def get_index_after_encoding(self, idx: int) -> int:
        if idx < 0 or idx >= self.sentence_length:
            return -1
        return self.encoder.offset + idx + sum(self._sub_words_lengths[:idx])

    def get_index_before_encoding(self, idx: int) -> int:
        if idx < 0 or idx >= self.emb_sentence_length:
            return -1
        return sum(self._sub_words_mask[:idx])

    def agree_index(self, idx: int) -> int:
        return self.get_index_after_encoding(self.get_index_before_encoding(idx))

    def __eq__(self, other) -> bool:
        if not isinstance(other, Sentence):
            raise NotImplemented

        return (self.triplets == other.triplets) and (self.sentence == other.sentence) and (
                self.encoded_sentence == other.encoded_sentence)

    def __hash__(self):
        return hash(self.raw_line)
