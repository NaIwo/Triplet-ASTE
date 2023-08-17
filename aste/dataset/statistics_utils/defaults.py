from ..domain.const import ASTELabels

from typing import Dict
from functools import partial

from .functions import (
    sentences_count,
    sentence_mean_length,
    specific_span_count,
    triplets_count,
    specific_span_encoder_count,
    specific_span_word_count,
    specific_sentiment_count,
    one_to_many_count,
    specific_span_word_length_count,
    triplet_word_length_count,
    specific_phrase_count,

    bad_pairing_count,
    bad_sentiment_count,
    opposite_sentiment_count,
    correct_triplets_length_count,
    triplets_length_based_metrics,
    metrics_for_sentences_with_multiple_sentiments,
    not_included_words,
    over_included_words
)

default_statistics_func: Dict = {
    'Number of sentences': sentences_count,
    'Mean sentence length': sentence_mean_length,
    'Number of opinion phrases': partial(specific_span_count, span_source='opinion_span'),
    'Number of aspect phrases': partial(specific_span_count, span_source='aspect_span'),
    'Number of triplets': triplets_count,
    'Mean length of opinion phrases (with encoder)': partial(specific_span_encoder_count, span_source='opinion_span'),
    'Mean length of opinion phrases (words)': partial(specific_span_word_count, span_source='opinion_span'),
    'Mean length of aspect phrases (with encoder)': partial(specific_span_encoder_count, span_source='aspect_span'),
    'Mean length of aspect phrases (words)': partial(specific_span_word_count, span_source='aspect_span'),
    'Number of positive sentiment': partial(specific_sentiment_count, sentiment=ASTELabels.POS),
    'Number of neutral sentiment': partial(specific_sentiment_count, sentiment=ASTELabels.NEU),
    'Number of negative sentiment': partial(specific_sentiment_count, sentiment=ASTELabels.NEG),
    'Number of one-to-many relations (opinions)': partial(one_to_many_count, span_source='opinion_span'),
    'Number of one-to-many relations (aspects)': partial(one_to_many_count, span_source='aspect_span'),
    'Number of opinions with length = 1': partial(
        specific_span_word_length_count, span_source='opinion_span',
        func=lambda el: len(el) == 1),
    'Number of aspects with length = 1': partial(
        specific_span_word_length_count, span_source='aspect_span',
        func=lambda el: len(el) == 1),
    'Number of opinions with length > 1': partial(
        specific_span_word_length_count, span_source='opinion_span',
        func=lambda el: len(el) > 1),
    'Number of aspects with length > 1': partial(
        specific_span_word_length_count, span_source='aspect_span',
        func=lambda el: len(el) > 1),
    'Number of triplets where length of each span = 1': partial(
        triplet_word_length_count,
        aspect_func=lambda el: len(el) == 1,
        opinion_func=lambda el: len(el) == 1),
    'Number of triplets where aspect span length > 1 and opinion span length = 1': partial(
        triplet_word_length_count,
        aspect_func=lambda el: len(el) > 1,
        opinion_func=lambda el: len(el) == 1),
    'Number of triplets where opinion span length > 1 and aspect span length = 1': partial(
        triplet_word_length_count,
        aspect_func=lambda el: len(el) == 1,
        opinion_func=lambda el: len(el) > 1),
    'Number of triplets where at least one span length > 1': partial(
        triplet_word_length_count,
        aspect_func=lambda el: len(el) > 1,
        opinion_func=lambda el: len(el) > 1,
        operation='or'),
}

default_results_func: Dict = {
    # if phrases are partially good extracted - prediction intersect with labels ends up with at least one word
    'Number of bad pairing': bad_pairing_count,
    'Bad sentiment': bad_sentiment_count,
    'Bad (opposite) sentiment': opposite_sentiment_count,

    'Metrics with length of aspect span == 1': partial(correct_triplets_length_count, k=1, span_sources='aspect_span', operation='=='),
    'Metrics with length of aspect span == 2': partial(correct_triplets_length_count, k=2, span_sources='aspect_span', operation='=='),
    'Metrics with length of aspect span == 3': partial(correct_triplets_length_count, k=3, span_sources='aspect_span', operation='=='),
    'Metrics with length of aspect span == 4': partial(correct_triplets_length_count, k=4, span_sources='aspect_span', operation='=='),
    'Metrics with length of aspect span == 5': partial(correct_triplets_length_count, k=5, span_sources='aspect_span', operation='=='),
    'Metrics with length of aspect span <= 3': partial(correct_triplets_length_count, k=3, span_sources='aspect_span', operation='<='),
    'Metrics with length of aspect span >= 3': partial(correct_triplets_length_count, k=3, span_sources='aspect_span', operation='>='),
    'Metrics with length of aspect span >= 6': partial(correct_triplets_length_count, k=6, span_sources='aspect_span', operation='>='),
    'Metrics with length of opinion span == 1': partial(correct_triplets_length_count, k=1, span_sources='opinion_span', operation='=='),
    'Metrics with length of opinion span == 2': partial(correct_triplets_length_count, k=2, span_sources='opinion_span', operation='=='),
    'Metrics with length of opinion span == 3': partial(correct_triplets_length_count, k=3, span_sources='opinion_span', operation='=='),
    'Metrics with length of opinion span == 4': partial(correct_triplets_length_count, k=4, span_sources='opinion_span', operation='=='),
    'Metrics with length of opinion span == 5': partial(correct_triplets_length_count, k=5, span_sources='opinion_span', operation='=='),
    'Metrics with length of opinion span <= 3': partial(correct_triplets_length_count, k=3, span_sources='opinion_span', operation='<='),
    'Metrics with length of opinion span >= 3': partial(correct_triplets_length_count, k=3, span_sources='opinion_span', operation='>='),
    'Metrics with length of opinion span >= 6': partial(correct_triplets_length_count, k=6, span_sources='opinion_span', operation='>='),
    'Metrics with length of aspect span & opinion span == 1': partial(correct_triplets_length_count, k=1, span_sources=('aspect_span', 'opinion_span'), operation='=='),
    'Metrics with length of aspect span & opinion span == 2': partial(correct_triplets_length_count, k=2, span_sources=('aspect_span', 'opinion_span'), operation='=='),
    'Metrics with length of aspect span & opinion span == 3': partial(correct_triplets_length_count, k=3, span_sources=('aspect_span', 'opinion_span'), operation='=='),
    'Metrics with length of aspect span & opinion span == 4': partial(correct_triplets_length_count, k=4, span_sources=('aspect_span', 'opinion_span'), operation='=='),
    'Metrics with length of aspect span & opinion span == 5': partial(correct_triplets_length_count, k=5, span_sources=('aspect_span', 'opinion_span'), operation='=='),
    'Metrics with length of aspect span & opinion span <= 3': partial(correct_triplets_length_count, k=3, span_sources=('aspect_span', 'opinion_span'), operation='<='),
    'Metrics with length of aspect span & opinion span >= 3': partial(correct_triplets_length_count, k=3, span_sources=('aspect_span', 'opinion_span'), operation='>='),
    'Metrics with length of aspect span & opinion span >= 6': partial(correct_triplets_length_count, k=6, span_sources=('aspect_span', 'opinion_span'), operation='>='),
    'Metrics for 1 triplet in sentence': partial(triplets_length_based_metrics, k=1, operation='=='),
    'Metrics for 2 triplets in sentence': partial(triplets_length_based_metrics, k=2, operation='=='),
    'Metrics for 3 triplets in sentence': partial(triplets_length_based_metrics, k=3, operation='=='),
    'Metrics for 4 triplets in sentence': partial(triplets_length_based_metrics, k=4, operation='=='),
    'Metrics for 5 triplets in sentence': partial(triplets_length_based_metrics, k=5, operation='=='),
    'Metrics for 3 or less triplets in sentence': partial(triplets_length_based_metrics, k=3, operation='<='),
    'Metrics for 3 or more triplets in sentence': partial(triplets_length_based_metrics, k=3, operation='>='),
    'Metrics for 5 or more triplets in sentence': partial(triplets_length_based_metrics, k=5, operation='>='),
    'Metrics for 1 or more triplets in sentence': partial(triplets_length_based_metrics, k=1, operation='>='),
    'Metrics for sentences with multiple triplets and sentiments': metrics_for_sentences_with_multiple_sentiments
}
default_phrases_func: Dict = {
    'Number of specific phrases': specific_phrase_count,
    'Number of specific phrases in aspect span': partial(specific_phrase_count, span_source='aspect_span'),
    'Number of specific phrases in opinion span': partial(specific_phrase_count, span_source='opinion_span'),
}

default_advanced_result_stats_func: Dict = {
    'Not included words': not_included_words,
    'Over included words': over_included_words,
}
