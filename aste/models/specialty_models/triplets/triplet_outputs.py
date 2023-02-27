from typing import List

from torch import Tensor

from ....dataset.domain import Sentence, Triplet, ASTELabels
from ....dataset.reader import Batch
from ...outputs import BaseModelOutput
from ...outputs.utils import construct_predicted_spans


class SampleTripletOutput(BaseModelOutput):
    def __init__(
            self,
            aspect_ranges: Tensor,
            opinion_ranges: Tensor,
            sentiments: Tensor,
            sentence: Sentence,
    ):
        super().__init__()
        self.sentence: Sentence = sentence
        self.triplets: List[Triplet] = list()

        for aspect_range, opinion_range, sentiment in zip(aspect_ranges, opinion_ranges, sentiments):
            triplet: Triplet = Triplet(
                aspect_span=construct_predicted_spans(aspect_range.unsqueeze(0), sentence)[0],
                opinion_span=construct_predicted_spans(opinion_range.unsqueeze(0), sentence)[0],
                sentiment=ASTELabels(int(sentiment)).name
            )
            self.triplets.append(triplet)


class TripletModelOutput(BaseModelOutput):
    def __init__(
            self,
            batch: Batch,
            triplets: List[SampleTripletOutput],
            similarities: Tensor,
            true_predicted_mask: Tensor,
            loss_mask: Tensor,
    ):
        super().__init__(batch=batch, features=similarities)
        self.triplets: List[SampleTripletOutput] = triplets

        self.true_predicted_mask: Tensor = true_predicted_mask
        self.loss_mask: Tensor = loss_mask

    def number_of_triplets(self) -> int:
        return sum(len(sample.triplets) for sample in self.triplets)
