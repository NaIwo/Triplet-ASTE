from ...dataset.reader import Batch
from ...models.specialty_models import SpanCreatorOutput, TripletModelOutput
from ...models.outputs import BaseModelOutput


class ModelOutput(BaseModelOutput):

    def __init__(
            self,
            batch: Batch,
            span_creator_output: SpanCreatorOutput,
            triplet_results: TripletModelOutput
    ):
        super().__init__(batch=batch)

        self.span_creator_output: SpanCreatorOutput = span_creator_output
        self.triplet_results: TripletModelOutput = triplet_results
