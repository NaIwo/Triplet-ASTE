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

        self.span_creator_output: SpanCreatorOutput = span_creator_output.release_memory()
        self.triplet_results: TripletModelOutput = triplet_results.release_memory()

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
