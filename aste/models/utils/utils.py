from typing import List

import torch
from torch import Tensor

from ...dataset.domain import Sentence, Span
from ..utils.const import TripletDimensions


def construct_predicted_spans(span_range: Tensor, sentence: Sentence) -> List[Span]:
    spans: List[Span] = list()
    for s_range in span_range:
        s_range = [
            sentence.get_index_before_encoding(s_range[0].item()),
            sentence.get_index_before_encoding(s_range[1].item())
        ]
        spans.append(Span.from_range(s_range, sentence.sentence))
    return spans


def create_random_tensor_with_one_one_in_each_row(size: int, device: torch.device) -> Tensor:
    tensor = torch.zeros(size).to(device)
    random_idx = torch.randint(0, size, (1,)).to(device)
    tensor[random_idx] = 1
    return tensor


def create_random_tensor_with_one_true_per_row(mask: Tensor, dim: int, device: torch.device) -> Tensor:
    if dim == TripletDimensions.ASPECT:
        mask = mask.permute(0, 2, 1)
    m1 = mask / (mask.sum(dim=TripletDimensions.OPINION, keepdims=True) + 1e-8)
    m2 = torch.where(mask.sum(TripletDimensions.OPINION, keepdims=True) == 0, torch.ones_like(m1) / m1.shape[TripletDimensions.OPINION], m1)
    m3 = torch.distributions.Categorical(m2).sample()[..., None].to(device)
    f = torch.zeros_like(mask).scatter_(TripletDimensions.OPINION, m3, 1).to(device)
    if dim == TripletDimensions.ASPECT:
        f = f.permute(0, 2, 1)
    return f
