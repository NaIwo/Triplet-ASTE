import torch
from torch import Tensor
from aste.utils import ignore_index
from typing import Optional


def one_hot(labels: Tensor, num_classes: int, device: torch.device, dtype: torch.dtype, eps: float = 1e-6) -> Tensor:
    if not isinstance(labels, Tensor):
        raise TypeError(f"Input labels type is not a Tensor. Got {type(labels)}")

    if not labels.dtype == torch.int64:
        raise ValueError(f"labels must be of the same dtype torch.int64. Got: {labels.dtype}")

    if num_classes < 1:
        raise ValueError("The number of classes must be bigger than one." " Got: {}".format(num_classes))

    shape = labels.shape
    one_hot = torch.zeros((shape[0], num_classes) + shape[1:], device=device, dtype=dtype)

    return one_hot.scatter_(1, labels.unsqueeze(1), 1.0) + eps


def focal_loss(
        input: Tensor, target: Tensor, alpha: float, gamma: float = 2.0, reduction: str = 'none',
        eps: float | None = None
) -> Tensor:
    n = input.shape[0]
    out_size = (n,) + input.shape[2:]

    # compute softmax over the classes axis
    input_soft: Tensor = input.softmax(1)
    log_input_soft: Tensor = input.log_softmax(1)

    # create the labels one hot tensor
    target_one_hot: Tensor = one_hot(target, num_classes=input.shape[1], device=input.device, dtype=input.dtype)

    # compute the actual focal loss
    weight = torch.pow(-input_soft + 1.0, gamma)

    focal = -alpha * weight * log_input_soft
    loss_tmp = torch.einsum('bc...,bc...->b...', (target_one_hot, focal))

    if reduction == 'none':
        loss = loss_tmp
    elif reduction == 'mean':
        loss = torch.mean(loss_tmp)
    elif reduction == 'sum':
        loss = torch.sum(loss_tmp)
    else:
        raise NotImplementedError(f"Invalid reduction mode: {reduction}")
    return loss


class FocalLoss(torch.nn.Module):

    def __init__(self, alpha: float, gamma: float = 2.0, reduction: str = 'mean', eps: float | None = None, ignore_index: Optional[int] = None) -> None:
        super().__init__()
        self.alpha: float = alpha
        self.gamma: float = gamma
        self.reduction: str = reduction
        self.eps: float | None = eps
        self.ignore_index = ignore_index

    @ignore_index
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return focal_loss(input, target, self.alpha, self.gamma, self.reduction, self.eps)
