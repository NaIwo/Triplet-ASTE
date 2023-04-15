import torch
from torch import Tensor

from ...utils.const import CreatedSpanCodes


def get_labels_for_task(labels: Tensor) -> Tensor:
    labels = labels.clone()
    con = (labels == CreatedSpanCodes.ADDED_TRUE) | (labels == CreatedSpanCodes.PREDICTED_TRUE)
    labels = torch.where(con, 1., labels).long()
    con = (labels == CreatedSpanCodes.ADDED_FALSE) | (labels == CreatedSpanCodes.PREDICTED_FALSE)
    labels = torch.where(con, 0., labels).long()
    return labels
