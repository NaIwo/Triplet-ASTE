import json
import os
from typing import Dict, List, Union

from torch import Tensor


def ignore_index(func):
    def ignore_and_call(self, *args, **kwargs):
        if self.ignore_index is not None:
            if 'preds' in kwargs.keys():
                preds = kwargs['preds']
                target = kwargs['target']
            else:
                preds = args[0]
                target = args[1]
            indices = (target != self.ignore_index).nonzero(as_tuple=False).flatten()
            preds_ignored = preds[indices]
            target_ignored = target[indices]
            if 'mask' in kwargs.keys():
                kwargs['mask'] = kwargs['mask'][indices]
            res = func(self, preds_ignored, target_ignored)
        else:
            res = func(self, *args, **kwargs)
        return res

    return ignore_and_call


def to_json(data_to_save: Union[Dict, List], path: str, mode: str = 'a') -> None:
    os.makedirs(path[:path.rfind(os.sep)], exist_ok=True)
    with open(path, mode=mode) as f:
        json.dump(data_to_save, f)
