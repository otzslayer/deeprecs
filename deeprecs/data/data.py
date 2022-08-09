from typing import Iterable, Optional, Tuple

import torch
from torch.utils.data import Dataset


class AEDataset(Dataset):
    """
    추천 모델에서 가장 기본적으로 사용할 데이터셋

    Parameters
    ----------
    X : Optional[torch.Tensor]
        모델의 입출력부분
    array : Optional[Iterable[Tuple[float, float]]]
        [(x1, y1), (x2, y2), ...] 형태의 iterable object

    Arguments
    ---------
    X : torch.Tensor
        모델의 입력부분
    y : torch.Tensor
        모델의 출력부분
    """

    def __init__(
        self,
        X: Optional[torch.Tensor] = None,
        array: Optional[Iterable[Tuple[float, float]]] = None,
    ):
        if X is not None:
            self.X, self.y = X, X
        elif array:
            self.X, self.y = zip(*array)
        else:
            raise ValueError("AEEncoder must got X or array.")

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[index], self.y[index]

    def __len__(self) -> int:
        return len(self.X)
