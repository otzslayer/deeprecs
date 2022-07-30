from typing import Tuple

import torch
from torch.utils.data import Dataset


class SampleDataset(Dataset):
    """
    추천 모델에서 가장 기본적으로 사용할 데이터셋

    Parameters
    ----------
    X : torch.Tensor
        모델의 입력부분
    y : torch.Tensor
        모델의 출력부분

    Arguments
    ---------
    X : torch.Tensor
        모델의 입력부분
    y : torch.Tensor
        모델의 출력부분
    """

    def __init__(self, X: torch.Tensor, y: torch.Tensor):
        self.X = X
        self.y = y

    def __getitem__(self, index: int) -> Tuple(torch.Tensor, torch.Tensor):
        return self.X[index], self.y[index]

    def __len__(self) -> int:
        return len(self.X)
