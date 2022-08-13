# pylint: disable=no-member
import torch
from torch import nn


class RMSELoss(nn.Module):
    """
    예측결과에 대한 RMSE 점수를 계산하기 위한 클래스입니다.

    Arguments
    ----------
    mse : _Loss
        torch의 MSELoss
    eps : float
        mse 값이 0이 되어 sqrt가 에러가 나는 경우 예방
    """

    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        self.eps = 1e-7

    def forward(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
        """
        실제값과 모델의 예측값을 이용하여 RMSE를 계산합니다.

        Parameters
        ----------
        y_true : torch.Tensor
            실제값
        y_pred : torch.Tensor
            모델의 예측값

        Returns
        -------
        float
            RMSE 결과값
        """
        return torch.sqrt(self.mse(y_true, y_pred) + self.eps)
