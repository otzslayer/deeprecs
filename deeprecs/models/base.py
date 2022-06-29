from abc import ABCMeta, abstractmethod

from torch import nn


class BaseRecommender(nn.Module):
    r"""추천 모델을 위한 베이스 클래스입니다.
    모든 모델은 이 클래스의 서브클래스여야 합니다.

    이 베이스 클래스를 상속받는 모든 클래스는 다음 메서드를 포함하고 있습니다.

    `forward`
        : Forward propagation을 하는 메서드
    `fit`
        : 모델을 데이터에 적합시키는 메서드로 `_train_one_epoch`을 반복하게 됩니다.
    `predict`
        : 추천 결과를 생성하는 메서드
    `_train_one_epoch`
        : 1 epoch을 학습하는 메서드
    """
    __metaclass__ = ABCMeta

    @abstractmethod
    def forward(self):
        r"""Forward propagation을 수행합니다."""

    @abstractmethod
    def fit(self):
        r"""모델을 데이터에 적합시키는 메서드로 `_train_one_epoch`을 반복합니다."""

    @abstractmethod
    def predict(self):
        r"""추천 결과를 생성합니다."""

    @abstractmethod
    def _train_one_epoch(self):
        r"""데이터에 대해 1 epoch을 학습합니다."""
