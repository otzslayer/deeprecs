from typing import Final, Union

import torch
from torch import nn
from torch.nn.modules.loss import _Loss
from torch.optim import Adam
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader, Dataset

from deeprecs.models.base import BaseRecommender
from deeprecs.utils.helper import get_classes_from_module

ACTIVATION: Final = Union[get_classes_from_module(nn.modules.activation)]


class AutoEncoder(BaseRecommender):
    """
    1-depth layer인 Auto-Encoder를 활용한 추천모델입니다.

    Parameters
    ----------
    `input_dim` : int
        AutoEncoder의 input/output layer의 dimension
    `hidden_dim` : int
        AutoEncoder의 hidden layer의 dimension
    `final_activation` : ACTIVATION, optional
        AutoEncoder의 output layer의 활성화 함수, by default nn.ReLU()

    Arguments
    ---------
    `_input_dim` : int
        AutoEncoder의 input/output layer의 dimension
    `_hidden_dim` : int
        AutoEncoder의 hidden layer의 dimension
    `_encoder` : nn.Linear
        AutoEncoder의 encoder 부분
    `_decoder` : nn.Linear
        AutoEncoder의 decoder 부분
    `_final_activation` : ACTIVATION
        AutoEncoder의 output layer의 활성화 함수

    References
    ----------
    [1] https://users.cecs.anu.edu.au/~akmenon/papers/autorec/autorec-paper.pdf
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        final_activation: ACTIVATION = nn.ReLU(),
        optimizer: Union[str, Optimizer] = "adam",
        loss: _Loss = nn.MSELoss(),
    ):
        super().__init__()
        self._input_dim = input_dim
        self._hidden_dim = hidden_dim
        self._encoder = nn.Linear(input_dim, hidden_dim)
        self._decoder = nn.Linear(hidden_dim, input_dim)
        self._final_activation = final_activation
        if isinstance(optimizer, str):
            self._set_optimizer(optimizer)
        elif isinstance(optimizer, Optimizer):
            self._optimizer = optimizer
        else:
            raise ValueError
        self._loss = loss

    def _set_optimizer(self, optimizer: str):
        """
        optimizer 이름을 실제 optimizer로 변경

        Parameters
        ----------
        optimizer : str
            optimizer 이름

        Raises
        ------
        ValueError
            미리 정해지지 않은 optimizer 이름을 사용할 경우 에러
        """
        optimizer = optimizer.lower()
        if optimizer == "adam":
            self._optimizer = Adam(self.parameters(), lr=0.001)
        else:
            raise ValueError

    def forward(self, input_data: torch.Tensor) -> torch.Tensor:
        """
        Forward propagation을 수행합니다.

        Parameters
        ----------
        input_data : torch.Tensor
            (n_samples, n_input_dim)의 형태.
            Forward propagation을 수행할 input samples.

        Returns
        -------
        torch.Tensor
            (n_samples, n_input_dim)의 형태.
            Forward propagation 수행이 끝난 후의 결과
        """
        encoded = self._encoder(input_data)
        decoded = self._decoder(encoded)
        output = self._final_activation(decoded)

        return output

    def fit(self, trainset: Dataset, epochs: int, batch_size: int):
        """
        모델을 데이터에 적합시키는 메서드로 `_train_one_epoch`을 반복합니다.

        Parameters
        ----------
        trainset : Dataset
            학습에 사용할 데이터셋
        epochs : int
            데이터셋을 반복할 횟수
        batch_size : int
            한 번 학습할 때 사용할 데이터의 크기
        """
        self.train()
        for _ in range(epochs):
            train_loader = DataLoader(trainset, batch_size=batch_size)
            self._train_one_epoch(train_loader)

    def predict(self, dataset: Dataset) -> torch.Tensor:
        """
        추천 결과를 생성합니다.

        Parameters
        ----------
        dataset : Dataset
            추천에 사용할 데이터셋

        Returns
        -------
        torch.Tensor
            추천 결과
        """
        self.eval()
        pred = self(dataset.X)

        return pred

    def _train_one_epoch(self, train_loader: DataLoader):
        """
        데이터에 대해 1 epoch을 학습합니다.

        Parameters
        ----------
        train_loader : DataLoader
            에폭 안에서 사용할 데이터로더
        """
        for X, y_true in train_loader:
            self._optimizer.zero_grad()
            y_pred = self(X)
            loss = self._loss(y_pred, y_true)
            loss.backward()
            self._optimizer.step()

    @property
    def input_dim(self) -> int:
        """
        AutoRec의 input dimension 값 출력

        Returns
        -------
        int
           AutoRec의 input dimension 값
        """
        return self._input_dim

    @input_dim.setter
    def input_dim(self, dim: int):
        """
        AutoRec의 input demension 값 변경

        Parameters
        ----------
        dim : _type_
            새로운 input dimension
        """
        self._input_dim = dim
        self._encoder.in_features = dim
        self._decoder.out_features = dim

    @property
    def hidden_dim(self) -> int:
        """
        AutoRec의 hidden dimension 값 출력

        Returns
        -------
        int
            AutoRec의 hidden dimension 값
        """
        return self._hidden_dim

    @hidden_dim.setter
    def hidden_dim(self, dim: int):
        """
        AutoRec의 hidden dimension 값 변경

        Parameters
        ----------
        dim : int
            새로운 hidden dimension
        """
        self._hidden_dim = dim
        self._decoder.in_features = dim
        self._encoder.out_features = dim

    @property
    def encoder(self) -> nn.Linear:
        """
        AutoRec의 encoder layer 부분 출력

        Returns
        -------
        nn.Linear
            AutoRec의 encoder layer
        """
        return self._encoder

    @property
    def decoder(self) -> nn.Linear:
        """
        AutoRec의 decoder layer 부분 출력

        Returns
        -------
        nn.Linear
            AutoRec의 decoder layer
        """
        return self._decoder

    @property
    def final_activation(self) -> ACTIVATION:
        """
        AutoRec의 마지막 활성화 함수 출력

        Returns
        -------
        ACTIVATION
            AutoRec의 마지막 활성화 함수
        """
        return self._final_activation

    @final_activation.setter
    def final_activation(self, activation: ACTIVATION):
        """
        AutoRec의 마지막 활성화 함수 변경

        Parameters
        ----------
        activation : ACTIVATION
            새로운 마지막 활성화 함수
        """
        self._final_activation = activation

    @property
    def optimizer(self) -> Optimizer:
        """
        AutoRec의 optimizer 출력

        Returns
        -------
        Optimizer
            AutoRec의 optimizer
        """
        return self._optimizer

    @optimizer.setter
    def optimizer(self, optimizer: Union[str, Optimizer]):
        """
        AutoRec의 optimizer 변경

        Parameters
        ----------
        optimizer : Union[str, Optimizer]
            optimizer의 이름 혹은 실제 optimizer

        Raises
        ------
        ValueError
            str이나 Optimizer가 아닌 경우 에러
        """
        if isinstance(optimizer, str):
            self._set_optimizer(optimizer)
        elif isinstance(optimizer, Optimizer):
            self._optimizer = optimizer
        else:
            raise ValueError

    @property
    def loss(self) -> _Loss:
        """
        AutoRec의 손실함수 출력

        Returns
        -------
        _Loss
            AutoRec의 손실함수
        """
        return self._loss

    @loss.setter
    def loss(self, loss: _Loss):
        """
        AutoRec의 손실함수 변경

        Parameters
        ----------
        loss : _Loss
            새로운 손실함수
        """
        self._loss = loss
