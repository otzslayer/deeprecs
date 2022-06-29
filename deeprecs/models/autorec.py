from typing import List

import torch
from torch import nn
from numpy.typing import ArrayLike

from deeprecs.models.base import BaseRecommender
class Encoder:
    """
    AutoRec의 Auto-Encoder에서 Encoder에 해당하는 부분에 관한 클래스입니다.

    Parameters
    ----------
    `layers` : ArrayLike
        encoder의 layer 부분

    Arguments
    ---------
    `_layers` : ArrayLike
        encoder의 layer 부분
    `_encoder` : 
        _layers로 구성한 신경망

    Examples
    --------
    encoder = Encoder([nn.Linear(10, 2), nn.ReLU()])

    References
    ----------
    https://users.cecs.anu.edu.au/~akmenon/papers/autorec/autorec-paper.pdf
    """
    def __init__(self, layers: ArrayLike):
        self._layers = layers
        self._encoder = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forwrad propagation을 수행합니다.

        Parameters
        ----------
        `x` : torch.Tensor
            Encoder instance의 입력 데이터

        Returns
        -------
        torch.Tensor
            Decoder instance의 입력이 될 Encoder instance의 출력
        """
        return self._encoder(x)

    def __repr__(self):
        return self._encoder

    @property
    def layer(self):
        return self._layers

    @property
    def encoder(self):
        return self._encoder


class Decoder:
    """
    AutoRec의 Auto-Encoder에서 Decoder에 해당하는 부분에 관한 클래스입니다.

    Parameters
    ----------
    `layers` : ArrayLike
        decoder의 layer 부분

    Arguments
    ---------
    `_layers` : ArrayLike
        decoder의 layer 부분
    `_decoder` : 
        _layers로 구성한 신경망

    Examples
    --------
    decoder = Decoder([nn.Linear(2, 10), nn.Sigmoid()])

    References
    ----------
    https://users.cecs.anu.edu.au/~akmenon/papers/autorec/autorec-paper.pdf
    """
    def __init__(self, layers):
        self._layers = layers
        self._decoder = nn.Sequential(*layers)

    def forward(self, x):
        """
        Forwrad propagation을 수행합니다.

        Parameters
        ----------
        `x` : torch.Tensor
            Decoder instance의 입력 데이터

        Returns
        -------
        torch.Tensor
            AutoREC의 결과가 될 Decoder instance의 출력
        """
        return self._decoder(x)

    def __repr__(self):
        return self._decoder

    @property
    def layer(self):
        return self._layers

    @property
    def decoder(self):
        return self._decoder


class AutoRec(BaseRecommender):
    """
    Auto-Encoder를 활용한 추천 모델 클래스

    Parameters
    ----------
    `encoder` : Encoder
        Auto-Encoder의 Encoder 부분에 해당하는 neural network
    `decoder` : Decoder
        Auto-Encoder의 Decoder 부분에 해당하는 neural network

    Arguments
    ---------
    `_encoder`
    `_decoder`

    Examples
    --------
    encoder = Encoder()
    decoder = Decoder()
    autorec = AutoRec(encoder, decoder)

    Notes
    -----
    기존 논문에서는 Encoder와 Decoder가 완전히 대칭이 되지만,
    Denoise Auto-Encoder, Melon의 추천 시스템에서 처럼 비대칭 Auto-Encoder도
    많이 연구가 되고 있기 때문에, 대칭성을 강제하지 않았음.
    비대칭적인 layer의 개수, node의 개수, Input/Output의 형태 등이 가능함.

    References
    ----------
    https://users.cecs.anu.edu.au/~akmenon/papers/autorec/autorec-paper.pdf
    """
    def __init__(self, encoder, decoder):
        self._encoder = encoder
        self._decoder = decoder

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward propagation을 수행합니다.

        Parameters
        ----------
        `x` : torch.Tensor
            Auto-Encoder의 입력 데이터

        Returns
        -------
        torch.Tensor
            Auto-Encoder의 출력 데이터
        """
        encoded = self._encoder(x)
        decoded = self._decoder(encoded)

        return decoded

    def fit(self):
        """모델을 데이터에 적합시키는 메서드로 `_train_one_epoch`을 반복합니다."""
        raise NotImplementedError

    def predict(self):
        """추천 결과를 생성합니다."""
        raise NotImplementedError

    def _train_one_epoch(self):
        """데이터에 대해 1 epoch을 학습합니다."""
        raise NotImplementedError

    @property
    def encoder(self):
        return self._encoder

    @property
    def decoder(self):
        return self._decoder