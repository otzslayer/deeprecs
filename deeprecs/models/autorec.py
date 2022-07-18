from typing import Final, Union

from torch import nn

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
    https://users.cecs.anu.edu.au/~akmenon/papers/autorec/autorec-paper.pdf
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        final_activation: ACTIVATION = nn.ReLU(),
    ):
        super().__init__()
        self._input_dim = input_dim
        self._hidden_dim = hidden_dim
        self._encoder = nn.Linear(input_dim, hidden_dim)
        self._decoder = nn.Linear(hidden_dim, input_dim)
        self._final_activation = final_activation

    def forward(self, input_data):
        """
        _summary_

        Parameters
        ----------
        input_data : _type_
            _description_

        Returns
        -------
        _type_
            _description_
        """
        encoded = self._encoder(input_data)
        decoded = self._decoder(encoded)
        output = self._final_activation(decoded)

        return output

    def fit(self):
        """
        _summary_

        Raises
        ------
        NotImplementedError
            _description_
        """
        raise NotImplementedError

    def predict(self):
        """
        _summary_

        Raises
        ------
        NotImplementedError
            _description_
        """
        raise NotImplementedError

    def _train_one_epoch(self):
        """
        _summary_

        Raises
        ------
        NotImplementedError
            _description_
        """
        raise NotImplementedError

    @property
    def input_dim(self):
        """
        input_dim

        Returns
        -------
        int
            int
        """
        return self._input_dim

    @input_dim.setter
    def input_dim(self, dim):
        """
        _summary_

        Parameters
        ----------
        dim : _type_
            _description_
        """
        self._input_dim = dim
        self._encoder.in_features = dim
        self._decoder.out_features = dim

    @property
    def hidden_dim(self):
        """
        _summary_

        Returns
        -------
        _type_
            _description_
        """
        return self._hidden_dim

    @hidden_dim.setter
    def hidden_dim(self, dim):
        """
        _summary_

        Parameters
        ----------
        dim : _type_
            _description_
        """
        self._hidden_dim = dim
        self._decoder.in_features = dim
        self._encoder.out_features = dim

    @property
    def encoder(self):
        """
        _summary_

        Returns
        -------
        _type_
            _description_
        """
        return self._encoder

    @property
    def decoder(self):
        """
        _summary_

        Returns
        -------
        _type_
            _description_
        """
        return self._decoder

    @property
    def final_activation(self):
        """
        _summary_

        Returns
        -------
        _type_
            _description_
        """
        return self._final_activation

    @final_activation.setter
    def final_activation(self, activation):
        """
        _summary_

        Parameters
        ----------
        activation : _type_
            _description_
        """
        self._final_activation = activation
