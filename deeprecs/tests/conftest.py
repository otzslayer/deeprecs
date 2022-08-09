# pylint: disable=import-error, unused-variable
import pytest

from deeprecs.models.autorec import AutoEncoder


@pytest.fixture
def autorec():
    """
    pytest에서 사용하게 될 AutoRec
    """

    def wrapper(input_dim: int, hidden_dim: int) -> AutoEncoder:
        """
        AutoRec의 추천모델을 반환하는 함수

        Parameters
        ----------
        input_dim : int
            AutoRec의 input dimension
        hidden_dim : int
            AutoRec의 hidden dimension

        Returns
        -------
        AutoEncoder
            AutoRec의 추천모델

        Notes
        -----
        데이터마다 input/hidden dimension이 다르므로,
        각 test마다 input/hidden을 입력하기 위해 wrapper 사용
        """
        autoencoder = AutoEncoder(input_dim=input_dim, hidden_dim=hidden_dim)
        return autoencoder

    return wrapper
