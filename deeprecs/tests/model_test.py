# pylint: disable=import-error
import pandas as pd
import pytest
import pytest_lazyfixture
import torch

from deeprecs.data.data import SampleDataset
from deeprecs.models.base import BaseRecommender


@pytest.mark.parametrize("model", [pytest_lazyfixture.lazy_fixture("autorec")])
def test_ml100k(model: BaseRecommender):
    """
    movielens 100k 데이터에 대해서 모델의 예측값이 잘 나오는지 확인하는 테스트함수

    Parameters
    ----------
    model : BaseRecommender
        추천 모델
    """
    model = model(input_dim=1683, hidden_dim=128)
    ml = pd.read_csv("./data/ml-100k/ml-100k_pivot.csv")
    ml_dataset = SampleDataset(torch.Tensor(ml.values), torch.Tensor(ml.values))
    model.fit(ml_dataset, epochs=1, batch_size=32)
    pred = model.predict(ml_dataset)

    assert pred.shape == ml_dataset.y.shape
