# pylint: disable=import-error
import pandas as pd
import pytest
import pytest_lazyfixture
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from deeprecs.data.data import AEDataset
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
    model = model(input_dim=1682, hidden_dim=128)

    # TODO: 데이터도 fixture로 관리 / train, test까지
    ml = pd.read_csv("./data/ml-100k/ml-100k_pivot.csv", index_col=0)
    train, test = map(
        AEDataset,
        train_test_split(
            torch.Tensor(ml.values), test_size=0.2, random_state=42
        ),
    )
    train_loader, test_loader = DataLoader(train), DataLoader(test)

    model.fit(train_loader, epochs=1)
    pred = model.predict(test_loader)

    # TODO: score 계산해서, base score 이상인 거만 테스트 통과하게끔
    # https://paperswithcode.com/sota/collaborative-filtering-on-movielens-100k
    # 기준 0.88은 넘어야하지 않을까?
    # u1 split이 뭔지는 조사 필요
    assert pred.reshape(-1, 1682).shape == test.y.shape
