# pylint: disable=import-error
import numpy as np
import pandas as pd
import pytest
import pytest_lazyfixture
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from deeprecs.data.data import AEDataset
from deeprecs.models.base import BaseRecommender
from deeprecs.utils.loss import RMSELoss


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
    ml = ml.fillna(0)
    train, test = map(
        AEDataset,
        train_test_split(
            torch.Tensor(ml.values), test_size=0.1, random_state=42
        ),
    )
    train_loader = DataLoader(train, batch_size=32)
    test_loader = DataLoader(test, batch_size=len(test))

    model.fit(train_loader, epochs=1)
    # TODO: reshape 하지 않고, test랑 형태 맞출 수 있는 방법 찾기
    #       batch_size 1일 때, pred -> (93, 1=batch_size, 1682) test -> (93, 1682)
    pred = model.predict(test_loader).reshape(-1, 1682)
    pred = np.clip(pred, 1, 5)

    rmse = RMSELoss()(test.y, pred)
    # https://paperswithcode.com/sota/collaborative-filtering-on-movielens-100k
    # 기준으로 가장 높은 rmse는 0.996
    assert rmse <= 1.15
