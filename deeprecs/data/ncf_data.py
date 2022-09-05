from ast import literal_eval
from typing import List

import numpy as np
import pandas as pd
import scipy.sparse as sp
from torch.utils import data


class NCFData(data.Dataset):
    r"""NCF 모델을 위한 데이터 클래스"""

    def __init__(
        self,
        features: List,
        num_item: int,
        train_mat: sp.base.spmatrix,
        num_ng: int,
        is_training: bool,
    ):

        super().__init__()

        self.features_ps = features
        self.features_ng = []
        self.features_fill = []
        self.num_item = num_item
        self.train_mat = train_mat
        self.num_ng = num_ng
        self.is_training = is_training
        self.labels = [0] * len(features)
        self.labels_fill = []

    def ng_sample(self):
        f"""학습을 위한 negative sample을 생성합니다."""

        assert self.is_training

        for x in self.features_ps:
            user_i = x[0]
            for _ in range(self.num_ng):
                item_j = np.random.randint(self.num_item)
                while (user_i, item_j) in self.train_mat:
                    item_j = np.random.randint(self.num_item)
                self.features_ng.append([user_i, item_j])

        labels_ps = [1] * len(self.features_ps)
        labels_ng = [0] * len(self.features_ng)

        self.features_fill = self.features_ps + self.features_ng
        self.labels_fill = labels_ps + labels_ng

    def __len__(self):
        return (self.num_ng + 1) * len(self.labels)

    def __getitem__(self, idx):
        features = self.features_fill if self.is_training else self.features_ps
        labels = self.labels_fill if self.is_training else self.labels

        user = features[idx][0]
        item = features[idx][1]
        label = labels[idx]
        return user, item, label


def load_data(train_data_path: str, test_data_path: str):
    train_data = pd.read_csv(
        train_data_path,
        sep="\t",
        header=None,
        names=["user", "item"],
        usecols=[0, 1],
        dtype={0: np.int32, 1: np.int32},
    )

    num_user = train_data["user"].max() + 1
    num_item = train_data["item"].max() + 1

    train_data = train_data.values.tolist()
    train_mat = sp.dok_matrix((num_user, num_item), dtype=np.float32)

    for x in train_data:
        train_mat[x[0], x[1]] = 1.0

    test_data = []
    with open(test_data_path) as f:
        line = f.readline()
        while line != None and line != "":
            arr = line.split("\t")
            user = literal_eval(arr[0])[0]
            test_data.append([user, literal_eval(arr[0])[1]])
            for i in arr[1:]:
                test_data.append([user, int(i)])
            line = f.readline()

    return train_data, test_data, num_user, num_item, train_mat
