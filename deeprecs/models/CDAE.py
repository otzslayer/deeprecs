import torch
import torch.nn.functional as F
from torch import nn

from deeprecs.models.base import BaseRecommender


class CDAE(BaseRecommender):
    r"""CDAE(Collaborative Denoising AutoEncoder) 추천 모델

    Parameters
    ----------
    num_users : int
        사용자 수
    num_items : int
        아이템 수
    hidden_dim : int
        은닉층 차원
    hparams : dictionary
        hyper-parameter들을 담은 dictionary
    References
    ----------
    [1] Wu, Yao, et al. "Collaborative Denoising Auto-Encoders for Top-N Recommender Systems."
        Proceedings of the 9th ACM International Conference on Web Search and Data Mining. 2016.
        https://alicezheng.org/papers/wsdm16-cdae.pdf
    """

    def __init__(self, num_users, num_items, hidden_dim, hparams, is_train):
        super().__init__()

        self.num_users = num_users
        self.num_items = num_items
        self.corruption_ratio = hparams["corruption_ratio"]
        self.activation = hparams["activation"]
        self.hidden_dim = hidden_dim

        self.user_embedding = nn.Embedding(
            num_embeddings=self.num_users,  # embedding할 노드 수
            embedding_dim=self.hidden_dim,  # embedding할 벡터 차원
        )
        self.encoder = nn.Linear(self.num_items, self.hidden_dim)
        self.decoder = nn.Linear(self.hidden_dim, self.num_items)

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.to(self.device)
        self.is_train = is_train

    def forward(self, user_idx, item_preference_vec):
        print("forward")
        # corrupted input vector
        corrupted_item_preference_vec = F.dropout(
            input=item_preference_vec,
            p=self.corruption_ratio,
            training=self.is_train,
        )
        # input layer: corrupted input vector + user specific node
        output = self.encoder(
            corrupted_item_preference_vec
        ) + self.user_embedding(user_idx)
        output = nn.ReLU(output)
        output = self.decoder(output)
        output = self.activation(output)
        return output

    def fit(self):
        print("fit")

    def predict(self):
        print("predict")

    def _train_one_epoch(self):
        print("_train_one_epoch")
