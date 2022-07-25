import torch
import torch.nn.functional as F
from torch import nn

from deeprecs.models.base import BaseRecommender


class CDAE(BaseRecommender):
    r"""CDAE model docstring
    Args:
        BaseRecommender (_type_): _description_
    """

    def __init__(
        self,
        num_users,
        num_items,
        latent_dim,
        hparams,
        is_train,
    ):
        super().__init__()
        self.corruption_ratio = hparams["corruption_ratio"]
        self.num_users = num_users
        self.num_items = num_items
        # self.encoder_dims = encoder_dims
        self.latent_dim = latent_dim
        self.user_embedding = nn.Embedding(
            num_embeddings=self.num_users,  # embedding할 노드 수
            embedding_dim=self.latent_dim,
        )  # embedding할 벡터 차원
        self.encoder = nn.Linear(self.num_items, self.latent_dim)
        self.decoder = nn.Linear(self.latent_dim, self.num_items)
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.to(self.device)
        self.is_train = is_train

    def forward(self, item_vector, user_vector):
        print("forward")
        # corrupted input vector
        corrupted_item_vector = F.dropout(
            input=item_vector, p=self.corruption_ratio, training=self.is_train
        )
        # input layer: corrupted input vector + user specific node
        # encode_layer = torch.append(corrupted_item_vector, user_vector)
        corrupted_item_vector.append(user_vector)

    def fit(self):
        print("fit")

    def predict(self):
        print("predict")

    def _train_one_epoch(self):
        print("_train_one_epoch")
