import torch
import torch.nn as nn

from deeprecs.models.base import BaseRecommender


class NCF(BaseRecommender):
    r"""NCF(Neural Collaborative Filtering) 추천 모델 클래스

    References
    ----------
    [1] TODO https://arxiv.org/pdf/1708.05031v2.pdf
    """

    def __init__(self, num_users: int, num_items: int, **kwargs):
        super().__init__()
        """
        model : 'MLP', 'GMF', 'NMF', and 'NMF-pretrained'
        """
        self.num_users = num_users
        self.num_items = num_items
        self.model = kwargs.get('model', 'NMF')
        self.num_factor = kwargs.get('num_factor', 8)
        self.layers = kwargs.get('layers', [16, 32, 16, 8])

        self.embed_user_gmf = nn.Embedding(self.num_users, self.num_factor)
        self.embed_item_gmf = nn.Embedding(self.num_items, self.num_factor)

        self.embed_user_mlp = nn.Embedding(self.num_users, self.layers[0]/2)
        self.embed_item_mlp = nn.Embedding(self.num_items, self.layers[0]/2)

        # ??
        self.fc_layers = nn.ModuleList()
        for _, (in_size, out_size) in enumerate(zip(self.layers[:-1], self.layers[1:])):
            self.fc_layers.append(torch.nn.Linear(in_size, out_size))
            self.fc_layers.append(nn.ReLU())

        self.affine_output = nn.Linear(
            in_features=self.layers[-1]+self.num_factor, out_features=1)
        self.logistic = torch.nn.Sigmoid()

        self._init_weight()

    def _init_weight(self):
        """
        모델의 초기 weight를 설정합니다.
        """

        if self.model != 'NeuMF-pretrained':
            nn.init.normal_(self.embed_user_gmf.weight, std=0.01)
            nn.init.normal_(self.embed_item_gmf.weight, std=0.01)
            nn.init.normal_(self.embed_user_mlp.weight, std=0.01)
            nn.init.normal_(self.embed_item_mlp.weight, std=0.01)

            for m in self.fc_layers:
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)

            nn.init.xavier_uniform_(self.affine_output.weight)

            for m in self.modules():
                if isinstance(m, nn.Linear) and m.bias is not None:
                    m.bias.data.zero_()

        else:  # TODO pretrained weight가 있는 경우
            pass

    def forward(self, user_ind: torch.Tensor, item_ind: torch.Tensor):
        r"""Forward propagation을 수행합니다."""

        user_embedding_gmf = self.embed_user_gmf(user_ind)
        item_embedding_gmf = self.embed_item_gmf(item_ind)
        user_embedding_mlp = self.embed_user_mlp(user_ind)
        item_embedding_mlp = self.embed_item_mlp(item_ind)

        mlp_vector = torch.cat(
            [user_embedding_mlp, item_embedding_mlp], dim=-1)

        gmf_vector = torch.mul(user_embedding_gmf, item_embedding_gmf)

        for idx, _ in enumerate(range(len(self.fc_layers))):
            mlp_vector = self.fc_layers[idx](mlp_vector)

        if self.model == "GMP":
            concat_vector = gmf_vector

        elif self.model == "MLP":
            concat_vector = self.MLP_layers(mlp_vector)

        else:
            concat_vector = torch.cat([mlp_vector, gmf_vector], dim=-1)

        logits = self.affine_output(concat_vector)
        rating = self.logistic(logits)
        return rating.squeeze()

    def fit(self):
        r"""모델을 데이터에 적합시키는 메서드로 `_train_one_epoch`을 반복합니다."""

    def predict(self):
        r"""추천 결과를 생성합니다."""

    def _train_one_epoch(self):
        r"""데이터에 대해 1 epoch을 학습합니다."""
