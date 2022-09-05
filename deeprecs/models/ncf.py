from typing import List


import torch
from torch.utils.data import DataLoader
from torch import nn, optim

from deeprecs.models.base import BaseRecommender
from deeprecs.data import NCFData
from deeprecs.utils import metrics

class NCF(BaseRecommender):
    r"""NCF(Neural Collaborative Filtering) 추천 모델 클래스

    Parameters
    ----------
    num_users : int
        사용자 수
    num_items : int
        아이템 수
    model : str, optional
        NCF 프레임워크로 선택 가능한 모델 옵션으로 기본값은 'NMF'
        'MLP', 'GMF', 'NMF', and 'NMF-pretrained'
    num_factor : int, optional
        NCF 마지막 히든 레이어(predictive factors) 사이즈 기본값은 8
    layers : List, optional
        MLP 레이어들의 사이즈로, layers[0]/2는 아이템, 사용자 임베딩 사이즈
    device : torch.device, optional
        모델에 사용할 디바이스

    References
    ----------
    [1] He, Xiangnan, et al. "Neural collaborative filtering."
        Proceedings of the 26th international conference on world wide web. 2017.
        https://arxiv.org/pdf/1708.05031v2.pdf
    """

    def __init__(
        self,
        num_users: int,
        num_items: int,
        model: str = "NMF",
        num_factor: int = 8,
        layers: List = None,
        lr: float = 0.001,
        device: torch.device = None,
        ):

        super().__init__()

        self.num_users = num_users
        self.num_items = num_items
        self.model = model
        self.num_factor = num_factor
        self.layers = layers
        self.lr = lr
        self.top_k = 10
        self.model_save = True
        self.model_path = '../results/'

        self.embed_user_gmf = nn.Embedding(self.num_users, self.num_factor)
        self.embed_item_gmf = nn.Embedding(self.num_items, self.num_factor)
        self.embed_user_mlp = nn.Embedding(self.num_users, self.layers[0] / 2)
        self.embed_item_mlp = nn.Embedding(self.num_items, self.layers[0] / 2)

        self.fc_layers = nn.ModuleList()
        for _, (in_size, out_size) in enumerate(
            zip(self.layers[:-1], self.layers[1:])
        ):
            self.fc_layers.append(torch.nn.Linear(in_size, out_size))
            self.fc_layers.append(nn.ReLU())

        self.affine_output = nn.Linear(
            in_features=self.layers[-1] + self.num_factor, out_features=1
        )
        self.logistic = torch.nn.Sigmoid()
        self.loss_func = nn.BCEWithLogitsLoss()

        if self.model == 'NeuMF-pretrained':
            self.optimizer = optim.SGD(self.parameters(), self.lr)
        else:
            self.optimizer = optim.Adam(self.parameters(), self.lr)

        if not (device != 'cuda' or torch.cuda.is_available()):
            self.devide = device
        else:
            raise RuntimeError(f'CUDA error: invalid argument {device}')

        self._init_weight()

    def _init_weight(self):
        r"""모델의 초기 weight를 설정합니다."""

        if self.model != "NeuMF-pretrained":
            nn.init.normal_(self.embed_user_gmf.weight, std=0.01)
            nn.init.normal_(self.embed_item_gmf.weight, std=0.01)
            nn.init.normal_(self.embed_user_mlp.weight, std=0.01)
            nn.init.normal_(self.embed_item_mlp.weight, std=0.01)

            for m_layer in self.fc_layers:
                if isinstance(m_layer, nn.Linear):
                    nn.init.xavier_uniform_(m_layer.weight)
                    # nn.init.normal_(m.weight)

            nn.init.xavier_uniform_(self.affine_output.weight)
            # nn.init.normal_(self.affine_output.weight)

        else:  # pretrained weight가 있는 경우
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
        return rating.view(-1)

    def fit(self, epochs: int, train_loader: DataLoader):
        r"""모델을 데이터에 적합시키는 메서드로 `_train_one_epoch`을 반복합니다."""

        for epoch in range(epochs):
            self.train()
            total_loss = 0
            train_loader.dataset.ng_sample()
            loss = self._train_one_epoch(train_loader)
            total_loss += loss

            print(f'[Training Epoch {epoch}] \
                    Epoch Loss : {loss}, \
                    Total Loss : {total_loss}')

            self.eval()
            best_hr = 0

            HR, NDCG = metrics(model, test_loader, self.top_k)
            print("HR: {:.3f}\tNDCG: {:.3f}".format(np.mean(HR), np.mean(NDCG)))

            if HR > best_hr:
                best_hr, best_ndcg, best_epoch = HR, NDCG, epoch
                if self.model_save:
                    if not os.path.exists(self.model_path):
                        os.mkdir(self.model_path)
                    torch.save(
                        model, "{}{}.pth".format(self.model_path, self.model)
                    )

        print("End. Best epoch {:03d}: HR = {:.3f}, NDCG = {:.3f}".format(
                    best_epoch, best_hr, best_ndcg))

    def predict(self):
        r"""추천 결과를 생성합니다."""

    def _train_one_epoch(self, train_loader: DataLoader):
        r"""데이터에 대해 1 epoch을 학습합니다."""

        for batch in train_loader:
            user = batch[0].to(self.device)
            item = batch[1].to(self.device)
            label = batch[2].float().to(self.device)

            self.zero_grad()
            pred = self.forward(user, item)
            loss = self.loss_func(pred, label)
            loss.backward()
            self.optimizer.step()
            return loss.items()
