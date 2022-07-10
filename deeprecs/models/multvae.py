from typing import List, Tuple

import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils import data

from deeprecs.models.base import BaseRecommender


class MultVAE(BaseRecommender):
    r"""Multinomial likelihood를 사용한 Variational Autoencoder 모델 클래스입니다.

    Parameters
    ----------
    num_users : int
        총 사용자 수
    num_items : int
        총 아이템 수
    encoder_dims : List
        인코더 차원
    latent_dim : int
        잠재 공간 차원 수
    decoder_dims : List, optional
        디코더 차원으로 값을 전달하지 않으면 인코더 차원을 역순으로 사용
    dropout : float, optional
        드롭아웃 확률로 기본값은 0.2
    total_anneal_steps : int, optional
        총 KL 어닐링 스텝 수로 기본값은 200000
    anneal_cap : float, optional
        최대 어닐링 수준으로 기본값은 0.2

    References
    ----------
    [1] Liang, Dawen, et al. "Variational autoencoders for collaborative filtering."
        Proceedings of the 2018 world wide web conference. 2018.
        https://arxiv.org/pdf/1802.05814
    """

    def __init__(
        self,
        num_users: int,
        num_items: int,
        encoder_dims: List,
        latent_dim: int,
        decoder_dims: List = None,
        dropout: float = 0.2,
        total_anneal_steps: int = 200000,
        anneal_cap: float = 0.2,
    ):

        super().__init__()

        self.num_users = num_users
        self.num_items = num_items
        self.encoder_dims = encoder_dims
        self.latent_dim = latent_dim

        if decoder_dims:
            assert (
                encoder_dims[0] == decoder_dims[-1]
            ), "Both dimensions of input and output should be equal each other for autoencoders."
            assert (
                encoder_dims[-1] == decoder_dims[0]
            ), "Latent dimension for the encoder and the decoder mismathces."
            self.decoder_dims = decoder_dims
        else:
            self.decoder_dims = encoder_dims[::-1]

        self.eps = 1e-6
        self.dropout = dropout
        self.anneal = 0.0
        self.total_anneal_steps = total_anneal_steps
        self.anneal_cap = anneal_cap

        # Note that an encoder contains latent feature layer!
        # The dimension of the last layer in an encoder is the double of latent dimension.
        encoder_latent_dims = encoder_dims + [latent_dim * 2]
        self.encoder_layers = nn.ModuleList(
            [
                nn.Linear(d_in, d_out)
                for d_in, d_out in zip(
                    encoder_latent_dims[:-1], encoder_latent_dims[1:]
                )
            ]
        )
        self.decoder_layers = nn.ModuleList(
            [
                nn.Linear(d_in, d_out)
                for d_in, d_out in zip(decoder_dims[:-1], decoder_dims[1:])
            ]
        )

    def forward(
        self, inputs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        r"""Forward propagation을 수행합니다.

        Parameters
        ----------
        input : torch.Tensor

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        """
        mu, logvar = self._encode(inputs)
        z = self._reparametrize(mu, logvar)
        return self._decode(z), mu, logvar

    def fit(
        self, train_loader: data.DataLoader, optimizer: optim.Optimizer
    ) -> torch.Tensor:
        raise NotImplementedError

    def predict(self):
        raise NotImplementedError

    def _train_one_epoch(
        self, train_loader: data.DataLoader, optimizer: optim.Optimizer
    ) -> torch.Tensor:
        r"""데이터에 대해 1 epoch을 학습합니다.

        Parameters
        ----------
        data_loader : data.DataLoader
            학습 데이터 `DataLoader`

        optimizer : optim.Optimizer

        Returns
        -------
        torch.Tensor
            1 epoch 학습 후 loss 값
        """
        update_count, loss = 0, 0

        # Turn training mode on
        self.train()

        for input_mat in train_loader:
            input_mat = input_mat.float().cuda()
            self.zero_grad()

            hat_x, mu, logvar = self.forward(input_mat)

            if self.total_anneal_steps > 0:
                anneal = min(
                    self.anneal_cap, update_count / self.total_anneal_steps
                )
            else:
                anneal = self.anneal_cap

            batch_loss = _elbo_loss(
                x=input_mat,
                hat_x=hat_x,
                mu=mu,
                logvar=logvar,
                anneal=anneal,
            )
            batch_loss.backward()
            optimizer.step()
            update_count += 1
            loss += batch_loss

        return loss / len(train_loader)

    def _encode(
        self, inputs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""MultVAE에서 인코더 연산을 수행합니다.

        Parameters
        ----------
        inputs : torch.Tensor
            인풋 텐서

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            `mu`, `var`
        """
        x = self.dropout(F.normalize(inputs))

        for i, layer in enumerate(self.encoder_layers):
            x = layer(x)
            if i != len(self.encoder_layers) - 1:
                x = F.tanh(x)
            else:
                mu = x[:, : self.latent_dim]
                logvar = x[:, self.latent_dim :]
        return mu, logvar

    def _decode(self, z: torch.Tensor) -> torch.Tensor:
        r"""Reparametrize한 레이어 값을 받아 디코더 연산을 수행합니다.

        Parameters
        ----------
        z : torch.Tensor
            인풋 텐서

        Returns
        -------
        torch.Tensor
        """
        x = z
        for i, layer in enumerate(self.decoder_layers):
            x = layer(x)
            if i != len(self.decoder_layers) - 1:
                x = F.tanh(x)
        return x

    def _reparametrize(
        self, mu: torch.Tensor, logvar: torch.Tensor
    ) -> torch.Tensor:
        r"""Reparametrization trick을 사용합니다.

        Parameters
        ----------
        mu : torch.Tensor
        logvar : torch.Tensor

        Returns
        -------
        torch.Tensor
        """
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        return mu


def _elbo_loss(
    x: torch.Tensor,
    hat_x: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    anneal: float,
) -> torch.Tensor:
    r"""ELBO loss를 계산합니다.

    Parameters
    ----------
    x : torch.Tensor
    hat_x : torch.Tensor
    mu : torch.Tensor
    logvar : torch.Tensor
    anneal : float
        KL 어닐링 계수

    Returns
    -------
    torch.Tensor

    References
    ----------
    [1] Kingma, Diederik P., and Max Welling. "Auto-encoding variational bayes."
        arXiv preprint arXiv:1312.6114 (2013).
    """
    # 1/2 * sum^J_1 (1 + log(sigma_j^2) - mu_j^2 - sigma_j^2 )
    bce = -torch.mean(torch.sum(F.log_softmax(hat_x, 1) * x, -1))
    kld = -0.5 * torch.mean(
        torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    )
    return bce + anneal * kld
