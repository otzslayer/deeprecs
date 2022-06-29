import torch.nn as nn

from deeprecs.models.base import BaseRecommender

class Encoder:
    def __init__(self, layers):
        self._layers = layers
        self._encoder = nn.Sequential(*layers)

    def forward(self, x):
        return self._encoder(x)

    def __repr__(self):
        return self._encoder

    @property
    def layer(self):
        return self._layers

    @property
    def encoder(self):
        return self._encoder


class Decoder:
    def __init__(self, layers):
        self._layers = layers
        self._decoder = nn.Sequential(*layers)

    def forward(self, x):
        return self._decoder(x)

    def __repr__(self):
        return self._decoder

    @property
    def layer(self):
        return self._layers

    @property
    def decoder(self):
        return self._decoder


class AutoRec(BaseRecommender):
    """_summary_

    Args:
        BaseRecommender (_type_): _description_
    """

    def __init__(self, encoder, decoder):
        self._encoder = encoder
        self._decoder = decoder

    def forward(self, x):
        encoded = self._encoder(x)
        decoded = self._decoder(encoded)

        return decoded

    def fit(self):
        pass

    def predict(self):
        pass

    def _train_one_epoch(self):
        pass

    @property
    def encoder(self):
        return self._encoder

    @property
    def decoder(self):
        return self._decoder