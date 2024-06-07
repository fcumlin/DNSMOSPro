import copy
import math
from typing import Any, Optional, Union, Type

import gin
import torch
import torch.nn as nn
import torch.nn.functional as F


def _get_conv_layer(
    in_channels: int,
    out_channels: int,
    kernel_size: tuple[int, int] = (3, 3),
    padding: tuple[int, int] = (1, 1),
    activation_fn: Any = nn.ReLU,
    max_pool_size: Optional[Union[tuple[int, int], int]] = 3,
    dropout: Optional[float] = 0.3,
    bn: bool = False,
) -> nn.Sequential:
    """Returns a CBAD layer: Convolution, Batch normalization, Activation, and Dropout."""
    layers = [nn.Conv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        padding=padding
    )]
    if bn:
        layers.append(nn.BatchNorm2d(out_channels))
    layers.append(activation_fn())
    if max_pool_size is not None:
        layers.append(nn.MaxPool2d(max_pool_size))
    if dropout is not None:
        layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)


@gin.configurable
def _dense_layer(
    in_dim: int,
    out_dim: int,
    use_ln: bool,
    use_activation: bool,
    activation_fn: Any = nn.ReLU,
) -> nn.Sequential:
    """Returns Sequential Dense-OptionalLayerNorm-OptionalActivation layer."""
    layers = [nn.Linear(in_dim, out_dim)]
    if use_ln:
        layers.append(nn.LayerNorm(out_dim))
    if use_activation:
        layers.append(activation_fn())
    return nn.Sequential(*layers)


@gin.configurable
class Encoder(nn.Module):
    
    def __init__(self, bn: bool = True, max_pool_size: int = 3, activation_fn: Any = nn.ReLU):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            _get_conv_layer(1, 32, bn=bn, max_pool_size=None, activation_fn=activation_fn),
            _get_conv_layer(32, 32, bn=bn, max_pool_size=max_pool_size, activation_fn=activation_fn),
            _get_conv_layer(32, 64, bn=bn, max_pool_size=None, activation_fn=activation_fn),
            _get_conv_layer(64, 64, bn=bn, max_pool_size=None, dropout=None, activation_fn=activation_fn),
        )
        self._flatten = nn.Flatten()

    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        # input speech_spectrum shape (batch, 1, max_seq_len, n_features)
        embeddings = self.encoder(spec) # shape (batch, 64, max_seq_len, n_features)
        embeddings = F.max_pool2d(embeddings, kernel_size=embeddings.size()[2:])
        return self._flatten(embeddings)


@gin.configurable
class Head(nn.Module):

    def __init__(self, use_ln: bool = False, activation_fn: Any = nn.ReLU, in_dim: int = 64):
        super(Head, self).__init__()
        self.head = nn.Sequential(
            _dense_layer(in_dim, 64, use_ln, True, activation_fn),
            _dense_layer(64, 64, use_ln, True, activation_fn),
            _dense_layer(64, 2, False, False),
        )

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        return self.head(embeddings)


@gin.configurable
class DnsmosPro(nn.Module):

    def __init__(self, encoder_cls: Type[nn.Module] = Encoder, head_cls: Type[nn.Module] = Head):
        super(DnsmosPro, self).__init__()
        self._encoder = encoder_cls()
        self._head = head_cls()
        self._softplus = nn.Softplus()

    def encoder(self, speech_spectrum: torch.Tensor) -> torch.Tensor:
        return self._encoder(speech_spectrum)

    def forward(self, speech_spectrum: torch.Tensor) -> torch.Tensor:
        embeddings = self._encoder(speech_spectrum)
        predictions = self._head(embeddings)
        mean_predictions = 2 * predictions[:, 0].unsqueeze(1) + 3
        var_predictions = 4 * self._softplus(predictions[:, 1].unsqueeze(1))
        predictions = torch.cat((mean_predictions, var_predictions), dim=1)
        return predictions


@gin.configurable
class DnsmosEncoder(nn.Module):
    
    def __init__(self):
        super(DnsmosEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(3,3), padding=(1,1)),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.3),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3,3), padding=(1,1)),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.3),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3,3), padding=(1,1)),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.3),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3,3), padding=(1,1)),
            nn.ReLU(),
        )
        
    def forward(self, speech_spectrum):
        # input speech_spectrum shape (batch, 1, max_seq_len, n_features)
        batch = speech_spectrum.shape[0]
        time = speech_spectrum.shape[2]
        speech_spectrum = self.encoder(speech_spectrum) # shape (batch, 64, max_seq_len, n_features)
        embeddings = F.max_pool2d(speech_spectrum, kernel_size=speech_spectrum.size()[2:])
        embeddings = embeddings.view(batch, -1) # shape (batch, 64)
        return embeddings


@gin.configurable
class DnsmosHead(nn.Module):

    def __init__(self):
        super(DnsmosHead, self).__init__()
        self.head = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, embeddings):
        # input embeddings shape (batch, 64)
        prediction = self.head(embeddings)
        return prediction


@gin.configurable
class DnsmosClassic(nn.Module):
    
    def __init__(self):
        super(DnsmosClassic, self).__init__()
        self.encoder = DnsmosEncoder()
        self.head = DnsmosHead()

    def forward(self, speech_spectrum: torch.Tensor) -> torch.Tensor:
        embeddings = self.encoder(speech_spectrum)
        prediction = self.head(embeddings)
        return prediction


@gin.configurable
class Mosnet(nn.Module):

    def __init__(self):
        super(Mosnet, self).__init__()
        self.mean_conv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(3,3), padding=(1,1)),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3,3), padding=(1,1)),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3,3), padding=(1,1), stride=(1,3)),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=32,  kernel_size=(3,3), padding=(1,1)),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3,3), padding=(1,1)),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3,3), padding=(1,1), stride=(1,3)),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3,3), padding=(1,1)),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3,3), padding=(1,1)),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3,3), padding=(1,1), stride=(1,3)),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3,3), padding=(1,1)),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3,3), padding=(1,1)),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3,3), padding=(1,1), stride=(1,3)),
            nn.ReLU()
        )

        self.mean_rnn = nn.LSTM(
            input_size=512,
            hidden_size=128,
            num_layers=1,
            batch_first=True,
            dropout=0.3,
            bidirectional=True
        )
        
        self.mean_MLP = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128,1)
        )

    def encoder(self, speech_spectrum: torch.Tensor) -> torch.Tensor:
        batch = speech_spectrum.shape[0]
        time = speech_spectrum.shape[2]
        speech_spectrum = self.mean_conv(speech_spectrum)  # shape (batch, 64, max_seq_len, n_features)
        embeddings = F.max_pool2d(speech_spectrum, kernel_size=speech_spectrum.size()[2:])
        embeddings = embeddings.view(batch, -1)  # shape (batch, 64)
        return embeddings

    def forward(self, speech_spectrum: torch.Tensor) -> torch.Tensor:
        # input speech_spectrum shape (batch, 1, max_seq_len, 257)
        batch = speech_spectrum.shape[0]
        time = speech_spectrum.shape[2]
        speech_spectrum = self.mean_conv(speech_spectrum)  # shape (batch, 128, max_seq_len, 4)
        speech_spectrum = speech_spectrum.view((batch, time, 512))  # shape (batch, max_seq_len, 512)
        speech_spectrum, (h, c) = self.mean_rnn(speech_spectrum)  # shape (batch, max_seq_len, 256)
        mos_mean = self.mean_MLP(speech_spectrum)  # shape (batch, max_seq_len, 1)
        return mos_mean
