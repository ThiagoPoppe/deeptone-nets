import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import List


class DenseLayer1D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, bottleneck_channels: int, kernel_size: int):
        """
        Implementation of a 1D dense layer, composed of a bottleneck and a convolutional layer.

        Arguments
        ---------
            - in_channels (int): number of the input channels (or features).
            - out_channels (int): number of the output channels (or features).
            - bottleneck_channels (int): number of channels produced by the bottleneck layer.
            - kernel_size (int): the convolutional layer kernel size.

        Notes
        -----
            - This module expects input shape = (n_batches, in_channels, seq_length) and returns an
              output with shape: (n_batches, out_channels, seq_length).
        """

        super(DenseLayer1D, self).__init__()

        self.bottleneck_layer = nn.Sequential(
            nn.ReLU(),
            nn.BatchNorm1d(in_channels),
            nn.Conv1d(in_channels, bottleneck_channels, kernel_size=1, padding='same')
        )

        self.convolutional_layer = nn.Sequential(
            nn.ReLU(),
            nn.BatchNorm1d(bottleneck_channels),
            nn.Conv1d(bottleneck_channels, out_channels, kernel_size=kernel_size, padding='same')
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_in = x
        out = self.bottleneck_layer(x)
        out = self.convolutional_layer(out)

        # Concatenation along the channels
        return torch.cat([x_in, out], dim=1)


class PoolingBlock1D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int):
        """
        Implementation of a 1D dense net pooling block layer.

        Arguments
        ---------
            - in_channels (int): number of the input channels (or features).
            - out_channels (int): number of the output channels (or features).
            - kernel_size (int): the convolutional layer kernel size.

        Notes
        -----
            - This module expects input shape = (n_batches, in_channels, seq_length) and returns an
              output with shape: (n_batches, out_channels, seq_length).
        """

        super(PoolingBlock1D, self).__init__()

        padding = math.floor((kernel_size - 1) / 2)
        self.avg_pool = nn.AvgPool1d(kernel_size=kernel_size, padding=padding)

        self.batch_norm = nn.BatchNorm1d(in_channels)
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=1, padding='same')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(self.batch_norm(x))
        x = self.avg_pool(x)

        return x


class DenseBlock(nn.Module):
    def __init__(self, num_layers: int, in_channels: int, out_channels: int, bottleneck_channels: int, kernel_size: int):
        """
        Implementation of a dense block (sequence of dense layers).

        Arguments
        ---------
            - num_layers (int): number of dense layers to be used.
            - in_channels (int): number of the input channels (or features).
            - out_channels (int): number of the output channels (or features).
            - bottleneck_channels (int): number of channels produced by the bottleneck layer.
            - kernel_size (int): the convolutional layer kernel size.

        Notes
        -----
            - This module expects input shape = (n_batches, in_channels, seq_length) and returns an
              output with shape: (n_batches, out_channels, seq_length).
        """

        super(DenseBlock, self).__init__()

        layers = []
        for i in range(num_layers):
            layers.append(DenseLayer1D(in_channels + i * out_channels, out_channels, bottleneck_channels, kernel_size))

        self.dense_layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dense_layers(x)


class DenseNet(nn.Module):
    params = {
        'block_1': {
            'num_layers': 3,
            'out_channels': 10,
            'bottleneck_channels': 32,
            'kernel_size': 7
        },

        'block_2': {
            'num_layers': 2,
            'out_channels': 4,
            'bottleneck_channels': 20,
            'kernel_size': 3
        },

        'block_3': {
            'num_layers': 2,
            'out_channels': 4,
            'bottleneck_channels': 20,
            'kernel_size': 3
        },

        'pooling_block_1': {
            'out_channels': 48,
            'kernel_size': 2
        },

        'pooling_block_2': {
            'out_channels': 48,
            'kernel_size': 2
        }
    }

    def __init__(self, in_channels: int):
        """
        Implementation of a DenseNet architecture (sequence of dense blocks and pooling blocks).

        Arguments
        ---------
            - in_channels (int): number of the input channels (or features).

        Notes
        -----
            - This module expects input shape = (n_batches, in_channels, seq_length) and returns an
              output with shape: (n_batches, out_channels, seq_length).
        """

        super(DenseNet, self).__init__()

        added_channels_block_1 = self.params['block_1']['num_layers'] * self.params['block_1']['out_channels']
        added_channels_block_2 = self.params['block_2']['num_layers'] * self.params['block_2']['out_channels']
        added_channels_block_3 = self.params['block_3']['num_layers'] * self.params['block_3']['out_channels']

        in_channels_block_2 = self.params['pooling_block_1']['out_channels']
        in_channels_block_3 = self.params['pooling_block_2']['out_channels']

        in_channels_pooling_block_1 = in_channels + added_channels_block_1
        in_channels_pooling_block_2 = in_channels_block_2 + added_channels_block_2

        self.out_channels = in_channels_block_3 + added_channels_block_3
        self.dense_block_1 = DenseBlock(in_channels=in_channels, **self.params['block_1'])
        self.dense_block_2 = DenseBlock(in_channels=in_channels_block_2, **self.params['block_2'])
        self.dense_block_3 = DenseBlock(in_channels=in_channels_block_3, **self.params['block_3'])

        self.avg_pool_block_1 = PoolingBlock1D(in_channels=in_channels_pooling_block_1, **self.params['pooling_block_1'])
        self.avg_pool_block_2 = PoolingBlock1D(in_channels=in_channels_pooling_block_2, **self.params['pooling_block_2'])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.avg_pool_block_1(self.dense_block_1(x))
        x = self.avg_pool_block_2(self.dense_block_2(x))
        x = self.dense_block_3(x)

        return x


class GRU(nn.Module):
    def __init__(self, in_features: int, out_features: int = 64, hidden_size: int = 178, bidirectional: bool = True, dropout: float = 0.2):
        """
        Implementation of a Gated Recurrent Unit (GRU).

        Arguments
        ---------
            - in_features (int): number of input features.
            - out_features (int, default=64): number of output features.
            - hidden_size (int, default=178): number of hidden recurrent units.
            - bidirectional (bool, default=True): whether to create a bidirectional GRU or not.
            - dropout (float, default=0.2): the input dropout rate.

        Notes
        -----
            - This module expects input shape = (n_batches, in_channels, seq_length) and returns an
              output with shape: (n_batches, out_channels, seq_length).
        """

        super(GRU, self).__init__()

        self.out_features = out_features
        self.num_directions = 2 if bidirectional else 1

        self.input_dropout = nn.Dropout(p=dropout)
        self.gru = nn.GRU(in_features, hidden_size=hidden_size, bidirectional=bidirectional, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(self.num_directions * hidden_size, out_features),
            nn.Tanh()
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Implementation of the forward method.

        Arguments
        ---------
            - x (torch.Tensor): input tensor.
            - mask (torch.Tensor, default=None): 1D mask for the input and output. By default, no mask is applied.

        Returns
        -------
            - A torch.Tensor with shape (n_batches, seq_length, num_directions * hidden_size). n_directions will be 1
            if bidirectional=False else n_directions will be equal to 2.
        """

        if mask is not None:
            x = x * mask.unsqueeze(-1)

        outputs, _ = self.gru(self.input_dropout(x))
        outputs = self.fc(outputs)

        if mask is not None:
            outputs = outputs * mask.unsqueeze(-1)

        return outputs


class CRNN(nn.Module):
    def __init__(self, in_features: int):
        """
        Implementation of a CRNN (Convolutional Recurrent Neural Network) backbone, composed of DenseNet + GRU.

        Arguments
        ---------
            - in_features (int): number of input features.

        Notes
        -----
            - This module expects input shape = (n_batches, in_channels, seq_length) and returns
              an output shape = (n_batches, seq_length, gru.out_features).
        """

        super(CRNN, self).__init__()

        self.dense_net = DenseNet(in_features)
        self.gru = GRU(self.dense_net.out_channels)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> List[torch.Tensor]:
        """
        Implementation of the forward method.

        Arguments
        ---------
            - x (torch.Tensor): input tensor.
            - mask (torch.Tensor, default=None): 1D mask for the input and output. By default, no mask is applied.

        Returns
        -------
            - torch.Tensor with shape (n_batches, seq_length, gru.out_features).
        """

        out = F.relu(self.dense_net(x))

        # Converting from (N, C, L) to (N, L, C) for GRU
        out = out.transpose(1, 2)
        out = self.gru(out, mask)
            
        return out
