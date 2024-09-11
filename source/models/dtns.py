import torch
import torch.nn as nn

from typing import List
from .crnn import CRNN
from .nade import NADE


class DeepToneNets(nn.Module):
    def __init__(self, in_features: int, label_sizes: List[int], use_key_bias: bool = True):
        """
        Implementation of the DeepToneNets model, composed of DenseNet + GRU + NADE block.

        Arguments
        ---------
            - in_features (int): number of input features.
            - label_sizes (List[int]): size of each label.
            - use_key_bias (bool, default=True): flag to use key similarity bias version.

        Notes
        -----
            - This module expects input shape = (n_batches, in_channels, seq_length) and returns
              a list of torch.Tensor with shape (n_batches, seq_length, label_size).
        """

        super(DeepToneNets, self).__init__()

        self.backbone = CRNN(in_features)
        self.classifier = NADE(self.backbone.gru.out_features, hidden_size=350,
                               label_sizes=label_sizes, use_key_bias=use_key_bias)

    def forward(self, x: torch.Tensor, is_train: bool, key_similarities: torch.Tensor = None, targets: torch.Tensor = None, mask: torch.Tensor = None) -> List[torch.Tensor]:
        """
        Implementation of the forward method.

        Arguments
        ---------
            - x (torch.Tensor): input tensor.
            - is_train (bool): flag indicator if we want to apply DeepToneNets in training or sampling mode.
            - key_similarities (torch.Tensor, default=None): key similarities tensor to use for bias method.
            - targets (torch.Tensor, default=None): targets tensor to use teacher forcing.
            - mask (torch.Tensor, default=None): 1D mask for the input and output. By default, no mask is applied.

        Returns
        -------
            - A list of torch.Tensor with shape (n_batches, seq_length, label_size).
        """

        out = self.backbone(x, mask)
        out = self.classifier(out, is_train=is_train, key_similarities=key_similarities, targets=targets)
        
        return out
