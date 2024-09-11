import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import List


class NADE(nn.Module):
    def __init__(self, in_size: int, hidden_size: int, label_sizes: List[int], use_key_bias: bool = True):
        """
        Auxiliary class to model NADE (Neural Autoregressive Density Estimator) output.

        Arguments
        ---------
            in_features (int): input features dimension.
            hidden_size (int): size of NADE hidden units.
            label_sizes (List[int]): size of each label.
            use_key_bias (bool, default=True): flag to use key similarity bias version.

        Notes
        -----
            - A list of torch.Tensor with shape (n_batches, seq_length, label_size).
        """

        super().__init__()

        self.num_blocks = len(label_sizes)
        self.hidden2vis = nn.ModuleList([nn.Linear(hidden_size, out_size, bias=False) for out_size in label_sizes])
        self.visible_biases = nn.ModuleList([nn.Linear(in_size, out_size) for out_size in label_sizes])
    
        self.hidden_bias = nn.Linear(in_size, hidden_size)
        self.vis2hidden = nn.ParameterList([nn.Linear(out_size, hidden_size, bias=False) for out_size in label_sizes])

        self._initialize_parameters()
        
        self.use_key_bias = use_key_bias
        self.bias_weights = nn.Parameter(torch.ones(24,))
    
    def forward(self, x: torch.Tensor, is_train: bool, key_similarities: torch.Tensor = None, targets: torch.Tensor = None) -> List[torch.Tensor]:
        if self.use_key_bias and key_similarities is None:
            raise ValueError('When using key similarity bias, you must inform the key_similarities tensor')
        
        if is_train:
            return self._train(x, key_similarities=key_similarities, targets=targets)

        return self._sample(x, key_similarities=key_similarities)

    def _train(self, x: torch.Tensor, key_similarities: torch.Tensor, targets: torch.Tensor):
        if targets is None:
            raise ValueError('When training, you must inform the targets tensor')

        hidden_bias = torch.sigmoid(self.hidden_bias(x))
        visible_bias = [torch.sigmoid(bias(x)) for bias in self.visible_biases]

        # Applying teacher forcing by replacing x_{<d} by the target
        teacher_forcing = [vis_layer(target) for vis_layer, target in zip(self.vis2hidden, targets)]

        outputs = []
        hidden = hidden_bias  # first iteration we don't have <d yet

        for i in range(self.num_blocks):
            if i == 0 and self.use_key_bias:
                output, hidden = self._train_step(hidden, self.hidden2vis[i], teacher_forcing[i], visible_bias[i], key_similarities=key_similarities)
            else:
                output, hidden = self._train_step(hidden, self.hidden2vis[i], teacher_forcing[i], visible_bias[i])
                
            outputs.append(output)
        
        return outputs

    def _train_step(self, hidden: torch.Tensor, hidden2vis: nn.Module, teacher_forcing: torch.Tensor, visible_bias: nn.Parameter, key_similarities: torch.Tensor = None):
        output = hidden2vis(torch.sigmoid(hidden)) + visible_bias
        if key_similarities is not None:
            output = output + self.bias_weights * key_similarities.transpose(1, 2)
        
        hidden = hidden + teacher_forcing # updating hidden state

        return output, hidden
    
    def _sample(self, x: torch.Tensor, key_similarities: torch.Tensor):
        hidden_bias = torch.sigmoid(self.hidden_bias(x))
        visible_bias = [torch.sigmoid(bias(x)) for bias in self.visible_biases]

        outputs = []
        hidden = hidden_bias  # in first iteration we don't have <d yet
        
        for i in range(self.num_blocks):
            if i == 0 and self.use_key_bias:
                output, hidden = self._sample_step(hidden, self.hidden2vis[i], self.vis2hidden[i], visible_bias[i], key_similarities=key_similarities)
            else:
                output, hidden = self._sample_step(hidden, self.hidden2vis[i], self.vis2hidden[i], visible_bias[i])
                
            outputs.append(output)

        return outputs
 
    def _sample_step(self, hidden: torch.Tensor, hidden2vis: nn.Module, vis2hidden: nn.Module, visible_bias: nn.Parameter, key_similarities: torch.Tensor = None):
        output = hidden2vis(torch.sigmoid(hidden)) + visible_bias
        if key_similarities is not None:
            output = output + self.bias_weights * key_similarities.transpose(1, 2)
        
        output_argmax = torch.argmax(output, dim=-1)

        num_classes = output.shape[-1]
        output_one_hot = F.one_hot(output_argmax, num_classes=num_classes)

        hidden = hidden + vis2hidden(output_one_hot.float())
        return output, hidden

    def _initialize_parameters(self):
        for param in self.parameters():
            nn.init.trunc_normal_(param)
