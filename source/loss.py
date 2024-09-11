import torch
import torch.nn as nn
from .data.constants import LABEL_SIZES


class MultiTaskCrossEntropyLoss(nn.Module):
    def __init__(self, reduction: str = 'sum', task_weights: torch.Tensor = None):
        """
        Cross Entropy Loss applyed to a MultiTask cenario.

        Arguments
        ---------
            - reduction (str, default="sum"): reduction to be used, either "sum" or "mean".
            - task_weights (torch.Tensor, default=None): weights to give to each task (must be in the same device as outputs and labels).
                If None is passed then weights will automatically be assigned to 1's.
        """

        super().__init__()

        assert reduction in ['sum', 'mean'], 'Reduction parameter must be either "sum" or "mean"'

        self.reduction = reduction
        self.task_weights = task_weights
        self.criterion = nn.CrossEntropyLoss(ignore_index=-1)

    def forward(self, outputs, targets):
        total_loss = 0.0
        n_tasks = len(LABEL_SIZES)

        if self.task_weights is None:
            device = outputs[0].device
            self.task_weights = torch.ones(n_tasks).to(device)

        for i in range(n_tasks):
            outputs_i = outputs[i].transpose(1, 2)
            loss = self.criterion(outputs_i, targets[:, i])
            total_loss += self.task_weights[i] * loss

        if self.reduction == 'mean':
            total_loss /= self.task_weights.sum()

        return total_loss
