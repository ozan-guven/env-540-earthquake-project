# From: https://www.kaggle.com/code/bigironsphere/loss-function-library-keras-pytorch
import torch
import torch.nn as nn
import torch.nn.functional as F


class IoULoss(nn.Module):
    """
    Intersection over Union (IoU) loss.
    """

    def __init__(self, weight: torch.Tensor = None, size_average: bool = True) -> None:
        """
        Initialize IoU loss.

        Args:
            weight (torch.Tensor, optional): The weight, defaults to None
            size_average (bool, optional): Whether to average the loss, defaults to True
        """
        super(IoULoss, self).__init__()

    def forward(
            self, inputs: torch.Tensor, targets: torch.Tensor, smooth: int = 1
        ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            inputs (torch.Tensor): The inputs
            targets (torch.Tensor): The targets
            smooth (int, optional): The smoothing factor, defaults to 1

        Returns:
            torch.Tensor: The loss
        """
        inputs = F.sigmoid(inputs).type(torch.float32)
        targets = targets.type(torch.float32)

        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        # intersection is equivalent to True Positive count
        # union is the mutually inclusive area of all labels & predictions
        intersection = (inputs * targets).sum()
        total = (inputs + targets).sum()
        union = total - intersection

        IoU = (intersection + smooth) / (union + smooth)

        return 1 - IoU
