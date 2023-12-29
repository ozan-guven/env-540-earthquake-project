# This file contains the Siamese model.

import sys

sys.path.append("../../")

from collections import OrderedDict
from typing import List

import torch
import torch.nn as nn

from src.models.encoder import Encoder


class SiameseBranch(nn.Module):
    """
    Siamese branch.
    """

    def __init__(
        self,
        conv_channels: List[List[int]] = [
            [3, 16, 16],
            [16, 32, 32],
            [32, 64, 64, 64],
            [64, 128, 128, 128],
        ],
        embedding_size: int = 128,
        activation: nn.Module = nn.ReLU(inplace=True),
        dropout_rate: float = 0.0,
    ):
        """
        Initialize the Siamese branch.

        Args:
            conv_channels (List[List[int]]): Each list of integers represents the number of channels for each convolutional layer
            embedding_size (int): The embedding size
            activation (nn.Module): The activation function
            dropout_rate (float): The dropout rate
        """
        super().__init__()

        self.siamese_branch = nn.Sequential(
            OrderedDict(
                [
                    (
                        "encoder",
                        Encoder(
                            conv_channels,
                            activation,
                            dropout_rate,
                            return_skipped_connections=False,
                        ),
                    ),
                    ("avg_pool", nn.AdaptiveAvgPool2d((1, 1))),
                    ("flatten", nn.Flatten()),
                    ("embedding", nn.Linear(conv_channels[-1][-1], embedding_size)),
                ]
            )
        )

    def forward(self, x: torch.tensor) -> torch.tensor:
        """
        Forward pass.

        Args:
            x (torch.tensor): The input tensor

        Returns:
            torch.tensor: The output tensor
        """
        return self.siamese_branch(x)


class Siamese(nn.Module):
    """
    Siamese model.
    """

    def __init__(
        self,
        conv_channels: List[List[int]] = [
            [3, 16, 16],
            [16, 32, 32],
            [32, 64, 64, 64],
            [64, 128, 128, 128],
        ],
        embedding_size: int = 128,
        activation: nn.Module = nn.ReLU(inplace=True),
        dropout_rate: float = 0.0,
    ):
        """
        Initialize the Siamese model.

        Args:
            conv_channels (List[List[int]]): Each list of integers represents the number of channels for each convolutional layer
            embedding_size (int): The embedding size
            activation (nn.Module): The activation function
            dropout_rate (float): The dropout rate
        """
        super().__init__()

        # Siamese Branch
        self.branch = SiameseBranch(
            conv_channels, embedding_size, activation, dropout_rate
        )

    def forward(self, pre: torch.tensor, post: torch.tensor) -> torch.tensor:
        """
        Forward pass.

        Args:
            pre (torch.tensor): The pre-event image
            post (torch.tensor): The post-event image

        Returns:
            torch.tensor: The pre-event embedding
            torch.tensor: The post-event embedding
        """
        # Forward pass on each branch
        pre_embeddings = self.branch(pre)
        post_embeddings = self.branch(post)

        return pre_embeddings, post_embeddings
