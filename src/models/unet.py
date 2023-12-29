# This file contains the UNet architecture.

from typing import List

import torch
import torch.nn as nn
from src.models.decoder import Decoder
from src.models.encoder import Encoder


class UNet(nn.Module):
    """
    UNet architecture.
    """

    def __init__(
        self,
        encoder_channels: List[List[int]] = [
            [6, 16, 16],
            [16, 32, 32],
            [32, 64, 64, 64],
            [64, 128, 128, 128],
        ],
        decoder_channels: List[List[int]] = [
            [256, 128, 128, 64],
            [128, 64, 64, 32],
            [64, 32, 16],
            [32, 16, 1],
        ],
        activation=nn.ReLU(inplace=True),
        dropout_rate: float = 0.0,
    ) -> None:
        """
        Initialize the UNet architecture.

        Args:
            encoder_channels (List[List[int]]): Each list of integers represents the number of channels for each convolutional layer in the encoder
            decoder_channels (List[List[int]]): Each list of integers represents the number of channels for each convolutional layer in the decoder
            activation (nn.Module): The activation function
            dropout_rate (float): The dropout rate
        """
        super().__init__()
        self.encoder_channels = encoder_channels
        self.decoder_channels = decoder_channels
        self.activation = activation
        self.dropout_rate = dropout_rate

        self.encoder = Encoder(
            conv_channels=encoder_channels,
            activation=activation,
            dropout_rate=dropout_rate,
        )

        self.decoder = Decoder(
            conv_channels=decoder_channels,
            n=2,
            activation=activation,
            dropout_rate=dropout_rate,
        )

    def forward(self, pre: torch.Tensor, post: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            pre (torch.Tensor): The pre-fire image
            post (torch.Tensor): The post-fire image

        Returns:
            x (torch.Tensor): The output tensor
        """
        # Concatenate pre and post images along the channel dimension
        x = torch.cat([pre, post], dim=1)

        # Encoding
        x, skip_outputs = self.encoder(x)

        # Decoding with skip connections
        for module in self.decoder.decoder:
            x = module(x)
            if isinstance(module, nn.ConvTranspose2d):
                x = torch.cat([x, skip_outputs.pop()], dim=1)

        return x
