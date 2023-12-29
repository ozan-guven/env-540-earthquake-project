# This file contains the Siamese UNet architecture with concatenation.

from typing import List

import torch
import torch.nn as nn
from src.models.decoder import Decoder
from src.models.encoder import Encoder


class SiameseUNetConc(nn.Module):
    """
    Siamese UNet architecture with concatenation.
    """

    def __init__(
        self,
        encoder_channels: List[List[int]] = [
            [3, 16, 16],
            [16, 32, 32],
            [32, 64, 64, 64],
            [64, 128, 128, 128],
        ],
        decoder_channels: List[List[int]] = [
            [384, 128, 128, 64],
            [192, 64, 64, 32],
            [96, 32, 16],
            [48, 16, 1],
        ],
        activation=nn.ReLU(inplace=True),
        dropout_rate: float = 0.0,
    ) -> None:
        """
        Initialize the Siamese UNet architecture.

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
            n=3,
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
        # Encoding
        pre, pre_skip_outputs = self.encoder(pre)
        x, post_skip_outputs = self.encoder(post)

        # Decoding with skip connections
        for module in self.decoder.decoder:
            x = module(x)
            if isinstance(module, nn.ConvTranspose2d):
                x = torch.cat(
                    [x, post_skip_outputs.pop(), pre_skip_outputs.pop()], dim=1
                )

        return x
