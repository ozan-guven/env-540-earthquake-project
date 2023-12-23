### OK
# This file contains the Siamese UNet architecture with difference.

from typing import List

import torch
import torch.nn as nn
from decoder import Decoder
from encoder import Encoder


class SiameseUNetDiff(nn.Module):
    """
    Siamese UNet architecture with difference.
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
            [256, 128, 128, 64],
            [128, 64, 64, 32],
            [64, 32, 16],
            [32, 16, 1],
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
        """
        # Encoding
        pre, pre_skip_outputs = self.encoder(pre)
        x, post_skip_outputs = self.encoder(post)

        # Make difference between skip outputs
        skip_outputs = []
        for pre_skip_output, post_skip_output in zip(
            pre_skip_outputs, post_skip_outputs
        ):
            skip_outputs.append(pre_skip_output - post_skip_output)

        # Decoding with skip connections
        for module in self.decoder.decoder:
            x = module(x)
            if isinstance(module, nn.ConvTranspose2d):
                x = torch.cat([x, skip_outputs.pop()], dim=1)

        return x
