from typing import List

import torch
import torch.nn as nn
from decoder import Decoder
from encoder import Encoder


class SiameseUNetConc(nn.Module):
    def __init__(
        self, 
        encoder_channels: List[List[int]] = [[3, 16, 16], [16, 32, 32], [32, 64, 64, 64], [64, 128, 128, 128]],
        decoder_channels: List[List[int]] = [[384, 128, 128, 64], [192, 64, 64, 32], [96, 32, 16], [48, 16, 1]],
        activation = nn.ReLU(inplace=True),
        dropout_rate: float = 0.0
    ):
        super().__init__()

        self.encoder = Encoder(
            conv_channels = encoder_channels,
            activation = activation, 
            dropout_rate = dropout_rate
        )

        self.decoder = Decoder(
            conv_channels = decoder_channels,
            n = 3,
            activation = activation,
            dropout_rate = dropout_rate
        )

    def forward(self, pre, post):
        # Encoding
        pre, pre_skip_outputs = self.encoder(pre)
        x, post_skip_outputs = self.encoder(post)

        # Decoding with skip connections
        for module in self.decoder.decoder:
            x = module(x)
            if isinstance(module, nn.ConvTranspose2d):
                x = torch.cat([x, post_skip_outputs.pop(), pre_skip_outputs.pop()], dim=1)

        return x