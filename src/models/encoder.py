from typing import List

import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(
            self,
            conv_channels: List[List[int]] = [[3, 16, 16], [16, 32, 32], [32, 64, 64, 64], [64, 128, 128, 128]],
            activation = nn.ReLU(inplace=True),
            dropout_rate: float = 0.0,
            return_skipped_connections: bool = True
        ):
        super().__init__()

        self.return_skipped_connections = return_skipped_connections
        
        self.encoder = nn.Sequential()

        # Convolutional Layers
        for i, channels in enumerate(conv_channels):
            for index, (in_conv, out_conv) in enumerate(zip(channels, channels[1:])):
                self.encoder.add_module(f'conv{i}_{index}', nn.Conv2d(in_conv, out_conv, kernel_size=3, stride=1, padding=1))
                self.encoder.add_module(f'bn{i}_{index}', nn.BatchNorm2d(out_conv))
                self.encoder.add_module(f'dropout{i}_{index}', nn.Dropout(dropout_rate))
                self.encoder.add_module(f'activation{i}_{index}', activation)

            self.encoder.add_module(f'maxpool{i}', nn.MaxPool2d(kernel_size=2, stride=2, padding=0))

    def forward(self, x: torch.tensor) -> torch.tensor:
        skipped_outputs = []
        for module in self.encoder:
            if isinstance(module, nn.MaxPool2d):
                skipped_outputs.append(x)
            x = module(x)

        if self.return_skipped_connections:
            return x, skipped_outputs
        else:
            return x