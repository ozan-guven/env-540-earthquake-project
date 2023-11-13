from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

class SiameseEncoder(nn.Module):
    def __init__(
            self, 
            input_channels: int = 3, 
            conv_channels: List[int] = [16, 32, 64, 128, 256, 512],
            activation = nn.ReLU(inplace=True),
            dropout_rate: float = 0.0
        ):
        super().__init__()

        self.conv_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.activation = activation
        self.dropout = nn.Dropout(dropout_rate)

        # Convolutional Layers
        in_channels = input_channels
        for out_channels in conv_channels:
            self.conv_layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=5, stride=1, padding=2))
            self.bn_layers.append(nn.BatchNorm2d(out_channels))
            in_channels = out_channels

    def forward(self, x: torch.tensor) -> torch.tensor:
        # Convolutional Layers
        for conv, bn in zip(self.conv_layers, self.bn_layers):
            x = conv(x)
            x = bn(x)
            self.activation(x)
            x = self.dropout(x)
            x = self.pool(x)

        return x

class SiameseBranch(nn.Module):
    def __init__(
            self, 
            input_channels: int = 3, 
            conv_channels: List[int] = [16, 32, 64, 128, 256, 512], 
            embedding_size: int = 128,
            activation: nn.Module = nn.ReLU(inplace=True),
            dropout_rate: float = 0.0
        ):
        super().__init__()

        # Siamese Encoder
        self.encoder = SiameseEncoder(input_channels, conv_channels, activation, dropout_rate)
        self.activation = activation
        self.dropout = nn.Dropout(dropout_rate)

        # Embedding Layer
        self.embedding = nn.Linear(conv_channels[-1], embedding_size)

    def forward(self, x):
        # Encoding
        x = self.encoder(x)

        # Global Average Pooling abd latten the output
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1) 

        # Embedding
        x = self.embedding(x)
        x = self.activation(x)
        x = self.dropout(x)

        return x
    
class SiameseNetwork(nn.Module):
    def __init__(
            self,
            input_channels: int = 3, 
            conv_channels: List[int] = [16, 32, 64, 128, 256, 512], 
            embedding_size: int = 128,
            activation: nn.Module = nn.ReLU(inplace=True),
            dropout_rate: float = 0.0
        ):
        super().__init__()

        # Siamese Branch
        self.branch = SiameseBranch(input_channels, conv_channels, embedding_size, activation, dropout_rate)

    def forward(self, pre, post):
        # Forward pass on each branch
        pre_embeddings = self.branch(pre)
        post_embeddings = self.branch(post)

        return pre_embeddings, post_embeddings