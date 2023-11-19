from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


class SiameseEncoder(nn.Module):
    def __init__(
            self, 
            input_channels: int = 3, 
            conv_channels: List[int] = [16, 16, 32, 32, 64, 64, 64, 128, 128, 128],
            max_pool_indices: List[int] = [1, 3, 6, 9],
            activation = nn.ReLU(inplace=True),
            dropout_rate: float = 0.0
        ):
        super().__init__()

        self.pipeline = nn.Sequential()

        # Convolutional Layers
        in_channels = input_channels
        for index, out_channels in enumerate(conv_channels):
            self.pipeline.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1))
            self.pipeline.append(nn.BatchNorm2d(out_channels))
            self.pipeline.append(nn.Dropout(dropout_rate))
            self.pipeline.append(activation)

            if index in max_pool_indices:
                self.pipeline.append(nn.MaxPool2d(kernel_size=2, stride=2, padding=0))
                
            in_channels = out_channels

    def forward(self, x: torch.tensor) -> torch.tensor:
        return self.pipeline(x)

class SiameseBranch(nn.Module):
    def __init__(
            self, 
            input_channels: int = 3, 
            conv_channels: List[int] = [16, 16, 32, 32, 64, 64, 64, 128, 128, 128],
            max_pool_indices: List[int] = [1, 3, 6, 9],
            embedding_size: int = 128,
            activation: nn.Module = nn.ReLU(inplace=True),
            dropout_rate: float = 0.0
        ):
        super().__init__()

        # Siamese Encoder
        self.encoder = SiameseEncoder(input_channels, conv_channels, max_pool_indices, activation, dropout_rate)
        self.activation = activation
        self.dropout = nn.Dropout(dropout_rate)

        # Embedding Layer
        self.embedding = nn.Linear(conv_channels[-1], embedding_size)

    def forward(self, x):
        # Encoding
        x = self.encoder(x)
        # Global Average Pooling and flatten the output
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1) 

        # Embedding
        x = self.embedding(x)

        return x
    
class SiameseNetwork(nn.Module):
    def __init__(
            self,
            input_channels: int = 3, 
            conv_channels: List[int] = [16, 16, 32, 32, 64, 64, 64, 128, 128, 128],
            max_pool_indices: List[int] = [1, 3, 6, 9],
            embedding_size: int = 128,
            activation: nn.Module = nn.ReLU(inplace=True),
            dropout_rate: float = 0.0
        ):
        super().__init__()

        # Siamese Branch
        self.branch = SiameseBranch(input_channels, conv_channels, max_pool_indices, embedding_size, activation, dropout_rate)

    def forward(self, pre, post):
        # Forward pass on each branch
        pre_embeddings = self.branch(pre)
        post_embeddings = self.branch(post)

        return pre_embeddings, post_embeddings