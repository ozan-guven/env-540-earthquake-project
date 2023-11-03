from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


class Autoencoder(nn.Module):
    """
    Autoencoder class.
    """
    def __init__(self, layer_sizes: List[int]):
        """Initialize the autoencoder.

        Args:
            layer_sizes (List[int]): list of layer sizes for the encoder and decoder. The first half of the list is used for the encoder, the second half for the decoder.

        Raises:
            ValueError: if the layer sizes list is not even
        """
        super(Autoencoder, self).__init__()
        
        # Check if the layer sizes list is even
        if len(layer_sizes) % 2 != 0:
            raise ValueError("❌ Layer sizes list should contain an even number of elements")
        
        # Define the encoder part
        self.encoder_layers = nn.ModuleList()
        for i in range(len(layer_sizes) // 2 - 1):
            self.encoder_layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            self.encoder_layers.append(nn.ReLU(True))
        self.encoder_layers.append(nn.Linear(layer_sizes[len(layer_sizes)//2 - 1], layer_sizes[len(layer_sizes)//2]))
        
        # Define the decoder part
        self.decoder_layers = nn.ModuleList()
        for i in range(len(layer_sizes) // 2, len(layer_sizes) - 1):
            self.decoder_layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            self.decoder_layers.append(nn.ReLU(True))
        self.decoder_layers.append(nn.Linear(layer_sizes[-2], layer_sizes[-1]))
        
    def forward(self, x):
        # Encoder
        for layer in self.encoder_layers:
            x = layer(x)
        
        # Decoder
        for layer in self.decoder_layers[:-1]:
            x = layer(x)
        # Apply sigmoid to the output layer to scale the output between 0 and 1
        x = torch.sigmoid(self.decoder_layers[-1](x))
        
        return x
