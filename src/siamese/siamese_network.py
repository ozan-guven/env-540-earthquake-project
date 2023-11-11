import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

class SiameseNetwork(nn.Module):
    def __init__(
            self, 
            image_size = 1024,
            input_channels = 3, 
            conv_channels = [8, 16, 32, 64, 128], 
            fc_layers = [512, 128], 
            activation = nn.ReLU(inplace=True),
            embedding_size = 50
        ):
        super(SiameseNetwork, self).__init__()

        self.conv_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()
        self.fc_layers = nn.ModuleList()
        self.activation = activation

        # Convolutional Layers
        in_channels = input_channels
        for out_channels in conv_channels:
            self.conv_layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=5, stride=2, padding=2))
            self.bn_layers.append(nn.BatchNorm2d(out_channels))
            in_channels = out_channels

        # Pooling Layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # Calculate the flat size after convolutions and pooling
        flat_size = self._get_conv_output(torch.zeros(1, input_channels, image_size, image_size))

        # Fully Connected Layers
        in_features = flat_size
        for out_features in fc_layers:
            self.fc_layers.append(nn.Linear(in_features, out_features))
            in_features = out_features

        # Embedding Layer
        self.emb_layer = nn.Linear(in_features, embedding_size)

    def _get_conv_output(self, input_shape):
        output = input_shape
        for conv, bn in zip(self.conv_layers, self.bn_layers):
            output = self.pool(bn(conv(output)))
        return int(np.prod(output.size()[1:]))
    
    def forward(self, x):
        # Forward pass for one leg of the Siamese network
        for conv, bn in zip(self.conv_layers, self.bn_layers):
            x = self.pool(self.activation(bn(conv(x))))
        
        # Flatten the output
        x = x.view(x.size(0), -1)

        # Fully connected layers
        for fc in self.fc_layers:
            x = self.activation(fc(x))

        x = self.emb_layer(x)
        return x