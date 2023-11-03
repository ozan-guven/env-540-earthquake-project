from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvAutoencoder(nn.Module):
    def __init__(self, encoder_channels: List[int], decoder_channels: List[int]):
        """
        Initialize the convolutional autoencoder.
        
        Args:
            encoder_channels (List[int]): list of channel sizes for the encoder, 
                                          including the input channel size.
            decoder_channels (List[int]): list of channel sizes for the decoder, 
                                          including the output channel size.
        """
        super(ConvAutoencoder, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential()
        for i in range(len(encoder_channels) - 1):
            self.encoder.add_module(
                f"enc_conv{i}",
                nn.Conv2d(encoder_channels[i], encoder_channels[i+1], kernel_size=3, stride=2, padding=1)
            )
            self.encoder.add_module(f"enc_relu{i}", nn.ReLU(True))
            self.encoder.add_module(f"enc_batchnorm{i}", nn.BatchNorm2d(encoder_channels[i+1]))

        # Decoder
        self.decoder = nn.Sequential()
        for i in range(len(decoder_channels) - 1):
            self.decoder.add_module(
                f"dec_convtrans{i}",
                nn.ConvTranspose2d(decoder_channels[i], decoder_channels[i+1], kernel_size=3, stride=2, padding=1, output_padding=1)
            )
            self.decoder.add_module(f"dec_relu{i}", nn.ReLU(True))
            self.decoder.add_module(f"dec_batchnorm{i}", nn.BatchNorm2d(decoder_channels[i+1]))
        
        # Last layer of decoder without ReLU to allow for the full range of pixel values
        self.decoder.add_module(
            "dec_final",
            nn.ConvTranspose2d(decoder_channels[-2], decoder_channels[-1], kernel_size=3, stride=2, padding=1, output_padding=1)
        )
        self.decoder.add_module("dec_final_activation", nn.Sigmoid())  # Assuming the input images are normalized between [0, 1]

    def forward(self, x):
        x = self.encoder(x)
        print(x.shape)
        x = self.decoder(x)
        return x