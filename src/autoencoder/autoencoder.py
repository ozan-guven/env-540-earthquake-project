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
        super().__init__()
        
        # Encoder
        self.encoder = nn.Sequential()
        for i in range(len(encoder_channels) - 1):
            self.encoder.add_module(
                f"enc_conv{i}",
                nn.Conv2d(encoder_channels[i], encoder_channels[i+1], kernel_size=3, stride=2, padding=1)
            )
            self.encoder.add_module(f"enc_relu{i}", nn.ReLU(True))

        # Decoder
        self.decoder = nn.Sequential()
        for i in range(len(decoder_channels) - 1):
            # Ensure that ConvTranspose2d layers are structured to upsample correctly
            self.decoder.add_module(
                f"dec_convtrans{i}",
                nn.ConvTranspose2d(
                    decoder_channels[i],
                    decoder_channels[i+1],
                    kernel_size=3,
                    stride=2,  # This might need to be adjusted for your specific architecture
                    padding=1,
                    output_padding=1  # This might need to be adjusted as well
                )
            )
            if i < len(decoder_channels) - 2:  # Add ReLU and BatchNorm only for intermediate layers
                self.decoder.add_module(f"dec_relu{i}", nn.ReLU(True))

    def forward(self, x):
        print(self.encoder)
        print(self.decoder)
        x = self.encoder(x)
        x = self.decoder(x)
        return x