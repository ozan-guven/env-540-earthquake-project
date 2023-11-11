from typing import List

import torch.nn as nn

class ConvAutoencoder(nn.Module):
    # H_out = (H_in - 1) * stride - 2 * padding + kernel_size + output_padding
    def __init__(
        self, 
        encoder_channels: List[int], 
        decoder_channels: List[int],
        kernel_size: int = 3,
        stride: int = 2,
        padding: int = 1,
        output_padding: int = 1
        ):
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
                nn.Conv2d(encoder_channels[i], encoder_channels[i+1], kernel_size=kernel_size, stride=stride, padding=padding)
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
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    output_padding=output_padding
                )
            )
            if i < len(decoder_channels) - 2:  # Add ReLU and BatchNorm only for intermediate layers
                self.decoder.add_module(f"dec_relu{i}", nn.ReLU(True))

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x