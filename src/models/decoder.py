from typing import List

import torch.nn as nn


class Decoder(nn.Module):
    def __init__(self,
                 conv_channels: List[List[int]] = [[256, 128, 128, 64], [128, 64, 64, 32], [64, 32, 16], [32, 16, 1]],
                 activation = nn.ReLU(inplace=True),
                 n: int = 2,
                 dropout_rate: float = 0.0
                 ):
        super().__init__()
        
        self.decoder = nn.Sequential()
        
        for i, convs in enumerate(conv_channels):
            trans = convs[0] // n
            self.decoder.add_module(f"conv_t{i}", nn.ConvTranspose2d(trans, trans, kernel_size=2, stride=2))
            for index, (in_conv, out_conv) in enumerate(zip(convs, convs[1:])):
                self.decoder.add_module(f"conv{i}_{index}", nn.Conv2d(in_conv, out_conv, kernel_size=3, padding=1))
                self.decoder.add_module(f"bn{i}_{index}", nn.BatchNorm2d(out_conv))
                self.decoder.add_module(f"activation{i}_{index}", activation)
                self.decoder.add_module(f"dropout{i}_{index}", nn.Dropout2d(dropout_rate))
    
    def forward(self, x):
        return self.decoder(x) # Not used