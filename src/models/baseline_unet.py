"""
Baseline U-Net for NeuralILT (target layout -> optimized mask).

This is a standard 4-level U-Net with skip connections, similar to
what LithoBench uses for their NeuralILT experiments [Zheng et al., 2023].

Architecture:
    Encoder: 1 -> 64 -> 128 -> 256 -> 512 (with maxpool between levels)
    Bottleneck: 512 -> 1024
    Decoder: mirrors encoder with transposed convs + skip connections
    Output: 1x1 conv -> sigmoid (mask values in [0,1])

We use this as our baseline to compare against the DS-CNN version.
"""

import torch
import torch.nn as nn

from src.models.blocks import DoubleConv
from src.models.constants import DEFAULT_FEATURES


class UNet(nn.Module):

    def __init__(self, in_channels=1, out_channels=1, features=None):
        super().__init__()

        if features is None:
            features = DEFAULT_FEATURES

        # encoder
        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()
        ch = in_channels
        for f in features:
            self.encoders.append(DoubleConv(ch, f))
            self.pools.append(nn.MaxPool2d(2))
            ch = f

        # bottleneck
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)

        # decoder
        self.upconvs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        reversed_features = list(reversed(features))
        ch = features[-1] * 2
        for f in reversed_features:
            self.upconvs.append(nn.ConvTranspose2d(ch, f, 2, stride=2))
            self.decoders.append(DoubleConv(f * 2, f))  # *2 because of skip concat
            ch = f

        self.final = nn.Conv2d(features[0], out_channels, 1)

    def forward(self, x):
        # save encoder outputs for skip connections
        skip_connections = []
        for enc, pool in zip(self.encoders, self.pools):
            x = enc(x)
            skip_connections.append(x)
            x = pool(x)

        x = self.bottleneck(x)

        # decoder with skip connections (reverse order)
        skip_connections = skip_connections[::-1]
        for up, dec, skip in zip(self.upconvs, self.decoders, skip_connections):
            x = up(x)
            # handle odd-sized feature maps
            if x.shape != skip.shape:
                x = nn.functional.interpolate(x, size=skip.shape[2:],
                                              mode='bilinear', align_corners=False)
            x = torch.cat([x, skip], dim=1)
            x = dec(x)

        return torch.sigmoid(self.final(x))
