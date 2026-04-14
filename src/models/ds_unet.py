<<<<<<< HEAD
import torch
import torch.nn as nn

class DSConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.depthwise = nn.Conv2d(in_ch, in_ch, 3, padding=1, groups=in_ch)
        self.pointwise = nn.Conv2d(in_ch, out_ch, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.pointwise(self.depthwise(x)))


class DSUNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.enc1 = DSConv(1, 64)
        self.pool = nn.MaxPool2d(2)
        self.enc2 = DSConv(64, 128)

        self.up = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = DSConv(128, 64)

        self.out = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        x1 = self.enc1(x)
        x2 = self.enc2(self.pool(x1))

        x = self.up(x2)
        x = torch.cat([x, x1], dim=1)
        x = self.dec1(x)

        return torch.sigmoid(self.out(x))
=======
"""
Depthwise Separable U-Net (DS-UNet) for NeuralILT.

This is our proposed model. The idea is simple: take the baseline U-Net
and swap out all the standard 3x3 convolutions for depthwise separable
convolutions. Everything else stays the same — same encoder-decoder
structure, same skip connections, same training setup.

This way we can do a fair apples-to-apples comparison and isolate the
effect of the convolution type on efficiency and accuracy.

Expected improvements (from our proposal):
    - ~8x fewer FLOPs (theoretical, for 3x3 kernels)
    - 2-4x faster inference on GPU
    - SSIM within 2-5% of baseline (hypothesis)
"""

import torch
import torch.nn as nn

from src.models.blocks import DoubleConvDS


class DSUNet(nn.Module):

    def __init__(self, in_channels=1, out_channels=1, features=None):
        super().__init__()

        if features is None:
            features = [64, 128, 256, 512]

        # encoder — same structure as baseline but with DS conv blocks
        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()
        ch = in_channels
        for f in features:
            self.encoders.append(DoubleConvDS(ch, f))
            self.pools.append(nn.MaxPool2d(2))
            ch = f

        # bottleneck
        self.bottleneck = DoubleConvDS(features[-1], features[-1] * 2)

        # decoder — upsampling still uses regular ConvTranspose2d
        # (only the double-conv blocks are replaced with DS versions)
        self.upconvs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        reversed_features = list(reversed(features))
        ch = features[-1] * 2
        for f in reversed_features:
            self.upconvs.append(nn.ConvTranspose2d(ch, f, 2, stride=2))
            self.decoders.append(DoubleConvDS(f * 2, f))
            ch = f

        self.final = nn.Conv2d(features[0], out_channels, 1)

    def forward(self, x):
        skip_connections = []
        for enc, pool in zip(self.encoders, self.pools):
            x = enc(x)
            skip_connections.append(x)
            x = pool(x)

        x = self.bottleneck(x)

        skip_connections = skip_connections[::-1]
        for up, dec, skip in zip(self.upconvs, self.decoders, skip_connections):
            x = up(x)
            if x.shape != skip.shape:
                x = nn.functional.interpolate(x, size=skip.shape[2:],
                                              mode='bilinear', align_corners=False)
            x = torch.cat([x, skip], dim=1)
            x = dec(x)

        return torch.sigmoid(self.final(x))
>>>>>>> 84fc7a7ec09a2c86a67fbb72f2212df9d89cf005
