"""
Building blocks for the U-Net architectures.

We define two types of conv blocks here:
1. Standard 3x3 conv (for baseline)
2. Depthwise separable conv (for our proposed DS-CNN)

The depthwise separable version splits a standard conv into two steps:
  - depthwise: one filter per input channel (spatial only)
  - pointwise: 1x1 conv to mix channels
This is the key idea from MobileNet [Howard et al., 2017] that we're
applying to the NeuralILT architecture.
"""

import torch.nn as nn


class ConvBlock(nn.Module):
    """Standard conv -> batchnorm -> relu block."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class DSConvBlock(nn.Module):
    """
    Depthwise separable conv block.

    This is the core of our proposed architecture. Instead of doing
    a full 3x3 conv (which costs H*W*Cin*Cout*K^2 FLOPs), we split it:
      1) depthwise 3x3 conv: H*W*Cin*K^2 FLOPs (spatial filtering)
      2) pointwise 1x1 conv: H*W*Cin*Cout FLOPs (channel mixing)

    For K=3 and Cout=64, this gives roughly 8x fewer FLOPs.
    """

    def __init__(self, in_ch, out_ch):
        super().__init__()
        # depthwise: groups=in_ch means each channel gets its own filter
        self.depthwise = nn.Conv2d(in_ch, in_ch, 3, padding=1,
                                   groups=in_ch, bias=False)
        self.bn1 = nn.BatchNorm2d(in_ch)
        # pointwise: 1x1 conv to combine channels
        self.pointwise = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.bn1(self.depthwise(x)))
        x = self.relu(self.bn2(self.pointwise(x)))
        return x


class DoubleConv(nn.Module):
    """Two standard conv blocks back to back (used in baseline U-Net)."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            ConvBlock(in_ch, out_ch),
            ConvBlock(out_ch, out_ch),
        )

    def forward(self, x):
        return self.block(x)


class DoubleConvDS(nn.Module):
    """Two DS conv blocks back to back (used in proposed DS-CNN U-Net)."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            DSConvBlock(in_ch, out_ch),
            DSConvBlock(out_ch, out_ch),
        )

    def forward(self, x):
        return self.block(x)
