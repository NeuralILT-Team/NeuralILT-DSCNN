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
