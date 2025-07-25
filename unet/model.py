# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.net(x)

class UNet(nn.Module):
    def __init__(self, in_channels=4, out_channels=1):
        super().__init__()
        # 下采样
        self.dconv_down1 = DoubleConv(in_channels, 64)
        self.dconv_down2 = DoubleConv(64, 128)
        self.dconv_down3 = DoubleConv(128, 256)
        self.dconv_down4 = DoubleConv(256, 512)
        self.maxpool     = nn.MaxPool2d(2)
        # 上采样
        self.upsample    = nn.Upsample(scale_factor=2,
                                       mode='bilinear',
                                       align_corners=True)
        self.dconv_up3   = DoubleConv(256+512, 256)
        self.dconv_up2   = DoubleConv(128+256, 128)
        self.dconv_up1   = DoubleConv(128+64, 64)
        self.conv_last   = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        # down
        x1 = self.dconv_down1(x)
        x2 = self.dconv_down2(self.maxpool(x1))
        x3 = self.dconv_down3(self.maxpool(x2))
        x4 = self.dconv_down4(self.maxpool(x3))
        # up
        x  = self.upsample(x4)
        x  = torch.cat([x, x3], dim=1)
        x  = self.dconv_up3(x)
        x  = self.upsample(x)
        x  = torch.cat([x, x2], dim=1)
        x  = self.dconv_up2(x)
        x  = self.upsample(x)
        x  = torch.cat([x, x1], dim=1)
        x  = self.dconv_up1(x)
        out= self.conv_last(x)
        return torch.sigmoid(out)
