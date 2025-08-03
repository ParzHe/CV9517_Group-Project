import torch
import torch.nn as nn


class REBNCONV(nn.Module):
    def __init__(self, in_ch=3, out_ch=3, dilation=1):
        super(REBNCONV, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=dilation, dilation=dilation),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.layer(x)


def _upsample_(src, tar):
    return nn.functional.interpolate(src, size=tar.shape[2:], mode='bilinear', align_corners=True)


class RSU1(nn.Module):
    def __init__(self, in_ch=3, inner_ch=12, out_ch=3):
        super(RSU1, self).__init__()
        self.rebnconvin = REBNCONV(in_ch, out_ch, dilation=1)

        self.rebnconv1 = REBNCONV(out_ch, inner_ch, dilation=1)

        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv2 = REBNCONV(inner_ch, inner_ch, dilation=1)

        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv3 = REBNCONV(inner_ch, inner_ch, dilation=1)

        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv4 = REBNCONV(inner_ch, inner_ch, dilation=1)

        self.pool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv5 = REBNCONV(inner_ch, inner_ch, dilation=1)

        self.pool5 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv6 = REBNCONV(inner_ch, inner_ch, dilation=1)

        self.rebnconv7 = REBNCONV(inner_ch, inner_ch, dilation=2)

        self.rebnconv6d = REBNCONV(inner_ch * 2, inner_ch, dilation=1)
        self.rebnconv5d = REBNCONV(inner_ch * 2, inner_ch, dilation=1)
        self.rebnconv4d = REBNCONV(inner_ch * 2, inner_ch, dilation=1)
        self.rebnconv3d = REBNCONV(inner_ch * 2, inner_ch, dilation=1)
        self.rebnconv2d = REBNCONV(inner_ch * 2, inner_ch, dilation=1)
        self.rebnconv1d = REBNCONV(inner_ch * 2, out_ch, dilation=1)

    def forward(self, x):
        hxin = self.rebnconvin(x)
        hx1 = self.rebnconv1(hxin)

        x = self.pool1(hx1)
        hx2 = self.rebnconv2(x)

        x = self.pool2(hx2)
        hx3 = self.rebnconv3(x)

        x = self.pool3(hx3)
        hx4 = self.rebnconv4(x)

        x = self.pool4(hx4)
        hx5 = self.rebnconv5(x)

        x = self.pool5(hx5)
        hx6 = self.rebnconv6(x)

        hx7 = self.rebnconv7(hx6)

        hx6d = self.rebnconv6d(torch.cat((hx7, hx6), dim=1))

        hx6dup = _upsample_(hx6d, hx5)
        hx5d = self.rebnconv5d(torch.cat((hx6dup, hx5), dim=1))

        hx5dup = _upsample_(hx5d, hx4)
        hx4d = self.rebnconv4d(torch.cat((hx5dup, hx4), dim=1))

        hx4dup = _upsample_(hx4d, hx3)
        hx3d = self.rebnconv3d(torch.cat((hx4dup, hx3), dim=1))

        hx3dup = _upsample_(hx3d, hx2)
        hx2d = self.rebnconv2d(torch.cat((hx3dup, hx2), dim=1))

        hx2dup = _upsample_(hx2d, hx1)
        hx1d = self.rebnconv1d(torch.cat((hx2dup, hx1), dim=1))

        return hx1d + hxin


class RSU2(nn.Module):
    def __init__(self, in_ch=3, inner_ch=12, out_ch=3):
        super(RSU2, self).__init__()
        self.rebnconvin = REBNCONV(in_ch, out_ch, dilation=1)

        self.rebnconv1 = REBNCONV(out_ch, inner_ch, dilation=1)

        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv2 = REBNCONV(inner_ch, inner_ch, dilation=1)

        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv3 = REBNCONV(inner_ch, inner_ch, dilation=1)

        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv4 = REBNCONV(inner_ch, inner_ch, dilation=1)

        self.pool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv5 = REBNCONV(inner_ch, inner_ch, dilation=1)

        self.rebnconv6 = REBNCONV(inner_ch, inner_ch, dilation=2)

        self.rebnconv5d = REBNCONV(inner_ch * 2, inner_ch, dilation=1)
        self.rebnconv4d = REBNCONV(inner_ch * 2, inner_ch, dilation=1)
        self.rebnconv3d = REBNCONV(inner_ch * 2, inner_ch, dilation=1)
        self.rebnconv2d = REBNCONV(inner_ch * 2, inner_ch, dilation=1)
        self.rebnconv1d = REBNCONV(inner_ch * 2, out_ch, dilation=1)

    def forward(self, x):
        hxin = self.rebnconvin(x)
        hx1 = self.rebnconv1(hxin)

        x = self.pool1(hx1)
        hx2 = self.rebnconv2(x)

        x = self.pool2(hx2)
        hx3 = self.rebnconv3(x)

        x = self.pool3(hx3)
        hx4 = self.rebnconv4(x)

        x = self.pool4(hx4)
        hx5 = self.rebnconv5(x)

        hx6 = self.rebnconv6(hx5)

        hx5d = self.rebnconv5d(torch.cat((hx6, hx5), dim=1))

        hx5dup = _upsample_(hx5d, hx4)
        hx4d = self.rebnconv4d(torch.cat((hx5dup, hx4), dim=1))

        hx4dup = _upsample_(hx4d, hx3)
        hx3d = self.rebnconv3d(torch.cat((hx4dup, hx3), dim=1))

        hx3dup = _upsample_(hx3d, hx2)
        hx2d = self.rebnconv2d(torch.cat((hx3dup, hx2), dim=1))

        hx2dup = _upsample_(hx2d, hx1)
        hx1d = self.rebnconv1d(torch.cat((hx2dup, hx1), dim=1))

        return hx1d + hxin


class RSU3(nn.Module):
    def __init__(self, in_ch=3, inner_ch=12, out_ch=3):
        super(RSU3, self).__init__()
        self.rebnconvin = REBNCONV(in_ch, out_ch, dilation=1)

        self.rebnconv1 = REBNCONV(out_ch, inner_ch, dilation=1)

        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv2 = REBNCONV(inner_ch, inner_ch, dilation=1)

        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv3 = REBNCONV(inner_ch, inner_ch, dilation=1)

        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv4 = REBNCONV(inner_ch, inner_ch, dilation=1)

        # self.pool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        # self.rebnconv5 = REBNCONV(inner_ch, inner_ch, dilation=1)

        self.rebnconv6 = REBNCONV(inner_ch, inner_ch, dilation=2)

        # self.rebnconv5d = REBNCONV(inner_ch * 2, inner_ch, dilation=1)
        self.rebnconv4d = REBNCONV(inner_ch * 2, inner_ch, dilation=1)
        self.rebnconv3d = REBNCONV(inner_ch * 2, inner_ch, dilation=1)
        self.rebnconv2d = REBNCONV(inner_ch * 2, inner_ch, dilation=1)
        self.rebnconv1d = REBNCONV(inner_ch * 2, out_ch, dilation=1)

    def forward(self, x):
        hxin = self.rebnconvin(x)
        hx1 = self.rebnconv1(hxin)

        x = self.pool1(hx1)
        hx2 = self.rebnconv2(x)

        x = self.pool2(hx2)
        hx3 = self.rebnconv3(x)

        x = self.pool3(hx3)
        hx4 = self.rebnconv4(x)

        hx5 = self.rebnconv6(hx4)

        hx4d = self.rebnconv4d(torch.cat((hx5, hx4), dim=1))

        hx4dup = _upsample_(hx4d, hx3)
        hx3d = self.rebnconv3d(torch.cat((hx4dup, hx3), dim=1))

        hx3dup = _upsample_(hx3d, hx2)
        hx2d = self.rebnconv2d(torch.cat((hx3dup, hx2), dim=1))

        hx2dup = _upsample_(hx2d, hx1)
        hx1d = self.rebnconv1d(torch.cat((hx2dup, hx1), dim=1))

        return hx1d + hxin


class RSU4(nn.Module):
    def __init__(self, in_ch=3, inner_ch=12, out_ch=3):
        super(RSU4, self).__init__()
        self.rebnconvin = REBNCONV(in_ch, out_ch, dilation=1)

        self.rebnconv1 = REBNCONV(out_ch, inner_ch, dilation=1)

        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv2 = REBNCONV(inner_ch, inner_ch, dilation=1)

        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv3 = REBNCONV(inner_ch, inner_ch, dilation=1)

        self.rebnconv6 = REBNCONV(inner_ch, inner_ch, dilation=2)

        self.rebnconv3d = REBNCONV(inner_ch * 2, inner_ch, dilation=1)
        self.rebnconv2d = REBNCONV(inner_ch * 2, inner_ch, dilation=1)
        self.rebnconv1d = REBNCONV(inner_ch * 2, out_ch, dilation=1)

    def forward(self, x):
        hxin = self.rebnconvin(x)
        hx1 = self.rebnconv1(hxin)

        x = self.pool1(hx1)
        hx2 = self.rebnconv2(x)

        x = self.pool2(hx2)
        hx3 = self.rebnconv3(x)

        hx4 = self.rebnconv6(hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4, hx3), dim=1))

        hx3dup = _upsample_(hx3d, hx2)
        hx2d = self.rebnconv2d(torch.cat((hx3dup, hx2), dim=1))

        hx2dup = _upsample_(hx2d, hx1)
        hx1d = self.rebnconv1d(torch.cat((hx2dup, hx1), dim=1))

        return hx1d + hxin


class RSU5F(nn.Module):
    def __init__(self, in_ch=3, inner_ch=12, out_ch=3):
        super(RSU5F, self).__init__()
        self.rebnconvin = REBNCONV(in_ch, out_ch, dilation=1)
        self.rebnconv1 = REBNCONV(out_ch, inner_ch, dilation=1)
        self.rebnconv2 = REBNCONV(inner_ch, inner_ch, dilation=2)
        self.rebnconv3 = REBNCONV(inner_ch, inner_ch, dilation=4)

        self.rebnconv4 = REBNCONV(inner_ch, inner_ch, dilation=8)

        self.rebnconv3d = REBNCONV(inner_ch * 2, inner_ch, dilation=4)
        self.rebnconv2d = REBNCONV(inner_ch * 2, inner_ch, dilation=2)
        self.rebnconv1d = REBNCONV(inner_ch * 2, out_ch, dilation=1)

    def forward(self, x):
        hxin = self.rebnconvin(x)

        hx1 = self.rebnconv1(hxin)
        hx2 = self.rebnconv2(hx1)
        hx3 = self.rebnconv3(hx2)

        hx4 = self.rebnconv4(hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4, hx3), dim=1))
        hx2d = self.rebnconv2d(torch.cat((hx3d, hx2), dim=1))
        hx1d = self.rebnconv1d(torch.cat((hx2d, hx1), dim=1))

        return hx1d + hxin

class U2net(nn.Module):
    def __init__(self, in_ch=3, out_ch=1):
        super(U2net, self).__init__()
        self.encoder1 = RSU1(in_ch, 32, 64)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.encoder2 = RSU2(64, 32, 128)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.encoder3 = RSU3(128, 64, 256)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.encoder4 = RSU4(256, 128, 512)
        self.pool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.encoder5 = RSU5F(512, 256, 512)
        self.pool5 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.encoder6 = RSU5F(512, 256, 512)

        self.decoder5 = RSU5F(1024, 256, 512)
        self.decoder4 = RSU4(1024, 128, 256)
        self.decoder3 = RSU3(512, 64, 128)
        self.decoder2 = RSU2(256, 32, 64)
        self.decoder1 = RSU1(128, 16, 64)

        self.side1 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side2 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side3 = nn.Conv2d(128, out_ch, 3, padding=1)
        self.side4 = nn.Conv2d(256, out_ch, 3, padding=1)
        self.side5 = nn.Conv2d(512, out_ch, 3, padding=1)
        self.side6 = nn.Conv2d(512, out_ch, 3, padding=1)

        self.out_conv = nn.Conv2d(6, out_ch, 1)

    def forward(self, x):
        hx1 = self.encoder1(x)
        x = self.pool1(hx1)

        hx2 = self.encoder2(x)
        x = self.pool2(hx2)

        hx3 = self.encoder3(x)
        x = self.pool3(hx3)

        hx4 = self.encoder4(x)
        x = self.pool4(hx4)

        hx5 = self.encoder5(x)
        x = self.pool5(hx5)

        hx6 = self.encoder6(x)
        hx6up = _upsample_(hx6, hx5)

        hx5d = self.decoder5(torch.cat((hx6up, hx5), dim=1))
        hx5dup = _upsample_(hx5d, hx4)

        hx4d = self.decoder4(torch.cat((hx5dup, hx4), dim=1))
        hx4dup = _upsample_(hx4d, hx3)

        hx3d = self.decoder3(torch.cat((hx4dup, hx3), dim=1))
        hx3dup = _upsample_(hx3d, hx2)

        hx2d = self.decoder2(torch.cat((hx3dup, hx2), dim=1))
        hx2dup = _upsample_(hx2d, hx1)

        hx1d = self.decoder1(torch.cat((hx2dup, hx1), dim=1))

        d1 = self.side1(hx1d)

        d2 = self.side2(hx2d)
        d2 = _upsample_(d2, d1)

        d3 = self.side3(hx3d)
        d3 = _upsample_(d3, d1)

        d4 = self.side4(hx4d)
        d4 = _upsample_(d4, d1)

        d5 = self.side5(hx5d)
        d5 = _upsample_(d5, d1)

        d6 = self.side6(hx6)
        d6 = _upsample_(d6, d1)

        d0 = self.out_conv(torch.cat((d1, d2, d3, d4, d5, d6), dim=1))

        return d0
    