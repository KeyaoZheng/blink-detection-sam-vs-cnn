
# seg/unet_tiny.py
import torch, torch.nn as nn

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(True)
        )
    def forward(self, x): return self.net(x)

class UNetTiny(nn.Module):
    def __init__(self, ch=16):
        super().__init__()
        self.down1 = DoubleConv(3, ch)
        self.pool1 = nn.MaxPool2d(2)
        self.down2 = DoubleConv(ch, ch*2); self.pool2 = nn.MaxPool2d(2)
        self.down3 = DoubleConv(ch*2, ch*4); self.pool3 = nn.MaxPool2d(2)
        self.mid  = DoubleConv(ch*4, ch*8)
        self.up3  = nn.ConvTranspose2d(ch*8, ch*4, 2, stride=2)
        self.dec3 = DoubleConv(ch*8, ch*4)
        self.up2  = nn.ConvTranspose2d(ch*4, ch*2, 2, stride=2)
        self.dec2 = DoubleConv(ch*4, ch*2)
        self.up1  = nn.ConvTranspose2d(ch*2, ch, 2, stride=2)
        self.dec1 = DoubleConv(ch*2, ch)
        self.outc = nn.Conv2d(ch, 1, 1)
    def forward(self, x):
        d1 = self.down1(x); p1 = self.pool1(d1)
        d2 = self.down2(p1); p2 = self.pool2(d2)
        d3 = self.down3(p2); p3 = self.pool3(d3)
        m  = self.mid(p3)
        u3 = self.up3(m);  x3 = torch.cat([u3, d3], dim=1); x3 = self.dec3(x3)
        u2 = self.up2(x3); x2 = torch.cat([u2, d2], dim=1); x2 = self.dec2(x2)
        u1 = self.up1(x2); x1 = torch.cat([u1, d1], dim=1); x1 = self.dec1(x1)
        return self.outc(x1)
