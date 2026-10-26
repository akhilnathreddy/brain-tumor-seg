"""The three architectures compared in this project.

- unet:      plain U-Net (Ronneberger et al. 2015), trained from scratch
- attunet:   Attention U-Net (Oktay et al. 2018): same U-Net, with attention gates
             that let the decoder ignore skip-connection features outside the tumor
- resunet34: U-Net with an ImageNet-pretrained ResNet34 encoder
             (segmentation_models_pytorch); its first conv is adapted to 4 channels

All take (B, 4, H, W) and return logits (B, 3, H, W) for WT, TC, ET.
"""
import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """(conv 3x3 -> BatchNorm -> ReLU) x 2"""

    def __init__(self, cin, cout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class AttentionGate(nn.Module):
    """Weights skip features by how relevant they look given the coarser decoder signal."""

    def __init__(self, c_skip, c_gate, c_mid):
        super().__init__()
        self.w_skip = nn.Sequential(nn.Conv2d(c_skip, c_mid, 1, bias=False), nn.BatchNorm2d(c_mid))
        self.w_gate = nn.Sequential(nn.Conv2d(c_gate, c_mid, 1, bias=False), nn.BatchNorm2d(c_mid))
        self.psi = nn.Sequential(nn.Conv2d(c_mid, 1, 1), nn.BatchNorm2d(1), nn.Sigmoid())

    def forward(self, skip, gate):
        alpha = self.psi(torch.relu(self.w_skip(skip) + self.w_gate(gate)))  # (B,1,H,W) in [0,1]
        return skip * alpha


class UNet(nn.Module):
    """4-level U-Net. `attention=True` turns it into an Attention U-Net."""

    def __init__(self, in_ch=4, out_ch=3, base=32, attention=False, dropout=0.1):
        super().__init__()
        ch = [base * 2 ** i for i in range(5)]           # 32, 64, 128, 256, 512
        self.downs = nn.ModuleList([ConvBlock(in_ch, ch[0])] +
                                   [ConvBlock(ch[i], ch[i + 1]) for i in range(3)])
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = nn.Sequential(ConvBlock(ch[3], ch[4]), nn.Dropout2d(dropout))
        self.ups = nn.ModuleList([nn.ConvTranspose2d(ch[i + 1], ch[i], 2, stride=2) for i in reversed(range(4))])
        self.gates = nn.ModuleList([AttentionGate(ch[i], ch[i], ch[i] // 2) for i in reversed(range(4))]) \
            if attention else None
        self.decs = nn.ModuleList([ConvBlock(ch[i] * 2, ch[i]) for i in reversed(range(4))])
        self.head = nn.Conv2d(ch[0], out_ch, 1)

    def forward(self, x):
        skips = []
        for down in self.downs:
            x = down(x)
            skips.append(x)
            x = self.pool(x)
        x = self.bottleneck(x)
        for i, (up, dec) in enumerate(zip(self.ups, self.decs)):
            x = up(x)
            skip = skips[-1 - i]
            if self.gates is not None:
                skip = self.gates[i](skip, x)
            x = dec(torch.cat([skip, x], 1))
        return self.head(x)


def build_model(name, pretrained=True):
    if name == "unet":
        return UNet(attention=False)
    if name == "attunet":
        return UNet(attention=True)
    if name == "resunet34":
        import segmentation_models_pytorch as smp
        # smp builds a 4-channel first conv from the pretrained RGB weights
        return smp.Unet("resnet34", encoder_weights="imagenet" if pretrained else None,
                        in_channels=4, classes=3)
    raise ValueError(f"unknown model {name!r}")


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
