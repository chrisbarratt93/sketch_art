"""TEED: Tiny and Efficient Edge Detector (Soria et al., ICCV-W 2023).

A 58K-parameter network trained on BIPED, a dataset of human-annotated edges
in urban street photographs. It returns an edge probability map that ignores
most texture (stone blocks, paving, cloud) that a gradient detector picks up.

Model definition condensed from https://github.com/xavysp/TEED (MIT License,
Copyright (c) 2022 Xavier Soria Poma). Layer names are kept so the published
weights load unchanged. Weights are downloaded on first use, see `load()`.
"""

import urllib.request
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn

WEIGHTS_URL = "https://raw.githubusercontent.com/xavysp/TEED/main/checkpoints/BIPED/5/5_model.pth"
CACHE = Path(__file__).resolve().parent.parent / "models"
BIPED_MEAN_BGR = np.array([103.939, 116.779, 123.68], np.float32)


def smish(x):
    return x * torch.tanh(torch.log(1 + torch.sigmoid(x)))


class Smish(nn.Module):
    def forward(self, x):
        return smish(x)


class DoubleFusion(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.DWconv1 = nn.Conv2d(in_ch, in_ch * 8, 3, padding=1, groups=in_ch)
        self.DWconv2 = nn.Conv2d(24, 24, 3, padding=1, groups=24)

    def forward(self, x):
        attn = self.DWconv1(smish(x))
        attn2 = self.DWconv2(smish(attn))
        return smish((attn2 + attn).sum(1, keepdim=True))


class DenseLayer(nn.Sequential):
    def __init__(self, cin, cout):
        super().__init__()
        self.add_module("conv1", nn.Conv2d(cin, cout, 3, padding=2))
        self.add_module("smish1", Smish())
        self.add_module("conv2", nn.Conv2d(cout, cout, 3))

    def forward(self, x):
        x1, x2 = x
        return 0.5 * (super().forward(smish(x1)) + x2), x2


class DenseBlock(nn.Sequential):
    def __init__(self, cin, cout):
        super().__init__()
        self.add_module("denselayer1", DenseLayer(cin, cout))


class UpConvBlock(nn.Module):
    def __init__(self, cin, up_scale):
        super().__init__()
        pad = [0, 0, 1, 3, 7][up_scale]
        layers = []
        for i in range(up_scale):
            cout = 1 if i == up_scale - 1 else 16
            layers += [nn.Conv2d(cin, cout, 1), Smish(),
                       nn.ConvTranspose2d(cout, cout, 2 ** up_scale, stride=2, padding=pad)]
            cin = cout
        self.features = nn.Sequential(*layers)

    def forward(self, x):
        return self.features(x)


class SingleConvBlock(nn.Module):
    def __init__(self, cin, cout, stride):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, 1, stride=stride)

    def forward(self, x):
        return self.conv(x)


class DoubleConvBlock(nn.Module):
    def __init__(self, cin, cmid, cout=None, stride=1, use_act=True):
        super().__init__()
        self.use_act = use_act
        self.conv1 = nn.Conv2d(cin, cmid, 3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(cmid, cout or cmid, 3, padding=1)

    def forward(self, x):
        x = self.conv2(smish(self.conv1(x)))
        return smish(x) if self.use_act else x


class TED(nn.Module):
    def __init__(self):
        super().__init__()
        self.block_1 = DoubleConvBlock(3, 16, 16, stride=2)
        self.block_2 = DoubleConvBlock(16, 32, use_act=False)
        self.dblock_3 = DenseBlock(32, 48)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        self.side_1 = SingleConvBlock(16, 32, 2)
        self.pre_dense_3 = SingleConvBlock(32, 48, 1)
        self.up_block_1 = UpConvBlock(16, 1)
        self.up_block_2 = UpConvBlock(32, 1)
        self.up_block_3 = UpConvBlock(48, 2)
        self.block_cat = DoubleFusion(3)

    def forward(self, x):
        b1 = self.block_1(x)
        b2 = self.block_2(b1)
        b2_down = self.maxpool(b2)
        b3, _ = self.dblock_3([b2_down + self.side_1(b1), self.pre_dense_3(b2_down)])
        outs = [self.up_block_1(b1), self.up_block_2(b2), self.up_block_3(b3)]
        return self.block_cat(torch.cat(outs, dim=1))


_model = None


def load():
    global _model
    if _model is None:
        path = CACHE / "teed_biped_5.pth"
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(WEIGHTS_URL, path)
        _model = TED()
        _model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
        _model.eval()
    return _model


def edge_probability(bgr, scale=1.0):
    """Edge probability (0..1) at the size of `bgr`. `scale` runs the network
    on a resized copy: >1 finds finer edges, <1 only the main structure."""
    h, w = bgr.shape[:2]
    H = int(round(h * scale / 16)) * 16
    W = int(round(w * scale / 16)) * 16
    img = cv2.resize(bgr, (W, H), interpolation=cv2.INTER_CUBIC).astype(np.float32) - BIPED_MEAN_BGR
    x = torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1)[None]))
    with torch.no_grad():
        prob = torch.sigmoid(load()(x))[0, 0].numpy()
    return cv2.resize(prob, (w, h), interpolation=cv2.INTER_LINEAR)
