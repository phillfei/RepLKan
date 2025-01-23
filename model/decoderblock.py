import torch
import torch.nn as nn
import numpy as np
import math
import torch.nn.functional as F
from torch import nn, Tensor
import os
import warnings
warnings.filterwarnings('ignore')
from model.block import ConcatBlock2
def normal_init(module, mean=0, std=1, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.normal_(module.weight, mean, std)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)
def constant_init(module, val, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.constant_(module.weight, val)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)
class DySample_UP(nn.Module):
    def __init__(self, in_channels, scale=2, groups=4, dyscope=False):
        super(DySample_UP,self).__init__()
        self.scale = scale
        self.groups = groups
        in_channels = in_channels // scale ** 2
        out_channels = 2 * groups
        self.offset = nn.Conv2d(in_channels, out_channels, 1)
        normal_init(self.offset, std=0.001)
        if dyscope:
            self.scope = nn.Conv2d(in_channels, out_channels, 1)
            constant_init(self.scope, val=0.)

        self.register_buffer('init_pos', self._init_pos())
    def _init_pos(self):
        h = torch.arange((-self.scale + 1) / 2, (self.scale - 1) / 2 + 1) / self.scale
        return torch.stack(torch.meshgrid([h, h])).transpose(1, 2).repeat(1, self.groups, 1).reshape(1, -1, 1, 1)

    def sample(self, x, offset):
        B, _, H, W = offset.shape
        offset = offset.view(B, 2, -1, H, W)
        coords_h = torch.arange(H) + 0.5
        coords_w = torch.arange(W) + 0.5
        coords = torch.stack(torch.meshgrid([coords_w, coords_h])
                             ).transpose(1, 2).unsqueeze(1).unsqueeze(0).type(x.dtype).to(x.device)
        normalizer = torch.tensor([W, H], dtype=x.dtype, device=x.device).view(1, 2, 1, 1, 1)
        coords = 2 * (coords + offset) / normalizer - 1
        coords = F.pixel_shuffle(coords.view(B, -1, H, W), self.scale).view(
            B, 2, -1, self.scale * H, self.scale * W).permute(0, 2, 3, 4, 1).contiguous().flatten(0, 1)
        return F.grid_sample(x.reshape(B * self.groups, -1, H, W), coords, mode='bilinear',
                             align_corners=False, padding_mode="border").view(B, -1, self.scale * H, self.scale * W)
    def forward_pl(self, x):
        x_ = F.pixel_shuffle(x, self.scale)
        if hasattr(self, 'scope'):
            offset = F.pixel_unshuffle(self.offset(x_) * self.scope(x_).sigmoid(), self.scale) * 0.5 + self.init_pos
        else:
            offset = F.pixel_unshuffle(self.offset(x_), self.scale) * 0.25 + self.init_pos
        # i=self.sample(x, offset)
        # print("pl",i.shape)
        return self.sample(x, offset)

    def forward(self, x):

        return self.forward_pl(x)


class DecoderBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, scale, scale_factor, m_channels=None, flag=True):
        super().__init__()
        self.flag = flag
        if m_channels == None:
            self.m_channels = in_channels
        else:
            self.m_channels = m_channels
        # if scale == 0:
        self.scale_factor = scale_factor
        self.scale = scale
        self.upsample = nn.Upsample((scale, scale), mode='bilinear', align_corners=True)

        self.upsample1 = DySample_UP(self.m_channels, scale_factor)
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.cat = ConcatBlock2(in_channels)

    def forward(self, x, x_concat=None, x_concat1=None):
        _, _, w, h = x.shape
        if w != self.scale:
            if w * self.scale_factor != self.scale:
                x = self.upsample(x)
            else:
                x = self.upsample1(x)
        if x_concat is not None:
            if self.flag == False:
                x = torch.cat([x_concat, x], dim=1)
            else:
                x = self.cat(x_concat, x)
        x = self.layer(x)
        return x