# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import numpy as np
from timm.models.layers import DropPath
from torch.nn import Dropout, Conv2d
from torch.nn.modules.utils import _pair


class Embeddings(nn.Module):
    # Construct the patch, position embeddings
    def __init__(self, config, patch_size, img_size, in_channels):
        super().__init__()
        img_size = _pair(img_size)
        patch_size = _pair(patch_size)
        n_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])
        self.patch_embeddings = Conv2d(in_channels=in_channels,
                                       out_channels=in_channels,
                                       kernel_size=patch_size,
                                       stride=patch_size)
        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches, in_channels))
        self.dropout = Dropout(0.1)

    def forward(self, x):
        if x is None:
            return None
        x = self.patch_embeddings(x)  # (B, hidden. n_patches^(1/2), n_patches^(1/2))
        x = x.flatten(2)
        x = x.transpose(-1, -2)  # (B, n_patches, hidden)
        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings

class ConvTransBN(nn.Module):  # (convolution => [BN] => ReLU)
    def __init__(self, in_channels, out_channels):
        super(ConvTransBN, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm = nn.BatchNorm1d(out_channels)
        self.activation = nn.ReLU()

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        return self.activation(out)

class Interactor(nn.Module):  # Transformer-branch
    def __init__(self, config, vis, img_size, channel_num, patch_size, embed_dim, depth=1, num_heads=8,
                 mlp_ratio=4., qkv_bias=True, num_classes=1, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.):
        super(Interactor, self).__init__()
        self.config = config
        self.embeddings = Embeddings(config=config, patch_size=patch_size, img_size=img_size, in_channels=channel_num)
        self.CTBN = ConvTransBN(in_channels=embed_dim, out_channels=embed_dim)
        self.attn_text = nn.MultiheadAttention(embed_dim, num_heads//2, dropout=0.1)
        self.attn_img = nn.MultiheadAttention(embed_dim, num_heads//2, dropout=0.1)

    def forward(self, img, text_feat, reconstruct=False):
        # b,c,h,w = img.shape
        img_feat = self.embeddings(img) # b, n, c
        text_feat = self.attn_text(text_feat.permute(1,0,2), img_feat.permute(1,0,2), img_feat.permute(1,0,2))[0].permute(1,0,2)
        img_feat = self.attn_img(img_feat.permute(1,0,2), text_feat.permute(1,0,2), text_feat.permute(1,0,2))[0].permute(1,0,2)

        text_feat = text_feat.transpose(1, 2)  # [B, c, n]
        text_feat = self.CTBN(text_feat)  # [B, c*2, n]
        text_feat = text_feat.transpose(1, 2)  # [B, n, c*2]

        return text_feat, img_feat