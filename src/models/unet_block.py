from __future__ import annotations

import warnings
from collections.abc import Sequence

import torch
import torch.nn as nn

from monai.networks.layers.factories import Act, Norm

from conv_moudules import get_conv_layer, get_norm_layer, get_act_layer

class Decoder(nn.Module):
    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        kernel_size: Sequence[int] | int,
        stride: int,
        norm_name: tuple | str,
        act_name: tuple | str = ("leakyrelu", {"inplace": True, "negative_slope": 0.01}),
        dropout: tuple | str | float | None = None,
    ):
        super().__init__()
        self.conv1 = get_conv_layer(
            spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            norm=norm_name,
            act=act_name,
            dropout=dropout,
            conv_only=False,
            is_transposed=True,
        )

    def forward(self, x, skip):
        x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        
        return x


class Encoder(nn.Module):
    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        kernel_size: Sequence[int] | int,
        stride: int,
        norm_name: tuple | str,
        act_name: tuple | str = ("leakyrelu", {"inplace": True, "negative_slope": 0.01}),
        dropout: tuple | str | float | None = None,
    ):
        super().__init__()
        self.conv = get_conv_layer(
            spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            norm=norm_name,
            act=act_name,
            dropout=dropout,
            conv_only=False,
            is_transposed=False,
        )

    def forward(self, x):
        return self.conv(x)