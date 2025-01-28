# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");

from __future__ import annotations

import warnings
from collections.abc import Sequence

import torch
import torch.nn as nn

# 불필요한 import 제거
# from monai.networks.blocks.convolutions import Convolution, ResidualUnit
# from monai.networks.layers.simplelayers import SkipConnection
from monai.networks.layers.factories import Act, Norm

from unet_block import Encoder, Decoder, get_conv_layer
from cbam_kaggle import CBAM3D


class c_Decoder(nn.Module):
    def __init__(
        self,
        spatial_dims: int,
        up_channels: int,
        skip_channels: int,
        out_channels: int,
        kernel_size: Sequence[int] | int,
        stride: int,
        norm_name: tuple | str,
        act_name: tuple | str = ("leakyrelu", {"inplace": True, "negative_slope": 0.01}),
        dropout: tuple | str | float | None = None,
        conv_only=False,
    ):
        super().__init__()
        self.conv1 = get_conv_layer(
            spatial_dims,
            in_channels=up_channels+skip_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            norm=norm_name,
            act=act_name,
            dropout=dropout,
            conv_only=conv_only,
            is_transposed=True,
        )
        self.skip_cbam = CBAM3D(channels = skip_channels, reduction=8, spatial_kernel_size=3)
        self.cbam = CBAM3D(channels = out_channels, reduction=8, spatial_kernel_size=3)

    def forward(self, x, skip):
        # print(x.shape, skip.shape)
        skip = self.skip_cbam(skip)
        # print(x.shape, skip.shape)
        x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.cbam(x)
        
        return x

class UNet_CBAM_bw(nn.Module):
    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        channels: Sequence[int],
        strides: Sequence[int],
        kernel_size: Sequence[int] | int = 3,
        up_kernel_size: Sequence[int] | int = 3,
        act: tuple | str = Act.PRELU,
        norm: tuple | str = Norm.INSTANCE,
        dropout: float = 0.0,
        bias: bool = True,
        adn_ordering: str = "NDA",
    ) -> None:
        super().__init__()

        if len(channels) < 2:
            raise ValueError("the length of `channels` should be no less than 2.")
        
        # 기존 코드와 동일한 검사
        delta = len(strides) - (len(channels) - 1)
        if delta < 0:
            raise ValueError("the length of `strides` should equal `len(channels) - 1`.")
        if delta > 0:
            warnings.warn(f"`len(strides) > len(channels) - 1`, the last {delta} values of strides will not be used.")
        
        if isinstance(kernel_size, Sequence) and len(kernel_size) != spatial_dims:
            raise ValueError("the length of `kernel_size` should equal to `dimensions`.")
        if isinstance(up_kernel_size, Sequence) and len(up_kernel_size) != spatial_dims:
            raise ValueError("the length of `up_kernel_size` should equal to `dimensions`.")

        self.dimensions = spatial_dims
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.channels = channels
        self.strides = strides
        self.kernel_size = kernel_size
        self.up_kernel_size = up_kernel_size
        self.act = act
        self.norm = norm
        self.dropout = dropout
        self.bias = bias
        self.adn_ordering = adn_ordering

        # ---------------------
        # Encoder
        # ---------------------
        self.encoder1 = Encoder(
            spatial_dims,
            in_channels,
            channels[0],
            kernel_size,
            strides[0],
            norm,
            act,
            dropout,
        )
        self.encoder2 = Encoder(
            spatial_dims,
            channels[0],
            channels[1],
            kernel_size,
            strides[1],
            norm,
            act,
            dropout,
        )
        self.encoder3 = Encoder(
            spatial_dims,
            channels[1],
            channels[2],
            kernel_size,
            strides[2],
            norm,
            act,
            dropout,
        )
        # encoder4는 strides[3]를 사용할 수 있도록 strides에 4개 값을 넣어주거나, 아래처럼 stride=1로 따로 설정 가능
        # 여기서는 strides에 4개 값을 넣어준다고 가정함
        self.bottleneck = Encoder(
            spatial_dims,
            channels[2],
            channels[3],
            kernel_size,
            1, 
            norm,
            act,
            dropout,
        )

        # self.cbam = CBAM3D(channels=channels[3], reduction=8, spatial_kernel_size=3)
        # ---------------------
        # Decoder
        # ---------------------
        self.decoder3 = c_Decoder(
            spatial_dims,
            channels[3],
            channels[2],
            channels[1],
            up_kernel_size,
            strides[2],
            norm,
            act,
            dropout,
        )
        self.decoder2 = c_Decoder(
            spatial_dims,
            channels[1],
            channels[1],
            channels[0],
            up_kernel_size,
            strides[1],
            norm,
            act,
            dropout,
        )
        self.decoder1 = c_Decoder(
            spatial_dims,
            channels[0],
            channels[0],
            out_channels,
            up_kernel_size,
            strides[0],
            conv_only=True,
            norm_name=None,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.encoder1(x)
        x2 = self.encoder2(x1)
        x3 = self.encoder3(x2)
        
        x = self.bottleneck(x3)
        # x4 = self.cbam(x4)

        x = self.decoder3(x, x3)
        x = self.decoder2(x, x2)
        x = self.decoder1(x, x1)

        return x

if __name__ == "__main__":
    # strides를 4개로 늘림
    net = UNet_CBAM_bw(
        spatial_dims=3,
        in_channels=1,
        out_channels=7,
        channels=(16, 32, 64, 128),
        strides=(2, 2, 2),  # 4개의 stride
    )
    print(net)

    input_tensor = torch.randn((1, 1, 64, 64, 64))
    with torch.no_grad():
        out = net(input_tensor)
    print("Output shape:", out.shape)