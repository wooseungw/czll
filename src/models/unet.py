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

from .unet_block import Encoder, Decoder, get_conv_layer

__all__ = ["UNet"]  # "Unet" 제거



class UNet(nn.Module):
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

        # ---------------------
        # Decoder
        # ---------------------
        self.decoder3 = Decoder(
            spatial_dims,
            channels[3] + channels[2],
            channels[1],
            up_kernel_size,
            strides[2],
            norm,
            act,
            dropout,
        )
        self.decoder2 = Decoder(
            spatial_dims,
            channels[1] + channels[1],
            channels[0],
            up_kernel_size,
            strides[1],
            norm,
            act,
            dropout,
        )
        self.decoder1 = Decoder(
            spatial_dims,
            channels[0] + channels[0],
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
        
        x4 = self.bottleneck(x3)

        x = self.decoder3(x4, x3)
        x = self.decoder2(x, x2)
        x = self.decoder1(x, x1)

        return x


if __name__ == "__main__":
    # strides를 4개로 늘림
    net = UNet(
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