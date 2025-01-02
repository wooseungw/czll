from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
import torch.nn as nn

from monai.networks.blocks.convolutions import Convolution
from monai.networks.layers.factories import Act, Norm
from monai.networks.layers.utils import get_act_layer, get_norm_layer

from monai.networks.blocks import UnetResBlock



def get_conv_layer(
    spatial_dims: int,
    in_channels: int,
    out_channels: int,
    kernel_size: Sequence[int] | int = 3,
    stride: Sequence[int] | int = 1,
    act: tuple | str | None = Act.PRELU,
    norm: tuple | str | None = Norm.INSTANCE,
    dropout: tuple | str | float | None = None,
    bias: bool = False,
    conv_only: bool = True,
    is_transposed: bool = False,
):
    padding = get_padding(kernel_size, stride)
    output_padding = None
    if is_transposed:
        output_padding = get_output_padding(kernel_size, stride, padding)
    return Convolution(
        spatial_dims,
        in_channels,
        out_channels,
        strides=stride,
        kernel_size=kernel_size,
        act=act,
        norm=norm,
        dropout=dropout,
        bias=bias,
        conv_only=conv_only,
        is_transposed=is_transposed,
        padding=padding,
        output_padding=output_padding,
    )


def get_padding(kernel_size: Sequence[int] | int, stride: Sequence[int] | int) -> tuple[int, ...] | int:
    kernel_size_np = np.atleast_1d(kernel_size)
    stride_np = np.atleast_1d(stride)
    padding_np = (kernel_size_np - stride_np + 1) / 2
    if np.min(padding_np) < 0:
        raise AssertionError("padding value should not be negative, please change the kernel size and/or stride.")
    padding = tuple(int(p) for p in padding_np)

    return padding if len(padding) > 1 else padding[0]


def get_output_padding(
    kernel_size: Sequence[int] | int, stride: Sequence[int] | int, padding: Sequence[int] | int
) -> tuple[int, ...] | int:
    kernel_size_np = np.atleast_1d(kernel_size)
    stride_np = np.atleast_1d(stride)
    padding_np = np.atleast_1d(padding)

    out_padding_np = 2 * padding_np + stride_np - kernel_size_np
    if np.min(out_padding_np) < 0:
        raise AssertionError("out_padding value should not be negative, please change the kernel size and/or stride.")
    out_padding = tuple(int(p) for p in out_padding_np)

    return out_padding if len(out_padding) > 1 else out_padding[0]

class Bottleneck(nn.Module):
    def __init__(self, c1, c2, norm_name, act_name, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__()
        c_ = int(c2 * e)  
        self.cv1 = get_conv_layer(3, c1, c_, kernel_size=k[0], stride=1, norm=norm_name, act=act_name) # spatial_dims를 3으로 변경
        self.cv2 = get_conv_layer(3, c_, c2, kernel_size=k[1], stride=1, norm=norm_name, act=act_name)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        y = self.cv2(self.cv1(x))
        return x + y if self.add else y
    
class CSPBlock(nn.Module):
    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        kernel_size: Sequence[int] | int,
        stride: Sequence[int] | int,
        norm_name: tuple | str,
        act_name: tuple | str = ("leakyrelu", {"inplace": True, "negative_slope": 0.01}),
        dropout: tuple | str | float | None = None,
        split_ratio: float = 0.5,
        n = 1
    ):
        super().__init__()
        self.split_channels = int(out_channels * split_ratio)
        self.conv1 = get_conv_layer(
            spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            norm=norm_name,
            act=act_name,
            dropout=dropout,
        )
        
        # First branch: conv1x1 -> Bottleneck -> Bottleneck
        self.left = nn.Sequential(
            get_conv_layer(
                spatial_dims,
                in_channels=self.split_channels,
                out_channels=self.split_channels,
                kernel_size=1,
                stride=1,
                norm=norm_name,
                act=act_name,
            ),
            *[Bottleneck(c1=self.split_channels, c2=self.split_channels, norm_name=norm_name, act_name=act_name, shortcut=True) for _ in range(n)]  # n번 반복
        )
        
        # Second branch: conv1x1
        self.right = get_conv_layer(
            spatial_dims,
            in_channels=self.split_channels,
            out_channels=self.split_channels,
            kernel_size=1,
            stride=1,
            norm=norm_name,
            act=act_name,
        )
        
        # Final transition: concatenate and conv1x1
        self.final = get_conv_layer(
            spatial_dims,
            in_channels=int(self.split_channels//split_ratio),
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            norm=norm_name,
            act=act_name,
        )

    def forward(self, x):
        x = self.conv1(x)
        x = torch.split(x, (self.split_channels,self.split_channels), dim=1)
        x1 = self.left(x[0])
        x2 = self.right(x[1])
        # print(f"x1 mean: {x1.mean().item()}, std: {x1.std().item()}")
        # print(f"x2 mean: {x2.mean().item()}, std: {x2.std().item()}")
        x = torch.cat([x1, x2], dim=1)
        return self.final(x)
    
class UPCSPBlock(nn.Module):
    """
    An upsampling module that can be used for UNETR: "Hatamizadeh et al.,
    UNETR: Transformers for 3D Medical Image Segmentation <https://arxiv.org/abs/2103.10504>"
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        kernel_size: Sequence[int] | int,
        upsample_kernel_size: Sequence[int] | int,
        norm_name: tuple | str,
        n=2
    ) -> None:
        """
        Args:
            spatial_dims: number of spatial dimensions.
            in_channels: number of input channels.
            out_channels: number of output channels.
            kernel_size: convolution kernel size.
            upsample_kernel_size: convolution kernel size for transposed convolution layers.
            norm_name: feature normalization type and arguments.
            res_block: bool argument to determine if residual block is used.

        """

        super().__init__()
        upsample_stride = upsample_kernel_size
        self.transp_conv = get_conv_layer(
            spatial_dims,
            in_channels,
            out_channels,
            kernel_size=upsample_kernel_size,
            stride=upsample_stride,
            conv_only=True,
            is_transposed=True,
        )

        self.conv_block = CSPBlock(
            spatial_dims,
            out_channels + out_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=1,
            norm_name=norm_name,
            n=n
        )
        
    def forward(self, inp, skip):
        # number of channels for skip should equals to out_channels
        out = self.transp_conv(inp)
        out = torch.cat((out, skip), dim=1)
        out = self.conv_block(out)
        return out


class UnetResBlock(nn.Module):
    """
    A skip-connection based module that can be used for DynUNet, based on:
    `Automated Design of Deep Learning Methods for Biomedical Image Segmentation <https://arxiv.org/abs/1904.08128>`_.
    `nnU-Net: Self-adapting Framework for U-Net-Based Medical Image Segmentation <https://arxiv.org/abs/1809.10486>`_.

    Args:
        spatial_dims: number of spatial dimensions.
        in_channels: number of input channels.
        out_channels: number of output channels.
        kernel_size: convolution kernel size.
        stride: convolution stride.
        norm_name: feature normalization type and arguments.
        act_name: activation layer type and arguments.
        dropout: dropout probability.

    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        kernel_size: Sequence[int] | int,
        stride: Sequence[int] | int,
        norm_name: tuple | str,
        act_name: tuple | str = ("leakyrelu", {"inplace": True, "negative_slope": 0.01}),
        dropout: tuple | str | float | None = None,
    ):
        super().__init__()
        self.conv1 = get_conv_layer(
            spatial_dims,
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            dropout=dropout,
            act=None,
            norm=None,
            conv_only=False,
        )
        self.conv2 = get_conv_layer(
            spatial_dims,
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=1,
            dropout=dropout,
            act=None,
            norm=None,
            conv_only=False,
        )
        self.lrelu = get_act_layer(name=act_name)
        self.norm1 = get_norm_layer(name=norm_name, spatial_dims=spatial_dims, channels=out_channels)
        self.norm2 = get_norm_layer(name=norm_name, spatial_dims=spatial_dims, channels=out_channels)
        self.downsample = in_channels != out_channels
        stride_np = np.atleast_1d(stride)
        if not np.all(stride_np == 1):
            self.downsample = True
        if self.downsample:
            self.conv3 = get_conv_layer(
                spatial_dims,
                in_channels,
                out_channels,
                kernel_size=1,
                stride=stride,
                dropout=dropout,
                act=None,
                norm=None,
                conv_only=False,
            )
            self.norm3 = get_norm_layer(name=norm_name, spatial_dims=spatial_dims, channels=out_channels)

    def forward(self, inp):
        residual = inp
        out = self.conv1(inp)
        out = self.norm1(out)
        out = self.lrelu(out)
        out = self.conv2(out)
        out = self.norm2(out)
        if hasattr(self, "conv3"):
            residual = self.conv3(residual)
        if hasattr(self, "norm3"):
            residual = self.norm3(residual)
        out += residual
        out = self.lrelu(out)
        return out

if __name__ == '__main__':
    block = CSPBlock(
        spatial_dims=3,
        in_channels=64,  # 입력 채널 수정
        out_channels=128,
        kernel_size=3,
        stride=2,
        norm_name="instance",
        act_name=("leakyrelu", {"inplace": True, "negative_slope": 0.01}),
        dropout=None,
        split_ratio=0.5,
        n=1
    )
    x = torch.randn(1, 64, 24, 24, 24)
    p = block(x)
    
    unetresblock = UnetResBlock(
        spatial_dims=3,
        in_channels=64,
        out_channels=128,
        kernel_size=3,
        stride=2,
        norm_name="instance",
        act_name=("leakyrelu", {"inplace": True, "negative_slope": 0.01}),
        dropout=None,
    )
    
    upbolck = UPCSPBlock(
        spatial_dims=3,
        in_channels=128,
        out_channels=64,
        kernel_size=3,
        upsample_kernel_size=2,
        norm_name="instance",
        n=1
    )
    print(f"Input shape: {x.shape}")
    print(f"CSPBlock Output shape: {p.shape}")
    print(f"UnetResBlock Output shape: {unetresblock(x).shape}")
    print(f"UP Output shape: {upbolck(p,x).shape}")
    print(f"CSPBlock mean: {torch.mean(p)}")
    print(f"UnetResBlock mean: {torch.mean(unetresblock(x))}")
    print(f"UPCSPBlock mean: {torch.mean(upbolck(p,x))}")
        