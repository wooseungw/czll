from __future__ import annotations

import itertools
from collections.abc import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from torch.nn import LayerNorm
from typing_extensions import Final

from monai.networks.blocks import MLPBlock as Mlp

from monai.networks.layers import DropPath, trunc_normal_
from monai.utils import ensure_tuple_rep, look_up_option, optional_import
from monai.utils.deprecate_utils import deprecated_arg
from monai.networks.blocks import PatchEmbed, UnetOutBlock, UnetrUpBlock

from .swin_transformer import SwinTransformer, MERGING_MODE
from .conv_moudules import CSPBlock, UPCSPBlock
from .segformer3D import MiT3D, cast_tuple
from functools import partial

class MiTCSPUnet(nn.Module):
    def __init__(
        self,
        *,
        img_size = 96,
        feature_size,
        heads=(1, 1, 2, 5, 8),
        ff_expansion=(2, 8, 8, 4, 4),
        reduction_ratio=(16, 8, 4, 2, 1),
        num_layers=2,
        channels=1,
        stage_kernel_stride_pad = ((3, 1, 1), (3, 2, 1), (3, 2, 1), (3, 2, 1), (3, 2, 1)),
        
        spatial_dims=3,
        out_channels=7,
        norm_name="instance",
        act_name = ("leakyrelu ", {"inplace": True, "negative_slope": 0.01}),
        n=2,
        
    ):
        super().__init__()
        dims = (feature_size, feature_size * 2, feature_size * 4, feature_size * 8, feature_size * 16)
        depth = len(stage_kernel_stride_pad)
        heads, ff_expansion, reduction_ratio, num_layers = map(
            partial(cast_tuple, depth=depth), (heads, ff_expansion, reduction_ratio, num_layers))
        assert all([*map(lambda t: len(t) == depth, (heads, ff_expansion, reduction_ratio, num_layers))]), \
            'only four stages are allowed, all keyword arguments must be either a single value or a tuple of 4 values'

        self.mit = MiT3D(  # Use the 3D MiT
            channels=channels,
            dims=dims,
            heads=heads,
            ff_expansion=ff_expansion,
            reduction_ratio=reduction_ratio,
            num_layers=num_layers,
            stage_kernel_stride_pad=stage_kernel_stride_pad
        )

        self.encoder1 = CSPBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size,
            out_channels=feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            act_name=act_name,
            dropout=None,
            split_ratio=0.5,
            n=n
        )

        self.encoder2 = CSPBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size*2,
            out_channels=feature_size*2,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            act_name=act_name,
            dropout=None,
            split_ratio=0.5,
            n=n
        )

        self.encoder3 = CSPBlock(
            spatial_dims=spatial_dims,
            in_channels=4 * feature_size,
            out_channels=4 * feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            act_name=act_name,
            dropout=None,
            split_ratio=0.5,
            n=n
        )

        self.encoder4 = CSPBlock(
            spatial_dims=spatial_dims,
            in_channels=8 * feature_size,
            out_channels=8 * feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            act_name=act_name,
            dropout=None,
            split_ratio=0.5,
            n=n
        )

        self.encoder10 = CSPBlock(
            spatial_dims=spatial_dims,
            in_channels=16 * feature_size,
            out_channels=16 * feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            act_name=act_name,
            dropout=None,
            split_ratio=0.5,
            n=n
        )

        self.decoder4 = UPCSPBlock(
            spatial_dims=spatial_dims,
            in_channels=16 * feature_size,
            out_channels=8 * feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            n=n
        )

        self.decoder3 = UPCSPBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 8,
            out_channels=feature_size * 4,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            n=n
        )

        self.decoder2 = UPCSPBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 4,
            out_channels=feature_size * 2,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            n=n
        )
        self.decoder1 = UPCSPBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 2,
            out_channels=feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            n=n
        )

        

        self.out = UnetOutBlock(spatial_dims=spatial_dims, in_channels=feature_size, out_channels=out_channels)


    def forward(self, x):
        # Extract features from MiT3D
        hidden_states_out = self.mit(x, return_layer_outputs=True)
        # print(len(hidden_states_out))
        # for i in range(len(hidden_states_out)):
        #     print(hidden_states_out[i].shape)
        enc0 = self.encoder1(hidden_states_out[0])
        
        enc1 = self.encoder2(hidden_states_out[1])
        
        enc2 = self.encoder3(hidden_states_out[2])
        
        enc3 = self.encoder4(hidden_states_out[3])
        
        mid = self.encoder10(hidden_states_out[4])
        
        dec3 = self.decoder4(mid, enc3)
        dec2 = self.decoder3(dec3, enc2)
        dec1 = self.decoder2(dec2, enc1)
        dec0 = self.decoder1(dec1, enc0)
        out = self.out(dec0)
        
        return out
