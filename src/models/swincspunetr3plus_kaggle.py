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

from swin_transformer import SwinTransformer, MERGING_MODE
from conv_moudules import CSPBlock, UPCSPBlock, get_conv_layer, UPCSPBlock3plus, Upsample3D


class SwinCSPUNETR3plus(nn.Module):
    """
    Swin UNETR based on: "Hatamizadeh et al.,
    Swin UNETR: Swin Transformers for Semantic Segmentation of Brain Tumors in MRI Images
    <https://arxiv.org/abs/2201.01266>"
    """

    patch_size: Final[int] = 2

    @deprecated_arg(
        name="img_size",
        since="1.3",
        removed="1.5",
        msg_suffix="The img_size argument is not required anymore and "
        "checks on the input size are run during forward().",
    )
    def __init__(
        self,
        img_size: Sequence[int] | int,
        in_channels: int,
        out_channels: int,
        depths: Sequence[int] = (2, 2, 2, 2),
        num_heads: Sequence[int] = (3, 6, 12, 24),
        feature_size: int = 24,
        norm_name: tuple | str = "instance",
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        dropout_path_rate: float = 0.0,
        normalize: bool = True,
        use_checkpoint: bool = False,
        spatial_dims: int = 3,
        downsample="merging",
        use_v2=False,
        n=2
    ) -> None:
        """
        Args:
            img_size: spatial dimension of input image.
                This argument is only used for checking that the input image size is divisible by the patch size.
                The tensor passed to forward() can have a dynamic shape as long as its spatial dimensions are divisible by 2**5.
                It will be removed in an upcoming version.
            in_channels: dimension of input channels.
            out_channels: dimension of output channels.
            feature_size: dimension of network feature size.
            depths: number of layers in each stage.
            num_heads: number of attention heads.
            norm_name: feature normalization type and arguments.
            drop_rate: dropout rate.
            attn_drop_rate: attention dropout rate.
            dropout_path_rate: drop path rate.
            normalize: normalize output intermediate features in each stage.
            use_checkpoint: use gradient checkpointing for reduced memory usage.
            spatial_dims: number of spatial dims.
            downsample: module used for downsampling, available options are `"mergingv2"`, `"merging"` and a
                user-specified `nn.Module` following the API defined in :py:class:`monai.networks.nets.PatchMerging`.
                The default is currently `"merging"` (the original version defined in v0.9.0).
            use_v2: using swinunetr_v2, which adds a residual convolution block at the beggining of each swin stage.

        Examples::

            # for 3D single channel input with size (96,96,96), 4-channel output and feature size of 48.
            >>> net = SwinUNETR(img_size=(96,96,96), in_channels=1, out_channels=4, feature_size=48)

            # for 3D 4-channel input with size (128,128,128), 3-channel output and (2,4,2,2) layers in each stage.
            >>> net = SwinUNETR(img_size=(128,128,128), in_channels=4, out_channels=3, depths=(2,4,2,2))

            # for 2D single channel input with size (96,96), 2-channel output and gradient checkpointing.
            >>> net = SwinUNETR(img_size=(96,96), in_channels=3, out_channels=2, use_checkpoint=True, spatial_dims=2)

        """

        super().__init__()

        img_size = ensure_tuple_rep(img_size, spatial_dims)
        patch_sizes = ensure_tuple_rep(self.patch_size, spatial_dims)
        window_size = ensure_tuple_rep(7, spatial_dims)

        if spatial_dims not in (2, 3):
            raise ValueError("spatial dimension should be 2 or 3.")

        self._check_input_size(img_size)

        if not (0 <= drop_rate <= 1):
            raise ValueError("dropout rate should be between 0 and 1.")

        if not (0 <= attn_drop_rate <= 1):
            raise ValueError("attention dropout rate should be between 0 and 1.")

        if not (0 <= dropout_path_rate <= 1):
            raise ValueError("drop path rate should be between 0 and 1.")

        if feature_size % 12 != 0:
            raise ValueError("feature_size should be divisible by 12.")

        self.normalize = normalize

        self.swinViT = SwinTransformer(
            in_chans=in_channels,
            embed_dim=feature_size,
            window_size=window_size,
            patch_size=patch_sizes,
            depths=depths,
            num_heads=num_heads,
            mlp_ratio=4.0,
            qkv_bias=True,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=dropout_path_rate,
            norm_layer=nn.LayerNorm,
            use_checkpoint=use_checkpoint,
            spatial_dims=spatial_dims,
            downsample=look_up_option(downsample, MERGING_MODE) if isinstance(downsample, str) else downsample,
            use_v2=use_v2,
        )

        self.encoder1 = CSPBlock(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            act_name=("leakyrelu", {"inplace": True, "negative_slope": 0.01}),
            dropout=None,
            split_ratio=0.5,
            n=n
        )

        self.encoder2 = CSPBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size,
            out_channels=feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            act_name=("leakyrelu", {"inplace": True, "negative_slope": 0.01}),
            dropout=None,
            split_ratio=0.5,
            n=n
        )

        self.encoder3 = CSPBlock(
            spatial_dims=spatial_dims,
            in_channels=2 * feature_size,
            out_channels=2 * feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            act_name=("leakyrelu", {"inplace": True, "negative_slope": 0.01}),
            dropout=None,
            split_ratio=0.5,
            n=n
        )

        self.encoder4 = CSPBlock(
            spatial_dims=spatial_dims,
            in_channels=4 * feature_size,
            out_channels=4 * feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            act_name=("leakyrelu", {"inplace": True, "negative_slope": 0.01}),
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
            act_name=("leakyrelu", {"inplace": True, "negative_slope": 0.01}),
            dropout=None,
            split_ratio=0.5,
            n=n
        )
        '''hidden stage 1'''
        self.s1_hd4 = nn.Sequential(
            Upsample3D(scale_factor=2, mode='trilinear', align_corners=False),
            get_conv_layer(
                spatial_dims,
                feature_size*16,
                feature_size,
                kernel_size=1,
                stride=1,
                norm=norm_name,
                act=("leakyrelu", {"inplace": True, "negative_slope": 0.01}),
            ))
        self.s1_hd3 = nn.Sequential(
            get_conv_layer(
                spatial_dims,
                feature_size*8,
                feature_size,
                kernel_size=1,
                stride=1,
                norm=norm_name,
                act=("leakyrelu", {"inplace": True, "negative_slope": 0.01}),
            ))
        self.s1_hd2 = nn.Sequential(
            nn.MaxPool3d(2, stride=2, ceil_mode=True),
            get_conv_layer(
                spatial_dims,
                feature_size*4,
                feature_size,
                kernel_size=1,
                stride=1,
                norm=norm_name,
                act=("leakyrelu", {"inplace": True, "negative_slope": 0.01}),
            ))
        self.s1_hd1 = nn.Sequential(
            nn.MaxPool3d(4, stride=4, ceil_mode=True),
            get_conv_layer(
                spatial_dims,
                feature_size*2,
                feature_size,
                kernel_size=1,
                stride=1,
                norm=norm_name,
                act=("leakyrelu", {"inplace": True, "negative_slope": 0.01}),
            ))
        self.s1_hd0 = nn.Sequential(
            nn.MaxPool3d(8, stride=8, ceil_mode=True),
            get_conv_layer(
                spatial_dims,
                feature_size,
                feature_size,
                kernel_size=1,
                stride=1,
                norm=norm_name,
                act=("leakyrelu", {"inplace": True, "negative_slope": 0.01}),
            ))
        '''hidden stage 2'''
        self.s2_hd4 = nn.Sequential(
            Upsample3D(scale_factor=4, mode='trilinear', align_corners=False),
            get_conv_layer(
                spatial_dims,
                feature_size*16,
                feature_size,
                kernel_size=1,
                stride=1,
                norm=norm_name,
                act=("leakyrelu", {"inplace": True, "negative_slope": 0.01}),
            ))
        self.s2_hd3 = nn.Sequential(
            Upsample3D(scale_factor=2, mode='trilinear', align_corners=False),
            get_conv_layer(
                spatial_dims,
                feature_size*5,
                feature_size,
                kernel_size=1,
                stride=1,
                norm=norm_name,
                act=("leakyrelu", {"inplace": True, "negative_slope": 0.01}),
            ))
        self.s2_hd2 = nn.Sequential(
            get_conv_layer(
                spatial_dims,
                feature_size*4,
                feature_size,
                kernel_size=1,
                stride=1,
                norm=norm_name,
                act=("leakyrelu", {"inplace": True, "negative_slope": 0.01}),
            ))
        self.s2_hd1 = nn.Sequential(
            nn.MaxPool3d(2, stride=2, ceil_mode=True),
            get_conv_layer(
                spatial_dims,
                feature_size*2,
                feature_size,
                kernel_size=1,
                stride=1,
                norm=norm_name,
                act=("leakyrelu", {"inplace": True, "negative_slope": 0.01}),
            ))
        self.s2_hd0 = nn.Sequential(
            nn.MaxPool3d(4, stride=4, ceil_mode=True),
            get_conv_layer(
                spatial_dims,
                feature_size,
                feature_size,
                kernel_size=1,
                stride=1,
                norm=norm_name,
                act=("leakyrelu", {"inplace": True, "negative_slope": 0.01}),
            ))
        '''hidden stage 3'''
        self.s3_hd4 = nn.Sequential(
            Upsample3D(scale_factor=8, mode='trilinear', align_corners=False),
            get_conv_layer(
                spatial_dims,
                feature_size*16,
                feature_size,
                kernel_size=1,
                stride=1,
                norm=norm_name,
                act=("leakyrelu", {"inplace": True, "negative_slope": 0.01}),
            ))
        self.s3_hd3 = nn.Sequential(
            Upsample3D(scale_factor=4, mode='trilinear', align_corners=False),
            get_conv_layer(
                spatial_dims,
                feature_size*5,
                feature_size,
                kernel_size=1,
                stride=1,
                norm=norm_name,
                act=("leakyrelu", {"inplace": True, "negative_slope": 0.01}),
            ))
        self.s3_hd2 = nn.Sequential(
            Upsample3D(scale_factor=2, mode='trilinear', align_corners=False),
            get_conv_layer(
                spatial_dims,
                feature_size*5,
                feature_size,
                kernel_size=1,
                stride=1,
                norm=norm_name,
                act=("leakyrelu", {"inplace": True, "negative_slope": 0.01}),
            ))
        self.s3_hd1 = nn.Sequential(
            get_conv_layer(
                spatial_dims,
                feature_size*2,
                feature_size,
                kernel_size=1,
                stride=1,
                norm=norm_name,
                act=("leakyrelu", {"inplace": True, "negative_slope": 0.01}),
            ))
        self.s3_hd0 = nn.Sequential(
            nn.MaxPool3d(2, stride=2, ceil_mode=True),
            get_conv_layer(
                spatial_dims,
                feature_size,
                feature_size,
                kernel_size=1,
                stride=1,
                norm=norm_name,
                act=("leakyrelu", {"inplace": True, "negative_slope": 0.01}),
            ))
        '''hidden stage 4'''
        self.s4_hd4 = nn.Sequential(
            Upsample3D(scale_factor=16, mode='trilinear', align_corners=False),
            get_conv_layer(
                spatial_dims,
                feature_size*16,
                feature_size,
                kernel_size=1,
                stride=1,
                norm=norm_name,
                act=("leakyrelu", {"inplace": True, "negative_slope": 0.01}),
            ))
        self.s4_hd3 = nn.Sequential(
            Upsample3D(scale_factor=8, mode='trilinear', align_corners=False),
            get_conv_layer(
                spatial_dims,
                feature_size*5,
                feature_size,
                kernel_size=1,
                stride=1,
                norm=norm_name,
                act=("leakyrelu", {"inplace": True, "negative_slope": 0.01}),
            ))
        self.s4_hd2 = nn.Sequential(
            Upsample3D(scale_factor=4, mode='trilinear', align_corners=False),
            get_conv_layer(
                spatial_dims,
                feature_size*5,
                feature_size,
                kernel_size=1,
                stride=1,
                norm=norm_name,
                act=("leakyrelu", {"inplace": True, "negative_slope": 0.01}),
            ))
        self.s4_hd1 = nn.Sequential(
            Upsample3D(scale_factor=2, mode='trilinear', align_corners=False),
            get_conv_layer(
                spatial_dims,
                feature_size*5,
                feature_size,
                kernel_size=1,
                stride=1,
                norm=norm_name,
                act=("leakyrelu", {"inplace": True, "negative_slope": 0.01}),
            ))
        self.s4_hd0 = nn.Sequential(
            get_conv_layer(
                spatial_dims,
                feature_size,
                feature_size,
                kernel_size=1,
                stride=1,
                norm=norm_name,
                act=("leakyrelu", {"inplace": True, "negative_slope": 0.01}),
            ))

        self.decoder5 = UPCSPBlock3plus(
            spatial_dims=spatial_dims,
            in_channels=5 * feature_size,
            out_channels=5 * feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            n=n
        )

        self.decoder4 = UPCSPBlock3plus(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 5,
            out_channels=feature_size * 5,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            n=n
        )

        self.decoder3 = UPCSPBlock3plus(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 5,
            out_channels=feature_size * 5,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            n=n
        )
        self.decoder2 = UPCSPBlock3plus(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 5,
            out_channels=feature_size * 5,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            n=n
        )

        self.decoder1 = UPCSPBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 5,
            out_channels=feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            n=n
        )

        self.out = UnetOutBlock(spatial_dims=spatial_dims, in_channels=feature_size, out_channels=out_channels)

    def load_from(self, weights):
        with torch.no_grad():
            self.swinViT.patch_embed.proj.weight.copy_(weights["state_dict"]["module.patch_embed.proj.weight"])
            self.swinViT.patch_embed.proj.bias.copy_(weights["state_dict"]["module.patch_embed.proj.bias"])
            for bname, block in self.swinViT.layers1[0].blocks.named_children():
                block.load_from(weights, n_block=bname, layer="layers1")
            self.swinViT.layers1[0].downsample.reduction.weight.copy_(
                weights["state_dict"]["module.layers1.0.downsample.reduction.weight"]
            )
            self.swinViT.layers1[0].downsample.norm.weight.copy_(
                weights["state_dict"]["module.layers1.0.downsample.norm.weight"]
            )
            self.swinViT.layers1[0].downsample.norm.bias.copy_(
                weights["state_dict"]["module.layers1.0.downsample.norm.bias"]
            )
            for bname, block in self.swinViT.layers2[0].blocks.named_children():
                block.load_from(weights, n_block=bname, layer="layers2")
            self.swinViT.layers2[0].downsample.reduction.weight.copy_(
                weights["state_dict"]["module.layers2.0.downsample.reduction.weight"]
            )
            self.swinViT.layers2[0].downsample.norm.weight.copy_(
                weights["state_dict"]["module.layers2.0.downsample.norm.weight"]
            )
            self.swinViT.layers2[0].downsample.norm.bias.copy_(
                weights["state_dict"]["module.layers2.0.downsample.norm.bias"]
            )
            for bname, block in self.swinViT.layers3[0].blocks.named_children():
                block.load_from(weights, n_block=bname, layer="layers3")
            self.swinViT.layers3[0].downsample.reduction.weight.copy_(
                weights["state_dict"]["module.layers3.0.downsample.reduction.weight"]
            )
            self.swinViT.layers3[0].downsample.norm.weight.copy_(
                weights["state_dict"]["module.layers3.0.downsample.norm.weight"]
            )
            self.swinViT.layers3[0].downsample.norm.bias.copy_(
                weights["state_dict"]["module.layers3.0.downsample.norm.bias"]
            )
            for bname, block in self.swinViT.layers4[0].blocks.named_children():
                block.load_from(weights, n_block=bname, layer="layers4")
            self.swinViT.layers4[0].downsample.reduction.weight.copy_(
                weights["state_dict"]["module.layers4.0.downsample.reduction.weight"]
            )
            self.swinViT.layers4[0].downsample.norm.weight.copy_(
                weights["state_dict"]["module.layers4.0.downsample.norm.weight"]
            )
            self.swinViT.layers4[0].downsample.norm.bias.copy_(
                weights["state_dict"]["module.layers4.0.downsample.norm.bias"]
            )

    @torch.jit.unused
    def _check_input_size(self, spatial_shape):
        img_size = np.array(spatial_shape)
        remainder = (img_size % np.power(self.patch_size, 5)) > 0
        if remainder.any():
            wrong_dims = (np.where(remainder)[0] + 2).tolist()
            raise ValueError(
                f"spatial dimensions {wrong_dims} of input image (spatial shape: {spatial_shape})"
                f" must be divisible by {self.patch_size}**5."
            )

    def forward(self, x_in):
        if not torch.jit.is_scripting() and not torch.jit.is_tracing():
            self._check_input_size(x_in.shape[2:])
        hidden_states_out = self.swinViT(x_in, self.normalize)
        
        
        enc0 = self.encoder1(x_in)
        
        enc1 = self.encoder2(hidden_states_out[0])
        
        enc2 = self.encoder3(hidden_states_out[1])
        
        enc3 = self.encoder4(hidden_states_out[2])
        
        dec4 = self.encoder10(hidden_states_out[4])

        s1_hd4 = self.s1_hd4(dec4)
        s1_hd3 = self.s1_hd3(hidden_states_out[3])
        s1_hd2 = self.s1_hd2(enc3)
        s1_hd1 = self.s1_hd1(enc2)
        s1_hd0 = self.s1_hd0(enc1)
        # print(s1_hd4.shape, s1_hd3.shape, s1_hd2.shape, s1_hd1.shape, s1_hd0.shape)
        s1_out = self.decoder5(s1_hd4, s1_hd3, s1_hd2, s1_hd1, s1_hd0)

        s2_hd4 = self.s2_hd4(dec4)
        s2_hd3 = self.s2_hd3(s1_out)
        s2_hd2 = self.s2_hd2(enc3)
        s2_hd1 = self.s2_hd1(enc2)
        s2_hd0 = self.s2_hd0(enc1)
        s2_out = self.decoder4(s2_hd4, s2_hd3, s2_hd2, s2_hd1, s2_hd0)

        s3_hd4 = self.s3_hd4(dec4)
        s3_hd3 = self.s3_hd3(s1_out)
        s3_hd2 = self.s3_hd2(s2_out)
        s3_hd1 = self.s3_hd1(enc2)
        s3_hd0 = self.s3_hd0(enc1)
        s3_out = self.decoder3(s3_hd4, s3_hd3, s3_hd2, s3_hd1, s3_hd0)

        s4_hd4 = self.s4_hd4(dec4)
        s4_hd3 = self.s4_hd3(s1_out)
        s4_hd2 = self.s4_hd2(s2_out)
        s4_hd1 = self.s4_hd1(s3_out)
        s4_hd0 = self.s4_hd0(enc1)
        s4_out = self.decoder2(s4_hd4, s4_hd3, s4_hd2, s4_hd1, s4_hd0)
        
        # dec3 = self.decoder5(dec4, hidden_states_out[3])
        # dec2 = self.decoder4(dec3, enc3)
        # dec1 = self.decoder3(dec2, enc2)
        # dec0 = self.decoder2(dec1, enc1)
        out = self.decoder1(s4_out, enc0)
        logits = self.out(out)
        return logits
    

# if __name__ == "__main__":
#     swin_unetr = SwinCSPUNETR3plus(
#         img_size=(96, 96, 96),
#         in_channels=1,
#         out_channels=7,
#         feature_size=48,
#         depths=(2, 2, 2, 2),
#         num_heads=(3, 6, 12, 24),
#         norm_name="instance",
#         drop_rate=0.0,
#         attn_drop_rate=0.0,
#         dropout_path_rate=0.0,
#         normalize=True,
#         use_checkpoint=True,
#         spatial_dims=3,
#         downsample="merging",
#         use_v2=True,
#         n=2,
#     )
#     input_tensor = torch.randn(1, 1, 96, 96, 96)
#     output = swin_unetr(input_tensor)
#     print(output.shape)