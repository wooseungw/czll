import warnings
from collections.abc import Sequence
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from monai.networks.blocks.convolutions import Convolution
from monai.networks.layers.factories import Act, Norm

# ------------------------------------------------------------------------
# (1) LayerNorm3D (옵션): 3D 텐서에 LayerNorm 적용 필요 시 사용
# ------------------------------------------------------------------------
class LayerNorm3D(nn.Module):
    def __init__(self, num_channels: int):
        super().__init__()
        self.layer_norm = nn.LayerNorm(num_channels)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        N, C, *spatial_dims = x.shape
        x = x.permute(0, 2, 3, 4, 1).contiguous()
        x = self.layer_norm(x)
        x = x.permute(0, 4, 1, 2, 3).contiguous()
        return x


# ------------------------------------------------------------------------
# (2) Helpler: padding 계산
# ------------------------------------------------------------------------
def get_padding(kernel_size: Sequence[int] | int, stride: Sequence[int] | int):
    kernel_size_np = np.atleast_1d(kernel_size)
    stride_np = np.atleast_1d(stride)
    padding_np = (kernel_size_np - stride_np + 1) / 2
    if np.min(padding_np) < 0:
        raise AssertionError("padding must not be negative.")
    padding = tuple(int(p) for p in padding_np)
    return padding if len(padding) > 1 else padding[0]

def get_output_padding(
    kernel_size: Sequence[int] | int, stride: Sequence[int] | int, padding: Sequence[int] | int
):
    kernel_size_np = np.atleast_1d(kernel_size)
    stride_np = np.atleast_1d(stride)
    padding_np = np.atleast_1d(padding)
    out_padding_np = 2 * padding_np + stride_np - kernel_size_np
    if np.min(out_padding_np) < 0:
        raise AssertionError("out_padding must not be negative.")
    out_padding = tuple(int(p) for p in out_padding_np)
    return out_padding if len(out_padding) > 1 else out_padding[0]


# ------------------------------------------------------------------------
# (3) get_conv_layer: MONAI Convolution + LayerNorm3D 등 처리
# ------------------------------------------------------------------------
def get_conv_layer(
    spatial_dims: int,
    in_channels: int,
    out_channels: int,
    kernel_size: Sequence[int] | int = 3,
    stride: Sequence[int] | int = 1,
    act: tuple | str | None = Act.PRELU,
    norm: tuple | str | None = Norm.INSTANCE,
    dropout: float | None = 0.0,
    bias: bool = True,
    conv_only: bool = False,
    is_transposed: bool = False,
):
    padding = get_padding(kernel_size, stride)
    output_padding = None
    if is_transposed:
        output_padding = get_output_padding(kernel_size, stride, padding)

    # LayerNorm 처리
    if norm == Norm.LAYER:
        norm = LayerNorm3D(num_channels=out_channels)

    return Convolution(
        spatial_dims=spatial_dims,
        in_channels=in_channels,
        out_channels=out_channels,
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


# ------------------------------------------------------------------------
# (4) Upsample 레이어 (nn.Upsample)
# ------------------------------------------------------------------------
class ResizeLayer(nn.Module):
    def __init__(self, mode="trilinear", align_corners=False):
        super().__init__()
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x: torch.Tensor, size: tuple[int, ...]) -> torch.Tensor:
        up = nn.Upsample(size=size, mode=self.mode, align_corners=self.align_corners)
        return up(x)


# ------------------------------------------------------------------------
# (5) SkipAlign: Upsample + optional 1x1 Conv
# ------------------------------------------------------------------------
class SkipAlign(nn.Module):
    def __init__(
        self,
        skip_in_channels: int,
        out_channels: int,
        spatial_dims: int = 3,
        channel_match: bool = True,
        mode: str = "trilinear",
        align_corners: bool = False,
    ):
        super().__init__()
        self.spatial_dims = spatial_dims
        self.channel_match = channel_match

        self.upsample = ResizeLayer(mode=mode, align_corners=align_corners)

        # 1×1 Conv (채널 매칭)
        if channel_match and (skip_in_channels != out_channels):
            if spatial_dims == 3:
                self.conv1x1 = nn.Conv3d(skip_in_channels, out_channels, kernel_size=1, bias=True)
            else:
                self.conv1x1 = nn.Conv2d(skip_in_channels, out_channels, kernel_size=1, bias=True)
        else:
            self.conv1x1 = None

    def forward(self, skip_tensor: torch.Tensor, target_tensor: torch.Tensor) -> torch.Tensor:
        # device/dtype 맞춤
        device = target_tensor.device
        dtype = target_tensor.dtype
        skip_tensor = skip_tensor.to(device=device, dtype=dtype)

        # Upsample
        D_out, H_out, W_out = target_tensor.shape[2:]
        if skip_tensor.shape[2:] != (D_out, H_out, W_out):
            skip_tensor = self.upsample(skip_tensor, (D_out, H_out, W_out))

        # 1×1 Conv
        if self.conv1x1 is not None:
            skip_tensor = self.conv1x1(skip_tensor)

        return skip_tensor


# ------------------------------------------------------------------------
# (6) build_conv_stack: 여러 Conv/ConvTranspose를 쌓는 헬퍼
# ------------------------------------------------------------------------
def build_conv_stack(
    spatial_dims: int,
    in_channels: int,
    out_channels: int,
    num_layers: int,
    kernel_size: Sequence[int] | int,
    stride: Sequence[int] | int,
    act: tuple | str | None,
    norm: tuple | str | None,
    dropout: float,
    bias: bool,
    is_transposed: bool = False,
):
    layers = []
    for i in range(num_layers):
        if i == 0:
            layers.append(
                get_conv_layer(
                    spatial_dims=spatial_dims,
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    act=act,
                    norm=norm,
                    dropout=dropout,
                    bias=bias,
                    conv_only=False,
                    is_transposed=is_transposed,
                )
            )
        else:
            layers.append(
                get_conv_layer(
                    spatial_dims=spatial_dims,
                    in_channels=out_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=1,
                    act=act,
                    norm=norm,
                    dropout=dropout,
                    bias=bias,
                    conv_only=False,
                    is_transposed=False,
                )
            )
    return nn.Sequential(*layers)


# ------------------------------------------------------------------------
# (7) SingleEncoderBlock
# ------------------------------------------------------------------------
class SingleEncoderBlock(nn.Module):
    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        num_layers: int,
        kernel_size: Sequence[int] | int,
        stride: Sequence[int] | int,
        act: tuple | str | None,
        norm: tuple | str | None,
        dropout: float,
        bias: bool = True,
    ):
        super().__init__()
        self.stack = build_conv_stack(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            num_layers=num_layers,
            kernel_size=kernel_size,
            stride=stride,
            act=act,
            norm=norm,
            dropout=dropout,
            bias=bias,
            is_transposed=False,
        )
    def forward(self, x):
        return self.stack(x)


# ------------------------------------------------------------------------
# (8) SingleDecoderBlock
# ------------------------------------------------------------------------
class SingleDecoderBlock(nn.Module):
    """
    - __init__ 시점에 main_up(ConvTranspose), skip_aligners(스킵 개수만큼), post_conv_stack 생성
    - forward에서 호출만 수행
    """
    def __init__(
        self,
        spatial_dims: int,
        main_in_channels: int,
        core_channels: int,
        out_channels: int,
        skip_in_channels_list: list[int],  # 스킵 텐서들의 in_channels
        num_layers: int,
        kernel_size: Sequence[int] | int,
        stride: Sequence[int] | int,
        act: tuple | str | None,
        norm: tuple | str | None,
        dropout: float,
        bias: bool,
        mode: str = "trilinear",
        align_corners: bool = False,
    ):
        super().__init__()
        self.spatial_dims = spatial_dims
        self.out_channels = out_channels
        self.skip_count = len(skip_in_channels_list)
        print(f"Skip count: {self.skip_count}")
        # 1) main_up
        self.main_up = get_conv_layer(
            spatial_dims=spatial_dims,
            in_channels=main_in_channels,
            out_channels=core_channels,
            kernel_size=kernel_size,
            stride=stride,
            act=act,
            norm=norm,
            dropout=dropout,
            bias=bias,
            conv_only=False,
            is_transposed=True,
        )

        # 2) skip aligners
        # skip_in_channels_list 크기만큼 SkipAlign
        self.skip_aligners = nn.ModuleList()
        for s_in_ch in skip_in_channels_list:
            aligner = SkipAlign(
                skip_in_channels=s_in_ch,
                out_channels=core_channels,
                spatial_dims=spatial_dims,
                channel_match=True,
                mode=mode,
                align_corners=align_corners,
            )
            self.skip_aligners.append(aligner)

        # 3) post_conv_stack
        self.post_conv_stack = build_conv_stack(
            spatial_dims=spatial_dims,
            in_channels=core_channels * (1 + self.skip_count),  # concat 후 1×1 Conv로 out_channels 만든다고 가정
            out_channels=out_channels,
            num_layers=max(num_layers - 1, 0),
            kernel_size=kernel_size,
            stride=1,
            act=act,
            norm=norm,
            dropout=dropout,
            bias=bias,
            is_transposed=False,
        )

    def forward(self, x_main: torch.Tensor, skip_tensors: list[torch.Tensor]) -> torch.Tensor:
        out_main = self.main_up(x_main)

        cat_list = [out_main]
        # skip_tensors 개수 == skip_aligners 개수
        for i, s in enumerate(skip_tensors):
            aligned_s = self.skip_aligners[i](s, out_main)
            print(aligned_s.shape)
            cat_list.append(aligned_s)

        cat_input = torch.cat(cat_list, dim=1)

        out = self.post_conv_stack(cat_input)
        return out


# ------------------------------------------------------------------------
# (9) FlexibleUNet
# ------------------------------------------------------------------------
class FlexibleUNet(nn.Module):
    """
    - __init__ 시점에 인코더 블록 생성
    - __init__ 시점에 디코더 블록 생성 -> skip_in_channels_list를 미리 계산
    - forward에서 실행
    """
    def __init__(
        self,
        spatial_dims: int = 3,
        in_channels: int = 1,
        out_channels: int = 2,
        encoder_channels: Sequence[int] = (32, 64, 128, 256),
        encoder_strides: Sequence[int] = (2, 2, 2),
        core_channels: int = 64,
        decoder_channels: Sequence[int] = (128, 64, 32),
        decoder_strides: Sequence[int] = (2, 2, 2),
        num_layers_encoder: Sequence[int] = (1, 1, 1, 1),
        num_layers_decoder: Sequence[int] = (1, 1, 1),
        skip_connections: dict[int, list[int]] | None = None,
        kernel_size: Sequence[int] | int = 3,
        up_kernel_size: Sequence[int] | int = 3,
        act: tuple | str = Act.LEAKYRELU,
        norm: tuple | str = Norm.BATCH,
        dropout: float = 0.0,
        bias: bool = True,
        mode: str = "trilinear",
        align_corners: bool = False,
    ):
        super().__init__()
        if len(encoder_channels) != len(num_layers_encoder):
            raise ValueError("encoder_channels와 num_layers_encoder 길이가 맞지 않습니다.")
        if len(encoder_strides) != len(encoder_channels) - 1:
            raise ValueError("encoder_strides 길이는 (len(encoder_channels) - 1)이어야 합니다.")
        if len(decoder_channels) != len(num_layers_decoder):
            raise ValueError("decoder_channels와 num_layers_decoder 길이가 맞지 않습니다.")
        if len(decoder_strides) != len(decoder_channels):
            raise ValueError("decoder_strides 길이는 len(decoder_channels)와 같아야 합니다.")

        self.spatial_dims = spatial_dims
        self.skip_connections = skip_connections if skip_connections else {}

        # ---------------------- 인코더 생성 ----------------------
        self.encoder_blocks = nn.ModuleList()
        prev_ch = in_channels
        for i, out_ch in enumerate(encoder_channels):
            stride = encoder_strides[i] if i < len(encoder_strides) else 1
            block = SingleEncoderBlock(
                spatial_dims=spatial_dims,
                in_channels=prev_ch,
                out_channels=out_ch,
                num_layers=num_layers_encoder[i],
                kernel_size=kernel_size,
                stride=stride,
                act=act,
                norm=norm,
                dropout=dropout,
                bias=bias,
            )
            self.encoder_blocks.append(block)
            prev_ch = out_ch

        # ---------------------- 디코더 생성 ----------------------
        self.decoder_blocks = nn.ModuleList()
        main_in_ch = encoder_channels[-1]  # bottleneck out
        for dec_i in range(len(decoder_channels)):
            out_ch = decoder_channels[dec_i]
            stride = decoder_strides[dec_i]
            nlayer = num_layers_decoder[dec_i]

            # skip 인덱스 -> skip_in_channels
            # 예) skip_connections[0] = [2, 1] -> skip_in_channels_list = [encoder_channels[2], encoder_channels[1]]
            skip_idx_list = self.skip_connections.get(dec_i, [])
            skip_in_channels_list = [encoder_channels[idx] for idx in skip_idx_list]

            block = SingleDecoderBlock(
                spatial_dims=spatial_dims,
                main_in_channels=main_in_ch,
                out_channels=out_ch,
                core_channels=core_channels,
                skip_in_channels_list=skip_in_channels_list,
                num_layers=nlayer,
                kernel_size=up_kernel_size,
                stride=stride,
                act=act,
                norm=norm,
                dropout=dropout,
                bias=bias,
                mode=mode,
                align_corners=align_corners,
            )
            self.decoder_blocks.append(block)
            main_in_ch = out_ch

        # ---------------------- 최종 Conv ----------------------
        self.final_conv = get_conv_layer(
            spatial_dims=spatial_dims,
            in_channels=decoder_channels[-1],
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            norm=None,
            act=None,
            dropout=0.0,
            bias=True,
            conv_only=True,
            is_transposed=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # device/dtype 통일
        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype
        x = x.to(device=device, dtype=dtype)

        # 인코더
        encoder_outputs = []
        out = x
        for enc in self.encoder_blocks:
            out = enc(out)
            encoder_outputs.append(out)

        # 디코더
        out = encoder_outputs[-1]  # bottleneck
        for dec_i, dec_block in enumerate(self.decoder_blocks):
            print(f"Out {out.shape}")
            # skip 인덱스
            skip_idx_list = self.skip_connections.get(dec_i, [])
            skip_list = [encoder_outputs[idx] for idx in skip_idx_list]
            out = dec_block(out, skip_list)

        # 최종
        out = self.final_conv(out)
        return out

# ------------------------------------------------------------------------
# (10) Test
# ------------------------------------------------------------------------
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    enc_channels = (32,64,128,256)
    enc_strides = (2,2,2)
    num_layers_enc = (1,1,1,1)
    core_channels = 64
    dec_channels = (128,64,32)
    dec_strides = (2,2,2)
    num_layers_dec = (1,1,1)

    skip_map = {
        0: [2,1],  # 디코더0 => 인코더2
        1: [1],  # 디코더1 => 인코더1
        2: [0],  # 디코더2 => 인코더0
    }

    net = FlexibleUNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=2,
        encoder_channels=enc_channels,
        encoder_strides=enc_strides,
        decoder_channels=dec_channels,
        decoder_strides=dec_strides,
        num_layers_encoder=num_layers_enc,
        num_layers_decoder=num_layers_dec,
        skip_connections=skip_map,
        kernel_size=3,
        up_kernel_size=3,
        act=Act.LEAKYRELU,
        norm=Norm.BATCH,
        dropout=0.0,
        bias=True,
        mode="trilinear",
        align_corners=False,
    ).to(device)

    print(net)

    x = torch.randn(1, 1, 64, 64, 32).to(device)
    with torch.no_grad():
        y = net(x)
    print("Output shape:", y.shape)  # ex) (1, 2, 64, 64, 32)
