import warnings
from collections.abc import Sequence
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from monai.networks.blocks.convolutions import Convolution
from monai.networks.layers.factories import Act, Norm

# ---------------------------
# 1) LayerNorm3D (옵션)
# ---------------------------
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


# ---------------------------
# 2) 헬퍼 함수: padding 계산
# ---------------------------
def get_padding(kernel_size: Sequence[int] | int, stride: Sequence[int] | int):
    kernel_size_np = np.atleast_1d(kernel_size)
    stride_np = np.atleast_1d(stride)
    padding_np = (kernel_size_np - stride_np + 1) / 2
    if np.min(padding_np) < 0:
        raise AssertionError("padding must not be negative.")
    padding = tuple(int(p) for p in padding_np)
    return padding if len(padding) > 1 else padding[0]

def get_output_padding(kernel_size, stride, padding):
    kernel_size_np = np.atleast_1d(kernel_size)
    stride_np = np.atleast_1d(stride)
    padding_np = np.atleast_1d(padding)
    out_padding_np = 2 * padding_np + stride_np - kernel_size_np
    if np.min(out_padding_np) < 0:
        raise AssertionError("out_padding must not be negative.")
    out_padding = tuple(int(p) for p in out_padding_np)
    return out_padding if len(out_padding) > 1 else out_padding[0]


# ---------------------------
# 3) get_conv_layer
# ---------------------------
def get_conv_layer(
    spatial_dims: int,
    in_channels: int,
    out_channels: int,
    kernel_size: int | Sequence[int] = 3,
    stride: int | Sequence[int] = 1,
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


# ---------------------------
# 4) ResizeLayer(nn.Upsample)
# ---------------------------
class ResizeLayer(nn.Module):
    def __init__(self, mode="trilinear", align_corners=False):
        super().__init__()
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x: torch.Tensor, size: tuple[int, ...]) -> torch.Tensor:
        up = nn.Upsample(size=size, mode=self.mode, align_corners=self.align_corners)
        return up(x)


# ---------------------------
# 5) SkipAlign
# ---------------------------
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

        if channel_match and (skip_in_channels != out_channels):
            if spatial_dims == 3:
                self.conv1x1 = nn.Conv3d(skip_in_channels, out_channels, kernel_size=1, bias=True)
            else:
                self.conv1x1 = nn.Conv2d(skip_in_channels, out_channels, kernel_size=1, bias=True)
        else:
            self.conv1x1 = None

    def forward(self, skip_tensor: torch.Tensor, target_tensor: torch.Tensor) -> torch.Tensor:
        device = target_tensor.device
        dtype = target_tensor.dtype
        skip_tensor = skip_tensor.to(device=device, dtype=dtype)

        # 업샘플
        D_out, H_out, W_out = target_tensor.shape[2:]
        if skip_tensor.shape[2:] != (D_out, H_out, W_out):
            skip_tensor = self.upsample(skip_tensor, (D_out, H_out, W_out))

        # 1×1 Conv
        if self.conv1x1 is not None:
            skip_tensor = self.conv1x1(skip_tensor)

        return skip_tensor


# ---------------------------
# 6) build_conv_stack
# ---------------------------
def build_conv_stack(
    spatial_dims: int,
    in_channels: int,
    out_channels: int,
    num_layers: int,
    kernel_size: int | Sequence[int],
    stride: int | Sequence[int],
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


# ---------------------------
# 7) SingleEncoderBlock
# ---------------------------
class SingleEncoderBlock(nn.Module):
    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        num_layers: int,
        kernel_size: int | Sequence[int],
        stride: int | Sequence[int],
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


# ---------------------------
# 8) SingleDecoderBlock
# ---------------------------
class SingleDecoderBlock(nn.Module):
    def __init__(
        self,
        spatial_dims: int,
        main_in_channels: int,
        core_channels: int,
        out_channels: int,
        skip_in_channels_list: list[int],
        num_layers: int,
        kernel_size: int | Sequence[int],
        stride: int | Sequence[int],
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
        
        # (1) Main input channel aligner (1x1 conv로 core_channels로 맞춤)
        if spatial_dims == 3:
            self.main_aligner = nn.Conv3d(main_in_channels, core_channels, kernel_size=1, bias=True)
        else:
            self.main_aligner = nn.Conv2d(main_in_channels, core_channels, kernel_size=1, bias=True)
        
        # (2) Skip aligners (채널 매칭을 위한)
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

        # (3) Main conv stack with transposed conv
        total_in_channels = core_channels * (1 + self.skip_count)  # main(core_ch) + skips(core_ch each)
        
        
        self.conv_stack = build_conv_stack(
            spatial_dims=spatial_dims,
            in_channels=total_in_channels,
            out_channels=out_channels,
            num_layers=num_layers,  # Already used one layer for upsampling
            kernel_size=kernel_size,
            stride=1,
            act=act,
            norm=norm,
            dropout=dropout,
            bias=bias,
            is_transposed=True,
        )

    def forward(self, x_main: torch.Tensor, skip_tensors: list[torch.Tensor]) -> torch.Tensor:
        # (1) Main input을 core_channels로 변환
        x_main = self.main_aligner(x_main)  # (N, core_channels, ...)
        print("x_main :", x_main.shape)
        # (2) 모든 skip connection을 현재 입력 크기로 맞춤
        aligned_skips = []
        
        for i, s in enumerate(skip_tensors):
            print("skip_tensor :", s.shape)
            aligned_s = self.skip_aligners[i](s, x_main)  # (N, core_channels, ...)
            aligned_skips.append(aligned_s)
            print(aligned_s.shape)
        
        # (3) Concatenate main input with aligned skips
        cat_list = [x_main] + aligned_skips
        cat_input = torch.cat(cat_list, dim=1)  # (N, core_channels * (1 + skip_count), ...)
        
        # (4) Apply transposed conv for upsampling
        out = self.conv_stack(cat_input)
        print("++++++++++")
        return out


# ---------------------------
# 9) FlexibleUNet
# ---------------------------
class FlexibleUNet(nn.Module):
    """
    디코더 간 스킵 연결:
      skip_connections = {
         dec_idx: [
           ("enc", enc_i),  # 인코더 레벨 enc_i
           ("dec", dec_j),  # 디코더 레벨 dec_j
         ],
         ...
      }
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
        skip_connections: dict[int, list[tuple[str, int]]] | None = None,
        kernel_size: int | Sequence[int] = 3,
        up_kernel_size: int | Sequence[int] = 3,
        act: tuple | str = Act.LEAKYRELU,
        norm: tuple | str = Norm.BATCH,
        dropout: float = 0.0,
        bias: bool = True,
        mode: str = "trilinear",
        align_corners: bool = False,
    ):
        """
        skip_connections: {
          decoder_index: [("enc", i), ("dec", j), ...],
          ...
        }
        """
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

        # ---------------------- 인코더 구성 ----------------------
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

        # ---------------------- 디코더 구성 ----------------------
        self.decoder_blocks = nn.ModuleList()
        main_in_ch = encoder_channels[-1]  # bottleneck
        for dec_i in range(len(decoder_channels)):
            out_ch = decoder_channels[dec_i]
            stride = decoder_strides[dec_i]
            nlayer = num_layers_decoder[dec_i]

            # skip 인덱스 -> skip_in_channels
            # "enc" -> encoder_channels[idx], "dec" -> decoder_channels[idx]
            skip_info_list = skip_connections.get(dec_i, [])
            skip_in_channels_list = []
            for (typ, idx) in skip_info_list:
                if typ == "enc":
                    skip_in_channels_list.append(encoder_channels[idx])
                elif typ == "dec":
                    # 디코더 레벨 idx의 출력 채널
                    skip_in_channels_list.append(decoder_channels[idx])
                else:
                    raise ValueError(f"Invalid skip type: {typ}, must be 'enc' or 'dec'.")
            
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

        # 최종 Conv
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
        # device/dtype
        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype
        x = x.to(device=device, dtype=dtype)

        # 1) 인코더
        encoder_outputs = []
        out = x
        for enc in self.encoder_blocks:
            out = enc(out)
            encoder_outputs.append(out)

        # 2) 디코더
        decoder_outputs = []
        # out = encoder_outputs[-1]  # bottleneck
        # decoder_outputs.append(out)  # 0번 디코더가 시작하기 전(bottleneck)에 쓸 수도 있지만, 여기선 i=0부터 맞춰줄 수도 있음.

        for dec_i, dec_block in enumerate(self.decoder_blocks):
            # skip에 "enc" => encoder_outputs, "dec" => decoder_outputs
            skip_info_list = self.skip_connections.get(dec_i, [])
            skip_list = []
            for (typ, idx) in skip_info_list:
                if typ == "enc":
                    skip_list.append(encoder_outputs[idx])
                elif typ == "dec":
                    # 디코더 idx는 0.. dec_i-1 범위여야함
                    skip_list.append(decoder_outputs[idx])
                else:
                    raise ValueError(f"Invalid skip type: {typ}.")

            out = dec_block(out, skip_list)
            # 디코더 i번 블록 결과를 decoder_outputs에 저장
            decoder_outputs.append(out)

        # 3) 최종 Conv
        out = self.final_conv(out)
        return out


# ---------------------------
# 10) 테스트
# ---------------------------
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    enc_channels = (32, 64, 128, 256)
    enc_strides = (2, 2, 2)
    num_layers_enc = (1, 1, 1, 1)

    core_channels = 64
    dec_channels = (128, 64, 32)
    dec_strides = (2, 2, 2)
    num_layers_dec = (1, 1, 1)

    # 디코더간 스킵 예시:
    #   디코더0: ("enc",2), ("dec", ?) -- 가능하지만 보통 dec_? < dec_0
    #   디코더1: ("enc",1), ("dec",0)
    #   디코더2: ("enc",0), ("dec",1)
    # 반드시 "dec", j => j < i 여야함
    skip_map = {
        0: [("enc", 2)],       # 디코더0 => 인코더2
        1: [("enc", 3), ("enc", 1)],  # 디코더1 => 인코더1 + 디코더0
        2: [("enc", 3), ("dec", 0), ("enc", 0)]   # 디코더2 => 인코더0 + 디코더1
    }

    net = FlexibleUNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=2,
        encoder_channels=enc_channels,
        encoder_strides=enc_strides,
        core_channels=core_channels,
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

    x = torch.randn((1, 1, 64, 64, 32), device=device)
    with torch.no_grad():
        out = net(x)
        print(out.shape)