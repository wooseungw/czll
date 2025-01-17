import warnings
from collections.abc import Sequence
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from monai.networks.blocks.convolutions import Convolution
from monai.networks.layers.factories import Act, Norm

class LayerNorm3D(nn.Module):
    """
    3D 입력을 위한 LayerNorm 래퍼 클래스.
    PyTorch의 LayerNorm은 마지막 차원을 기준으로 정규화하므로,
    5D 입력 텐서에 적용되도록 재구성합니다.
    """
    def __init__(self, num_channels: int):
        super().__init__()
        self.layer_norm = nn.LayerNorm(num_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [N, C, D, H, W] 형태의 텐서를 [N, D*H*W, C]로 변환하여 LayerNorm 적용
        N, C, *spatial_dims = x.shape
        x = x.permute(0, 2, 3, 4, 1).contiguous()  # [N, D, H, W, C]
        x = self.layer_norm(x)  # [N, D, H, W, C]
        x = x.permute(0, 4, 1, 2, 3).contiguous()  # 원래 형태로 복원 [N, C, D, H, W]
        return x


def get_padding(kernel_size: Sequence[int] | int, stride: Sequence[int] | int) -> tuple[int, ...] | int:
    kernel_size_np = np.atleast_1d(kernel_size)
    stride_np = np.atleast_1d(stride)
    padding_np = (kernel_size_np - stride_np + 1) / 2
    if np.min(padding_np) < 0:
        raise AssertionError("padding value should not be negative.")
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
        raise AssertionError("out_padding value should not be negative.")
    out_padding = tuple(int(p) for p in out_padding_np)
    return out_padding if len(out_padding) > 1 else out_padding[0]


def get_conv_layer(
    spatial_dims: int,
    in_channels: int,
    out_channels: int,
    kernel_size: Sequence[int] | int = 3,
    stride: Sequence[int] | int = 1,
    act: tuple | str | None = Act.PRELU,
    norm: tuple | str | None = Norm.INSTANCE,
    dropout: tuple | str | float | None = None,
    bias: bool = True,
    conv_only: bool = False,
    is_transposed: bool = False,
):
    """
    MONAI의 Convolution 레이어를 감싸는 헬퍼 함수.
    """
    padding = get_padding(kernel_size, stride)
    output_padding = None
    if is_transposed:
        output_padding = get_output_padding(kernel_size, stride, padding)

    # Norm.LAYER 처리: LayerNorm을 위한 LayerNorm3D 사용
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



class SkipAlign(nn.Module):
    """
    해상도 & 채널 자동 맞춤 모듈:
      - interpolate로 해상도 맞춤
      - 필요 시 1x1 Conv로 채널 맞춤
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        spatial_dims: int = 3,
        channel_match: bool = True,      # True면 채널도 1x1 Conv로 맞춤
        mode: str = "trilinear",
        align_corners: bool = False,
    ):
        super().__init__()
        self.mode = mode
        self.align_corners = align_corners
        self.spatial_dims = spatial_dims
        self.channel_match = channel_match

        if channel_match and (in_channels != out_channels):
            if spatial_dims == 3:
                self.conv1x1 = nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=True)
            elif spatial_dims == 2:
                self.conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=True)
            else:
                raise NotImplementedError("Only 2D/3D supported in this example.")
        else:
            self.conv1x1 = None

    def forward(self, skip_tensor: torch.Tensor, target_shape: torch.Size) -> torch.Tensor:
        """
        skip_tensor: shape (N, C_in, ...)
        target_shape: (N, C_target, D_out, H_out, W_out) (3D 가정)
        1) 해상도 맞추기
        2) 채널 맞추기
        """
        # 해상도 맞춤
        target_spatial = target_shape[2:]  # (D_out, H_out, W_out) in 3D
        if skip_tensor.shape[2:] != target_spatial:
            skip_tensor = F.interpolate(skip_tensor, size=target_spatial, mode=self.mode, align_corners=self.align_corners)

        # 채널 맞춤
        if self.conv1x1 is not None:
            skip_tensor = self.conv1x1(skip_tensor)

        return skip_tensor


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
    """
    블록 내 여러 Conv/ConvTranspose 스택을 만드는 헬퍼 함수
    """
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
            # 이후 레이어: out_channels 유지, stride=1
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


class SingleEncoderBlock(nn.Module):
    """
    인코더 블록
      - build_conv_stack로 num_layers만큼 레이어 쌓고,
        첫 레이어에서 stride/downsampling 수행
    """
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


class SingleDecoderBlock(nn.Module):
    """
    디코더 블록
      - 1) ConvTranspose (stride)로 업샘플
      - 2) 여러 스킵 텐서를 받아서 자동 해상도/채널 정렬(align) 후 concat
      - 3) (옵션) 1×1 Conv로 concat 결과 채널을 맞춤 (이 코드에서는 "post_conv_stack"이 담당)
      - 4) 이후, num_layers-1 만큼 추가 Conv
    """
    def __init__(
        self,
        spatial_dims: int,
        main_in_channels: int,
        out_channels: int,
        num_layers: int,
        kernel_size: Sequence[int] | int,
        stride: Sequence[int] | int,
        act: tuple | str | None,
        norm: tuple | str | None,
        dropout: float,
        bias: bool,
        align_skip_channel: bool = True,  # skip 채널도 out_channels에 맞출지 여부
    ):
        super().__init__()
        self.out_channels = out_channels
        self.spatial_dims = spatial_dims
        self.align_skip_channel = align_skip_channel

        # 첫 레이어: ConvTranspose로 (main_in_channels -> out_channels)
        self.main_up = get_conv_layer(
            spatial_dims=spatial_dims,
            in_channels=main_in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            act=act,
            norm=norm,
            dropout=dropout,
            bias=bias,
            conv_only=False,
            is_transposed=True,
        )

        # 나머지 레이어(num_layers-1)는 build_conv_stack 사용
        # 단, concat 후의 채널 크기가 "out_channels + (skip개수 * out_channels)"가 될 수 있으므로,
        # 여기서는 concat 직후에 1×1 Conv로 다시 out_channels로 맞춘 뒤 남은 레이어를 통과시키는 방식을 택할 수 있음.
        # => 아래와 같이 분리 구현

        # concat 후 (가변 채널)-> out_channels로 매핑
        # 이후 (num_layers - 1)번 conv stack
        self.post_conv_stack = build_conv_stack(
            spatial_dims=spatial_dims,
            in_channels=out_channels,  # concat 직후에 1x1 conv로 out_channels로 줄인다고 가정
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

    def forward(self, x_main: torch.Tensor, skip_list: list[torch.Tensor]) -> torch.Tensor:
        # 1) main_up
        out_main = self.main_up(x_main)  # (N, out_channels, D?, H?, W?)

        # 2) skip align + concat
        if len(skip_list) == 0:
            cat_input = out_main
        else:
            cat_tensors = [out_main]
            for s in skip_list:
                aligner = SkipAlign(
                    in_channels=s.shape[1],
                    out_channels=self.out_channels if self.align_skip_channel else s.shape[1],
                    spatial_dims=self.spatial_dims,
                    channel_match=self.align_skip_channel,
                    mode="trilinear",  # 또는 다른 모드
                    align_corners=False
                )
                aligned_s = aligner(s, out_main.shape)
                cat_tensors.append(aligned_s)

            cat_input = torch.cat(cat_tensors, dim=1)  # (N, out_channels*(1 + n_skips), ...)

            # concat된 채널 수가 out_channels가 아닐 수도 있으므로, 
            # 1x1 conv로 일단 out_channels로 줄이는 과정을 삽입
            # => post_conv_stack 시작 전 처리
            in_ch = cat_input.shape[1]
            if in_ch != self.out_channels:
                if self.spatial_dims == 3:
                    conv1x1 = nn.Conv3d(in_ch, self.out_channels, kernel_size=1, bias=True)
                else:
                    conv1x1 = nn.Conv2d(in_ch, self.out_channels, kernel_size=1, bias=True)
                cat_input = conv1x1(cat_input)

        # 3) post_conv_stack
        out = self.post_conv_stack(cat_input)
        return out


class FlexibleUNet(nn.Module):
    """
    FlexibleUNet: 인코더와 디코더가 자유롭게 설정 가능한 U-Net 변형.
    """
    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        encoder_channels: Sequence[int],
        encoder_strides: Sequence[int],
        decoder_channels: Sequence[int],
        decoder_strides: Sequence[int],
        num_layers_encoder: Sequence[int],
        num_layers_decoder: Sequence[int],
        skip_connections: dict[int, list[int]] | None = None,
        kernel_size: int | Sequence[int] = 3,
        up_kernel_size: int | Sequence[int] = 3,
        act: tuple | str = Act.LEAKYRELU,
        norm: tuple | str = Norm.BATCH,
        dropout: float = 0.0,
        bias: bool = True,
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
        self.encoder_blocks = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()

        # ------------------------------
        # 1) 인코더 구성
        # ------------------------------
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

        # ------------------------------
        # 2) 디코더 구성
        # ------------------------------
        main_in_ch = encoder_channels[-1]
        for i in range(len(decoder_channels)):
            out_ch = decoder_channels[i]
            stride = decoder_strides[i]
            nlayer = num_layers_decoder[i]

            block = SingleDecoderBlock(
                spatial_dims=spatial_dims,
                main_in_channels=main_in_ch,
                out_channels=out_ch,
                num_layers=nlayer,
                kernel_size=up_kernel_size,
                stride=stride,
                act=act,
                norm=norm,
                dropout=dropout,
                bias=bias,
                align_skip_channel=True,
            )
            self.decoder_blocks.append(block)
            main_in_ch = out_ch

        # ------------------------------
        # 3) 최종 출력 Conv
        # ------------------------------
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
        # 인코더
        encoder_outputs = []
        out = x
        for enc in self.encoder_blocks:
            out = enc(out)
            encoder_outputs.append(out)

        # 디코더
        out = encoder_outputs[-1]  # bottleneck
        for i, dec in enumerate(self.decoder_blocks):
            skip_idx_list = self.skip_connections.get(i, [])
            skip_list = [encoder_outputs[idx] for idx in skip_idx_list]
            out = dec(out, skip_list)

        # 최종 출력
        out = self.final_conv(out)
        return out


# -------------------------
# 간단 테스트
# -------------------------
if __name__ == "__main__":
    # 인코더
    enc_channels = (32,64,128,256)   # 3단 인코더 마지막 출력은 항상 Stride=1
    enc_strides = (2, 2)         # 2개 스트라이드
    num_layers_enc = (1, 1, 1,2)   # 각 인코더 블록당 2 레이어

    # 디코더
    dec_channels = (80, 80,80)  # 디코더 채널은 임의로 설정
    dec_strides = (2, 2,2)       # 3단 디코더, 각각 stride=2
    num_layers_dec = (2, 2,2)

    # skip 연결: 
    #   디코더 0번 -> 인코더 2번 출력
    #   디코더 1번 -> 인코더 1번 출력
    #   디코더 2번 -> 인코더 0번 출력
    skip_map = {
        0: [0],
        1: [1],
        2: [2,1,0],
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
    )

    print(net)
    x = torch.randn(1, 1, 64, 64, 64)
    with torch.no_grad():
        y = net(x)
    print("Output shape:", y.shape)
