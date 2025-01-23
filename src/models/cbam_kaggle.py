import torch
import torch.nn as nn
import torch.nn.functional as F

from unet_block import get_conv_layer

class ChannelAttention3D(nn.Module):
    """
    3D Channel Attention 모듈
    입력 텐서 크기: (B, F, D, W, H)
    """
    def __init__(self, channels, reduction=16):
        super(ChannelAttention3D, self).__init__()
        self.channels = channels
        self.reduction = reduction
        
        # 채널 축소 -> 복원 MLP
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels)
        )

    def forward(self, x):
        # x.shape = (B, F, D, W, H)
        b, f, d, w, h = x.shape
        
        # 1) Global Average Pooling
        avg_pool = F.adaptive_avg_pool3d(x, (1, 1, 1)).view(b, f)  # (B, F)
        
        # 2) Global Max Pooling
        # max_pool = F.adaptive_max_pool3d(x, (1, 1, 1)).view(b, f)  # (B, F)
        min_pool = -F.adaptive_max_pool3d(-x, (1, 1, 1)).view(b, f)  # (B, F)
        
        # 3) 각각 MLP 통과
        avg_out = self.mlp(avg_pool)  # (B, F)
        # max_out = self.mlp(max_pool)  # (B, F)
        min_out = self.mlp(min_pool)
        
        # 4) 두 결과 더하고 시그모이드
        # out = torch.sigmoid(avg_out + max_out)  # (B, F)
        out = torch.sigmoid(avg_out + min_out)
        
        # 5) (B, F) -> (B, F, 1, 1, 1)
        out = out.view(b, f, 1, 1, 1)
        
        # 6) 입력 x에 채널 가중치 곱
        return x * out

class SpatialAttention3D(nn.Module):
    """
    3D Spatial Attention 모듈
    입력 텐서 크기: (B, F, D, W, H)
    """
    def __init__(self, kernel_size=7):
        super(SpatialAttention3D, self).__init__()
        
        # kernel_size가 7이면, padding=3을 주어 입력 크기가 유지되도록 할 수 있음
        # 다만 3D에서 D차원까지 7로 크게 잡으면 연산량이 커질 수 있으니
        # 3 등으로 줄여보는 것도 방법.
        # 여기서는 예시로 일단 kernel_size=3, padding=1로 구현
        # 필요에 따라 변경 가능
        self.conv = get_conv_layer(
            spatial_dims=3,
            in_channels=2,    # 평균풀링, 최대풀링을 합친 2채널
            out_channels=1,
            kernel_size=7,    # 필요시 7이나 다른 값으로 변경
            bias=False
        )

    def forward(self, x):
        # x.shape = (B, F, D, W, H)
        b, f, d, w, h = x.shape
        
        # 1) 채널 차원에 대한 평균 풀링, 최대 풀링
        avg_pool = torch.mean(x, dim=1, keepdim=True)        # (B, 1, D, W, H)
        # max_pool, _ = torch.max(x, dim=1, keepdim=True)      # (B, 1, D, W, H)
        min_pool, _ = torch.min(x, dim=1, keepdim=True)      # (B, 1, D, W, H)
        
        # 2) 채널 차원으로 합치기
        # cat = torch.cat([avg_pool, max_pool], dim=1)         # (B, 2, D, W, H)
        cat = torch.cat([avg_pool, min_pool], dim=1)         # (B, 2, D, W, H)
        
        # 3) 3D Conv + 시그모이드
        attention_map = torch.sigmoid(self.conv(cat))        # (B, 1, D, W, H)
        
        # 4) 입력 x에 곱하기
        return x * attention_map

class CBAM3D(nn.Module):
    """
    3D CBAM: ChannelAttention3D + SpatialAttention3D
    """
    def __init__(self, channels, reduction=16, spatial_kernel_size=7):
        super(CBAM3D, self).__init__()
        self.channel_attention = ChannelAttention3D(channels, reduction=reduction)
        self.spatial_attention = SpatialAttention3D(kernel_size=spatial_kernel_size)

    def forward(self, x):
        # 1) 채널 어텐션
        out = self.channel_attention(x)
        # 2) 공간 어텐션
        out = self.spatial_attention(out)
        return out
    
if __name__ == "__main__":
    # strides를 4개로 늘림
        # (B, F, D, W, H) = (2, 16, 8, 16, 16) 형태의 텐서 예시
    x = torch.randn(2, 16, 8, 16, 16)
    
    # CBAM3D 선언
    cbam_3d = CBAM3D(channels=16, reduction=8, spatial_kernel_size=3)
    
    # 순전파
    y = cbam_3d(x)
    print(f"입력 크기: {x.shape}, 출력 크기: {y.shape}")