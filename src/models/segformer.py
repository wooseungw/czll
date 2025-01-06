from math import sqrt
from functools import partial
import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, reduce
from einops.layers.torch import Rearrange
from torchprofile import profile_macs

from monai.networks.blocks.convolutions import Convolution
from collections.abc import Sequence
from monai.networks.layers.factories import Act, Norm
from monai.networks.layers.utils import get_act_layer, get_norm_layer



def exists(val):
    return val is not None

def cast_tuple(val, depth):
    return val if isinstance(val, tuple) else (val,) * depth

# classes
'''
1. DsConv2d (Depthwise Separable Convolution)
입력 차원(dim_in)에서 출력 차원(dim_out)으로 매핑하는 두 단계 컨볼루션을 수행합니다.
첫 번째 단계는 depthwise convolution으로, 각 입력 채널에 대해 독립적으로 컨볼루션을 수행합니다.
두 번째 단계는 pointwise convolution(1x1 컨볼루션)으로, depthwise 단계의 출력을 결합하여 최종 출력 채널을 생성합니다
'''
class DsConv2d(nn.Module):
    def __init__(self, dim_in, dim_out, kernel_size, padding, stride = 1, bias = True):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim_in, dim_in, kernel_size = kernel_size, padding = padding, groups = dim_in, stride = stride, bias=bias),
            nn.Conv2d(dim_in, dim_out, kernel_size = 1, bias = bias)
        )
    def forward(self, x):
        return self.net(x)
'''
2. LayerNorm & PreNorm
LayerNorm은 채널 차원을 따라 정규화를 수행합니다. 이는 학습 과정을 안정화하고 가속화하는 데 도움을 줍니다.
PreNorm은 주어진 함수(fn, 예: 어텐션 또는 피드포워드 네트워크)를 적용하기 전에 입력을 정규화합니다.
'''    
class LayerNorm(nn.Module):
    def __init__(self, dim, eps = 1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        std = torch.var(x, dim = 1, unbiased = False, keepdim = True).sqrt()
        mean = torch.mean(x, dim = 1, keepdim = True)
        return (x - mean) / (std + self.eps) * self.g + self.b

    
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = LayerNorm(dim)

    def forward(self, x):
        return self.fn(self.norm(x))

'''
3. EfficientSelfAttention
Transformer 아키텍처의 핵심인 자기 주의 메커니즘을 구현합니다.
입력 이미지를 쿼리(q), 키(k), 값(v)으로 변환하고, 쿼리와 키의 유사도를 계산하여 어텐션 맵을 생성합니다.
이 어텐션 맵을 사용하여 값(v)을 가중 평균하여 출력을 생성합니다.
이 과정은 입력 특징의 중요한 부분을 강조하고 덜 중요한 부분을 억제합니다.
'''
class EfficientSelfAttention(nn.Module):
    def __init__(
        self,
        *,
        dim,
        heads,
        reduction_ratio
    ):
        super().__init__()
        self.scale = (dim // heads) ** -0.5
        self.heads = heads

        self.to_q = nn.Conv2d(dim, dim, 1, bias = False)
        self.to_kv = nn.Conv2d(dim, dim * 2, reduction_ratio, stride = reduction_ratio, bias = False)
        self.to_out = nn.Conv2d(dim, dim, 1, bias = False)

    def forward(self, x):
        h, w = x.shape[-2:]
        heads = self.heads

        q, k, v = (self.to_q(x), *self.to_kv(x).chunk(2, dim = 1))
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> (b h) (x y) c', h = heads), (q, k, v))

        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale
        attn = sim.softmax(dim = -1)

        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) (x y) c -> b (h c) x y', h = heads, x = h, y = w)
        return self.to_out(out)

'''
4. MixFeedForward
피드포워드 네트워크는 각 위치에서 독립적으로 동작하는 완전 연결 레이어입니다.
입력 차원을 확장한 후, Depthwise Separable Convolution을 적용하고, 다시 원래 차원으로 축소합니다.
'''    
class MixFeedForward(nn.Module):
    def __init__(
        self,
        *,
        dim,
        expansion_factor
    ):
        super().__init__()
        hidden_dim = dim * expansion_factor
        self.net = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, 1),
            DsConv2d(hidden_dim, hidden_dim, 3, padding = 1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, dim, 1)
        )

    def forward(self, x):
        return self.net(x)
'''
5. MiT (Mixer Transformer)
이미지를 여러 스테이지로 처리합니다. 각 스테이지는 이미지를 패치로 나누고, 패치를 임베딩한 후, 여러 개의 Transformer 레이어를 적용합니다.
이 과정은 이미지의 다양한 해상도에서 특징을 추출합니다. 
'''    
class MiT(nn.Module):
    def __init__(
        self,
        *,
        channels,
        dims,
        heads,
        ff_expansion,
        reduction_ratio,
        num_layers,
        stage_kernel_stride_pad = ((3, 2, 1),  
                                   (3, 2, 1), 
                                   (3, 2, 1), 
                                   (3, 2, 1))
    ):
        super().__init__()


        dims = (channels, *dims)
        dim_pairs = list(zip(dims[:-1], dims[1:]))

        self.stages = nn.ModuleList([])

        for (dim_in, dim_out), (kernel, stride, padding), num_layers, ff_expansion, heads, reduction_ratio in zip(
            dim_pairs, stage_kernel_stride_pad, num_layers, ff_expansion, heads, reduction_ratio):
            #여기서 너비와 높이가 같은 정사각형 패치를 사용합니다.
            get_overlap_patches = nn.Unfold(kernel, stride = stride, padding = padding)
            overlap_patch_embed = nn.Conv2d(dim_in * kernel ** 2, dim_out, 1)

            layers = nn.ModuleList([])

            for _ in range(num_layers):
                layers.append(nn.ModuleList([
                    PreNorm(dim_out, EfficientSelfAttention(dim = dim_out, heads = heads, reduction_ratio = reduction_ratio)),
                    PreNorm(dim_out, MixFeedForward(dim = dim_out, expansion_factor = ff_expansion)),
                ]))

            self.stages.append(nn.ModuleList([
                get_overlap_patches,
                overlap_patch_embed,
                layers
            ]))

    def forward(
        self,
        x,
        return_layer_outputs = False
    ):
        h, w = x.shape[-2:]

        layer_outputs = []
        for (get_overlap_patches, overlap_embed, layers) in self.stages:
            x = get_overlap_patches(x)

            num_patches = x.shape[-1]
            ratio = int(sqrt((h * w) / num_patches))
            x = rearrange(x, 'b c (h w) -> b c h w', h = h // ratio)

            x = overlap_embed(x)
            for (attn, ff) in layers:
                x = attn(x) + x
                x = ff(x) + x

            layer_outputs.append(x)

        ret = x if not return_layer_outputs else layer_outputs
        return ret

'''
6. Segformer
MiT를 통해 추출된 여러 스케일의 특징을 결합하고, 최종적으로 세그멘테이션 맵을 생성합니다.
각 스테이지의 출력을 디코더 차원으로 매핑하고, 업샘플링하여 동일한 해상도로 만든 후, 이를 결합합니다.
결합된 특징 맵을 사용하여 최종 세그멘테이션 맵을 생성합니다.
'''
class Segformer(nn.Module):
    def __init__(
        self,
        *,
        dims=(32, 64, 160, 256),
        heads=(1, 2, 5, 8),
        ff_expansion=(8, 8, 4, 4),
        reduction_ratio=(8, 4, 2, 1),
        num_layers=2,
        channels=3,
        decoder_dim=256,
        num_classes=19
    ):
        super().__init__()
        dims, heads, ff_expansion, reduction_ratio, num_layers = map(
            partial(cast_tuple, depth=4), (dims, heads, ff_expansion, reduction_ratio, num_layers))
        assert all([*map(lambda t: len(t) == 4, (dims, heads, ff_expansion, reduction_ratio, num_layers))]), \
            'only four stages are allowed, all keyword arguments must be either a single value or a tuple of 4 values'

        self.mit = MiT(
            channels=channels,
            dims=dims,
            heads=heads,
            ff_expansion=ff_expansion,
            reduction_ratio=reduction_ratio,
            num_layers=num_layers
        )

        self.to_fused = nn.ModuleList([nn.Sequential(
            nn.Conv2d(dim, decoder_dim, 1),
            nn.Upsample(scale_factor=2 ** i)
        ) for i, dim in enumerate(dims)])

        self.to_segmentation = nn.Sequential(
            nn.Conv2d(4 * decoder_dim, decoder_dim, 1),
            nn.Conv2d(decoder_dim, num_classes, 1),
        )

    def forward(self, x):
        layer_outputs = self.mit(x, return_layer_outputs=True)

        fused = [to_fused(output) for output, to_fused in zip(layer_outputs, self.to_fused)]
        fused = torch.cat(fused, dim=1)
        return self.to_segmentation(fused)
    
class Segwithcarbon(nn.Module):
    def __init__(
        self,
        *,
        dims=(64,128),
        heads=(1, 8),
        ff_expansion=(8, 4),
        reduction_ratio=(8, 4),
        num_layers=2,
        channels=3,
        decoder_dim=128,
        num_classes=19,
        stage_kernel_stride_pad = ((3, 1, 1), 
                                   (7, 4, 3))
    ):
        super().__init__()
        dim_len = len(dims)
        dims, heads, ff_expansion, reduction_ratio, num_layers = map(
            partial(cast_tuple, depth=dim_len), (dims, heads, ff_expansion, reduction_ratio, num_layers))
        assert all([*map(lambda t: len(t) == dim_len, (dims, heads, ff_expansion, reduction_ratio, num_layers))])

        self.mit = MiT(
            channels=channels,
            dims=dims,
            heads=heads,
            ff_expansion=ff_expansion,
            reduction_ratio=reduction_ratio,
            num_layers=num_layers,
            stage_kernel_stride_pad = stage_kernel_stride_pad
        )

        self.to_fused = nn.ModuleList([nn.Sequential(
            nn.Upsample(scale_factor=2 ** (i)),
            nn.Conv2d(dim, decoder_dim, 1),
            
        ) for i, dim in enumerate(dims)])

        self.segmentation_head = nn.Sequential(
            nn.Conv2d(dim_len * decoder_dim, decoder_dim, 1),
            nn.Conv2d(decoder_dim, num_classes, 1),
        )
        self.regression_head = nn.Sequential(
            nn.Conv2d(dim_len * decoder_dim,decoder_dim, 1),
            nn.Conv2d(decoder_dim, 1, 1),
        )

    def forward(self, x):
        layer_outputs = self.mit(x, return_layer_outputs=True)

        fused = [to_fused(output) for output, to_fused in zip(layer_outputs, self.to_fused)]
        print(fused[0].shape)
        print(fused[1].shape)
        print(fused[2].shape)
        print(fused[3].shape)
        
        fused = torch.cat(fused, dim=1)
        gt_preds = self.segmentation_head(fused)
        carbon_preds = self.regression_head(fused)
        return gt_preds, carbon_preds
    

''' 
데이터 흐름
입력 이미지는 MiT 모듈을 통해 여러 스테이지에 걸쳐 처리됩니다. 각 스테이지는 이미지를 더 작은 패치로 나누고, 이 패치들을 임베딩하여 Transformer 레이어에 입력합니다.
Transformer 레이어는 이미지의 글로벌한 컨텍스트를 모델링하며, 각 스테이지의 출력은 다음 스테이지의 입력으로 사용됩니다.
모든 스테이지를 거친 후, Segformer는 각 스테이지의 출력을 디코더 차원으로 매핑하고, 업샘플링하여 결합합니다.
결합된 특징 맵을 통해 최종적으로 각 픽셀의 클래스를 예측하는 세그멘테이션 맵을 생성합니다.
'''
if __name__ == "__main__":
    args = {
        'dims': (32, 64, 160, 256),
        'heads': (1, 2, 5, 8),
        'ff_expansion': (8, 8, 4, 4),
        'reduction_ratio': (8, 4, 2, 1),
        'num_layers': 2,
        'channels': 3,
        'decoder_dim': 256,
        'num_classes': 4
    }
    model = Segformer(**args)
    model.eval()
    input_tensor = torch.randn(1, 3, 224, 224)
    # Print the number of parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of parameters: {num_params}")
    output = model(input_tensor)
    print(output.shape)
    flops = profile_macs(model, input_tensor)
    print(f"FLOPs: {flops}")
    