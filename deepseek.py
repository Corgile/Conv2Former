import timm.models.layers
import torch
import torch.nn as nn
from einops import rearrange
import torch.nn.functional as F
from timm.models.layers import DropPath, trunc_normal_

from timm.models.layers import LayerNorm2d


class LayerNorm2D(nn.Module):
    """
    LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class ChannelAttention(nn.Module):
    """
    通道注意力模型: 通道维度不变，压缩空间维度。该模块关注输入图片中有意义的信息。
    1）假设输入的数据大小是(b,c,w,h)
    2）通过自适应平均池化使得输出的大小变为(b,c,1,1)
    3）通过2d卷积和sigmod激活函数后，大小是(b,c,1,1)
    4）将上一步输出的结果和输入的数据相乘，输出数据大小是(b,c,w,h)。
    """

    # Channel-attention module https://github.com/open-mmlab/mmdetection/tree/v3.0.0rc1/configs/rtmdet
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(channels, channels, 1, 1, 0, bias=True)
        self.act = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.act(self.fc(self.pool(x)))


class SpatialAttention(nn.Module):
    """
    空间注意力模块：空间维度不变，压缩通道维度。该模块关注的是目标的位置信息。
    1） 假设输入的数据x是(b,c,w,h)，并进行两路处理。
    2）其中一路在通道维度上进行求平均值，得到的大小是(b,1,w,h)；另外一路也在通道维度上进行求最大值，得到的大小是(b,1,w,h)。
    3） 然后对上述步骤的两路输出进行连接，输出的大小是(b,2,w,h)
    4）经过一个二维卷积网络，把输出通道变为1，输出大小是(b,1,w,h)
    4）将上一步输出的结果和输入的数据x相乘，最终输出数据大小是(b,c,w,h)。
    """

    # Spatial-attention module
    def __init__(self, kernel_size=7):
        super().__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.cv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.act = nn.Sigmoid()

    def forward(self, x):
        return x * self.act(self.cv1(torch.cat([torch.mean(x, 1, keepdim=True), torch.max(x, 1, keepdim=True)[0]], 1)))


class CBAM(nn.Module):
    def __init__(self, n_channels, kernel_size=7):  # ch_in, kernels
        super().__init__()
        self.channel_attention = ChannelAttention(n_channels)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        return self.spatial_attention(self.channel_attention(x))


class ConvModulation(nn.Module):

    def __init__(self, dim, kernel_size=3):
        super().__init__()
        self.dw_conv = nn.Conv2d(dim, dim, kernel_size, padding=kernel_size // 2, groups=dim)
        # TODO: 这里单通道,应该用空间注意力，改成CBAM
        # TODO: 后面得增加到3通道，然后用channel att
        self.cbam_atten = CBAM(n_channels=1, kernel_size=kernel_size)
        self.norm = LayerNorm2d(dim)

    def forward(self, x):
        # B, C, H, W = x.shape
        local_feat = self.dw_conv(x)
        channel_scale = self.cbam_atten(local_feat)
        modulated_feat = local_feat * channel_scale
        # modulated_feat = rearrange(modulated_feat, "b c h w -> b (h w) c")
        modulated_feat = self.norm(modulated_feat)
        # modulated_feat = rearrange(modulated_feat, "b (h w) c -> b c h w", h=H, w=W)
        return modulated_feat + x


class Block(nn.Module):
    def __init__(self, dim, kernel_size=3, mlp_ratio=4.0, drop_path_prob=0.05):  # 降低drop_path概率
        super().__init__()
        self.norm1 = LayerNorm2d(dim)
        self.conv_mod = ConvModulation(dim, kernel_size)
        self.drop_path1 = DropPath(drop_path_prob)

        self.norm2 = LayerNorm2d(dim)
        hidden_dim = int(dim * mlp_ratio)
        self.ffn = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, 1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, dim, 1)
        )
        self.drop_path2 = DropPath(drop_path_prob)

    def forward(self, x):
        B, C, H, W = x.shape
        x_norm = rearrange(x, "b c h w -> b (h w) c")
        x_norm = self.norm1(x_norm)
        x_norm = rearrange(x_norm, "b (h w) c -> b c h w", h=H, w=W)
        x = x + self.drop_path1(self.conv_mod(x_norm))

        x_norm = rearrange(x, "b c h w -> b (h w) c")
        x_norm = self.norm2(x_norm)
        x_norm = rearrange(x_norm, "b (h w) c -> b c h w", h=H, w=W)
        x = x + self.drop_path2(self.ffn(x_norm))
        return x


class ModFormer(nn.Module):
    def __init__(self, in_chans=1, num_classes=6, dims=None, depths=None):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], 4, stride=4),
            LayerNorm2d([dims[0], 64 // 4, 64 // 4])  # 需调整 shape
        )
        self.stages = nn.ModuleList()
        for i in range(len(dims)):
            # 深层逐渐增加 drop_path_prob
            stage = nn.Sequential(
                *[Block(dims[i], drop_path_prob=0.06) for _ in range(depths[i])],  # 降低drop_path概率
                nn.Conv2d(dims[i], dims[i + 1] if i < len(dims) - 1 else dims[i], 2, stride=2)
            )
            self.stages.append(stage)

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            LayerNorm2d(dims[-1]),
            nn.Linear(dims[-1], num_classes)
        )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            # nn.init.constant_(m.bias, 0)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, timm.models.layers.LayerNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        x = self.stem(x)
        for stage in self.stages:
            x = stage(x)
        x = self.head(x)
        return x
