import torch.nn as nn
from einops import rearrange
from timm.layers import DropPath


class ConvModulation(nn.Module):
    """卷积调制层（核心模块）"""

    def __init__(self, dim, kernel_size=3):
        super().__init__()
        self.dw_conv = nn.Conv2d(dim, dim, kernel_size, padding=kernel_size // 2, groups=dim)
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim // 4, 1),
            nn.GELU(),
            nn.Conv2d(dim // 4, dim, 1),
            nn.Sigmoid()
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        B, C, H, W = x.shape
        local_feat = self.dw_conv(x)
        channel_scale = self.channel_att(local_feat)
        modulated_feat = local_feat * channel_scale
        modulated_feat = rearrange(modulated_feat, "b c h w -> b (h w) c")
        modulated_feat = self.norm(modulated_feat)
        modulated_feat = rearrange(modulated_feat, "b (h w) c -> b c h w", h=H, w=W)
        return modulated_feat + x


class Block(nn.Module):
    def __init__(self, dim, kernel_size=3, mlp_ratio=4.0, drop_path_prob=0.05):  # 降低drop_path概率
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.conv_mod = ConvModulation(dim, kernel_size)
        self.drop_path1 = DropPath(drop_path_prob)

        self.norm2 = nn.LayerNorm(dim)
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
            nn.LayerNorm([dims[0], 64 // 4, 64 // 4])  # 需调整 shape
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
            nn.LayerNorm(dims[-1]),
            nn.Linear(dims[-1], num_classes)
        )

    def forward(self, x):
        x = self.stem(x)
        for stage in self.stages:
            x = stage(x)
        x = self.head(x)
        return x
