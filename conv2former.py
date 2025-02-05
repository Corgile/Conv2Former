import torch
import torch.nn as nn
from einops import rearrange


class ConvModulation(nn.Module):
    """卷积调制层（核心模块）"""

    def __init__(self, dim, kernel_size=3):
        super().__init__()
        # 深度可分离卷积提取局部特征
        self.dw_conv = nn.Conv2d(dim, dim, kernel_size, padding=kernel_size // 2, groups=dim)
        # 通道注意力调制
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim // 8, 1),
            nn.GELU(),
            nn.Conv2d(dim // 8, dim, 1),
            nn.Sigmoid()
        )
        # Layer Norm
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        B, C, H, W = x.shape
        # 局部卷积特征
        local_feat = self.dw_conv(x)
        # 通道注意力权重
        channel_scale = self.channel_att(local_feat)
        # 调制后的特征
        modulated_feat = local_feat * channel_scale
        # LayerNorm（需转为通道最后形式）
        modulated_feat = rearrange(modulated_feat, "b c h w -> b (h w) c")
        modulated_feat = self.norm(modulated_feat)
        modulated_feat = rearrange(modulated_feat, "b (h w) c -> b c h w", h=H, w=W)
        return modulated_feat + x  # 残差连接


class Conv2FormerBlock(nn.Module):
    """Conv2Former 基础块"""

    def __init__(self, dim, kernel_size=3, mlp_ratio=4.0):
        super().__init__()
        self.conv_mod = ConvModulation(dim, kernel_size)
        # 前馈网络（FFN）
        hidden_dim = int(dim * mlp_ratio)
        self.ffn = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, 1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, dim, 1)
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        # 卷积调制分支
        x = self.conv_mod(x)
        # FFN分支
        B, C, H, W = x.shape
        ffn_input = rearrange(x, "b c h w -> b (h w) c")
        ffn_input = self.norm(ffn_input)
        ffn_input = rearrange(ffn_input, "b (h w) c -> b c h w", h=H, w=W)
        ffn_out = self.ffn(ffn_input)
        return x + ffn_out  # 残差连接


class Conv2Former(nn.Module):
    """完整的 Conv2Former 模型"""

    def __init__(self, in_chans=1, num_classes=6, dims=[64, 128, 256, 512], depths=[2, 2, 6, 2]):
        super().__init__()
        # 输入 Stem
        self.stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], 4, stride=4),  # 64x64 -> 16x16
            # nn.LayerNorm(dims[0]),
            nn.BatchNorm2d(dims[0]),
        )
        # 分阶段构建网络
        self.stages = nn.ModuleList()
        for i in range(len(dims)):
            stage = nn.Sequential(
                *[Conv2FormerBlock(dims[i]) for _ in range(depths[i])],
                nn.Conv2d(dims[i], dims[i + 1] if i < len(dims) - 1 else dims[i], 2, stride=2)  # 下采样
            )
            self.stages.append(stage)

        # 分类头
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


# ----------------------------------------
# 测试模型 & 可视化
# ----------------------------------------
from torchviz import make_dot

if __name__ == "__main__":
    model = Conv2Former(num_classes=6)
    print(model)
    dummy_input = torch.randn(2, 1, 64, 64)
    output = model.forward(dummy_input)
    print(output.shape)  # 应为 torch.Size([2, 6])
    dot = make_dot(output, params=dict(model.named_parameters()))
    # 生成 conv2former_model.png 文件
    dot.render("intermediates/conv2former_model", format="png")
