import torch
from timm.models import SwinTransformer,SwinTransformerV2
from conv2former import Conv2Former
from ModFormer import ModFormer
from ablation.Att_ModFormer import ModFormer as AttModFormer
from ablation.Mod_ModFormer import ModFormer as ModModFormer

if __name__ == '__main__':
    # 参数量计算示例
    from thop import profile

    input = torch.randn(1, 1, 64, 64)

    model = Conv2Former(in_chans=1, dims=[96, 192, 384, 768], depths=[3, 3, 9, 3], num_classes=2, kernel_size=3)
    flops, params = profile(model, inputs=(input,), verbose=False)
    print(f"Conv2Former FLOPs: {flops / 1e9:.2f}G | Params: {params / 1e6:.2f}M")

    model = ModFormer(in_chans=1, dims=[96, 192, 384, 768], depths=[3, 3, 9, 3], num_classes=2, kernel_size=3)
    flops, params = profile(model, inputs=(input,), verbose=False)
    print(f"ModFormer FLOPs: {flops / 1e9:.2f}G | Params: {params / 1e6:.2f}M")

    model = SwinTransformer(img_size=64, in_chans=1, dims=[96, 192, 384, 768], depths=(3, 3, 9, 3), num_classes=2, kernel_size=3)
    flops, params = profile(model, inputs=(input,), verbose=False)
    print(f"SwinTransformer FLOPs: {flops / 1e9:.2f}G | Params: {params / 1e6:.2f}M")

    model = AttModFormer(in_chans=1, dims=[96, 192, 384, 768], depths=(3, 3, 9, 3), num_classes=2, kernel_size=3)
    flops, params = profile(model, inputs=(input,), verbose=False)
    print(f"AttModFormer FLOPs: {flops / 1e9:.2f}G | Params: {params / 1e6:.2f}M")

    model = ModModFormer(in_chans=1, dims=[96, 192, 384, 768], depths=(3, 3, 9, 3), num_classes=2, kernel_size=3)
    flops, params = profile(model, inputs=(input,), verbose=False)
    print(f"ModModFormer FLOPs: {flops / 1e9:.2f}G | Params: {params / 1e6:.2f}M")

