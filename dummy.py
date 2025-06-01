from models.hovernet.net_desc_SwinViT import create_model
import torch
import torch.nn as nn

dummy = torch.randn(1, 3, 256, 256)
model = create_model(input_ch=3, nr_types=4, mode="fast")  # 替换为你的 SwinViT 构造函数
model.eval()
out = model(dummy)
np_out, hv_out, tp_out = out["np"], out["hv"], out["tp"]
print(f"Final Print: {np_out.shape}, {hv_out.shape}, {tp_out.shape}")