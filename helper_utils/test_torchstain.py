import torch
from torchvision import transforms
import torchstain
import cv2
import numpy as np


target = cv2.cvtColor(cv2.imread("/data3/davidqu/python_project/hover_net/All_Test/Images/0277.png"), cv2.COLOR_BGR2RGB)
to_transform = cv2.cvtColor(cv2.imread("/data3/davidqu/python_project/hover_net/All_Train/Images/0090.png"), cv2.COLOR_BGR2RGB)

T = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x*255)
])

normalizer = torchstain.normalizers.MacenkoNormalizer(backend='torch')
normalizer.fit(T(target))

t_to_transform = T(to_transform)
norm, H, E = normalizer.normalize(I=t_to_transform, stains=True)

print("norm shape:", norm.shape)

# 1. 将 norm 从 tensor 转为 numpy
norm_np = norm.detach().cpu().numpy()

print("norm_np shape:", norm_np.shape) 

# 2. Clamp 到 [0, 255] 并转为 uint8
norm_np = np.clip(norm_np, 0, 255).astype(np.uint8)

print("norm_np shape after clip:", norm_np.shape)  # 应该是 torch.Size([3, H, W])

# 3. 转回 BGR（因为 OpenCV 用 BGR 保存）
norm_bgr = cv2.cvtColor(norm_np, cv2.COLOR_RGB2BGR)

# 4. 保存
cv2.imwrite("/data3/davidqu/python_project/hover_net/to_fransform/0090_normalized.png", norm_bgr)
print("Save the stain normalized image successfully!")