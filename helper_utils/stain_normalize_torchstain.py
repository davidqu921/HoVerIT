import os
import torch
from torchvision import transforms
import torchstain
import cv2
import numpy as np
from tqdm import tqdm

# --- 1. 配置路径 ---
target_path = "/data3/davidqu/python_project/hover_net/All_Test/Images/0277.png"  # 用作归一化的参考图像
input_dir = "/data3/davidqu/python_project/hover_net/All_Train/Images"            # 要归一化的 PNG 图像文件夹
output_dir = "/data3/davidqu/python_project/hover_net/PanNuke_stainNormalized/train"         # 输出保存路径

os.makedirs(output_dir, exist_ok=True)

# --- 2. 加载 target 图像并初始化 normalizer ---
target = cv2.cvtColor(cv2.imread(target_path), cv2.COLOR_BGR2RGB)
T = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x * 255)
])

normalizer = torchstain.normalizers.MacenkoNormalizer(backend='torch')
normalizer.fit(T(target))

# --- 3. 遍历 input_dir 中的所有 png 图像 ---
for filename in tqdm(os.listdir(input_dir)):
    if not filename.lower().endswith(".png"):
        continue

    input_path = os.path.join(input_dir, filename)
    output_path = os.path.join(output_dir, filename)

    # 读取要处理的图像
    to_transform = cv2.cvtColor(cv2.imread(input_path), cv2.COLOR_BGR2RGB)
    t_to_transform = T(to_transform)

    # 执行颜色归一化
    try:
        norm, _, _ = normalizer.normalize(I=t_to_transform, stains=True)
    except Exception as e:
        print(f"[WARNING] Skipping {filename} due to error: {e}")
        continue

    # 转为 numpy，转为 [H, W, C]
    norm_np = norm.detach().cpu().numpy()
    norm_np = np.clip(norm_np, 0, 255).astype(np.uint8)

    # 转为 BGR 并保存
    norm_bgr = cv2.cvtColor(norm_np, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, norm_bgr)

print("Save the stain normalized images successfully:", output_dir)