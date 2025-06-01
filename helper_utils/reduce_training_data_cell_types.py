import numpy as np
import os
import shutil
from tqdm import tqdm

def process_and_save_npy_files(input_dir, output_dir):
    """
    处理 input_dir 中的所有 .npy 文件，修改第四个和第五个通道的值，
    并将结果保存到 output_dir 中。
    """
    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 获取输入目录中的所有 .npy 文件
    npy_files = [f for f in os.listdir(input_dir) if f.endswith('.npy')]

    # 使用 tqdm 创建进度条
    for file_name in tqdm(npy_files, desc="Processing files", unit="file"):
        # 构建输入和输出文件路径
        input_path = os.path.join(input_dir, file_name)
        output_path = os.path.join(output_dir, file_name)

        # 加载 .npy 文件
        data = np.load(input_path)

        # 检查数据形状是否符合预期
        if data.shape != (256, 256, 5):
            raise ValueError(f"Unexpected shape for file {file_name}: {data.shape}")

        # 提取第四个和第五个通道
        inst_map = data[..., 3]  # 实例标注
        type_map = data[..., 4]  # 类别标注

        # 找到类别标注中值为 3、4、5 的位置
        mask = np.isin(type_map, [3, 4, 5])

        # 将这些位置的类别标注和实例标注都设置为 3
        type_map[mask] = np.float64(3)
        inst_map[mask] = np.float64(3)

        # 将修改后的通道写回原数组
        data[..., 3] = inst_map
        data[..., 4] = type_map

        # 保存修改后的数组到输出目录
        np.save(output_path, data)

    print(f"Processed {len(npy_files)} files. Results saved to {output_dir}")

# 示例用法
input_directory = "/data3/davidqu/python_project/hover_net/Training_Data/pannuke/pannuke/valid/256x256_256x256"  # 输入文件夹路径
output_directory = "/data3/davidqu/python_project/hover_net/Training_Data/pannuke/three_types/valid/256x256_256x256"  # 输出文件夹路径

process_and_save_npy_files(input_directory, output_directory)