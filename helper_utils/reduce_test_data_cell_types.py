import numpy as np
import scipy.io as sio
import os
from tqdm import tqdm

def process_and_save_mat_files(input_dir, output_dir):
    """
    处理 input_dir 中的所有 .mat 文件，修改 mask 变量的第二个通道中数值为 3、4、5 的位置，
    并将第一个通道中对应位置的实例标注也设置为背景值 0。
    修改后的文件将保存到 output_dir 中。
    """
    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 获取输入目录中的所有 .mat 文件
    mat_files = [f for f in os.listdir(input_dir) if f.endswith('.mat')]

    # 使用 tqdm 创建进度条
    for file_name in tqdm(mat_files, desc="Processing files", unit="file"):
        # 构建输入和输出文件路径
        input_path = os.path.join(input_dir, file_name)
        output_path = os.path.join(output_dir, file_name)

        # 加载 .mat 文件
        data = sio.loadmat(input_path)
        mask = data['mask']  # mask 是一个 (256, 256, 2) 的数组

        # 检查 mask 的形状是否符合预期
        if mask.shape != (256, 256, 2):
            raise ValueError(f"Unexpected shape for file {file_name}: {mask.shape}")

        # 提取第一个通道（实例分割标注）和第二个通道（类别分类标注）
        inst_map = mask[..., 0]
        type_map = mask[..., 1]

        # 找到类别分类标注中数值为 3、4、5 的位置
        mask_to_remove = np.isin(type_map, [3, 4, 5])

        # 将这些位置的类别分类标注和实例分割标注都设置为背景值 0
        type_map[mask_to_remove] = np.float64(3)
        inst_map[mask_to_remove] = np.float64(3)

        # 将修改后的通道写回原数组
        mask[..., 0] = inst_map
        mask[..., 1] = type_map

        # 保存修改后的数组到输出目录
        sio.savemat(output_path, {'mask': mask})

    print(f"Processed {len(mat_files)} files. Results saved to {output_dir}")

# 示例用法
input_directory = "/data3/davidqu/python_project/hover_net/All_Test/Lables"  # 输入文件夹路径
output_directory = "/data3/davidqu/python_project/hover_net/Test_Data/test_lable_3_types"       # 输出文件夹路径

process_and_save_mat_files(input_directory, output_directory)