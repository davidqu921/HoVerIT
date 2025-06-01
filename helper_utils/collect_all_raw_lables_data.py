import os
import shutil
from tqdm import tqdm

def merge_mat_files(folder_paths, output_folder):
    """
    将多个文件夹中的所有 .mat 文件按路径顺序统一复制到一个整合的大文件夹中，
    使用全新的编号来命名文件，并打印详细信息。
    """
    # 确保输出目录存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 初始化文件编号计数器
    file_counter = 1

    # 获取所有文件夹中的 .mat 文件，并按路径顺序排列
    all_files = []
    for folder_path in folder_paths:
        files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.mat')]
        files.sort()  # 确保文件顺序一致
        all_files.extend(files)

    # 使用 tqdm 创建进度条
    for file_path in tqdm(all_files, desc="Copying files", unit="file"):
        # 构建目标文件路径，文件名格式为 "number.mat"
        target_file_name = f"{file_counter:04d}.mat"
        target_path = os.path.join(output_folder, target_file_name)
        # 复制文件
        shutil.copy(file_path, target_path)
        # 打印详细信息
        print(f"Copied {file_path} to {target_path}")
        # 更新文件编号计数器
        file_counter += 1

    print(f"All files have been merged into {output_folder}")


# 示例用法
folder_paths = [
    "/data3/davidqu/python_project/hover_net/F1_Train/Lables",
    "/data3/davidqu/python_project/hover_net/F2_Train/Lables",
    "/data3/davidqu/python_project/hover_net/F3_Train/Lables"
]
output_folder = "/data3/davidqu/python_project/hover_net/All_Train/Lables"

merge_mat_files(folder_paths, output_folder)