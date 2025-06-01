import os
import shutil

def merge_png_files(folder_paths, output_folder):
    """
    将多个文件夹中的所有 .png 文件按路径顺序统一复制到一个整合的大文件夹中。
    保证每个文件夹中的文件顺序不变。

    :param folder_paths: 包含文件夹路径的列表
    :param output_folder: 输出文件夹路径
    """
    # 确保输出目录存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 用于记录当前文件编号
    file_counter = 1

    # 遍历每个文件夹路径
    for folder_path in folder_paths:
        # 获取当前文件夹中的所有 .png 文件
        png_files = [f for f in os.listdir(folder_path) if f.endswith('.png')]
        # 按文件名排序，确保顺序一致
        png_files.sort()

        # 遍历每个文件
        for file_name in png_files:
            # 构建源文件路径和目标文件路径
            source_path = os.path.join(folder_path, file_name)
            # 生成目标文件名，格式为 "number.png"，确保顺序
            target_file_name = f"{file_counter:04d}.png"
            target_path = os.path.join(output_folder, target_file_name)

            # 复制文件
            shutil.copy(source_path, target_path)
            print(f"Copied {source_path} to {target_path}")

            # 更新文件编号
            file_counter += 1

    print(f"All files have been merged into {output_folder}")
# 示例用法
folder_paths = [
    "/data3/davidqu/python_project/hover_net/F1_Train/Images",
    "/data3/davidqu/python_project/hover_net/F2_Train/Images",
    "/data3/davidqu/python_project/hover_net/F3_Train/Images"
]
output_folder = "/data3/davidqu/python_project/hover_net/All_Train/Images"

merge_png_files(folder_paths, output_folder)