import scipy.io as sio
import numpy as np
import os
from tqdm import tqdm

class PatchExtractor:
    def __init__(self, win_size=(540, 540), step_size=(164, 164), debug=False):
        """
        初始化 PatchExtractor。
        :param win_size: 窗口大小 (height, width)
        :param step_size: 步长 (height, width)
        :param debug: 是否为调试模式
        """
        self.win_size = win_size
        self.step_size = step_size
        self.debug = debug

    def __extract_mirror(self, x):
        """
        使用镜像填充提取子块。
        :param x: 输入图像，形状为 (H, W, C)
        :return: 子块列表，每个子块的类型与 x 相同
        """
        # 计算填充大小
        diff_h = self.win_size[0] - self.step_size[0]
        padt = diff_h // 2
        padb = diff_h - padt

        diff_w = self.win_size[1] - self.step_size[1]
        padl = diff_w // 2
        padr = diff_w - padl

        # 选择填充类型
        pad_type = "constant" if self.debug else "reflect"
        x = np.pad(x, ((padt, padb), (padl, padr), (0, 0)), pad_type)

        return x

def process_and_save_mat_files(input_directory, output_directory):
    """
    处理输入文件夹中的所有 .mat 文件，将每个文件中的 'mask' 数据通过镜像填充扩展到 540x540x2，
    并保存到指定文件夹中。
    """
    # 确保输出目录存在
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # 获取输入目录中的所有 .mat 文件
    mat_files = [f for f in os.listdir(input_directory) if f.endswith('.mat')]

    # 初始化 PatchExtractor
    extractor = PatchExtractor(win_size=(540, 540), step_size=(164, 164), debug=False)

    # 使用 tqdm 创建进度条
    for file_name in tqdm(mat_files, desc="Processing files", unit="file"):
        input_path = os.path.join(input_directory, file_name)
        output_path = os.path.join(output_directory, file_name)

        # 加载 .mat 文件
        data = sio.loadmat(input_path)

        # 提取 'mask'
        if 'mask' not in data:
            raise KeyError(f"'mask' not found in {input_path}")

        mask = data['mask']

        # 检查 mask 的形状是否符合预期
        if mask.shape != (256, 256, 2):
            raise ValueError(f"Expected 'mask' to have shape (256, 256, 2), but got shape {mask.shape}")

        # 使用镜像填充扩展 mask
        expanded_mask = extractor._PatchExtractor__extract_mirror(mask)

        # 保存扩展后的 mask 到 .mat 文件
        sio.savemat(output_path, {'mask': expanded_mask})
        print(f"Saved expanded mask to {output_path}")

# 示例用法
input_directory = "/data3/davidqu/python_project/hover_net/Test_Data/test_lable_type_reduced"  # 输入文件夹路径
output_directory = "/data2/davidqu/hover_net/mirror_data/mirror_gt_lable"  # 输出文件夹路径

process_and_save_mat_files(input_directory, output_directory)