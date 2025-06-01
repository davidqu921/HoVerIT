import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from PIL import Image

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

def process_and_save_images(input_directory, output_directory):
    """
    处理输入文件夹中的所有 PNG 图像文件，将每个图像扩展到 540x540 并保存到指定文件夹中。
    """
    # 确保输出目录存在
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # 获取输入目录中的所有 PNG 文件
    png_files = [f for f in os.listdir(input_directory) if f.endswith('.png')]

    # 初始化 PatchExtractor
    extractor = PatchExtractor(win_size=(540, 540), step_size=(164, 164), debug=False)

    # 使用 tqdm 创建进度条
    for file_name in tqdm(png_files, desc="Processing files", unit="file"):
        input_path = os.path.join(input_directory, file_name)
        output_path = os.path.join(output_directory, file_name)

        # 加载 PNG 图像
        img = np.array(Image.open(input_path))

        # 使用镜像填充扩展图像
        expanded_img = extractor._PatchExtractor__extract_mirror(img)

        # 保存扩展后的图像
        plt.imsave(output_path, expanded_img, cmap='gray')
        print(f"Saved expanded image to {output_path}")

# 示例用法
input_directory = "/data3/davidqu/python_project/hover_net/All_Test/Images"  # 输入文件夹路径
output_directory = "/data2/davidqu/hover_net/mirror_data/mirror_image"  # 输出文件夹路径

process_and_save_images(input_directory, output_directory)