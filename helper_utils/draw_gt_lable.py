import os
import numpy as np
from scipy.io import loadmat
from PIL import Image
import cv2

def annotate_images(mat_folder, png_folder, output_folder):
    """
    根据.mat文件中的实例分割和类别标签，在对应的.png图像上进行标注。

    参数:
        mat_folder: 包含.mat文件的文件夹路径
        png_folder: 包含.png文件的文件夹路径
        output_folder: 保存标注后的图像的文件夹路径
    """
    # 创建输出文件夹（如果不存在）
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 获取文件夹中的文件名
    mat_files = sorted(os.listdir(mat_folder))
    png_files = sorted(os.listdir(png_folder))

    # 确保文件数量一致
    if len(mat_files) != len(png_files):
        raise ValueError("MAT文件和PNG文件的数量不匹配！")

    # 定义颜色映射
    color_map = {
        0: (0, 0, 0),    # 背景：黑色
        1: (255, 0, 0),  # 类别1：红色
        2: (0, 255, 0),  # 类别2：绿色
        3: (0, 0, 255),   # 类别3：蓝色
        4: (255, 255, 0),  #黄
        5: (255, 165, 0)   #橙
    }

    color_map_bgr = {k: (v[2], v[1], v[0]) for k, v in color_map.items()}

    # 遍历文件
    for mat_file, png_file in zip(mat_files, png_files):
        if not mat_file.endswith('.mat') or not png_file.endswith('.png'):
            continue

        # 加载.mat文件
        mat_path = os.path.join(mat_folder, mat_file)
        mat_data = loadmat(mat_path)
        mask = mat_data['mask']  # 假设 'mask' 是实例分割和类别标签的键

        # 加载.png文件
        png_path = os.path.join(png_folder, png_file)
        image = cv2.imread(png_path)  # 使用OpenCV加载图像

        # 遍历每个实例
        for instance_id in np.unique(mask[..., 0]):
            if instance_id == 0:  # 跳过背景
                continue

            # 获取当前实例的掩码
            instance_mask = mask[..., 0] == instance_id
            instance_type = mask[..., 1][instance_mask][0]  # 获取类别标签cd

            # 获取实例的轮廓
            contours, _ = cv2.findContours(instance_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # 根据类别选择颜色
            color = color_map_bgr.get(instance_type, (0, 0, 0))

            # 在图像上绘制轮廓
            cv2.drawContours(image, contours, -1, color, thickness=2)

        # 保存标注后的图像
        output_path = os.path.join(output_folder, png_file)
        cv2.imwrite(output_path, image)

        print(f"Processed {png_file} and saved to {output_path}")

    print("标注完成！")

# 示例调用
mat_folder = '/data3/davidqu/python_project/hover_net/All_Test/Lables'
png_folder = '/data3/davidqu/python_project/hover_net/All_Test/Images'
output_folder = '/data2/davidqu/hover_net/gt_img_5_types'
annotate_images(mat_folder, png_folder, output_folder)