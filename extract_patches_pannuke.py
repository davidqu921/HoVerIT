"""extract_patches.py

Patch extraction script.
"""

import os  # 用于操作系统相关功能

import cv2
import numpy as np
import os
import json
import random
from tqdm import tqdm
from PIL import Image
from scipy.io import savemat
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

category_info = {
    0: "Background",
    1: "Neoplastic",
    2: "Non-Neoplastic Epithelial",
    3: "Inflammatory",
    4: "Connective",
    5: "Dead"
}

def check_dir(path):
    os.makedirs(path, exist_ok=True)

def npy2png():
    # 设置输入和输出文件夹路径
    input_folder_img = r"D:\PycharmProjects\zhongshan_hospital\cell_classifier\datasets\Fold 1\images\fold1\split_sub"
    output_folder_img = r"D:\PycharmProjects\zhongshan_hospital\cell_classifier\datasets\Fold 1\images\fold1\split_sub_png"
    input_folder_mask = r"D:\PycharmProjects\zhongshan_hospital\cell_classifier\datasets\Fold 1\masks\fold1\split_sub"
    output_folder_mask = r"D:\PycharmProjects\zhongshan_hospital\cell_classifier\datasets\Fold 1\masks\fold1\split_sub_png"
    output_folder_class = r"D:\PycharmProjects\zhongshan_hospital\cell_classifier\datasets\Fold 1\masks\fold1\class_label"
    output_folder_pixel = r"D:\PycharmProjects\zhongshan_hospital\cell_classifier\datasets\Fold 1\masks\fold1\pixel_inst_label"

    # 创建输出文件夹（如果不存在）
    os.makedirs(output_folder_mask, exist_ok=True)
    os.makedirs(output_folder_img, exist_ok=True)
    os.makedirs(output_folder_class, exist_ok=True)
    os.makedirs(output_folder_pixel, exist_ok=True)
    # 颜色映射（6 类别，每个类别指定 RGB 颜色）
    color_map = np.array([
        [255, 0, 0],  # 类别 0 - 红色 肿瘤
        [255, 255, 255],  # 类别 1 - 白色 炎症
        [0, 255, 0],  # 类别 2 - 绿色 结缔
        [0, 0, 255],  # 类别 3 - 蓝色
        [255, 255, 0],  # 类别 4 - 黄色
        [0, 0, 0]  # 类别 5 - 背景 黑色
    ], dtype=np.uint8)  # 颜色值范围 0-255
    # 遍历所有 .npy 文件
    for file_name_mask in os.listdir(input_folder_mask):
        if file_name_mask.endswith(".npy"):  # 只处理 .npy 文件
            file_path_mask = os.path.join(input_folder_mask, file_name_mask)
            file_name_img = file_name_mask.replace("mask", "image")
            file_path_img = os.path.join(input_folder_img, file_name_img)
            # 读取 .npy 文件
            data_mask = np.load(file_path_mask)
            # 变换形状 (H, W, C) -> (C, H, W)
            data_transposed = np.transpose(data_mask, (2, 0, 1))
            # 确保数据形状正确 (H, W, 6)
            if data_transposed.shape[0] == 6:

                # 取类别索引 (H, W)
                data_convert = np.argmax(data_mask, axis=-1).astype(np.uint8)
                unique_values = np.unique(data_convert)
                if len(unique_values) == 1 and (unique_values[0] == 0 or unique_values[0] == 5):
                    continue
                # 取实例像素（H,W）
                data_pixel_inst = np.sum(data_mask, axis=-1)  # 结果形状: (256, 256)
                output_path_pixel = os.path.join(output_folder_pixel, file_name_mask.replace(".npy", ".png"))
                Image.fromarray(data_pixel_inst.astype(np.uint8)).save(output_path_pixel)

                output_path_class = os.path.join(output_folder_class, file_name_mask.replace(".npy", ".png"))
                Image.fromarray(data_convert.astype(np.uint8)).save(output_path_class)
                # 应用颜色映射，将类别索引转换为 RGB 颜色图
                color_img = color_map[data_convert]  # 形状 (H, W, 3)
                # 保存为 PNG
                output_path = os.path.join(output_folder_mask, file_name_mask.replace(".npy", ".png"))
                Image.fromarray(color_img).save(output_path)

                print(f"转换 {file_name_mask} -> {output_path}")
            print(f"跳过 {file_name_mask}, 因为形状不匹配: {data_mask.shape}")
            data_img = np.load(file_path_img)
            # 归一化数据到 0-255（如果数据范围不是0-255，需要调整）
            data_img = (data_img - data_img.min()) / (data_img.max() - data_img.min()) * 255
            data_img = data_img.astype(np.uint8)  # 转换为 8-bit 格式

            # 转换为 PIL 图像
            img = Image.fromarray(data_img)

            # 保存为 .png 文件
            output_path = os.path.join(output_folder_img, file_name_img.replace(".npy", ".png"))
            img.save(output_path)

            print(f"转换 {file_name_img} -> {output_path}")



    print("所有 .npy 文件已转换为 .png！")

def save_numpy_images_with_folds(data, output_dir, train_dir, val_dir, num_folds=5, seed=42):
    """
    遍历 (2656, 256, 256, 3) 的 NumPy 数组，随机划分 5 折，并保存到 train/val 目录。
    最后一折的索引保存在 val_indices.json 中。

    参数：
    - data: (2656, 256, 256, 3) 形状的 NumPy 数组
    - output_dir: 图像保存的根目录
    - num_folds: 折数，默认为 5
    - seed: 随机种子，保证划分一致
    """

    # 生成索引并打乱
    indices = list(range(data.shape[0]))
    random.seed(seed)
    random.shuffle(indices)

    # 进行 5 折划分
    fold_size = len(indices) // num_folds
    val_indices = indices[-fold_size:]  # 取最后一折作为验证集
    train_indices = indices[:-fold_size]  # 其他折作为训练集

    # 记录 val 索引到 JSON 文件
    json_path = os.path.join(output_dir, "val_indices.json")
    with open(json_path, "w") as f:
        json.dump(val_indices, f, indent=4)

    print(f"验证集大小: {len(val_indices)}，已保存索引至 {json_path}")

    # 遍历所有索引并保存图像
    for idx in tqdm(indices, desc="保存图像"):
        img_array = data[idx]  # 形状 (256, 256, 3)
        data_img = (img_array - img_array.min()) / (img_array.max() - img_array.min()) * 255
        img = Image.fromarray(data_img.astype(np.uint8))
        # img = Image.fromarray((img_array * 255).astype(np.uint8))  # 归一化到 0-255 并转换为 PIL 图像
        img_filename = f"{idx:04d}.png"

        # 按索引分配到 train 或 val 目录
        if idx in val_indices:
            img.save(os.path.join(val_dir, img_filename))
        else:
            img.save(os.path.join(train_dir, img_filename))

    print(f"所有图像已保存至 {output_dir}/train 和 {output_dir}/val")


def process_and_save_masks(masks, output_dir, train_dir, val_dir):
    """
    处理并保存 PanNuke mask 数据，根据 JSON 划分训练/验证集，并转换后存为 .mat 文件。

    处理逻辑：
    - 读取 val_indices.json，划分训练/验证集。
    - 对 (256, 256, 6) 的 mask 进行处理：
        - 0-4 通道：只要对应像素值大于 0，就变为该通道索引（0,1,2,3,4）。
        - 5 通道：所有 255 变为 0。
    - 保存为 .mat 文件：
        - 训练集保存到 `output_data/train/`
        - 验证集保存到 `output_data/val/`

    参数:
        masks (numpy.ndarray): 形状为 (2656, 256, 256, 6) 的 mask 数据
        val_json_path (str): 验证集索引 JSON 文件路径
        output_dir (str): 训练/验证数据的输出目录
    """

    # 读取 val_indices.json
    val_json_path = os.path.join(output_dir, "val_indices.json")
    with open(val_json_path, "r") as f:
        val_indices = json.load(f)

    # 计算训练集索引
    all_indices = set(range(masks.shape[0]))
    train_indices = list(all_indices - set(val_indices))

    # 创建输出目录
    # train_dir = os.path.join(output_dir, "train")
    # val_dir = os.path.join(output_dir, "val")
    # os.makedirs(train_dir, exist_ok=True)
    # os.makedirs(val_dir, exist_ok=True)

    def process_mask(mask):
        # 合并前 5 个通道（0-4），生成一个 (256, 256, 2) 的数组
        combined_mask = np.sum(mask[..., :5], axis=-1)

        # 检查要分割的是不是全0
        need_delete= False
        unique_values = np.unique(combined_mask)
        if len(unique_values) == 1 and unique_values[0] == 0:
            need_delete = True

        # plt.imshow(combined_mask, cmap='gray')
        # plt.axis('off')  # 隐藏坐标轴
        # plt.show()
        expanded_array = combined_mask[:, :, np.newaxis]
        # expanded_array = np.stack([combined_mask] * 4, axis=-1)
        # print(f"expanded_array shape:{expanded_array.shape}")

        processed_mask = np.zeros((256, 256), dtype=np.uint8)
        # 遍历最后 5 个通道，在像素值大于 0 的地方，将通道索引赋值给最后一个通道
        for class_idx in range(5):
            mask_layer = mask[..., class_idx]
            processed_mask[mask_layer > 0] = class_idx + 1  # 将最后一个通道的相应位置赋值为通道的索引

        # # 定义颜色映射：0 映射为黑色，1-5 映射为随机颜色
        # cmap = mcolors.ListedColormap(['black', 'red', 'green', 'blue', 'yellow', 'purple'])
        # # 定义归一化，将 0-5 的值映射到颜色映射的索引
        # norm = mcolors.BoundaryNorm(boundaries=[-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5], ncolors=cmap.N)
        # # 显示图像
        # plt.imshow(processed_mask, cmap=cmap, norm=norm)
        # plt.colorbar(ticks=[0, 1, 2, 3, 4, 5], label='Category')
        # plt.axis('off')  # 隐藏坐标轴
        # plt.show()

        # 最后将 (256, 256) 添加到 processed_mask 变为 (256, 256, 5)
        processed_mask = processed_mask[:, :, np.newaxis]
        final_mask = np.concatenate((expanded_array, processed_mask), axis=-1)
        # print(f"final_mask.shape:{final_mask.shape}")
        return final_mask, need_delete

    # 处理并保存训练集
    print("正在处理训练集...")
    for idx in tqdm(train_indices, desc="训练集处理"):
        mask_data, need_delete = process_mask(masks[idx])  # 处理 mask
        if need_delete:
            print("没有任何分割类别")
            os.remove(os.path.join(os.path.dirname(train_dir), 'Images', f"{idx:04d}.png"))
        else:
            savemat(os.path.join(train_dir, f"{idx:04d}.mat"), {"mask": mask_data})  # 保存为 .mat

    # 处理并保存验证集
    print("正在处理验证集...")
    for idx in tqdm(val_indices, desc="验证集处理"):
        mask_data, need_delete = process_mask(masks[idx])  # 处理 mask
        if need_delete:
            print("没有任何分割类别")
            os.remove(os.path.join(os.path.dirname(val_dir), 'Images', f"{idx:04d}.png"))
        else:
            savemat(os.path.join(val_dir, f"{idx:04d}.mat"), {"mask": mask_data})  # 保存为 .mat

    print(f"数据已保存至 {output_dir}/train 和 {output_dir}/val")


# -------------------------------------------------------------------------------------
if __name__ == "__main__":
    image_file = r"/data3/davidqu/python_project/hover_net/Fold_3/Fold_3/images/fold3/images.npy"
    mask_file = r"/data3/davidqu/python_project/hover_net/Fold_3/Fold_3/masks/fold3/masks.npy"

    dataset_train_img_path = r"/data3/davidqu/python_project/hover_net/F3_Train/Images"
    dataset_train_mask_path = r"/data3/davidqu/python_project/hover_net/F3_Train/Lables"

    dataset_val_img_path = r"/data3/davidqu/python_project/hover_net/F3_Test/Images"
    dataset_val_mask_path = r"/data3/davidqu/python_project/hover_net/F3_Test/Lables"

    check_dir(dataset_train_img_path)
    check_dir(dataset_train_mask_path)

    check_dir(dataset_val_img_path)
    check_dir(dataset_val_mask_path)

    # #先提取image
    image_data = np.load(image_file)  # 读取 .npy 文件
    save_numpy_images_with_folds(image_data,r"/data4/userFolder/davidqu/study/hover_net-master/dataset/pannuke", dataset_train_img_path, dataset_val_img_path)

    #再提取mask
    mask_data = np.load(mask_file)  # 读取 .npy 文件
    process_and_save_masks(mask_data, r"/data4/userFolder/davidqu/study/hover_net-master/dataset/pannuke", dataset_train_mask_path, dataset_val_mask_path)
