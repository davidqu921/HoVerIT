import numpy as np
import matplotlib.pyplot as plt
import os
'''
def save_npy_channels_as_grayscale(input_path, output_directory):
    """
    将输入路径中的 .npy 文件的第四个和第五个通道以灰度图的形式保存到指定目录。
    :param input_path: 输入文件的路径。
    :param output_directory: 输出文件的目录。
    """
    # 加载 .npy 文件
    data = np.load(input_path)

    # 检查数据形状是否符合预期
    if data.ndim != 3 or data.shape[-1] != 5:
        raise ValueError(f"Expected a 5D array with shape (..., 5), but got shape {data.shape}")

    # 提取第四个通道（实例标注）和第五个通道（类别标注）
    inst_map = data[..., 3]
    type_map = data[..., 4]

    # 将数据归一化到 [0, 255] 范围
    inst_map = (inst_map - np.min(inst_map)) / (np.max(inst_map) - np.min(inst_map)) * 255
    type_map = (type_map - np.min(type_map)) / (np.max(type_map) - np.min(type_map)) * 255

    # 转换为 uint8 类型
    inst_map = inst_map.astype(np.uint8)
    type_map = type_map.astype(np.uint8)

    # 构建输出文件路径
    file_name = os.path.basename(input_path)
    file_name_without_ext = os.path.splitext(file_name)[0]
    inst_output_path = os.path.join(output_directory, f"{file_name_without_ext}_inst.png")
    type_output_path = os.path.join(output_directory, f"{file_name_without_ext}_type.png")

    # 使用 matplotlib 保存为灰度图
    plt.imsave(inst_output_path, inst_map, cmap='gray')
    plt.imsave(type_output_path, type_map, cmap='gray')

    print(f"Saved instance map to {inst_output_path}")
    print(f"Saved type map to {type_output_path}")

# 示例用法
input_path = "/data3/davidqu/python_project/hover_net/Training_Data/pannuke/type_reduced/train/540x540_164x164/0001_001.npy"  # 输入文件路径
output_directory = "/data4/userFolder/davidqu/study/hover_net-master/plot"  # 输出文件夹路径

# 确保输出目录存在
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

save_npy_channels_as_grayscale(input_path, output_directory)
'''



# 加载 .npy 文件
data = np.load("/data3/davidqu/python_project/hover_net/Training_Data/pannuke/type_reduced/train/256x256_256x256/0001_000.npy")

# 查看数组的形状
print("输入模型的数组形状:", data.shape)

# 查看数组的数据类型
print("输入模型的数据类型:", data.dtype)

# 提取第五个通道
channel_5 = data[:, :, 4]

# 查看提取的通道形状
print("第五个通道的形状:", channel_5.shape)

# 获取第五个通道的唯一值
unique_values = np.unique(channel_5)

print("第五个通道的唯一值:", unique_values)

# 提取第4个通道
channel_4 = data[:, :, 3]

# 查看提取的通道形状
print("第4个通道的形状:", channel_4.shape)

# 获取第4个通道的唯一值
unique_values = np.unique(channel_4)

print("第五个通道的唯一值:", unique_values)