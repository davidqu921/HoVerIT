import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import os

'''
def save_mat_mask_as_grayscale(input_path, output_directory):
    """
    将输入路径中的 .mat 文件中的 'mask' 提取出来，并将第一个通道（inst_map）和第二个通道（type_map）
    分别以灰度图的形式保存到指定目录。
    :param input_path: 输入文件的路径。
    :param output_directory: 输出文件的目录。
    """
    # 加载 .mat 文件
    data = sio.loadmat(input_path)
    
    # 打印所有键名及其形状
    print(f"Contents of {input_path}:")
    for key in data.keys():
        if not key.startswith('__'):  # 忽略内部键
            print(f"Key: {key}, Shape: {data[key].shape}")

    # 提取 'mask'
    if 'mask' not in data:
        raise KeyError(f"'mask' not found in {input_path}")
    
    mask = data['mask']
    
    # 检查 mask 的形状是否符合预期
    if mask.shape != (256, 256, 2):
        raise ValueError(f"Expected 'mask' to have shape (256, 256, 2), but got shape {mask.shape}")
    
    # 提取第一个通道（inst_map）和第二个通道（type_map）
    inst_map = mask[..., 0]
    type_map = mask[..., 1]

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

    print(f"Saved 'inst_map' as grayscale image to {inst_output_path}")
    print(f"Saved 'type_map' as grayscale image to {type_output_path}")

# 示例用法
input_path = "/data3/davidqu/python_project/hover_net/All_Train/Lables/0001.mat"  # 输入文件路径
output_directory = "/data4/userFolder/davidqu/study/hover_net-master/plot"  # 输出文件夹路径

# 确保输出目录存在
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

save_mat_mask_as_grayscale(input_path, output_directory)


'''
input_path = '/data3/davidqu/HoVerIT/infer_out_HoVerITt/mat/0020.mat'
    # 加载 .mat 文件
data = sio.loadmat(input_path)
    
    # 打印所有键名及其形状
print(f"Contents of {input_path}:")
for key in data.keys():
    if not key.startswith('__'):  # 忽略内部键
        print(f"Key: {key}, Shape: {data[key].shape}")

    # 提取 'mask'
if 'mask' not in data:
    raise KeyError(f"'mask' not found in {input_path}")
    
mask = data['mask']
    
    # 检查 mask 的形状是否符合预期
if mask.shape != (256, 256, 2):
    raise ValueError(f"Expected 'mask' to have shape (256, 256, 2), but got shape {mask.shape}")
    
    # 提取第一个通道（inst_map）和第二个通道（type_map）
inst_map = mask[..., 0]
type_map = mask[..., 1]

# 查看提取的通道形状
print("inst_map的形状:", inst_map.shape)

# 获取inst_map的唯一值
unique_values = np.unique(inst_map)

print("inst_map的唯一值:", unique_values)

# 查看提取的通道形状
print("type_map的形状:", type_map.shape)

# 获取inst_map的唯一值
unique_values = np.unique(type_map)

print("type_map的唯一值:", unique_values)

# 查看数组的数据类型
print("mask的数据类型:", mask.dtype)

