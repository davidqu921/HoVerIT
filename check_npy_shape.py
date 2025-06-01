import numpy as np

# 加载.npy文件
file_path = '/data3/davidqu/python_project/hover_net/Training_Data/pannuke/pannuke/train/256x256_256x256/0001_000.npy'  # 替换为你的.npy文件路径
data = np.load(file_path)

# 获取并打印数据的形状
shape = data.shape
print("The shape of the npy file is:", shape)