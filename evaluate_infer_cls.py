import numpy as np
import os
from scipy.io import loadmat
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.widgets import Slider

def handle_output_mat(handle_mat_path):
    data = loadmat(handle_mat_path)

    # 获取相关数据
    inst_map = data['inst_map']  # shape: (256, 256)
    inst_type = data['inst_type']  # shape: (26, 1)

    # 初始化一个新的标签图像，大小与 inst_map 相同
    output_map = np.zeros_like(inst_map)

    # 遍历每个实例，将其类别赋值到对应的像素位置
    for i in range(1, inst_type.shape[0] + 1):  # 假设实例编号从1开始
        # 找到 inst_map 中所有属于该实例的像素位置
        mask = (inst_map == i)

        # 获取该实例的类别
        inst_category = inst_type[i - 1]  # i-1 因为索引从0开始

        # 将这些位置的像素值更新为该实例的类别
        output_map[mask] = inst_category

    return output_map

'''
def handle_output_mat_instance(handle_mat_path):
    data = loadmat(handle_mat_path)

    # 获取相关数据
    inst_map = data['inst_map']
    inst_type = data['inst_type']  

    return inst_type
'''

def compute_dice_multiclass(mask_ref: np.ndarray, mask_pred: np.ndarray, ignore_mask: np.ndarray = None) -> dict:
    """
    计算多类别标签的Dice系数，每个类别的Dice系数分别计算。

    参数:
        mask_ref (np.ndarray): 真实标签的多类别掩码，像素值为0到N的整数。
        mask_pred (np.ndarray): 预测标签的多类别掩码，像素值为0到N的整数。
        ignore_mask (np.ndarray, optional): 需要忽略的区域的布尔掩码，True表示忽略对应的位置。默认为None。

    返回:
        dict: 包含每个类别Dice系数的字典。键为类别标签，值为对应的Dice系数。当分母为零时值为np.nan。
    """
    # 确定所有可能的类别（0到最大标签值）
    max_class = max(mask_ref.max(), mask_pred.max())
    classes = range(max_class + 1)
    dice_scores = {}

    # 处理忽略掩码
    if ignore_mask is not None:
        use_mask = ~ignore_mask
    else:
        use_mask = np.ones_like(mask_ref, dtype=bool)

    for k in classes:
        # 生成当前类别的二值掩码
        ref_binary = (mask_ref == k)
        pred_binary = (mask_pred == k)

        # 计算交集和各自的总和（仅在有效区域内）
        intersection = np.sum((ref_binary & pred_binary) & use_mask)
        sum_ref = np.sum(ref_binary & use_mask)
        sum_pred = np.sum(pred_binary & use_mask)

        # 计算Dice系数
        if sum_ref == 0 and sum_pred == 0:
            # 如果该类别在 mask_ref 和 mask_pred 中都不存在，则Dice系数为1
            dice_scores[k] = 1.0
        else:
            denominator = sum_ref + sum_pred
            if denominator == 0:
                dice_scores[k] = np.nan
            else:
                dice_scores[k] = (2.0 * intersection) / denominator

    return dice_scores


# def compute_dice_for_folders(ref_dir: str, pred_dir: str,
#                              ignore_mask: np.ndarray = None) -> dict:
#     """
#     遍历两个文件夹下的同名.mat文件并计算Dice系数，同时返回每个类别的平均Dice
#
#     参数:
#         ref_dir (str): 参考数据文件夹路径
#         pred_dir (str): 预测数据文件夹路径
#         data_key (str): .mat文件中存储掩码数据的键名，默认为'data'
#         ignore_mask (np.ndarray): 需要忽略的区域的布尔掩码
#
#     返回:
#         dict: 包含两个键的字典：
#             - 'per_file': 各文件的Dice系数（字典的字典）
#             - 'per_class_mean': 各类别平均Dice系数（字典）
#     """
#     # 获取共同文件列表
#     ref_files = {f for f in os.listdir(ref_dir) if f.endswith('.mat')}
#     pred_files = {f for f in os.listdir(pred_dir) if f.endswith('.mat')}
#     common_files = ref_files & pred_files
#
#     results = {}
#     class_scores = defaultdict(list)
#
#     for fname in common_files:
#         # 加载并处理数据
#         ref_path = os.path.join(ref_dir, fname)
#         pred_path = os.path.join(pred_dir, fname)
#
#
#         # mask_ref = loadmat(ref_path)
#         # print(f"mask_ref keys:{loadmat(ref_path).keys()}")
#         # print(f"mask_ref keys:{loadmat(ref_path).keys()}")
#         mask_ref_or = loadmat(ref_path)['mask'].astype(np.uint8)
#         mask_ref = mask_ref_or[:, :, 1]
#
#         # 定义颜色映射：0 映射为黑色，1-5 映射为随机颜色
#         cmap = mcolors.ListedColormap(['black', 'red', 'green', 'blue', 'yellow', 'purple'])
#         # 定义归一化，将 0-5 的值映射到颜色映射的索引
#         norm = mcolors.BoundaryNorm(boundaries=[-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5], ncolors=cmap.N)
#         # # 显示图像
#         # plt.imshow(mask_ref, cmap=cmap, norm=norm)
#         # plt.colorbar(ticks=[0, 1, 2, 3, 4, 5], label='Category')
#         # plt.axis('off')  # 隐藏坐标轴
#         # plt.show()
#
#
#         # mask_pred= loadmat(pred_path)['inst_type'].astype(np.uint8)
#         mask_pred = handle_output_mat(pred_path)
#         # print(f"mask_pred keys:{loadmat(pred_path).keys()}")
#         # print(f"mask_pred inst_map keys:{loadmat(pred_path)['inst_map'].shape}")
#         # print(f"mask_pred inst_uid keys:{loadmat(pred_path)['inst_uid'].shape}")
#         # print(f"mask_pred inst_centroid keys:{loadmat(pred_path)['inst_centroid'].shape}")
#         # print(f"mask_pred inst_type keys:{loadmat(pred_path)['inst_type'].shape}")
#
#         # # 创建2个子图显示mask_ref和mask_pred
#         # fig, axs = plt.subplots(1, 2, figsize=(12, 6))
#         #
#         # # 显示mask_ref
#         # im0 = axs[0].imshow(mask_ref, cmap=cmap, norm=norm)
#         # axs[0].set_title("Reference Mask")
#         # axs[0].axis('off')  # 隐藏坐标轴
#         #
#         # # 显示mask_pred
#         # im1 = axs[1].imshow(mask_pred, cmap=cmap, norm=norm)
#         # axs[1].set_title("Predicted Mask")
#         # axs[1].axis('off')  # 隐藏坐标轴
#         #
#         # # 添加 colorbar 在右侧
#         # cbar = fig.colorbar(im0, ax=axs, orientation='vertical', fraction=0.02, pad=0.04)
#         # cbar.set_ticks([0, 1, 2, 3, 4, 5])
#         # cbar.set_label('Category')
#         #
#         # # 显示两张图
#         # plt.tight_layout()
#         # plt.show()
#
#
#
#         # 计算Dice
#         dice_scores = compute_dice_multiclass(mask_ref, mask_pred, ignore_mask)
#         print(f"{ref_path} VS {pred_path} DICE: {dice_scores}")
#         results[fname] = dice_scores
#
#         # 收集各类别分数
#         for cls, score in dice_scores.items():
#             class_scores[cls].append(score)
#
#     # 计算类别平均（自动跳过NaN）
#     class_means = {cls: np.nanmean(scores) for cls, scores in class_scores.items()}
#
#     return {
#         'per_file': results,
#         'per_class_mean': class_means
#     }

def compute_dice_for_folders(ref_dir: str, pred_dir: str,
                             ignore_mask: np.ndarray = None) -> dict:
    """
    遍历两个文件夹下的同名.mat文件并计算Dice系数，同时返回每个类别的平均Dice

    参数:
        ref_dir (str): 参考数据文件夹路径
        pred_dir (str): 预测数据文件夹路径
        ignore_mask (np.ndarray): 需要忽略的区域的布尔掩码

    返回:
        dict: 包含两个键的字典：
            - 'per_file': 各文件的Dice系数（字典的字典）
            - 'per_class_mean': 各类别平均Dice系数（字典）
    """
    # 获取共同文件列表并排序
    ref_files = {f for f in os.listdir(ref_dir) if f.endswith('.mat')}
    pred_files = {f for f in os.listdir(pred_dir) if f.endswith('.mat')}
    common_files = sorted(ref_files & pred_files)

    results = {}
    class_scores = defaultdict(list)
    image_data = []  # 存储图像数据和文件名

    # 预处理所有数据
    for idx, fname in enumerate(common_files):
        # 加载数据
        ref_path = os.path.join(ref_dir, fname)
        pred_path = os.path.join(pred_dir, fname)

        # 处理参考图像(获取真实标注的类别分类)
        mask_ref_or = loadmat(ref_path)['mask'].astype(np.uint8)
        mask_ref = mask_ref_or[:, :, 1]

        # 处理预测图像（假设handle_output_mat已定义）
        mask_pred = handle_output_mat(pred_path)

        # 存储数据
        image_data.append({
            "ref": mask_ref,
            "pred": mask_pred,
            "fname": fname,
            "ref_path": ref_path,
            "pred_path": pred_path
        })

        # 计算Dice系数
        dice_scores = compute_dice_multiclass(mask_ref, mask_pred, ignore_mask)
        results[fname] = dice_scores
        print(f"{idx + 1}/{len(common_files)} Processed: {fname}")

        # 收集有效分数
        for cls, score in dice_scores.items():
            if not np.isnan(score):
                class_scores[cls].append(score)

    # 创建可视化界面
    if image_data:
        # 创建图形和子图
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
        plt.subplots_adjust(bottom=0.25)

        # 初始化颜色映射
        cmap = mcolors.ListedColormap(['black', 'red', 'green', 'blue', 'yellow', 'purple'])
        norm = mcolors.BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5], cmap.N)

        # 初始化图像显示
        initial_idx = 0
        im1 = ax1.imshow(image_data[initial_idx]["ref"], cmap=cmap, norm=norm)
        im2 = ax2.imshow(image_data[initial_idx]["pred"], cmap=cmap, norm=norm)

        # 设置初始标题
        title1 = ax1.set_title(f"Reference\n{image_data[initial_idx]['fname']}")
        title2 = ax2.set_title(f"Prediction\n{image_data[initial_idx]['fname']}")
        plt.setp([title1, title2], fontsize=10)

        # 添加colorbar
        cax = fig.add_axes([0.15, 0.85, 0.7, 0.03])  # 水平colorbar
        cbar = fig.colorbar(im1, cax=cax, orientation='horizontal')
        cbar.set_ticks([0, 1, 2, 3, 4, 5])
        cbar.set_label('Class Labels')

        # 创建滑动条
        ax_slider = plt.axes([0.2, 0.1, 0.6, 0.03])
        slider = Slider(
            ax=ax_slider,
            label='Image Index',
            valmin=0,
            valmax=len(image_data) - 1,
            valinit=0,
            valstep=1
        )

        # 更新函数
        def update(val):
            idx = int(slider.val)
            im1.set_data(image_data[idx]["ref"])
            im2.set_data(image_data[idx]["pred"])
            title1.set_text(f"Reference\n{image_data[idx]['fname']}")
            title2.set_text(f"Prediction\n{image_data[idx]['fname']}")
            fig.canvas.draw_idle()

        slider.on_changed(update)

        # 添加键盘事件支持
        def key_press(event):
            if event.key == 'right':
                new_val = min(slider.val + 1, slider.valmax)
            elif event.key == 'left':
                new_val = max(slider.val - 1, slider.valmin)
            else:
                return
            slider.set_val(new_val)

        fig.canvas.mpl_connect('key_press_event', key_press)
        plt.savefig('/data4/userFolder/davidqu/study/hover_net-master/plot/only_1and2_type.png')
        plt.show()

    # 计算平均Dice
    class_means = {cls: np.nanmean(scores) for cls, scores in class_scores.items()}

    return {
        'per_file': results,
        'per_class_mean': class_means
    }



results = compute_dice_for_folders(
    ref_dir=r"/data3/davidqu/python_project/hover_net/Test_Data/test_lable_3_types",
    pred_dir=r"/data3/davidqu/HoVerIT/infer_out_HoVerIT3.0/mat"
)

# 访问单个文件结果
# print(results['per_file']['case001.mat'][0])  # 输出case001.mat的类别0 Dice

# 访问类别平均结果
print(results['per_class_mean'][1])  # 输出类别1的平均Dice

# 打印所有类别平均
for cls, mean_score in results['per_class_mean'].items():
    print(f"Class {cls}: {mean_score:.4f}")