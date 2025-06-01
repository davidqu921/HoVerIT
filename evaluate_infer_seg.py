import numpy as np
import scipy.io as sio
import os
from sklearn.metrics import precision_score, recall_score, f1_score, jaccard_score

def load_pred_mat_file(file_path):
    """加载预测结果的 .mat 文件"""
    data = sio.loadmat(file_path)
    # 提取实例分割的预测结果
    pred_seg = data.get('inst_map', None)

    #print(f"Shape of pred seg: {pred_seg.shape}")
    #print("The pred seg has inst:")
    #print(np.unique(pred_seg))
    if pred_seg is None:
        raise ValueError(f"No 'inst_map' key found in {file_path}")
    
    # 提取类别分类的预测结果
    pred_cls = data.get('inst_type', None)
    #print(pred_cls.shape)
    #print(pred_cls)

    if pred_cls is None:
        raise ValueError(f"No 'inst_type' key found in {file_path}")
    
    return pred_seg, pred_cls

def load_gt_mat_file(file_path):
    """加载真实标注的 .mat 文件"""
    data = sio.loadmat(file_path)
    
    mask = data.get('mask', None)
    #print(f"Shape of gt mask: {mask.shape}")
    
    # 提取实例分割的真实标注（第一个通道）
    gt_seg = mask[..., 0]
    #print(f"Shape of gt_seg: {gt_seg.shape}")
    # print("The gt seg has inst:")
    # print(np.unique(gt_seg))
    # print(gt_seg.shape)    
    gt_cls = mask[..., 1]
    # print("The gt cls has inst:")
    # print(np.unique(gt_cls))
    # print(gt_cls.shape)
    if gt_seg is None:
        raise ValueError(f"No 'mask' key found in {file_path}")
    
    return gt_seg, gt_cls



def calculate_dice(pred, gt):
    """计算实例分割的 DICE 分数"""
    
    smooth=1e-5
    clipped_pred = np.clip(pred, a_min=None, a_max=1)   # 二值化
    clipped_gt = np.clip(gt, a_min=None, a_max=1)
    intersection = np.sum(clipped_pred * clipped_gt)
    #bool_pred = clipped_pred.astype(bool)
    #bool_gt = clipped_gt.astype(bool)
    #intersection = bool_pred & bool_gt
    #intersection_num = np.sum(intersection)
    return (2.0 * intersection + smooth) / ((np.sum(clipped_pred) + np.sum(clipped_gt)) + smooth)

'''
def calculate_classification_metrics(pred, gt, num_classes):
    """计算类别分类的精确率、召回率和 F1 分数"""
    pred_flat = pred.flatten()
    gt_flat = gt.flatten()

    precision = precision_score(gt_flat, pred_flat, average='macro', zero_division=0)
    recall = recall_score(gt_flat, pred_flat, average='macro', zero_division=0)
    f1 = f1_score(gt_flat, pred_flat, average='macro', zero_division=0)

    return precision, recall, f1
'''


def evaluate_predictions(pred_dir, gt_dir, num_classes):
    """评估预测结果"""
    pred_files = sorted(os.listdir(pred_dir))
    gt_files = sorted(os.listdir(gt_dir))

    total_dice = 0
    '''
    total_precision = 0
    total_recall = 0
    total_f1 = 0
    '''

    for pred_file, gt_file in zip(pred_files, gt_files):
        pred_path = os.path.join(pred_dir, pred_file)
        gt_path = os.path.join(gt_dir, gt_file)

        # 加载预测结果
        pred_seg, pred_cls = load_pred_mat_file(pred_path)
        # print(f"The shape of pred_cls is {pred_cls.shape}")

        # 加载真实标注
        gt_seg, gt_cls = load_gt_mat_file(gt_path)
        # print(f"The shape of gt_cls is {gt_cls.shape}")

        # 计算实例分割的 DICE 分数
        dice = calculate_dice(pred_seg, gt_seg)
        total_dice += dice
           
        '''
        # 计算类别分类的性能指标
        precision, recall, f1 = calculate_classification_metrics(pred_cls, gt_cls, num_classes)
        total_precision += precision
        total_recall += recall
        total_f1 += f1
        '''

    avg_dice = total_dice / len(pred_files)
    '''
    avg_precision = total_precision / len(pred_files)
    avg_recall = total_recall / len(pred_files)
    avg_f1 = total_f1 / len(pred_files)'
    '''

    print(f"Average Segmentation DICE: {avg_dice:.4f}")
    '''
    print(f"Average Precision: {avg_precision:.4f}")
    print(f"Average Recall: {avg_recall:.4f}")
    print(f"Average F1 Score: {avg_f1:.4f}")'
    '''

if __name__ == "__main__":
    pred_dir = "/data3/davidqu/HoVerIT/infer_out_HoVerIT3.0/mat"  # 预测结果目录
    gt_dir = "/data3/davidqu/python_project/hover_net/Test_Data/test_lable_3_types"  # 真实标注目录
    num_classes = 3  # 不包含背景0

    evaluate_predictions(pred_dir, gt_dir, num_classes)
    
