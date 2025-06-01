import os
import numpy as np
from scipy.io import loadmat
import cv2

def calculate_tp_fp_fn(gt, pred, class_id):
    gt_class = (gt == class_id)
    pred_class = (pred == class_id)
    
    tp = np.sum(gt_class & pred_class)
    fp = np.sum(~gt_class & pred_class)
    fn = np.sum(gt_class & ~pred_class)
    
    return tp, fp, fn

def load_and_process_data(gt_path, pred_path):
    # Load ground truth data
    pred_data = loadmat(pred_path)
    pred_inst_map = pred_data['inst_map']
    pred_type_map = pred_data['inst_type']
    
    # Load prediction data
    gt_data = loadmat(gt_path)
    gt_inst_map = gt_data['mask'][..., 0]
    gt_type_map = gt_data['mask'][..., 1]

    # 初始化一个新的标签图像，大小与 inst_map 相同
    reshaped_type_map = np.zeros_like(pred_inst_map)

    # 遍历每个实例，将其类别赋值到对应的像素位置
    for i in range(1, pred_type_map.shape[0] + 1):  # 假设实例编号从1开始
        # 找到 inst_map 中所有属于该实例的像素位置
        mask = (pred_inst_map == i)

        # 获取该实例的类别
        inst_category = pred_type_map[i - 1]  # i-1 因为索引从0开始

        # 将这些位置的像素值更新为该实例的类别
        reshaped_type_map[mask] = inst_category
    
    return gt_inst_map, gt_type_map, pred_inst_map, reshaped_type_map

def calculate_precision_recall_f1(tp, fp, fn):
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return precision, recall, f1_score

def calculate_average_precision_recall_f1(gt_folder, pred_folder, num_classes):
    precision_scores = {class_id: [] for class_id in range(1, num_classes + 1)}
    recall_scores = {class_id: [] for class_id in range(1, num_classes + 1)}
    f1_scores = {class_id: [] for class_id in range(1, num_classes + 1)}
    
    # Get list of files
    gt_files = sorted(os.listdir(gt_folder))
    pred_files = sorted(os.listdir(pred_folder))
    
    # Ensure files are matching
    if len(gt_files) != len(pred_files):
        raise ValueError("Number of ground truth and prediction files do not match.")
    
    for gt_file, pred_file in zip(gt_files, pred_files):
        gt_path = os.path.join(gt_folder, gt_file)
        pred_path = os.path.join(pred_folder, pred_file)
        
        # Load and process data
        gt_inst_map, gt_type_map, pred_inst_map, pred_type_map = load_and_process_data(gt_path, pred_path)
        
        # Calculate TP, FP, FN for each class
        for class_id in range(1, num_classes + 1):
            tp, fp, fn = calculate_tp_fp_fn(gt_type_map, pred_type_map, class_id)
            precision, recall, f1_score = calculate_precision_recall_f1(tp, fp, fn)
            
            precision_scores[class_id].append(precision)
            recall_scores[class_id].append(recall)
            f1_scores[class_id].append(f1_score)
    
    # Calculate average Precision, Recall, and F1-score
    average_precision = {class_id: np.mean(precision_scores[class_id]) for class_id in range(1, num_classes + 1)}
    average_recall = {class_id: np.mean(recall_scores[class_id]) for class_id in range(1, num_classes + 1)}
    average_f1_score = {class_id: np.mean(f1_scores[class_id]) for class_id in range(1, num_classes + 1)}
    
    return average_precision, average_recall, average_f1_score

# Example usage
gt_folder = '/data3/davidqu/python_project/hover_net/Test_Data/test_lable_3_types'
pred_folder = '/data3/davidqu/HoVerIT/infer_out_HoVerIT3.0/mat'
num_classes = 3  # Adjust based on your dataset, exclude background

average_precision, average_recall, average_f1_score = calculate_average_precision_recall_f1(gt_folder, pred_folder, num_classes)

for class_id in range(1, num_classes + 1):
    print(f"Class {class_id} Average Precision: {average_precision[class_id]:.4f}")
    print(f"Class {class_id} Average Recall: {average_recall[class_id]:.4f}")
    print(f"Class {class_id} Average F1-score: {average_f1_score[class_id]:.4f}")