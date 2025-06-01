"""
================================================================================
Program Name: Nuclei Segmentation and Classification with HoverSwinNet
Version: 1.0
Author: David Qu
Institution: Yangtze River Delta Guozhi(Shanghai) Intelligient Medical Technology Co., Ltd
Email: davidqu01921@outlook.com
Date: 2025-04-08
================================================================================

Description:
This network is improved based on original HoVerNet architecture for nuclear segmentation and classfication in histopathology images.
It includes modules for training, testing, and visualizing segmentation results.

Usage:
To run this program, you can use the following command:
    python run_train.py --gpu='0'

Dependencies:
- Python 3.9
- PyTorch 1.10
- NumPy
- OpenCV

License:
This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.

Contact:
For any questions, suggestions, or bug reports, please contact David Qu at davidqu01921@outlook.com.

================================================================================
"""

import math
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .net_utils import (DenseBlock, Net, ResidualBlock, TFSamepaddingLayer,
                        UpSample2x)
from .utils import crop_op, crop_to_shape

from monai.networks.nets import SwinUNETR

class HoVerNetBranch(nn.Module):
    """
    通用的 HoVerNet 分支模块：
    接收 backbone 输出的特征图，经过几个卷积 + BN + ReLU，输出目标任务的结果。
    """

    def __init__(self, in_channels: int, out_channels: int, num_convs: int = 2, dropout_p=0.3):
        """
        Args:
            in_channels (int): 输入通道数，一般为 SwinUNETR 输出特征的通道数。
            out_channels (int): 输出通道数（np: 1, hv: 2, tp: 分类数）。
            num_convs (int): 中间卷积层数，默认为 2。
        """
        super(HoVerNetBranch, self).__init__()

        layers = []
        for i in range(num_convs - 1):
            layers.append(nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(in_channels))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout2d(p=dropout_p))  # ← Add dropout here

        # 最后一层卷积输出到目标维度
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=1))

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class HoverSwinNet(nn.Module):
    def __init__(
        self,
        img_size=256,
        in_channels=3,
        feature_size=48,
        out_channels_dict={"np": 2, "hv": 2, "tp": 4},
    ):
        super(HoverSwinNet, self).__init__()
        self.nr_types = (
            out_channels_dict["tp"] if "tp" in out_channels_dict else None
        )

        self.swin_unetr = SwinUNETR(
            img_size=img_size,
            in_channels=in_channels,
            out_channels=feature_size,  # 最终 decoder 输出通道
            feature_size=feature_size,
            spatial_dims=2,
        )

        # HoVerNet 分支输出，保留分支结构不变
        self.np_branch = HoVerNetBranch(in_channels=feature_size, out_channels=out_channels_dict["np"], dropout_p=0.3)
        self.hv_branch = HoVerNetBranch(in_channels=feature_size, out_channels=out_channels_dict["hv"], dropout_p=0.3)
        self.tp_branch = HoVerNetBranch(in_channels=feature_size, out_channels=out_channels_dict["tp"], dropout_p=0.3)

    def forward(self, x):
        # 1. SwinUNETR 的输出
        feat, skip_feat = self.swin_unetr(x)  # now returns: (final_output, decoder3)  # Shape: B x C x H x W
        print("Shape of feat:", feat.shape)

        # 2. 三头分支
        np_out = self.np_branch(feat)
        hv_out = self.hv_branch(feat)
        tp_out = self.tp_branch(feat)


        # ⬇ 添加裁剪，让输出变为 [B, C, 164, 164]
        np_out = crop_op(np_out, [92, 92])   
        hv_out = crop_op(hv_out, [92, 92])
        tp_out = crop_op(tp_out, [92, 92])

        print("np_out shape", np_out.shape)
        print("hv_out shape", hv_out.shape)
        print("tp_out shape", tp_out.shape)

        return {
            "np": np_out,
            "hv": hv_out,
            "tp": tp_out
        }


####
def create_model(input_ch=3, nr_types=None, freeze=False, mode=None, **kwargs):
    if mode not in ['original', 'fast', 'swin']:
        raise ValueError(f"Unknown model mode: {mode}")

    # 构造输出通道字典
    out_channels_dict = {
        "np": 2,                # Nuclei Pixel map (2 channels)
        "hv": 2,                # Horizontal/Vertical map (2 channels)
        "tp": nr_types     # Type prediction map (default 4 if None)
    }

    return HoverSwinNet(
        img_size=256,
        in_channels=input_ch,
        feature_size=48,
        out_channels_dict=out_channels_dict
    )
