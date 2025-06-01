"""
================================================================================
Program Name: Nuclei Segmentation and Classification with HoVerNetEnhenced
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

from .net_utils import (DenseBlock, Net, TFSamepaddingLayer, EnhancedResBlock,
                        UpSample2x)
from .utils import crop_op, crop_to_shape

####
class EnhencedHoVerNet(Net):
    """Initialise HoVer-Net."""

    def __init__(self, input_ch=3, nr_types=None, freeze=False, mode='original'):  # <----- 将input channel改为5通道? No
        super().__init__()
        self.mode = mode
        self.freeze = freeze
        self.nr_types = nr_types
        self.output_ch = 3 if nr_types is None else 4

        assert mode == 'original' or mode == 'fast', \
                'Unknown mode `%s` for HoVerNet %s. Only support `original` or `fast`.' % mode
        # ==================== 编码器改进 ====================
        # 初始卷积层（严格保持与原始一致）
        module_list = [
            ("/", nn.Conv2d(input_ch, 64, 7, stride=1, padding=0, bias=False)),
            ("bn", nn.BatchNorm2d(64, eps=1e-5)),
            ("relu", nn.ReLU(inplace=True)),
        ]
        if mode == 'fast':
            module_list = [("pad", TFSamepaddingLayer(ksize=7, stride=1))] + module_list
        self.conv0 = nn.Sequential(OrderedDict(module_list))
        
        
        # 增强的残差阶段（保持原始下采样率）
        self.d0 = self._make_enhanced_res_stage(64, 256, blocks=3)
        self.d1 = self._make_enhanced_res_stage(256, 512, blocks=4, use_se=True, stride=2)
        self.d2 = self._make_enhanced_res_stage(512, 1024, blocks=6, use_se=True, stride=2)
        self.d3 = self._make_enhanced_res_stage(1024, 2048, blocks=3, stride=2)

        self.conv_bot = nn.Conv2d(2048, 1024, 1, stride=1, padding=0, bias=False)
        
        def create_decoder_branch(out_ch=2, ksize=5):
            module_list = [ 
                ("conva", nn.Conv2d(1024, 256, ksize, stride=1, padding=0, bias=False)),
                ("dense", DenseBlock(256, [1, ksize], [128, 32], 8, split=4)),
                ("convf", nn.Conv2d(512, 512, 1, stride=1, padding=0, bias=False),),
            ]
            u3 = nn.Sequential(OrderedDict(module_list))

            module_list = [ 
                ("conva", nn.Conv2d(512, 128, ksize, stride=1, padding=0, bias=False)),
                ("dense", DenseBlock(128, [1, ksize], [128, 32], 4, split=4)),
                ("convf", nn.Conv2d(256, 256, 1, stride=1, padding=0, bias=False),),
            ]
            u2 = nn.Sequential(OrderedDict(module_list))

            module_list = [ 
                ("conva/pad", TFSamepaddingLayer(ksize=ksize, stride=1)),
                ("conva", nn.Conv2d(256, 64, ksize, stride=1, padding=0, bias=False),),
            ]
            u1 = nn.Sequential(OrderedDict(module_list))

            module_list = [ 
                ("bn", nn.BatchNorm2d(64, eps=1e-5)),
                ("relu", nn.ReLU(inplace=True)),
                ("conv", nn.Conv2d(64, out_ch, 1, stride=1, padding=0, bias=True),),
            ]
            u0 = nn.Sequential(OrderedDict(module_list))

            decoder = nn.Sequential(
                OrderedDict([("u3", u3), ("u2", u2), ("u1", u1), ("u0", u0),])
            )
            return decoder

        ksize = 5 if mode == 'original' else 3
        if nr_types is None:
            self.decoder = nn.ModuleDict(
                OrderedDict(
                    [
                        ("np", create_decoder_branch(ksize=ksize,out_ch=2)),
                        ("hv", create_decoder_branch(ksize=ksize,out_ch=2)),
                    ]
                )
            )
        else:
            self.decoder = nn.ModuleDict(
                OrderedDict(
                    [
                        ("tp", create_decoder_branch(ksize=ksize, out_ch=nr_types)),
                        ("np", create_decoder_branch(ksize=ksize, out_ch=2)),
                        ("hv", create_decoder_branch(ksize=ksize, out_ch=2)),
                    ]
                )
            )

        self.upsample2x = UpSample2x()
        # TODO: pytorch still require the channel eventhough its ignored
        self.weights_init()

    def _make_enhanced_res_stage(self, in_ch, out_ch, blocks, use_se=False, stride=1):
            layers = []
            # 第一个块处理下采样
            layers.append(EnhancedResBlock(in_ch, out_ch, use_se=use_se, stride=stride))
            # 后续块保持尺寸
            for _ in range(1, blocks):
                layers.append(EnhancedResBlock(out_ch, out_ch, use_se=use_se))
            return nn.Sequential(*layers)

    def forward(self, imgs):
        imgs = imgs / 255.0
        
        # 编码器前向
        d0 = self.conv0(imgs)
        d0 = self.d0(d0)
        d1 = self.d1(d0)
        d2 = self.d2(d1)
        d3 = self.d3(d2)
        d3 = self.conv_bot(d3)
        d = [d0, d1, d2, d3]
        #print("---d0 shape:", d0.shape)
        #print("---d1 shape:", d1.shape)
        # print("---d2 shape:", d2.shape)
        # print("---d3 shape:", d3.shape)

        # 严格保持原始裁剪逻辑
        if self.mode == 'original':
            d[0] = crop_op(d[0], [184, 184])
            d[1] = crop_op(d[1], [72, 72])
        else:
            d[0] = crop_op(d[0], [92, 92])
            d[1] = crop_op(d[1], [36, 36])
        #print("---d[0] croped shape:", d[0].shape)
        #print("---d[1] croped shape:", d[1].shape)

        # 解码器保持不变
        out_dict = OrderedDict()
        for branch_name, branch_desc in self.decoder.items():
            u3 = self.upsample2x(d[-1]) + d[-2]
            u3 = branch_desc[0](u3)
            u2 = self.upsample2x(u3) + d[-3]
            u2 = branch_desc[1](u2)
            u1 = self.upsample2x(u2) + d[-4]
            u1 = branch_desc[2](u1)
            u0 = branch_desc[3](u1)
            out_dict[branch_name] = u0

        return out_dict


####
def create_model(mode=None, **kwargs):
    if mode not in ['original', 'fast']:
        assert "Unknown Model Mode %s" % mode
    return EnhencedHoVerNet(mode=mode, **kwargs)