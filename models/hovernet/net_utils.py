import numpy as np
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoderLayer

from collections import OrderedDict

from .utils import crop_op, crop_to_shape
from config import Config
from einops import rearrange
from einops.layers.torch import Rearrange
from einops import repeat

####
class Net(nn.Module):
    """ A base class provides a common weight initialisation scheme."""
    def weights_init(self):
        for m in self.modules():
            classname = m.__class__.__name__

            # ! Fixed the type checking
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

            if "norm" in classname.lower():
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

            if "linear" in classname.lower():
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        return x


####
class TFSamepaddingLayer(nn.Module):
    """To align with tf `same` padding. 
    
    Putting this before any conv layer that need padding
    Assuming kernel has Height == Width for simplicity
    """

    def __init__(self, ksize, stride):
        super(TFSamepaddingLayer, self).__init__()
        self.ksize = ksize
        self.stride = stride

    def forward(self, x):
        if x.shape[2] % self.stride == 0:
            pad = max(self.ksize - self.stride, 0)
        else:
            pad = max(self.ksize - (x.shape[2] % self.stride), 0)

        if pad % 2 == 0:
            pad_val = pad // 2
            padding = (pad_val, pad_val, pad_val, pad_val)
        else:
            pad_val_start = pad // 2
            pad_val_end = pad - pad_val_start
            padding = (pad_val_start, pad_val_end, pad_val_start, pad_val_end)
        # print(x.shape, padding)
        x = F.pad(x, padding, "constant", 0)
        # print(x.shape)
        return x


####
class ResidualBlock(Net):
    """Residual block as defined in:

    He, Kaiming, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. "Deep residual learning 
    for image recognition." In Proceedings of the IEEE conference on computer vision 
    and pattern recognition, pp. 770-778. 2016.

    """

    def __init__(self, in_ch, unit_ksize, unit_ch, unit_count, stride=1):
        super(ResidualBlock, self).__init__()
        assert len(unit_ksize) == len(unit_ch), "Unbalance Unit Info"

        self.nr_unit = unit_count
        self.in_ch = in_ch
        self.unit_ch = unit_ch

        # ! For inference only so init values for batchnorm may not match tensorflow
        unit_in_ch = in_ch
        self.units = nn.ModuleList()
        for idx in range(unit_count):
            unit_layer = [
                ("preact/bn", nn.BatchNorm2d(unit_in_ch, eps=1e-5)),
                ("preact/relu", nn.ReLU(inplace=True)),
                (
                    "conv1",
                    nn.Conv2d(
                        unit_in_ch,
                        unit_ch[0],
                        unit_ksize[0],
                        stride=1,
                        padding=0,
                        bias=False,
                    ),
                ),
                ("conv1/bn", nn.BatchNorm2d(unit_ch[0], eps=1e-5)),
                ("conv1/relu", nn.ReLU(inplace=True)),
                (
                    "conv2/pad",
                    TFSamepaddingLayer(
                        ksize=unit_ksize[1], stride=stride if idx == 0 else 1
                    ),
                ),
                (
                    "conv2",
                    nn.Conv2d(
                        unit_ch[0],
                        unit_ch[1],
                        unit_ksize[1],
                        stride=stride if idx == 0 else 1,
                        padding=0,
                        bias=False,
                    ),
                ),
                ("conv2/bn", nn.BatchNorm2d(unit_ch[1], eps=1e-5)),
                ("conv2/relu", nn.ReLU(inplace=True)),
                (
                    "conv3",
                    nn.Conv2d(
                        unit_ch[1],
                        unit_ch[2],
                        unit_ksize[2],
                        stride=1,
                        padding=0,
                        bias=False,
                    ),
                ),
            ]
            # * has bna to conclude each previous block so
            # * must not put preact for the first unit of this block
            unit_layer = unit_layer if idx != 0 else unit_layer[2:]
            self.units.append(nn.Sequential(OrderedDict(unit_layer)))
            unit_in_ch = unit_ch[-1]

        if in_ch != unit_ch[-1] or stride != 1:
            self.shortcut = nn.Conv2d(in_ch, unit_ch[-1], 1, stride=stride, bias=False)
        else:
            self.shortcut = None

        self.blk_bna = nn.Sequential(
            OrderedDict(
                [
                    ("bn", nn.BatchNorm2d(unit_in_ch, eps=1e-5)),
                    ("relu", nn.ReLU(inplace=True)),
                ]
            )
        )

        # print(self.units[0])
        # print(self.units[1])
        # exit()

    def out_ch(self):
        return self.unit_ch[-1]

    def forward(self, prev_feat, freeze=False):
        if self.shortcut is None:
            shortcut = prev_feat
        else:
            shortcut = self.shortcut(prev_feat)

        for idx in range(0, len(self.units)):
            new_feat = prev_feat
            if self.training:
                with torch.set_grad_enabled(not freeze):
                    new_feat = self.units[idx](new_feat)
            else:
                new_feat = self.units[idx](new_feat)
            prev_feat = new_feat + shortcut
            shortcut = prev_feat
        feat = self.blk_bna(prev_feat)
        return feat


class DenseBlock(Net):
    """Dense Block as defined in:

    Huang, Gao, Zhuang Liu, Laurens Van Der Maaten, and Kilian Q. Weinberger. 
    "Densely connected convolutional networks." In Proceedings of the IEEE conference 
    on computer vision and pattern recognition, pp. 4700-4708. 2017.

    Only performs `valid` convolution.

    """

    def __init__(self, in_ch, unit_ksize, unit_ch, unit_count, split=1, extra_features=0):
        super(DenseBlock, self).__init__()
        assert len(unit_ksize) == len(unit_ch), "Unbalance Unit Info"

        self.nr_unit = unit_count
        self.in_ch = in_ch
        self.unit_ch = unit_ch
        self.extra_features = extra_features  # 添加额外特征通道

        # ! For inference only so init values for batchnorm may not match tensorflow
        unit_in_ch = in_ch + extra_features  # 在初始输入通道中加入额外特征通道
        self.units = nn.ModuleList()
        for idx in range(unit_count):
            self.units.append(
                nn.Sequential(
                    OrderedDict(
                        [
                            ("preact_bna/bn", nn.BatchNorm2d(unit_in_ch, eps=1e-5)),
                            ("preact_bna/relu", nn.ReLU(inplace=True)),
                            (
                                "conv1",
                                nn.Conv2d(
                                    unit_in_ch,
                                    unit_ch[0],
                                    unit_ksize[0],
                                    stride=1,
                                    padding=0,
                                    bias=False,
                                ),
                            ),
                            ("conv1/bn", nn.BatchNorm2d(unit_ch[0], eps=1e-5)),
                            ("conv1/relu", nn.ReLU(inplace=True)),
                            # ('conv2/pool', TFSamepaddingLayer(ksize=unit_ksize[1], stride=1)),
                            (
                                "conv2",
                                nn.Conv2d(
                                    unit_ch[0],
                                    unit_ch[1],
                                    unit_ksize[1],
                                    groups=split,
                                    stride=1,
                                    padding=0,
                                    bias=False,
                                ),
                            ),
                        ]
                    )
                )
            )
            unit_in_ch += unit_ch[1]

        self.blk_bna = nn.Sequential(
            OrderedDict(
                [
                    ("bn", nn.BatchNorm2d(unit_in_ch, eps=1e-5)),
                    ("relu", nn.ReLU(inplace=True)),
                ]
            )
        )

    def out_ch(self):
        return self.in_ch + self.nr_unit * self.unit_ch[-1] + self.extra_features

    def forward(self, prev_feat):
        # 如果有额外特征通道，将其添加到输入特征图中
        if self.extra_features > 0:
            extra_feat = torch.zeros_like(prev_feat[:, :self.extra_features, :, :])
            prev_feat = torch.cat([prev_feat, extra_feat], dim=1)

        for idx in range(self.nr_unit):
            if prev_feat.shape[2] < 3 or prev_feat.shape[3] < 3:
                prev_feat = F.interpolate(prev_feat, size=(4, 4), mode='bilinear', align_corners=False)
            new_feat = self.units[idx](prev_feat)
            prev_feat = crop_to_shape(prev_feat, new_feat)
            prev_feat = torch.cat([prev_feat, new_feat], dim=1)
        prev_feat = self.blk_bna(prev_feat)

        return prev_feat

####
class UpSample2x(nn.Module):
    """Upsample input by a factor of 2.
    
    Assume input is of NCHW, port FixedUnpooling from TensorPack.
    """

    def __init__(self):
        super(UpSample2x, self).__init__()
        # correct way to create constant within module
        self.register_buffer(
            "unpool_mat", torch.from_numpy(np.ones((2, 2), dtype="float32"))
        )
        self.unpool_mat.unsqueeze(0)

    def forward(self, x):
        input_shape = list(x.shape)
        # unsqueeze is expand_dims equivalent
        # permute is transpose equivalent
        # view is reshape equivalent
        x = x.unsqueeze(-1)  # bchwx1
        mat = self.unpool_mat.unsqueeze(0)  # 1xshxsw
        ret = torch.tensordot(x, mat, dims=1)  # bxcxhxwxshxsw
        ret = ret.permute(0, 1, 2, 4, 3, 5)
        ret = ret.reshape((-1, input_shape[1], input_shape[2] * 2, input_shape[3] * 2))
        return ret


####
class EnhancedResBlock(nn.Module):
    """支持 stride 下采样的增强残差块"""
    def __init__(self, in_ch, out_ch, stride=1, use_se=False, dilation=1):
        super().__init__()
        self.stride = stride
        mid_ch = out_ch // 4

        # 主分支
        self.conv1 = nn.Conv2d(in_ch, mid_ch, kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_ch)
        
        # 关键修改：stride 应用于第二个卷积
        self.conv2 = nn.Conv2d(
            mid_ch, mid_ch, 
            kernel_size=3, 
            stride=stride,  # 下采样发生在此处
            padding=dilation, 
            dilation=dilation,
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(mid_ch)
        
        self.conv3 = nn.Conv2d(mid_ch, out_ch, kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_ch)
        
        # SE注意力
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_ch, out_ch//16, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(out_ch//16, out_ch, kernel_size=1),
            nn.Sigmoid()
        ) if use_se else None

        # 多头自注意力，注意力头设置成

        # 跳跃连接处理下采样
        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch)
            )

    def forward(self, x):
        residual = self.shortcut(x)
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x, inplace=True)
        
        x = self.conv2(x)  # 下采样发生在此层
        x = self.bn2(x)
        x = F.relu(x, inplace=True)
        
        x = self.conv3(x)
        x = self.bn3(x)
        
        if self.se is not None:
            x = x * self.se(x)
        
        x += residual
        return F.relu(x, inplace=True)

class MultiHeadSelfAttention2D(nn.Module):
    """2D多头自注意力模块，适用于特征图"""
    def __init__(self, embed_dim, num_heads=8):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        # 线性变换矩阵
        self.qkv = nn.Conv2d(embed_dim, embed_dim * 3, kernel_size=1)
        self.out = nn.Conv2d(embed_dim, embed_dim, kernel_size=1)
        
        # 相对位置编码
        self.rel_pos_enc = nn.Parameter(torch.randn(num_heads, 2 * 16 - 1, 2 * 16 - 1) * 0.02)
        
    def forward(self, x):
        B, C, H, W = x.shape
        
        # 生成Q,K,V
        qkv = self.qkv(x).chunk(3, dim=1)  # 3个[B, C, H, W]张量
        q, k, v = map(lambda t: t.view(B, self.num_heads, self.head_dim, H * W), qkv)
        
        # 计算注意力分数
        attn = torch.einsum('bhdn,bhdm->bhnm', q, k)  # [B, num_heads, H*W, H*W]
        attn = attn * (self.head_dim ** -0.5)
        
        # 添加相对位置编码
        h_rel = torch.arange(H, device=x.device).view(1, 1, H, 1) - torch.arange(H, device=x.device).view(1, 1, 1, H)
        w_rel = torch.arange(W, device=x.device).view(1, 1, W, 1) - torch.arange(W, device=x.device).view(1, 1, 1, W)
        
        # 使用相对位置编码
        h_idx = h_rel + 16 - 1  # 偏移到非负索引
        w_idx = w_rel + 16 - 1
        pos_bias = self.rel_pos_enc[:, h_idx, w_idx]  # [num_heads, H, W, H, W]
        pos_bias = pos_bias.permute(1, 2, 3, 4, 0).reshape(H*W, H*W, self.num_heads)
        attn = attn + pos_bias.permute(2, 0, 1).unsqueeze(0)
        
        # 计算注意力权重
        attn = F.softmax(attn, dim=-1)
        
        # 应用注意力到V
        out = torch.einsum('bhnm,bhdm->bhdn', attn, v)  # [B, num_heads, head_dim, H*W]
        out = out.reshape(B, C, H, W)
        
        # 输出投影
        out = self.out(out)
        return out

class EnhancedResBlock(nn.Module):
    """支持 stride 下采样的增强残差块，包含多头自注意力"""
    def __init__(self, in_ch, out_ch, stride=1, use_se=False, dilation=1, use_mhsa=False):
        super().__init__()
        self.stride = stride
        mid_ch = out_ch // 4
        self.use_mhsa = use_mhsa

        # 主分支
        self.conv1 = nn.Conv2d(in_ch, mid_ch, kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_ch)
        
        # 关键修改：stride 应用于第二个卷积
        self.conv2 = nn.Conv2d(
            mid_ch, mid_ch, 
            kernel_size=3, 
            stride=stride,  # 下采样发生在此处
            padding=dilation, 
            dilation=dilation,
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(mid_ch)
        
        self.conv3 = nn.Conv2d(mid_ch, out_ch, kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_ch)
        
        # SE注意力
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_ch, out_ch//16, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(out_ch//16, out_ch, kernel_size=1),
            nn.Sigmoid()
        ) if use_se else None

        # 多头自注意力
        # self.mhsa = MultiHeadSelfAttention2D(mid_ch) if use_mhsa else None

        # 跳跃连接处理下采样
        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch)
            )

    def forward(self, x):
        residual = self.shortcut(x)
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x, inplace=True)
        
        x = self.conv2(x)  # 下采样发生在此层
        x = self.bn2(x)
        
        # # 应用多头自注意力
        # if self.use_mhsa and self.mhsa is not None:
        #     x = x + self.mhsa(x)  # 残差连接
        
        x = F.relu(x, inplace=True)
        
        x = self.conv3(x)
        x = self.bn3(x)
        
        if self.se is not None:
            x = x * self.se(x)
        
        x += residual
        return F.relu(x, inplace=True)


class DenseEnhancement(nn.Module):
    """增强版密集连接模块"""
    def __init__(self, in_ch, growth_rate=32, layers=4):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.BatchNorm2d(in_ch + i*growth_rate),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_ch + i*growth_rate, growth_rate, 3, padding=1, bias=False)
            ) for i in range(layers)
        ])
        self.fuse = nn.Conv2d(in_ch + layers*growth_rate, in_ch, 1)  # 通道复原

    def forward(self, x):
        features = [x]
        # print(features)
        for layer in self.layers:
            new_feat = layer(torch.cat(features, dim=1))
            features.append(new_feat)
        return self.fuse(torch.cat(features, dim=1)) + x  # 残差连接