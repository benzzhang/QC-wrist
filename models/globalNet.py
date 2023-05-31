'''
Author: Peng Bo
Date: 2022-04-27 16:09:11
LastEditTime: 2023-05-31 11:43:46
Description: 

@inproceedings{zhu2021you,
  title={You only Learn Once: Universal Anatomical Landmark Detection},
  author={Zhu, Heqin and Yao, Qingsong and Xiao, Li and Zhou, S. Kevin},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
  pages={85--95},
  year={2021},
  organization={Springer}
}
    reference: https://github.com/MIRACLE-Center/YOLO_Universal_Anatomical_Landmark_Detection
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['globalNet']

class myConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1):
        super(myConv2d, self).__init__()
        padding = (kernel_size-1)//2
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=kernel_size, padding=padding)

    def forward(self, x):
        return self.conv(x)


class dilatedConv(nn.Module):
    ''' stride == 1 '''

    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1):
        super(dilatedConv, self).__init__()
        # f = (kernel_size-1) * d +1
        # new_width = (width - f + 2 * padding)/stride + stride
        padding = (kernel_size-1) * dilation // 2
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size, dilation=dilation, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class globalNet(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=0.25, kernel_size=3):
        super(globalNet, self).__init__()

        self.scale_factor = scale_factor
        mid_channels = 128
        self.in_conv = myConv2d(in_channels+out_channels, mid_channels, 1)
        self.dilated_conv = dilatedConv(mid_channels, mid_channels, kernel_size, dilation=5)
        self.out_conv = myConv2d(mid_channels, out_channels, 1)

    def forward(self, x):
        size = x.size()[2:]
        sf = self.scale_factor
        x = F.interpolate(x, scale_factor=sf)

        x = self.in_conv(x)
        x = self.dilated_conv(x)
        x = self.out_conv(x)

        x = F.interpolate(x, size=size)
        return torch.sigmoid(x)
        # return x
