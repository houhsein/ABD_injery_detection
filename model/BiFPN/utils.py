import math

from torch import nn
import torch.nn.functional as F

class Conv3dStaticSamePadding(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, bias=True, groups=1, dilation=1, **kwargs):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride,
                              bias=bias, groups=groups)
        self.stride = self.conv.stride
        self.kernel_size = self.conv.kernel_size
        self.dilation = self.conv.dilation

        if isinstance(self.stride, int):
            self.stride = [self.stride] * 3
        elif len(self.stride) == 1:
            self.stride = [self.stride[0]] * 3

        if isinstance(self.kernel_size, int):
            self.kernel_size = [self.kernel_size] * 3
        elif len(self.kernel_size) == 1:
            self.kernel_size = [self.kernel_size[0]] * 3

    def forward(self, x):
        # channel first
        h, w, d = x.shape[-3:]
        
        extra_w = (math.ceil(w / self.stride[1]) - 1) * self.stride[1] - w + self.kernel_size[1]
        extra_h = (math.ceil(h / self.stride[0]) - 1) * self.stride[0] - h + self.kernel_size[0]
        extra_d = (math.ceil(d / self.stride[2]) - 1) * self.stride[2] - h + self.kernel_size[2]
        
        # 平均分配padding在前後左右高低
        left = extra_w // 2
        right = extra_w - left
        top = extra_h // 2
        bottom = extra_h - top
        low = extra_d // 2
        high = extra_d - low

        x = F.pad(x, [left, right, top, bottom, low, high])

        x = self.conv(x)
        return x


class MaxPool3dStaticSamePadding(nn.Module):

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.pool = nn.MaxPool3d(*args, **kwargs)
        self.stride = self.pool.stride
        self.kernel_size = self.pool.kernel_size

        if isinstance(self.stride, int):
            self.stride = [self.stride] * 3
        elif len(self.stride) == 1:
            self.stride = [self.stride[0]] * 3

        if isinstance(self.kernel_size, int):
            self.kernel_size = [self.kernel_size] * 3
        elif len(self.kernel_size) == 1:
            self.kernel_size = [self.kernel_size[0]] * 3

    def forward(self, x):
        h, w, d = x.shape[-3:]
        
        extra_w = (math.ceil(w / self.stride[1]) - 1) * self.stride[1] - w + self.kernel_size[1]
        extra_h = (math.ceil(h / self.stride[0]) - 1) * self.stride[0] - h + self.kernel_size[0]
        extra_d = (math.ceil(d / self.stride[2]) - 1) * self.stride[2] - h + self.kernel_size[2]

        left = extra_w // 2
        right = extra_w - left
        top = extra_h // 2
        bottom = extra_h - top
        low = extra_d // 2
        high = extra_d - low

        x = F.pad(x, [left, right, top, bottom, low, high])

        x = self.pool(x)
        return x