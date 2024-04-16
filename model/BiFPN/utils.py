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
        extra_d = (math.ceil(d / self.stride[2]) - 1) * self.stride[2] - d + self.kernel_size[2]
        
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
        extra_d = (math.ceil(d / self.stride[2]) - 1) * self.stride[2] - d + self.kernel_size[2]

        left = extra_w // 2
        right = extra_w - left
        top = extra_h // 2
        bottom = extra_h - top
        low = extra_d // 2
        high = extra_d - low

        x = F.pad(x, [left, right, top, bottom, low, high])

        x = self.pool(x)
        return x
# TODO 有問題
class LSEPooling3dStaticSamePadding(nn.Module):
    def __init__(self, stride, kernel, r=1):
        """
        初始化 LSE 池化层
        :param r: 平滑参数，r > 0。r 越大，池化操作越接近于最大池化；r 越小，操作越接近于平均池化。
        """
        super(LSEPooling3D, self).__init__()
        self.r = r

    def forward(self, x):
        """
        前向传播
        :param x: 输入张量，形状为 (batch_size, channels, depth, height, width)
        :return: LSE 池化后的输出张量
        """
        h, w, d = x.shape[-3:]
        
        extra_w = (math.ceil(w / self.stride[1]) - 1) * self.stride[1] - w + self.kernel_size[1]
        extra_h = (math.ceil(h / self.stride[0]) - 1) * self.stride[0] - h + self.kernel_size[0]
        extra_d = (math.ceil(d / self.stride[2]) - 1) * self.stride[2] - d + self.kernel_size[2]

        left = extra_w // 2
        right = extra_w - left
        top = extra_h // 2
        bottom = extra_h - top
        low = extra_d // 2
        high = extra_d - low

        x = F.pad(x, [left, right, top, bottom, low, high])

        x_exp = torch.exp(x * self.r)
        x_sum_exp = torch.sum(x_exp, dim=[2, 3, 4], keepdim=True)  
        x_lse = torch.log(x_sum_exp) / self.r

        return x_lse