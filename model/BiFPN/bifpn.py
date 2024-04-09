import torch
import torch.nn as nn
from .utils import Conv3dStaticSamePadding, MaxPool3dStaticSamePadding
import torch.nn.functional as F


class SeparableConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels=None):
        super(SeparableConvBlock, self).__init__()
        if out_channels is None:
            out_channels = in_channels

        # Q: whether separate conv
        #  share bias between depthwise_conv and pointwise_conv
        #  or just pointwise_conv apply bias.
        # A: Confirmed, just pointwise_conv applies bias, depthwise_conv has no bias.

        self.depthwise_conv = Conv3dStaticSamePadding(in_channels, in_channels,
                                                      kernel_size=3, stride=1, groups=in_channels, bias=False)
        self.pointwise_conv = Conv3dStaticSamePadding(in_channels, out_channels, kernel_size=1, stride=1)

        self.bn = nn.BatchNorm3d(num_features=out_channels, momentum=0.01, eps=1e-3)
        self.act = nn.ReLU()


    def forward(self, x):
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)
        x = self.bn(x)
        x = self.act(x)

        return x

class ConvBlock(nn.Module):
    """
    Convolution block with Batch Normalization and ReLU activation.
    
    """
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1):
        super(ConvBlock,self).__init__()
        self.conv = Conv3dStaticSamePadding(in_channels, out_channels, kernel_size, stride=stride)
        self.bn = nn.BatchNorm3d(out_channels, momentum=0.01, eps=1e-3)
        self.pool = MaxPool3dStaticSamePadding(3, 2)
        self.act = nn.ReLU()

    def forward(self, inputs):
        x = self.conv(inputs)
        x = self.bn(x)
        x = self.pool(x)
        x = self.act(x)
        return x

class BiFPNBlock(nn.Module):
    """
    Bi-directional Feature Pyramid Network
    """
    def __init__(self, feature_size=64, epsilon=0.0001):
        super(BiFPNBlock, self).__init__()
        self.epsilon = epsilon
        
        self.p3_td = SeparableConvBlock(feature_size, feature_size)
        self.p4_td = SeparableConvBlock(feature_size, feature_size)
        self.p5_td = SeparableConvBlock(feature_size, feature_size)
        self.p6_td = SeparableConvBlock(feature_size, feature_size)
        
        self.p4_out = SeparableConvBlock(feature_size, feature_size)
        self.p5_out = SeparableConvBlock(feature_size, feature_size)
        self.p6_out = SeparableConvBlock(feature_size, feature_size)
        #self.p7_out = SeparableConvBlock(feature_size, feature_size)

        self.upsample = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.downsample = MaxPool3dStaticSamePadding(3, 2)

        # TODO: Init weights
        # self.w1 = nn.Parameter(torch.Tensor(2, 4))
        self.w1 = nn.Parameter(torch.Tensor(2, 3))
        self.w1_relu = nn.ReLU()
        # self.w2 = nn.Parameter(torch.Tensor(3, 4))
        self.w2 = nn.Parameter(torch.Tensor(3, 3))
        self.w2_relu = nn.ReLU()
    
    def forward(self, inputs):
        """
        illustration of a minimal bifpn unit
            P7_0 -------------------------> P7_2 -------->
               |-------------|                ↑
                             ↓                |
            P6_0 ---------> P6_1 ---------> P6_2 -------->
               |-------------|--------------↑ ↑
                             ↓                |
            P5_0 ---------> P5_1 ---------> P5_2 -------->
               |-------------|--------------↑ ↑
                             ↓                |
            P4_0 ---------> P4_1 ---------> P4_2 -------->
               |-------------|--------------↑ ↑
                             |--------------↓ |
            P3_0 -------------------------> P3_2 -------->
        """
        p3_x, p4_x, p5_x, p6_x = inputs
        
        # Calculate Top-Down Pathway
        w1 = self.w1_relu(self.w1)
        w1 = w1 / (torch.sum(w1, dim=0) + self.epsilon)
        w2 = self.w2_relu(self.w2)
        w2 = w2 / (torch.sum(w2, dim=0) + self.epsilon)
        
        #p7_td = p7_x
        # p6_td = self.p6_td(w1[0, 0] * p6_x + w1[1, 0] * self.upsample(p7_td))        
        # p5_td = self.p5_td(w1[0, 1] * p5_x + w1[1, 1] * self.upsample(p6_td))
        # p4_td = self.p4_td(w1[0, 2] * p4_x + w1[1, 2] * self.upsample(p5_td))
        # p3_td = self.p3_td(w1[0, 3] * p3_x + w1[1, 3] * self.upsample(p3_td))
        #p6_td = self.p6_td(w1[0, 0] * p6_x + w1[1, 0] * F.interpolate(p7_td, size=p6_x.shape[2:], mode='trilinear', align_corners=True))
        p6_td = p6_x        
        p5_td = self.p5_td(w1[0, 0] * p5_x + w1[1, 0] * F.interpolate(p6_td, size=p5_x.shape[2:], mode='trilinear', align_corners=True))
        p4_td = self.p4_td(w1[0, 1] * p4_x + w1[1, 1] * F.interpolate(p5_td, size=p4_x.shape[2:], mode='trilinear', align_corners=True))
        p3_td = self.p3_td(w1[0, 2] * p3_x + w1[1, 2] * F.interpolate(p4_td, size=p3_x.shape[2:], mode='trilinear', align_corners=True))
        
        # Calculate Bottom-Up Pathway
        p3_out = p3_td
        p4_out = self.p4_out(w2[0, 0] * p4_x + w2[1, 0] * p4_td + w2[2, 0] * self.downsample(p3_out))
        p5_out = self.p5_out(w2[0, 1] * p5_x + w2[1, 1] * p5_td + w2[2, 1] * self.downsample(p4_out))
        p6_out = self.p6_out(w2[0, 2] * p6_x + w2[1, 2] * p6_td + w2[2, 2] * self.downsample(p5_out))
        #p7_out = self.p7_out(w2[0, 3] * p7_x + w2[1, 3] * p7_td + w2[2, 3] * self.downsample(p6_out))

        return [p3_out, p4_out, p5_out, p6_out]

class BiFPN(nn.Module):
    '''
    Remove p7
    Add fpn feature map concat type
    '''
    def __init__(self, size, feature_size=64, num_layers=2, epsilon=0.0001):
        super(BiFPN, self).__init__()
        # 從backbone network中間層得到，已經過BN了，所以不需要再用
        self.p3 = nn.Conv3d(size[0], feature_size, kernel_size=1, stride=1, padding=0)
        self.p4 = nn.Conv3d(size[1], feature_size, kernel_size=1, stride=1, padding=0)
        self.p5 = nn.Conv3d(size[2], feature_size, kernel_size=1, stride=1, padding=0)
        
        #self.p6 = ConvBlock(size[2], feature_size, kernel_size=1, stride=1)
        #self.p7 = nn.Sequential(MaxPool3dStaticSamePadding(3, 2))
        self.p6 = nn.Conv3d(size[2], feature_size, kernel_size=3, stride=2, padding=1)
        #self.p7 = nn.Sequential(MaxPool3dStaticSamePadding(3, 2))
        #self.p7 = ConvBlock(feature_size, feature_size, kernel_size=3, stride=2)

        bifpns = []
        for _ in range(num_layers):
            bifpns.append(BiFPNBlock(feature_size))
        self.bifpn = nn.Sequential(*bifpns)
    
    def forward(self, inputs):
        # First time
        c3, c4, c5 = inputs
        
        # Calculate the input column of BiFPN
        p3_x = self.p3(c3)     
        p4_x = self.p4(c4)
        p5_x = self.p5(c5)
        p6_x = self.p6(c5)
        #p7_x = self.p7(p6_x)
        
        features = [p3_x, p4_x, p5_x, p6_x]
        # othere time
        return self.bifpn(features)