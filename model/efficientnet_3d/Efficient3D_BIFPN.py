from ..BiFPN.bifpn import BiFPN
from model_3d import EfficientNet3D
import torch
import torch.nn as nn

'''
https://github.com/tristandb/EfficientDet-PyTorch
https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch
'''
class EfficientNet3D_3_input(nn.Module):
    def __init__(self, image_size, normal=False):
        super(EfficientNet3D_3_input, self).__init__()
        # num_classes不重要，都會重新取在linear前進行BiFPN
        self.efficientnet = EfficientNet3D.from_name(f"efficientnet-b0", in_channels=1, num_classes=2, image_size=size, normal=normal)

    def forward(self, x1, x2, x3):
        # 对每个输入独立执行EfficientNet处理
        out1 = self.process_input(x1)
        out2 = self.process_input(x2)
        out3 = self.process_input(x3)
        return out1, out2, out3

     def process_input(self, x):
        # 使用EfficientNet处理输入
        x = self.efficientnet(x)
        return x


class EfficientNet3D_BiFPN(nn.Module):
    def __init__(self, image_size, n_input_channels=4, class_mum=9, fpn_loss='concat', normal=False):
        super(EfficientNet3D_BiFPN, self).__init__()
        self.fpn_loss = fpn_loss
        
        self.efficientnet =  EfficientNet3D_3_input
        self.bifpn = BiFPN()
        self.classifier = nn.Linear(class_num*3, class_num)

    def forward(self, x1,x2,x3):
        features, features2, features3 = self.efficientnet(x1,x2,x3)
