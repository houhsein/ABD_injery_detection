from ..BiFPN.bifpn import BiFPN
from model_3d import EfficientNet3D
import torch
import torch.nn as nn

class EfficientNet3D_BiFPN(nn.Module):
    def __init__(self, image_size, n_input_channels=4, class_mum=7, fpn_loss='concat', normal=False):
        super(EfficientNet3D_BiFPN, self).__init__()
        self.fpn_loss = fpn_loss
        # num_classes不重要，都會重新取在linear前進行BiFPN
        efficientnet =  EfficientNet3D.from_name(f"efficientnet-b0", in_channels=1, num_classes=2, image_size=size, normal=normal)
