from model.BiFPN.bifpn import BiFPN
from .model_3d import EfficientNet3D
import torch
import torch.nn as nn
from ..BiFPN.utils import LSEPooling3dStaticSamePadding
'''
https://github.com/tristandb/EfficientDet-PyTorch
https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch
'''
class EfficientNet3D_3_input(nn.Module):
    def __init__(self, structure_num, size, num_classes, normal=True):
        super(EfficientNet3D_3_input, self).__init__()
        # num_classes不重要，都會重新取在linear前進行BiFPN
        self.efficientnet = EfficientNet3D.from_name(f"efficientnet-{structure_num}", in_channels=1, num_classes=num_classes, image_size=size, normal=normal)

    def forward(self, x1, x2, x3):
        # 对每个输入独立执行EfficientNet处理
        out1 = self.process_input(x1)
        out2 = self.process_input(x2)
        out3 = self.process_input(x3)
        return out1, out2, out3

    def process_input(self, x):
        # 使用EfficientNet处理输入
        x, _ = self.efficientnet.extract_endpoints(x)
        return x

class BiFPN_3_input(nn.Module):
    '''
    num_layer在B0的時候預設是3層，設置的要減一層
    '''
    def __init__(self, num_classes, feature_size=64, num_layers=2, epsilon=0.0001, dropout=0.5, fpn_type='label_concat'):
        super(BiFPN_3_input, self).__init__()
        # 要能取得efficient 架構
        self.bifpn = BiFPN([24,40,112], feature_size, num_layers, epsilon)
        self.fpn_type = fpn_type
        self.classifier_3 = nn.Sequential(
            nn.Linear(64*16*16*16, 1000),
            nn.ReLU(),
            nn.Linear(1000, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes)
        )
        self.classifier_4 = nn.Sequential(
            nn.Linear(64*8*8*8, 1000),
            nn.ReLU(),
            nn.Linear(1000, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes)
        )
        self.classifier_5 = nn.Sequential(
            nn.Linear(64*4*4*4, 1000),
            nn.ReLU(),
            nn.Linear(1000, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes)
        )
        self.classifier_6 = nn.Sequential(
            # nn.Linear(64*2*2*1, 1000),
            # nn.ReLU(),
            # nn.Linear(1000, 512),
            # nn.ReLU(),
            # nn.Dropout(dropout),
            # nn.Linear(512, num_classes)
            nn.Linear(64*2*2*2, num_classes)
        )
        self.upsample = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.downsample = nn.

    def forward(self, x1, x2, x3):
        outputs = {}
        outputs_tmp = {}
        outputs_final = {}
        outputs["out_liv3"], outputs["out_liv4"], outputs["out_liv5"], outputs["out_liv6"] = self.bifpn(x1)
        outputs["out_spl3"], outputs["out_spl4"], outputs["out_spl5"], outputs["out_spl6"] = self.bifpn(x2)
        outputs["out_kid3"], outputs["out_kid4"], outputs["out_kid5"], outputs["out_kid6"] = self.bifpn(x3)
        organ_list = ['liv','spl','kid']
        if self.fpn_type == 'label_concat':
            for organ in organ_list:
                for fpn_layer in [3,4,5,6]:
                    key = f'out_{organ}{fpn_layer}'
                    outputs[key] = outputs[key].view(outputs[key].size(0), -1)
                    # 使用 getattr 動態獲取 classifier 方法
                    classifier = getattr(self, f'classifier_{fpn_layer}')
                    outputs[f'feature_concated_{fpn_layer}'] = classifier(outputs[key])
                outputs_final[organ] = torch.cat((outputs["feature_concated_3"],outputs["feature_concated_4"],outputs["feature_concated_5"],outputs["feature_concated_6"]), dim=1)
        elif self.fpn_type == 'feature_concat':
            # feature size: 16, 8, 4, 2
            
            
        elif self.fpn_type == 'split':
            for organ in organ_list:
                for fpn_layer in [3,4,5,6]:
                    key = f'out_{organ}{fpn_layer}'
                    outputs[key] = outputs[key].view(outputs[key].size(0), -1)
                    # 使用 getattr 動態獲取 classifier 方法
                    classifier = getattr(self, f'classifier_{fpn_layer}')
                    outputs_tmp[f'feature_concated_{fpn_layer}'] = classifier(outputs[key])
                outputs_final[organ] = outputs_tmp

        return outputs_final


class EfficientNet3D_BiFPN(nn.Module):
    def __init__(self, size, structure_num, class_num, dropout=0.2, fpn_type='label_concat', normal=True):
        super(EfficientNet3D_BiFPN, self).__init__()
        self.fpn_type = fpn_type
        self.efficientnet =  EfficientNet3D_3_input(structure_num, size, class_num, normal)
        self.bifpn = BiFPN_3_input(num_classes=class_num, dropout=dropout, fpn_type=fpn_type)
        self.classifier = nn.Linear(class_num*4, class_num)

    def forward(self, x1,x2,x3):
        if self.fpn_type == 'label_concat':
            liv_all, spl_all, kid_all = self.efficientnet(x1,x2,x3)
            reduction_keys = ['reduction_3', 'reduction_4', 'reduction_5']
            liv = tuple(liv_all[key] for key in reduction_keys)
            spl = tuple(spl_all[key] for key in reduction_keys)
            kid = tuple(kid_all[key] for key in reduction_keys)
            fpn_layer = self.bifpn(liv, spl, kid)
            output_liv = self.classifier(fpn_layer['liv'])
            output_spl = self.classifier(fpn_layer['spl'])
            output_kid = self.classifier(fpn_layer['kid'])

        return output_liv, output_spl, output_kid