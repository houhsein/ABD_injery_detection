from model.BiFPN.bifpn import BiFPN
from .model_3d import EfficientNet3D
import torch
import torch.nn as nn
from model.BiFPN.utils import LSEPooling3dStaticSamePadding
'''
https://github.com/tristandb/EfficientDet-PyTorch
https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch
'''
class EfficientNet3D_3_input(nn.Module):
    def __init__(self, structure_num, size, num_classes, depth_coefficient, normal=True, in_channels=1):
        super(EfficientNet3D_3_input, self).__init__()
        # num_classes不重要，都會重新取在linear前進行BiFPN
        self.efficientnet = EfficientNet3D.from_name(f"efficientnet-{structure_num}", in_channels=in_channels, num_classes=num_classes, image_size=size, normal=normal,  depth_coefficient=depth_coefficient)

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
    def __init__(self, class_num, feature_size=64, num_layers=2, epsilon=0.0001, dropout=0.5, fpn_type='label_concat'):
        super(BiFPN_3_input, self).__init__()
        # 要能取得efficient 架構
        self.bifpn = BiFPN([24, 40, 112], feature_size, num_layers, epsilon)
        self.fpn_type = fpn_type
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.classifier = nn.Linear(feature_size, class_num)
        self.feature_cat_classifier = nn.Linear(feature_size*4, class_num)
        # self.classifier_3 = nn.Sequential(
        #     nn.Linear(64*16*16*16, 1000),
        #     # nn.ReLU(),
        #     nn.Linear(1000, 512),
        #     # nn.ReLU(),
        #     nn.Dropout(dropout),
        #     nn.Linear(512, num_classes)
        # )
        # self.classifier_4 = nn.Sequential(
        #     nn.Linear(64*8*8*8, 1000),
        #     # nn.ReLU(),
        #     nn.Linear(1000, 512),
        #     # nn.ReLU(),
        #     nn.Dropout(dropout),
        #     nn.Linear(512, num_classes)
        # )
        # self.classifier_5 = nn.Sequential(
        #     nn.Linear(64*4*4*4, 1000),
        #     # nn.ReLU(),
        #     nn.Linear(1000, 512),
        #     # nn.ReLU(),
        #     nn.Dropout(dropout),
        #     nn.Linear(512, num_classes)
        # )
        # self.classifier_6 = nn.Sequential(
        #     # nn.Linear(64*2*2*1, 1000),
        #     # nn.ReLU(),
        #     # nn.Linear(1000, 512),
        #     # nn.ReLU(),
        #     # nn.Dropout(dropout),
        #     # nn.Linear(512, num_classes)
        #     nn.Linear(64*2*2*2, num_classes)
        # )
        self.upsample = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        # self.downsample = nn.

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
                    # classifier = getattr(self, f'classifier_{fpn_layer}')
                    # outputs[f'feature_concated_{fpn_layer}'] = classifier(outputs[key])
                    outputs[key] = self.avg_pool(outputs[key])
                    outputs[key] = torch.flatten(outputs[key], 1)
                    outputs[f'feature_concated_{fpn_layer}'] = self.classifier(outputs[key])
                outputs_final[organ] = torch.cat((outputs["feature_concated_3"],outputs["feature_concated_4"],outputs["feature_concated_5"],outputs["feature_concated_6"]), dim=1)
        
        elif self.fpn_type == 'feature_concat':
            # feature size: 16, 8, 4, 2
            for organ in organ_list:
                pooled_features = []
                for fpn_layer in [3,4,5,6]:
                    key = f'out_{organ}{fpn_layer}'
                    outputs[key] = self.avg_pool(outputs[key])
                    outputs[key] = torch.flatten(outputs[key], 1)
                    pooled_features.append(outputs[key])
                concatenated_features = torch.cat(pooled_features, dim=1)
                outputs_final[organ] = self.feature_cat_classifier(concatenated_features)    
        
        elif self.fpn_type == 'split':
            pass
            # for organ in organ_list:
            #     for fpn_layer in [3,4,5,6]:
            #         key = f'out_{organ}{fpn_layer}'
            #         outputs[key] = outputs[key].view(outputs[key].size(0), -1)
            #         # 使用 getattr 動態獲取 classifier 方法
            #         classifier = getattr(self, f'classifier_{fpn_layer}')
            #         outputs_tmp[f'feature_concated_{fpn_layer}'] = classifier(outputs[key])
            #     outputs_final[organ] = outputs_tmp

        return outputs_final


class EfficientNet3D_BiFPN(nn.Module):
    def __init__(self, size, structure_num, class_num, dropout=0.2, depth_coefficient=0.75, fpn_type='label_concat', normal=True, in_channels=1):
        super(EfficientNet3D_BiFPN, self).__init__()
        self.fpn_type = fpn_type
        self.efficientnet =  EfficientNet3D_3_input(structure_num, size, class_num, normal, depth_coefficient, in_channels=in_channels)
        self.bifpn = BiFPN_3_input(class_num=class_num, dropout=dropout, fpn_type=fpn_type)
        self.classifier = nn.Linear(class_num*4, class_num)

    def forward(self, x1,x2,x3):
        liv_all, spl_all, kid_all = self.efficientnet(x1,x2,x3)
        reduction_keys = ['reduction_3', 'reduction_4', 'reduction_5']
        liv = tuple(liv_all[key] for key in reduction_keys)
        spl = tuple(spl_all[key] for key in reduction_keys)
        kid = tuple(kid_all[key] for key in reduction_keys)

        if self.fpn_type == 'label_concat':
            fpn_layer = self.bifpn(liv, spl, kid)
            output_liv = self.classifier(fpn_layer['liv'])
            output_spl = self.classifier(fpn_layer['spl'])
            output_kid = self.classifier(fpn_layer['kid'])

        elif self.fpn_type == 'feature_concat':
            # feature size: 16, 8, 4, 2
            fpn_layer = self.bifpn(liv, spl, kid)
            output_liv = fpn_layer['liv']
            output_spl = fpn_layer['spl']
            output_kid = fpn_layer['kid']
            
        elif self.fpn_type == 'split':
            # 3,4,5,6 layer mutiple output
            fpn_layer = self.bifpn(liv, spl, kid)
            output_liv = fpn_layer['liv']
            output_spl = fpn_layer['spl']
            output_kid = fpn_layer['kid']

        return output_liv, output_spl, output_kid

class FPN3D(nn.Module):
    def __init__(self, input_channels, output_channels, class_num, fpn_type, normalize=False):
        super(FPN3D, self).__init__()
        self.output_channels = output_channels
        self.normalize = normalize
        self.fpn_type = fpn_type
        self.conv1 = nn.Conv3d(input_channels[0], output_channels, kernel_size=1)
        self.conv2 = nn.Conv3d(input_channels[1], output_channels, kernel_size=1)
        self.conv3 = nn.Conv3d(input_channels[2], output_channels, kernel_size=1)
        self.conv4 = nn.Conv3d(input_channels[3], output_channels, kernel_size=1)
        self.upsample = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.smooth = nn.Conv3d(output_channels, output_channels, kernel_size=3, stride=1, padding=1)
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.classifier = nn.Linear(output_channels, class_num)
        self.feature_cat_classifier = nn.Linear(output_channels*3, class_num)
        # self.classifier_4 = nn.Sequential(
        #     nn.Linear(256*2*2*2, 1000), 
        #     nn.Linear(1000, 512), 
        #     nn.Dropout(dropout),
        #     nn.Linear(512, class_num)   
        # )
        # self.classifier_3 = nn.Sequential(
        #     nn.Linear(256*4*4*4, 1000), 
        #     nn.Linear(1000, 512), 
        #     nn.Dropout(dropout),
        #     nn.Linear(512, class_num)   
        # )
        # self.classifier_2 = nn.Sequential(
        #     nn.Linear(256*8*8*8, 1000), 
        #     nn.Linear(1000, 512), 
        #     nn.Dropout(dropout),
        #     nn.Linear(512, class_num)  
        # )
        # self.classifier_1 = nn.Sequential(
        #     nn.Linear(3*256*16*16*16, 1000), 
        #     nn.Linear(1000, 512), 
        #     nn.Dropout(0.2),
        #     nn.Linear(512, class_num)  
        # )

    def forward(self, x1, x2, x3):
        # out_liv1, out_liv2, out_liv3, out_liv4 = self.process_input(x1)
        # out_spl1, out_spl2, out_spl3, out_spl4 = self.process_input(x2)
        # out_kid1, out_kid2, out_kid3, out_kid4 = self.process_input(x3)
        outputs = {}
        outputs_final = {}
        outputs["out_liv2"], outputs["out_liv3"], outputs["out_liv4"] = self.process_input(x1)
        outputs["out_spl2"], outputs["out_spl3"], outputs["out_spl4"] = self.process_input(x2)
        outputs["out_kid2"], outputs["out_kid3"], outputs["out_kid4"] = self.process_input(x3)
        organ_list = ['liv','spl','kid']
        
        # TODO　各Input的feature整合，可以有其他方法
        # feature_concated_1 = torch.cat((out_liv1, out_spl1, out_kid1), dim=1) 
        # feature_concated_2 = torch.cat((out_liv2, out_spl2, out_kid2), dim=1) 
        # feature_concated_3 = torch.cat((out_liv3, out_spl3, out_kid3), dim=1)
        # feature_concated_4 = torch.cat((out_liv4, out_spl4, out_kid4), dim=1)
        if self.fpn_type == 'label_concat':
            for organ in organ_list:
                for fpn_layer in [2,3,4]:
                    key = f'out_{organ}{fpn_layer}'
                    # outputs[key] = outputs[key].view(outputs[key].size(0), -1)
                # 使用 getattr 動態獲取 classifier 方法
                    # classifier = getattr(self, f'classifier_{fpn_layer}')
                # 各自全連接分類結果
                # feature_concated_1 = self.classifier_1(feature_concated_1)
                    # outputs[f'feature_concated_{fpn_layer}'] = classifier(outputs[key])
                    if self.normalize:
                        outputs[key] = self.normalize_features(outputs[key])
                    outputs[key] = self.avg_pool(outputs[key])
                    outputs[key] = torch.flatten(outputs[key], 1)
                    outputs[f'feature_concated_{fpn_layer}'] = self.classifier(outputs[key])
                outputs_final[organ] = torch.cat((outputs["feature_concated_2"],outputs["feature_concated_3"],outputs["feature_concated_4"]), dim=1)
        elif self.fpn_type == 'feature_concat':
            # 將展平的特徵圖先cat再分類
            for organ in organ_list:
                pooled_features = []
                for fpn_layer in [2,3,4]:
                    key = f'out_{organ}{fpn_layer}'
                    if self.normalize:
                        outputs[key] = self.normalize_features(outputs[key])
                    outputs[key] = self.avg_pool(outputs[key])
                    outputs[key] = torch.flatten(outputs[key], 1)
                    pooled_features.append(outputs[key])
                concatenated_features = torch.cat(pooled_features, dim=1)
                outputs_final[organ] = self.feature_cat_classifier(concatenated_features) 
        #return out
        # return feature_concated_1, feature_concated_2, feature_concated_3, feature_concated_4
        return outputs_final

    def normalize_features(self, features):
        mean = features.mean(dim=[2, 3, 4], keepdim=True)
        std = features.std(dim=[2, 3, 4], keepdim=True)
        return (features - mean) / (std + 1e-5)

    def process_input(self, x):
        # x is a list of feature maps from efficientnet at different scales
        x2, x3, x4 = x

        # x1 = self.conv1(x1)
        x2 = self.conv2(x2)
        x3 = self.conv3(x3)
        x4 = self.conv4(x4)
        # 上采样到下一层级的尺寸并加和
        
        x4_up = self.upsample(x4)  # 将 x4 上采样到 x3 的尺寸
        x3 = x4_up + x3
        x3_up = self.upsample(x3)  # 将 x3 上采样到 x2 的尺寸
        x2 = x3_up + x2
        # x2_up = self.upsample(x2)  # 将 x2 上采样到 x1 的尺寸
        # x1 = x2_up + x1
        # gradcam 取的convd層
        x4 = self.smooth(x4) 
        x3 = self.smooth(x3)
        x2 = self.smooth(x2)
        # x1 = self.smooth(x1)

        # return x1, x2, x3, x4
        return x2, x3, x4

class EfficientNet3D_FPN(nn.Module):
    def __init__(self, size, structure_num, class_num, depth_coefficient=0.75, fpn_type='label_concat', normal=True, normalize=False, in_channels=1):
        super(EfficientNet3D_FPN, self).__init__()
        self.fpn_type = fpn_type
        self.efficientnet =  EfficientNet3D_3_input(structure_num, size, class_num, normal, depth_coefficient, in_channels=in_channels)
        self.fpn = FPN3D(input_channels=[16, 24, 40, 112], output_channels=256, class_num=class_num, normalize=normalize, fpn_type=fpn_type)
        self.classifier = nn.Linear(class_num*3, class_num)

    def forward(self, x1,x2,x3):
        liv_all, spl_all, kid_all = self.efficientnet(x1,x2,x3)
        reduction_keys = ['reduction_3', 'reduction_4', 'reduction_5']
        liv = tuple(liv_all[key] for key in reduction_keys)
        spl = tuple(spl_all[key] for key in reduction_keys)
        kid = tuple(kid_all[key] for key in reduction_keys)

        if self.fpn_type == 'label_concat':
            fpn_layer = self.fpn(liv, spl, kid)
            output_liv = self.classifier(fpn_layer['liv'])
            output_spl = self.classifier(fpn_layer['spl'])
            output_kid = self.classifier(fpn_layer['kid'])

        elif self.fpn_type == 'feature_concat':
            fpn_layer = self.fpn(liv, spl, kid)
            output_liv = fpn_layer['liv']
            output_spl = fpn_layer['spl']
            output_kid = fpn_layer['kid']

        elif self.fpn_type == 'split':
            # 3,4,5,6 layer mutiple output
            fpn_layer = self.fpn(liv, spl, kid)
            output_liv = fpn_layer['liv']
            output_spl = fpn_layer['spl']
            output_kid = fpn_layer['kid']
            

        return output_liv, output_spl, output_kid

class FPN(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(FPN, self).__init__()
        self.output_channels = output_channels
        self.conv1 = nn.Conv3d(input_channels[0], output_channels, kernel_size=1)
        self.conv2 = nn.Conv3d(input_channels[1], output_channels, kernel_size=1)
        self.conv3 = nn.Conv3d(input_channels[2], output_channels, kernel_size=1)
        self.conv4 = nn.Conv3d(input_channels[3], output_channels, kernel_size=1)
        self.upsample = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.smooth = nn.Conv3d(output_channels, output_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        # x is a list of feature maps from efficientnet at different scales
        x2, x3, x4 = x

        # x1 = self.conv1(x1)
        x2 = self.conv2(x2)
        x3 = self.conv3(x3)
        x4 = self.conv4(x4)
        # 上采样到下一层级的尺寸并加和
        
        x4_up = self.upsample(x4)  # 将 x4 上采样到 x3 的尺寸
        x3 = x4_up + x3
        x3_up = self.upsample(x3)  # 将 x3 上采样到 x2 的尺寸
        x2 = x3_up + x2
        # x2_up = self.upsample(x2)  # 将 x2 上采样到 x1 的尺寸
        # x1 = x2_up + x1
        # gradcam 取的convd層
        x4 = self.smooth(x4) 
        x3 = self.smooth(x3)
        x2 = self.smooth(x2)
        # x1 = self.smooth(x1)

        # return x1, x2, x3, x4
        return x2, x3, x4


class EfficientNet_FPN(nn.Module):
    def __init__(self, size, structure_num, class_num, depth_coefficient=1, fpn_type='label_concat', normal=True, in_channels=1):
        super(EfficientNet_FPN, self).__init__()
        self.fpn_type = fpn_type
        self.efficientnet =  EfficientNet3D.from_name(f"efficientnet-{structure_num}", in_channels=in_channels, num_classes=class_num, image_size=size, normal=normal,  depth_coefficient=depth_coefficient)
        self.fpn = FPN(input_channels=[16, 24, 40, 112], output_channels=256)
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.classifier = nn.Linear(256, class_num) # output_channels, class_num
        self.feature_cat_classifier = nn.Linear(256*3, class_num) # output_channels*3, class_num
        self.label_cat_classifier = nn.Linear(class_num*3, class_num)

    def forward(self, x):
        outputs = {}
        x, _ = self.efficientnet.extract_endpoints(x)
        reduction_keys = ['reduction_3', 'reduction_4', 'reduction_5']
        x = tuple(x[key] for key in reduction_keys)
        outputs['x2'], outputs['x3'], outputs['x4'] = self.fpn(x)
        
        if self.fpn_type == 'label_concat':
            for fpn_layer in ['x2', 'x3', 'x4']:
                outputs[fpn_layer] = self.avg_pool(outputs[fpn_layer])
                outputs[fpn_layer] = torch.flatten(outputs[fpn_layer], 1)
                outputs[fpn_layer] = self.classifier(outputs[fpn_layer])
            output = torch.cat((outputs['x2'], outputs['x3'], outputs['x4']), dim=1)
            output = self.label_cat_classifier(output)

        elif self.fpn_type == 'feature_concat':
                pooled_features = []
                for fpn_layer in ['x2', 'x3', 'x4']:
                    outputs[fpn_layer] = self.avg_pool(outputs[fpn_layer])
                    outputs[fpn_layer] = torch.flatten(outputs[fpn_layer], 1)
                    pooled_features.append(outputs[fpn_layer])
                concatenated_features = torch.cat(pooled_features, dim=1)
                output = self.feature_cat_classifier(concatenated_features) 

        return output

class GhostModule(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, ratio=2, dw_kernel_size=3, stride=1, relu=True):
        super(GhostModule, self).__init__()
        self.out_channels = out_channels
        self.init_channels = int(out_channels / ratio)
        self.new_channels = self.init_channels * (ratio - 1)

        # Primary convolution
        self.primary_conv = nn.Sequential(
            nn.Conv3d(in_channels, self.init_channels, kernel_size, stride, kernel_size//2, bias=False),
            nn.BatchNorm3d(self.init_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential()
        )

        # Cheap operation for generating ghost feature maps
        self.cheap_operation = nn.Sequential(
            nn.Conv3d(self.init_channels, self.new_channels, dw_kernel_size, 1, dw_kernel_size//2, groups=self.init_channels, bias=False),
            nn.BatchNorm3d(self.new_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential()
        )

    def forward(self, x):
        primary_features = self.primary_conv(x)
        ghost_features = self.cheap_operation(primary_features)
        out = torch.cat([primary_features, ghost_features], dim=1)
        return out

class UpTransition(nn.Module):
    def __init__(self, inChans, outChans):
        super(UpTransition, self).__init__()
        self.up_conv = nn.ConvTranspose3d(inChans, outChans, kernel_size=2, stride=2)
        self.ops = GhostModule(in_channels=outChans * 2, out_channels=outChans, kernel_size=3)

    def forward(self, x, skip_x, upsample=True):
        if upsample:
            x = self.up_conv(x)
        concat = torch.cat((x, skip_x), 1)
        out = self.ops(concat)
        return out

class UNetEfficientFPN(nn.Module):
    def __init__(self, size, structure_num, class_num, depth_coefficient=1, fpn_type='label_concat', normal=True, in_channels=1):
        super(UNetEfficientFPN, self).__init__()
        self.fpn_type = fpn_type
        self.efficientnet =  EfficientNet3D.from_name(f"efficientnet-{structure_num}", 
            in_channels=in_channels, num_classes=class_num, image_size=size, normal=normal, depth_coefficient=depth_coefficient)

        self.fpn = FPN(input_channels=[16, 24, 40, 112], output_channels=256)

        self.decoder4 = UpTransition(256, 256)
        self.adjust_channels_x4 = nn.Conv3d(112, 256, kernel_size=1)
        self.decoder3 = UpTransition(256, 256)
        self.decoder2 = UpTransition(256, 256)
        self.adjust_channels_x1 = nn.Conv3d(16, 256, kernel_size=1)
        self.decoder1 = UpTransition(256, 256)
        self.adjust_channels_x0 = nn.Conv3d(32, 256, kernel_size=1)
        self.decoder0 = UpTransition(256, 256)

        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.classifier = nn.Linear(256, class_num) # output_channels, class_num
        self.feature_cat_classifier = nn.Linear(256*3, class_num) # output_channels*3, class_num
        self.label_cat_classifier = nn.Linear(class_num*3, class_num)
        self.final_conv = nn.Conv3d(256, class_num, kernel_size=1)
        
    def forward(self, x):
        outputs = {}
        x, _ = self.efficientnet.extract_endpoints(x)

        # FPN skip first layer, so reduction_1 name as x0 
        x0 = x['reduction_1']
        x1 = x['reduction_2']  
        x2 = x['reduction_3']
        x3 = x['reduction_4']
        x4 = x['reduction_5']
 
        # reduction_keys = ['reduction_3', 'reduction_4', 'reduction_5']
        # x = tuple(x[key] for key in reduction_keys)

        fpn_features = self.fpn((x2, x3, x4))
        
        outputs['x2'], outputs['x3'], outputs['x4'] = [x.clone() for x in fpn_features]
        
        # Classification
        if self.fpn_type == 'label_concat':
            for fpn_layer in ['x2', 'x3', 'x4']:
                outputs[fpn_layer] = self.avg_pool(outputs[fpn_layer])
                outputs[fpn_layer] = torch.flatten(outputs[fpn_layer], 1)
                outputs[fpn_layer] = self.classifier(outputs[fpn_layer])
            output = torch.cat((outputs['x2'], outputs['x3'], outputs['x4']), dim=1)
            cls_output = self.label_cat_classifier(output)

        elif self.fpn_type == 'feature_concat':
            pooled_features = []
            for fpn_layer in ['x2', 'x3', 'x4']:
                outputs[fpn_layer] = self.avg_pool(outputs[fpn_layer])
                outputs[fpn_layer] = torch.flatten(outputs[fpn_layer], 1)
                pooled_features.append(outputs[fpn_layer])
            concatenated_features = torch.cat(pooled_features, dim=1)
            cls_output = self.feature_cat_classifier(concatenated_features) 

        # Segmentation
        x4_adjusted = self.adjust_channels_x4(x4)
        x1_adjusted = self.adjust_channels_x1(x1)
        x0_adjusted = self.adjust_channels_x0(x0)

        d4 = self.decoder4(x4_adjusted, fpn_features[2], upsample=False)
        d3 = self.decoder3(d4, fpn_features[1])
        d2 = self.decoder2(d3, fpn_features[0])
        d1 = self.decoder1(d2, x1_adjusted)
        d0 = self.decoder0(d1, x0_adjusted)

        seg_output = self.final_conv(d0)

        return (cls_output, seg_output)

class UNetEfficient(nn.Module):
    def __init__(self, size, structure_num, class_num, depth_coefficient=1, normal=True, in_channels=1):
        super(UNetEfficient, self).__init__()
        self.type = 'normal' if normal else 'abnormal'
        self.efficientnet =  EfficientNet3D.from_name(f"efficientnet-{structure_num}", 
            in_channels=in_channels, num_classes=class_num, image_size=size, normal=normal, depth_coefficient=depth_coefficient)

        # self.fpn = FPN(input_channels=[16, 24, 40, 112], output_channels=256)

        self.adjust_channels_x4 = nn.Conv3d(self.size[0], 256, kernel_size=1)
        self.adjust_channels_x3 = nn.Conv3d(self.size[1], 128, kernel_size=1)
        self.adjust_channels_x2 = nn.Conv3d(self.size[2], 64, kernel_size=1)
        self.adjust_channels_x1 = nn.Conv3d(self.size[3], 32, kernel_size=1)
        self.adjust_channels_x0 = nn.Conv3d(self.size[4], 16, kernel_size=1)

        self.decoder3 = UpTransition(256, 128)  
        self.decoder2 = UpTransition(128, 64)
        self.decoder1 = UpTransition(64, 32)
        self.decoder0 = UpTransition(32, 16)


        self.classifier = nn.Linear(1280, class_num) # output_channels, class_num
        self.final_conv = nn.Conv3d(16, class_num, kernel_size=1)
    @property
    def size(self):
        size_dict = {'normal': [1280, 112, 40, 24, 16], 'abnormal':[112, 40, 24, 16, 32]}
        return size_dict[self.type]

    def forward(self, x):
        outputs = {}
        x, _ = self.efficientnet.extract_endpoints(x)

        # FPN skip first layer, so reduction_1 name as x0 
        x0 = x['reduction_1']
        x1 = x['reduction_2']  
        x2 = x['reduction_3']
        x3 = x['reduction_4']
        x4 = x['reduction_5']
      
        # Classification
        cls_output = self.classifier(x4.mean(dim=[2, 3, 4]))
    
        # Segmentation
        x4_adjusted = self.adjust_channels_x4(x4)  
        x3_adjusted = self.adjust_channels_x3(x3)
        d3 = self.decoder3(x4_adjusted, x3_adjusted)
        x2_adjusted = self.adjust_channels_x2(x2)
        d2 = self.decoder2(d3, x2_adjusted)
        x1_adjusted = self.adjust_channels_x1(x1)
        d1 = self.decoder1(d2, x1_adjusted)
        x0_adjusted = self.adjust_channels_x0(x0)
        d0 = self.decoder0(d1, x0_adjusted)


        # Final segmentation output
        seg_output = self.final_conv(d0)

        return (cls_output, seg_output)