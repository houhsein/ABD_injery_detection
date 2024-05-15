import torch
import torch.nn as nn
import torch.nn.functional as F
# from torchvision.models.densenet import _DenseBlock, _Transition
# from torchvision.models.densenet import _Transition

class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm3d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv3d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool3d(kernel_size=2, stride=2))


class _DenseLayer(nn.Module):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate, memory_efficient=False):
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm3d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv3d(num_input_features, bn_size * growth_rate, kernel_size=1, stride=1, bias=False)),
        self.add_module('norm2', nn.BatchNorm3d(bn_size * growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv3d(bn_size * growth_rate, growth_rate, kernel_size=3, stride=1, padding=1, bias=False)),
        self.drop_rate = drop_rate
        self.memory_efficient = memory_efficient

    def forward(self, *prev_features):
        bn_function = nn.Sequential(self.norm1, self.relu1, self.conv1, self.norm2, self.relu2, self.conv2)
        if self.memory_efficient and any(prev_feature.requires_grad for prev_feature in prev_features):
            bottleneck_output = cp.checkpoint(bn_function, *prev_features)
        else:
            concated_features = torch.cat(prev_features, 1)
            bottleneck_output = bn_function(concated_features)
        if self.drop_rate > 0:
            bottleneck_output = F.dropout(bottleneck_output, p=self.drop_rate, training=self.training)
        return bottleneck_output


class _DenseBlock(nn.ModuleDict):
    _version = 2

    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate, memory_efficient=False):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(
                num_input_features + i * growth_rate,
                growth_rate,
                bn_size,
                drop_rate,
                memory_efficient
            )
            self.add_module('denselayer%d' % (i + 1), layer)

    def forward(self, init_features):
        features = [init_features]
        for name, layer in self.items():
            new_features = layer(*features)
            features.append(new_features)
        return torch.cat(features, 1)
    
    
# class DenseNet3D(nn.Module):
#     def __init__(self, n_input_channels=1, num_init_features=64, growth_rate=32, block_config=(6, 12, 24), bn_size=4, drop_rate=0):
#         super(DenseNet3D, self).__init__()

#         # Initial convolution
#         self.features = nn.Sequential(
#             nn.Conv3d(n_input_channels, num_init_features, kernel_size=7, stride=2, padding=3, bias=False),
#             nn.BatchNorm3d(num_init_features),
#             nn.ReLU(inplace=True),
#             nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
#         )

#         # Each denseblock
#         num_features = num_init_features
#         for i, num_layers in enumerate(block_config):
#             block = _DenseBlock(num_layers=num_layers, num_input_features=num_features,
#                                 bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
#             self.features.add_module('denseblock%d' % (i + 1), block)
#             num_features = num_features + num_layers * growth_rate

#             if i != len(block_config) - 1:
#                 trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
#                 self.features.add_module('transition%d' % (i + 1), trans)
#                 num_features = num_features // 2

#         # Final batch norm
#         self.features.add_module('norm5', nn.BatchNorm3d(num_features))

#     def forward(self, x):
#         features = []

#         x = self.features[0](x)  # 初始卷积
#         x = self.features[1](x)  # 初始批归一化
#         x = self.features[2](x)  # 初始ReLU
#         x = self.features[3](x)  # 初始最大池化

#         for i in range(4, len(self.features), 2):
#             dense_block = self.features[i]
#             transition_layer = self.features[i + 1]

#             x = dense_block(x)
#             features.append(x)  # 保存每个密集块的输出

#             if i + 1 < len(self.features):
#                 x = transition_layer(x)

#         return features

# class FPN3D(nn.Module):
#     def __init__(self, input_channels, output_channels,class_mum,dropout=0.2):
#         super(FPN3D, self).__init__()
#         self.output_channels = output_channels
#         self.conv1 = nn.Conv3d(input_channels[0], output_channels, kernel_size=1)
#         self.conv2 = nn.Conv3d(input_channels[1], output_channels, kernel_size=1)
#         self.conv3 = nn.Conv3d(input_channels[2], output_channels, kernel_size=1)
# #         self.conv4 = nn.Conv3d(input_channels[3], output_channels, kernel_size=1)
#         self.upsample = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        
# #         self.classifier_4 = nn.Sequential(
# #             nn.Linear(128*4*4*2, 1000), 
# #             nn.Linear(1000, 512), 
# #             nn.Dropout(0.2),
# #             nn.Linear(512, class_mum)
            
# #         )
#         self.classifier_3 = nn.Sequential(
#             nn.Linear(128*8*8*4, 1000), 
#             nn.Linear(1000, 512), 
#             nn.Dropout(0.2),
#             nn.Linear(512, class_mum)
            
#         )
        
#         self.classifier_2 = nn.Sequential(
#             nn.Linear(128*16*16*8, 1000), 
#             nn.Linear(1000, 512), 
#             nn.Dropout(0.2),
#             nn.Linear(512, class_mum)
            
#         )
        
#         self.classifier_1 = nn.Sequential(
#             nn.Linear(128*32*32*16, 1000), 
#             nn.Linear(1000, 512), 
#             nn.Dropout(0.2),
#             nn.Linear(512, class_mum)
            
#         )
        
#         self.classifier = nn.Sequential(
#             nn.Linear(class_mum*3, class_mum)
#         )

#     def forward(self, x):
#         # x is a list of feature maps from DenseNet3D at different scales
# #         x1, x2, x3, x4 = x
#         x1, x2, x3 = x

#         x1 = self.conv1(x1)
#         x2 = self.conv2(x2)
#         x3 = self.conv3(x3)
# #         x4 = self.conv4(x4)
        
    
#         # 上采样到下一层级的尺寸并加和
# #         x4_up = self.upsample(x4)  # 将 x4 上采样到 x3 的尺寸
# #         x3 = x4_up + x3
# #         x4 = x4.view(x4.size(0), -1)
# #         x4 = self.classifier_4(x4)

#         x3_up = self.upsample(x3)  # 将 x3 上采样到 x2 的尺寸
#         x2 = x3_up + x2
#         x3 = x3.view(x3.size(0), -1)
#         x3 = self.classifier_3(x3)
        

#         x2_up = self.upsample(x2)  # 将 x2 上采样到 x1 的尺寸
#         x1 = x2_up + x1
        
#         x2 = x2.view(x2.size(0), -1)
#         x2 = self.classifier_2(x2)
        
#         x1 = x1.view(x1.size(0), -1)
#         x1 = self.classifier_1(x1)
        
# #         out = torch.cat((x4, x3, x2, x1), dim=1)
#         out = torch.cat((x3, x2, x1), dim=1)
# #         print(out.shape)
#         out = self.classifier(out)
        
        

#         return out

# class DenseNet3D_FPN(nn.Module):
#     def __init__(self,n_input_channels=4,dropout=0.2,class_mum=7):
#         super(DenseNet3D_FPN, self).__init__()
#         self.densenet3d = DenseNet3D(n_input_channels=4)
#         self.fpn = FPN3D(input_channels=[256, 512, 1024, 1024], output_channels=128,dropout=0.2,class_mum=7)

#     def forward(self, x):
#         features = self.densenet3d(x)  # 这将是一个特征图列表
#         out = self.fpn(features)       # FPN3D 处理特征图列表
#         return out    


class DenseNet3D(nn.Module):
    def __init__(self, n_input_channels=1, num_init_features=64, growth_rate=32, block_config=(6, 12, 24, 16), bn_size=4, drop_rate=0):
        super(DenseNet3D, self).__init__()

        # Initial convolution
        self.features = nn.Sequential(
            nn.Conv3d(n_input_channels, num_init_features, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm3d(num_init_features),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        )

        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers=num_layers, num_input_features=num_features,
                                bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate

            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2

        # Final batch norm
        self.features.add_module('norm5', nn.BatchNorm3d(num_features))

    def forward(self, x1, x2, x3):
        # 对每个输入独立执行相同的操作
        out1 = self.process_input(x1)
        out2 = self.process_input(x2)
        out3 = self.process_input(x3)
        return out1, out2, out3

    def process_input(self, x):
        # 原来的 forward 方法逻辑
        features = []

        x = self.features[0](x)  # 初始卷积
        x = self.features[1](x)  # 初始批归一化
        x = self.features[2](x)  # 初始ReLU
        x = self.features[3](x)  # 初始最大池化

        for i in range(4, len(self.features), 2):
            dense_block = self.features[i]
            transition_layer = self.features[i + 1]

            x = dense_block(x)
            features.append(x)

            if i + 1 < len(self.features):
                x = transition_layer(x)

        return features

class FPN3D(nn.Module):
    def __init__(self, input_channels, output_channels, dropout, class_num):
        super(FPN3D, self).__init__()
        self.output_channels = output_channels
        self.conv1 = nn.Conv3d(input_channels[0], output_channels, kernel_size=1)
        self.conv2 = nn.Conv3d(input_channels[1], output_channels, kernel_size=1)
        self.conv3 = nn.Conv3d(input_channels[2], output_channels, kernel_size=1)
        self.conv4 = nn.Conv3d(input_channels[3], output_channels, kernel_size=1)
        self.upsample = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.smooth = nn.Conv3d(output_channels, output_channels, kernel_size=3, stride=1, padding=1)
        
        self.classifier_4 = nn.Sequential(
            nn.Linear(256*2*2*2, 1000), 
            nn.Linear(1000, 512), 
            nn.Dropout(dropout),
            nn.Linear(512, class_num)   
        )
        self.classifier_3 = nn.Sequential(
            nn.Linear(256*4*4*4, 1000), 
            nn.Linear(1000, 512), 
            nn.Dropout(dropout),
            nn.Linear(512, class_num)   
        )
        self.classifier_2 = nn.Sequential(
            nn.Linear(256*8*8*8, 1000), 
            nn.Linear(1000, 512), 
            nn.Dropout(dropout),
            nn.Linear(512, class_num)  
        )
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
        outputs["out_liv2"], outputs["out_liv3"], outputs["out_liv4"] = self.process_input(x1)
        outputs["out_spl2"], outputs["out_spl3"], outputs["out_spl4"] = self.process_input(x2)
        outputs["out_kid2"], outputs["out_kid3"], outputs["out_kid4"] = self.process_input(x3)
        organ_list = ['liv','spl','kid']
        
        # TODO　各Input的feature整合，可以有其他方法
        # feature_concated_1 = torch.cat((out_liv1, out_spl1, out_kid1), dim=1) 
        # feature_concated_2 = torch.cat((out_liv2, out_spl2, out_kid2), dim=1) 
        # feature_concated_3 = torch.cat((out_liv3, out_spl3, out_kid3), dim=1)
        # feature_concated_4 = torch.cat((out_liv4, out_spl4, out_kid4), dim=1)
        for organ in organ_list:
            for fpn_layer in [2,3,4]:
                key = f'out_{organ}{fpn_layer}'
            # feature_concated_1 = feature_concated_1.view(feature_concated_1.size(0), -1)
                outputs[key] = outputs[key].view(outputs[key].size(0), -1)
            # feature_concated_3 = feature_concated_3.view(feature_concated_3.size(0), -1)
            # feature_concated_4 = feature_concated_4.view(feature_concated_4.size(0), -1)
            # 使用 getattr 動態獲取 classifier 方法
                classifier = getattr(self, f'classifier_{fpn_layer}')
            # 各自全連接分類結果
            # feature_concated_1 = self.classifier_1(feature_concated_1)
                outputs[f'feature_concated_{fpn_layer}'] = classifier(outputs[key])
            # feature_concated_2 = self.classifier_2(feature_concated_2)
            # feature_concated_3 = self.classifier_3(feature_concated_3)
            # feature_concated_4 = self.classifier_4(feature_concated_4)
            outputs[organ] = torch.cat((outputs["feature_concated_2"],outputs["feature_concated_3"],outputs["feature_concated_4"]), dim=1)
        
        #return out
        # return feature_concated_1, feature_concated_2, feature_concated_3, feature_concated_4
        return outputs
        
    def process_input(self, x):
        # x is a list of feature maps from DenseNet3D at different scales
        x1, x2, x3, x4 = x


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
                          
class DenseNet3D_FPN(nn.Module):
    def __init__(self, num_init_features, n_input_channels=4, dropout=0.2, class_num=9, fpn_type='label_concat'):
        super(DenseNet3D_FPN, self).__init__()
        self.fpn_type = fpn_type
        self.densenet3d = DenseNet3D(num_init_features = num_init_features)
        self.fpn = FPN3D(input_channels=[256, 512, 1024, 1024], output_channels=256, dropout=0.2, class_num=class_num)
        # class_num * FPN layer層數
        self.classifier = nn.Linear(class_num*3, class_num)

    def forward(self, x1,x2,x3):
        features, features2, features3 = self.densenet3d(x1,x2,x3)  # 这将是一个特征图列表
        # fpn_layer1, fpn_layer2, fpn_layer3, fpn_layer4  = self.fpn(features, features2, features3)  # FPN3D 处理特征图列表
        fpn_layer = self.fpn(features, features2, features3)
        # TODO concat 有問題，應該可以直接刪除
        if self.fpn_type == 'concat':
            # out_concat = torch.cat((fpn_layer1, fpn_layer2, fpn_layer3, fpn_layer4), dim=1)
            out_concat = torch.cat((fpn_layer2, fpn_layer3, fpn_layer4), dim=1)
            outputs = self.classifier(out_concat)
            return outputs

        elif self.fpn_type == 'label_concat':
            #out_concat = torch.cat((fpn_layer2, fpn_layer3, fpn_layer4), dim=1)
            output_liv = self.classifier(fpn_layer['liv'])
            output_spl = self.classifier(fpn_layer['spl'])
            output_kid = self.classifier(fpn_layer['kid'])
            return output_liv, output_spl, output_kid

        elif self.fpn_type == 'indivudial':
            return fpn_layer2, fpn_layer3, fpn_layer4