import os
import sys
import numpy as np
import argparse
import pprint
import pdb
import time
from datetime import datetime
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.utils.prune as prune
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import pandas as pd 
import cv2
import csv
import nibabel as nib
import matplotlib.pyplot as plt
import sys
# 路徑要根據你的docker路徑來設定
sys.path.append("/tf/jacky831006/ABD_classification/model/")
from efficientnet_3d.model_3d import EfficientNet3D
from efficientnet_3d.Efficient3D_BIFPN import EfficientNet3D_BiFPN
from resnet_3d import resnet_3d
from resnet_3d.resnet_3d_new import resnet101, ResNetWithClassifier
from SuPreM.model.Universal_model import SwinUNETRClassifier
from DenseNet3D_FPN import DenseNet3D_FPN
# 此架構參考這篇
# https://github.com/fei-aiart/NAS-Lung
sys.path.append("/tf/jacky831006/ABD_classification/model/NAS-Lung/") 
from models.cnn_res import ConvRes
from models.net_sphere import AngleLoss

import utils.config as config
import configparser
import gc
import math
import json
from utils.training_torch_utils import(
    train_mul_fpn, 
    valid_mul_fpn, 
    plot_loss_metric
)
from utils.grad_cam_torch_utils import(
    test,
    plot_confusion_matrix,
    plot_multi_class_roc,
    plot_roc,
    multi_label_progress,
    plot_dis,
    confusion_matrix_CI
)
import pickle
# Data augmnetation module (based on MONAI)
from monai.networks.nets import UNet, densenet, SENet, ViT
from monai.apps import download_and_extract
from monai.data import CacheDataset, DataLoader, Dataset
from monai.config import print_config
from monai.utils import first, set_determinism
from monai.transforms import (
    LoadImage,
    AsDiscrete,
    AsDiscreted,
    EnsureChannelFirstd,
    Compose,
    CropForegroundd,
    Rand3DElasticd,
    LoadImaged,
    Orientationd,
    RandCropByPosNegLabeld,
    ScaleIntensityRanged,
    Spacingd,
    Invertd,
    FillHoles,
    Resized,
    RepeatChanneld,
    HistogramNormalized
)
import functools
# let all of print can be flush = ture
print = functools.partial(print, flush=True)

def str2num_or_false(v):
    if v.isdigit():
        return int(v)
    else:
        return False

def get_parser():
    parser = argparse.ArgumentParser(description='liver classification')
    parser.add_argument('-k', '--class_type', help=" The class of data. (liver, kidney, spleen, rsna) ", type=str)
    parser.add_argument('-f', '--file', help=" The config file name. ", type=str)
    parser.add_argument('-s', '--select', help=" The selection of data file number. ", type=str2num_or_false, default=False)
    parser.add_argument('-c', '--cam_type', help=" The CAM type (LayerCAM(L) or GradCAM(G)). ", type=str)
#    parser.add_argument('-g','--gt', help="  The test file use ground truth (Default is false)", default=False, type=bool)
    parser.add_argument('-l','--label', help="  The Cam map show as label", action='store_const', const=True, default=False)
    parser.add_argument('-t','--test', help="  Gradcam test for selection (10 pos, 10 neg) ", action='store_const', const=True, default=False)
    parser.add_argument('-lb', '--label_type', help=" The label of data. (binary or multiple) ", type=str)
    return parser

def data_progress_all(file, dicts, class_type, label_type, image_type, attention_mask=False):
    dicts = []
    if image_type=='bbox':
        dir = "/SSD/TotalSegmentator/rsna_selected_crop_bbox"
    elif image_type=='cropping_convex':
        dir = "/Data/TotalSegmentator/rsna_selected_crop_dilation_new"
    elif image_type == 'gaussian_filter_channel_connected':
        dir = "/Data/TotalSegmentator/rsna_selected_crop_gaussian_channel"
    
    mask_dir = "/SSD/rsna-2023/train_images_new"
    for index, row in file.iterrows():
        # dirs = os.path.dirname(row['file_paths'])
        output = os.path.basename(row['file_paths'])[:-7]
        image_liv = os.path.join(dir,"liv",output)+".nii.gz"
        image_spl = os.path.join(dir,"spl",output)+".nii.gz"
        image_kid_r = os.path.join(dir,"kid",output)+"_r.nii.gz"
        image_kid_l = os.path.join(dir,"kid",output)+"_l.nii.gz"
        ID = str(row['patient_id'])
        Slice_ID = row['file_paths'].split('/')[-2]
        mask_liv = os.path.join(mask_dir,ID,Slice_ID,"liver.nii.gz")
        mask_spl = os.path.join(mask_dir,ID,Slice_ID,"spleen.nii.gz")
        mask_kid_r = os.path.join(mask_dir,ID,Slice_ID,"kidney_right.nii.gz")
        mask_kid_l = os.path.join(mask_dir,ID,Slice_ID,"kidney_left.nii.gz")

        if class_type=='liver':
            if label_type == 'binary':
                label = 0 if row['liver_healthy'] == 1 else 1
            else:
                label = np.array([row["liver_healthy"],row["liver_low"],row["liver_high"]])
            if attention_mask:
                dicts.append({'image':image_liv, 'mask':mask_liv, 'label':label})
            else:
                dicts.append({'image':image_liv, 'label':label})
        elif class_type=='spleen':
            if label_type == 'binary':
                label = 0 if row['spleen_healthy'] == 1 else 1
            else:
                label = np.array([row["spleen_healthy"],row["spleen_low"],row["spleen_high"]])
            if attention_mask:
                dicts.append({'image':image_spl, 'mask':mask_spl, 'label':label})
            else:
                dicts.append({'image':image_spl, 'label':label})
        elif class_type=='kidney':
            # 目前kidney都以單邊受傷為主
            # Negative資料可能會沒有kidney 要做判斷
            # label = int(row['kidney_inj_no'])
            if label_type == 'binary':
                label = 0 if row['kidney_healthy'] == 1 else 1
            else:
                label = np.array([row["kidney_healthy"],row["kidney_low"],row["kidney_high"]])
            if attention_mask:
                dicts.append({'image_r':image_kid_r,'image_l':image_kid_l,'mask_r':mask_kid_r,'mask_l':mask_kid_l,'label':label})
            else:
                dicts.append({'image_r':image_kid_r,'image_l':image_kid_l,'label':label})
            
    return dicts    

def convert_date(x):
    if pd.isna(x):  # Check if the value is NaN
        return x  # If it's NaN, return it as-is
    else:
        return pd.to_datetime(int(x), format='%Y%m%d')
# 將positive進行複製
def duplicate(df, col_name, num_sample, pos_sel=True):
    if pos_sel:
        df_inj_tmp = df[df[col_name] == 1]
    else:
        df_inj_tmp = df

    # 進行重複
    df_inj_tmp_duplicated = pd.concat([df_inj_tmp]*num_sample, ignore_index=True)

    # 將原始的df和複製後的df結合
    df_new = pd.concat([df, df_inj_tmp_duplicated], ignore_index=True)

    return df_new

# 依據positive情況進行資料切分
def train_valid_test_split(df, ratio=(0.7, 0.1, 0.2), seed=0, test_fix=None):

    df['group_key'] = df.apply(
    lambda row: (
        f"{row['liver_low']}_"
        f"{row['liver_high']}_"
        f"{row['spleen_low']}_"
        f"{row['spleen_high']}_"
        f"{row['kidney_low']}_"
        f"{row['kidney_high']}_"
        f"{row['any_injury']}"
    ),
    axis=1)
    
    df = df.reset_index()
    
    train_df = df.groupby("group_key", group_keys=False).sample(
            frac=ratio[0], random_state=seed
        )
    df_sel = df.drop(train_df.index.to_list())
    valid_df = df_sel.groupby("group_key", group_keys=False).sample(
            frac=(ratio[1]/(ratio[2]+ratio[1])), random_state=seed)
    test_df = df_sel.drop(valid_df.index.to_list())
    
    return train_df, valid_df, test_df

def label_trans(test_df, class_type, label_type):
    if label_type == 'binary':
        y_label = test_df[f'{class_type}_healthy'].replace({0: 1, 1: 0}).values
    else:
        y_label = test_df.loc[:,[f'{class_type}_healthy',f'{class_type}_low',f'{class_type}_high']].values
    return y_label

def get_transforms(keys, size, prob, sigma_range, magnitude_range, translate_range, rotate_range, scale_range, valid=False):
    if 'image_r' in keys and 'image_l' in keys:
        other_key = ["image_r","image_l"]
        size = size[0],size[1],size[2]//2
    else:
        other_key = ["image"]

    if 'mask_r' in keys:
        CropForegroundd_list = [
        CropForegroundd(keys=['image_r','mask_r'], source_key='image_r'),
        CropForegroundd(keys=['image_l','mask_l'], source_key='image_l')
        ]
    elif 'mask' in keys:
        CropForegroundd_list = [
        CropForegroundd(keys=['image','mask'], source_key='image')
        ]
    else:
        CropForegroundd_list = [CropForegroundd(keys=[key], source_key=key) for key in keys]
        
    common_transforms = [
            LoadImaged(keys=keys),
            EnsureChannelFirstd(keys=keys),
            # ImgAggd(keys=["image","bbox"], Hu_range=img_hu),
            ScaleIntensityRanged(
                #keys=["image"], a_min=-57, a_max=164, b_min=0.0, b_max=1.0, clip=True,
                keys=other_key, a_min=-50, a_max=250, b_min=0.0, b_max=1.0, clip=True,
            ),
            #Dulicated_new(keys=["image"], num_samples=num_samples, pos_sel=True),
            #RepeatChanneld(keys=["image","label"], repeats = num_sample),
            HistogramNormalized(keys=other_key, num_bins=64, min=0, max=1.0),
            Spacingd(keys=keys, pixdim=(1.5, 1.5, 2.0), mode=("bilinear")),
            Orientationd(keys=keys, axcodes="RAS"),
            *CropForegroundd_list, # Expand the list here
            Resized(keys=keys, spatial_size = size, mode=("trilinear"))
        ]
    augmentation_transforms = [
            Rand3DElasticd(
                    keys=keys,
                    mode=("bilinear"),
                    prob=prob,
                    sigma_range=sigma_range,
                    magnitude_range=magnitude_range,
                    spatial_size=size,
                    translate_range=translate_range,
                    rotate_range=rotate_range,
                    scale_range=scale_range,
                    padding_mode="border")
        ]
    if valid:
        return Compose(common_transforms)
    else:
        return Compose(common_transforms + augmentation_transforms)

parser = get_parser()
args = parser.parse_args()
class_type = args.class_type
label_type = args.label_type
if args.file.endswith('ini'):
    cfgpath = f'/tf/jacky831006/ABD_classification/config/{class_type}/{label_type}/{args.file}'
else:
    cfgpath = f'/tf/jacky831006/ABD_classification/config/{class_type}/{label_type}/{args.file}.ini'


conf = configparser.ConfigParser()
conf.read(cfgpath)

# Augmentation
size = eval(conf.get('Augmentation','size'))
num_samples = conf.getint('Augmentation','num_sample')
prob = conf.getfloat('Rand3DElasticd','prob')
sigma_range = eval(conf.get('Rand3DElasticd','sigma_range'))
magnitude_range = eval(conf.get('Rand3DElasticd','magnitude_range'))
translate_range = eval(conf.get('Rand3DElasticd','translate_range'))
rotate_range = eval(conf.get('Rand3DElasticd','rotate_range'))
scale_range = eval(conf.get('Rand3DElasticd','scale_range'))

# Data_setting
architecture = conf.get('Data_Setting','architecture')
if architecture == 'efficientnet':
    structure_num = conf.get('Data_Setting', 'structure_num')
gpu_num = conf.get('Data_Setting','gpu')
seed = conf.getint('Data_Setting','seed')
cross_kfold = conf.getint('Data_Setting','cross_kfold')
normal_structure = conf.getboolean('Data_Setting','normal_structure')
data_split_ratio = eval(conf.get('Data_Setting','data_split_ratio'))
# imbalance_data_ratio = conf.getint('Data_Setting','imbalance_data_ratio')
epochs = conf.getint('Data_Setting','epochs')
early_stop = conf.getint('Data_Setting','early_stop')
traning_batch_size = conf.getint('Data_Setting','traning_batch_size')
valid_batch_size = conf.getint('Data_Setting','valid_batch_size')
testing_batch_size = conf.getint('Data_Setting','testing_batch_size')
dataloader_num_workers = conf.getint('Data_Setting','dataloader_num_workers')
#init_lr = conf.getfloat('Data_Setting','init_lr')
init_lr = json.loads(conf.get('Data_Setting','init_lr'))
#optimizer = conf.get('Data_Setting','optimizer')
lr_decay_rate = conf.getfloat('Data_Setting','lr_decay_rate')
lr_decay_epoch = conf.getint('Data_Setting','lr_decay_epoch')
# whole, cropping_normal, cropping_convex, cropping_dilation
image_type = conf.get('Data_Setting','image_type')
# loss_type = conf.get('Data_Setting','loss')
# fpn_type = conf.get('Data_Setting','fpn_type')
# use_amp = conf.getboolean('Data_Setting','use_amp')
attention_mask = conf.getboolean('Data_Setting','attention_mask')
# bbox = conf.getboolean('Data_Setting','bbox')
# HU range: ex 0,100
# img_hu = eval(conf.get('Data_Setting','img_hu'))

# Data output
data_file_name = eval(conf.get('Data output','data file name'))
data_acc = eval(conf.get('Data output','best accuracy'))
if conf.has_option('Data output', 'cutoff'):
    cutoff = conf.getfloat('Data output','cutoff')

if args.select is not False:
    data_file_name = [data_file_name[args.select]]
    data_acc = [data_acc[args.select]]

# set parameter 
grad_cam_only = False

# heatmap_type: detail:每個Z軸各自為一張圖片, one_picture:每筆病人畫成一張圖片, all: 兩種都畫
# cam type: LayerCAM, GradCAM
if args.cam_type not in ['L','G','GradCAM','LayerCAM']:
    raise ValueError("Input error! Only GradCAM(G) and LayerCAM(L) type")
elif args.cam_type == 'L':
    cam_type = 'LayerCAM'
elif args.cam_type == 'G':
    cam_type = 'GradCAM'
else:
    cam_type = args.cam_type

if cam_type=='LayerCAM' and attention_mask:
        raise ValueError("Attention mask cannot be visualized using LayerCAM to present the results.")

# 參數條件
# split_num : 由於grad cam會將GPU的ram塞滿，需要切成小筆資料避免OOM 
heatmap_type = 'all'
input_shape = size
split_num = 16

# Setting cuda environment
if gpu_num != 'all':
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_num

# Data progressing
All_data = pd.read_csv("/SSD/rsna-2023/rsna_train_new_v2.csv")
test_df = pd.read_csv('/tf/jacky831006/ABD_classification/rsna_test_20240531.csv')
pos_data = All_data[All_data['any_injury']==1]
# neg_data = All_data[All_data['any_injury']==0].sample(n=300, random_state=seed)
neg_data = All_data[All_data['any_injury']==0]
neg_data = neg_data.sample(n=int(len(neg_data)*0.5), random_state=seed)
All_data = pd.concat([pos_data, neg_data])
no_seg_kid = pd.read_csv("/SSD/rsna-2023/nosegmentation_kid.csv")
no_seg = pd.read_csv("/SSD/rsna-2023/nosegmentation.csv")
All_data = All_data[~All_data['file_paths'].isin(no_seg_kid['file_paths'])]
All_data = All_data[~All_data['file_paths'].isin(no_seg['file_paths'])]

test_fix = False
if test_fix:
    All_data = All_data[~All_data.file_paths.isin(test_df.file_paths.values)]
df_all = All_data

if attention_mask:
    if class_type =='kidney':
        keys = ["image_r","image_l","mask_r","mask_l"]
    else:
        keys = ["image","mask"] 
else:
    if class_type =='kidney':
        keys = ["image_r","image_l"]
    else:
        keys = ["image"]
    
test_transforms = get_transforms(keys, size, prob, sigma_range, magnitude_range, 
                            translate_range, rotate_range, scale_range, valid=True)

for k in range(len(data_file_name)):
    # cross_validation fold number
    fold = k
    file_name = cfgpath.split("/")[-1][:-4]

    if not args.label:
        file_name = f'{file_name}_predicted'
    if args.test:
        file_name = f'{file_name}_test'

    if len(data_file_name)==1:
        dir_path = f'/tf/jacky831006/ABD_classification/grad_cam_image/{class_type}/{label_type}/{file_name}/'
    else:
        dir_path = f'/tf/jacky831006/ABD_classification/grad_cam_image/{class_type}/{label_type}/{file_name}/{fold}'
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)

    load_weight = f'/tf/jacky831006/ABD_classification/training_checkpoints/{class_type}/{label_type}/{data_file_name[k]}/{data_acc[k]}.pth'

    print(f'Fold:{fold}, file:{data_file_name}, acc:{data_acc}')

    print("Collecting:", datetime.now(), flush=True)

    if test_fix:
        test_df = test_df
    else:
        _, _, test_df = train_valid_test_split(df_all, ratio = data_split_ratio, seed = seed)

    # test_data_dicts = data_progress_all(test_df, 'test_data_dict', class_type)
    
    test_data_dicts  = data_progress_all(test_df, 'test_data_dict',   class_type, label_type, image_type, attention_mask)

    test_ds = CacheDataset(data=test_data_dicts, transform=test_transforms, cache_rate=1, num_workers=dataloader_num_workers)
    test_data = DataLoader(test_ds, batch_size=testing_batch_size, num_workers=dataloader_num_workers)

    device = torch.device("cuda",0)

    if label_type == 'binary':
        label_num = 2
    else:
        label_num = 3

    if architecture == 'densenet':
        if normal_structure:
            # Normal DenseNet121
            if attention_mask:
                model = densenet.densenet121(spatial_dims=3, in_channels=2, out_channels=label_num).to(device)
            else:
                model = densenet.densenet121(spatial_dims=3, in_channels=1, out_channels=label_num).to(device)
        else:
            if attention_mask:
            # Delete last dense block
                model = densenet.DenseNet(spatial_dims=3, in_channels=2, out_channels=label_num, block_config=(6, 12, 40)).to(device)
            else:
                model = densenet.DenseNet(spatial_dims=3, in_channels=1, out_channels=label_num, block_config=(6, 12, 40)).to(device)

    elif architecture == 'resnet':
        if attention_mask:
            model = resnet_3d.generate_model(101,normal=normal_structure,n_input_channels=2,n_classes =label_num).to(device)
        else:
            model = resnet_3d.generate_model(101,normal=normal_structure,n_classes=label_num).to(device)

    elif architecture == 'efficientnet':
        if attention_mask:
            model = EfficientNet3D.from_name(f"efficientnet-{structure_num}", in_channels=2, num_classes=label_num, image_size=size, normal=normal_structure).to(device)
        else:
            model = EfficientNet3D.from_name(f"efficientnet-{structure_num}", in_channels=1, num_classes=label_num, image_size=size, normal=normal_structure).to(device)

    elif architecture == 'CBAM':
        if size[0] == size[1] == size[2]:
            if attention_mask:
                model = ConvRes(size[0], [[64, 64, 64], [128, 128, 256], [256, 256, 256, 512]], input_channel=2, num_classes=label_num ,normal=normal_structure).to(device)
            else:
                model = ConvRes(size[0], [[64, 64, 64], [128, 128, 256], [256, 256, 256, 512]], num_classes=label_num ,normal=normal_structure).to(device)
        else:
            raise RuntimeError("CBAM model need same size in x,y,z axis")
    
    if gpu_num == 'all':
        model = nn.DataParallel(model).to(device)
    else:
        model.to(device)

    model.load_state_dict(torch.load(load_weight))

    # testing predicet
    y_pre = test(model, test_data, device)
    # y_pre
    # "kidney_healthy","kidney_low","kidney_high,
    # "liver_healthy","liver_low","liver_high",
    # "spleen_healthy","spleen_low","spleen_high"
    
    y_label = label_trans(test_df, class_type, label_type)

    # ROC curve figure
    # optimal_th = plot_roc(y_pre, y_label, dir_path, f'{file_name}_{fold}')
    if label_type == 'binary':
        y_label_b = test_df[f'{class_type}_healthy'].replace({0: 1, 1: 0}).values
        optimal_th = plot_roc(y_pre, y_label_b, dir_path, f'{file_name}_{fold}')
    else:
        index_ranges = {"other": (0, 3)}
        y_label_b = multi_label_progress(y_label, index_ranges)
        y_label_b = [item for sublist in y_label_b for item in sublist]
        optimal_th = plot_multi_class_roc(y_pre, y_label_b, label_num, class_type, dir_path, f'{file_name}_{fold}')

    if label_type == 'binary':
        # Data distributions
        pos_list = []
        neg_list = []
        for i in zip(y_label, y_pre):
            if i[0] == 1:
                pos_list.append(i[1][1])
            else:
                neg_list.append(i[1][1])

        plot_dis(pos_list, neg_list, dir_path, f'{file_name}_{fold}')

    if 'cutoff' in locals():
        print(f'cutoff value:{cutoff}')
        print('Original cutoff')
    else:
        print(f'cutoff value:{optimal_th}')
    
    if label_type == 'binary':
        y_pre_n = list()
        for i in range(y_pre.shape[0]):
            if 'cutoff' in locals():
                if y_pre[i][1] < cutoff:
                    y_pre_n.append(0)
                else:
                    y_pre_n.append(1)
            else:
                if y_pre[i][1] < optimal_th:
                    y_pre_n.append(0)
                else:
                    y_pre_n.append(1)
    else:
        y_pre_n = multi_label_progress(y_pre, index_ranges, [optimal_th])
        y_pre_n = [item for sublist in y_pre_n for item in sublist]

    # write csv
    test_df['pre_label'] = y_pre_n
    test_df['ori_pre'] = y_pre.tolist()
    test_df = test_df[test_df.columns.drop(list(test_df.filter(regex='Unnamed')))]
    test_df.to_csv(f"{dir_path}/{file_name}_{fold}.csv",index = False)
    test_df_path = f"{dir_path}/{file_name}_{fold}.csv"
    
    if label_type == 'binary':
        classes = ['Healthy', 'Injury']
    else:
        classes = ['Healthy', 'Low', 'High']

    cm = confusion_matrix(y_label_b, y_pre_n)
    full_cm = np.zeros((label_num, label_num), dtype=int)

    if cm.shape[0] == 1:
        full_cm[0, 0] = cm
    elif cm.shape[0] == 2:
        full_cm[0:2, 0:2] = cm
    else:
        full_cm = cm

    plot_confusion_matrix(full_cm, classes=classes, title=f'{class_type} Confusion matrix')       
    plt.savefig(f"{dir_path}/{file_name}_{class_type}_{fold}.png")
    plt.close()
    if label_type == 'binary':
        #plt.show()
        # 取小數點到第二位
        y_list = y_label_b.tolist()
        (tn, fp, fn, tp) = confusion_matrix(y_list, y_pre_n).ravel()
        ACC, PPV, NPV, Sensitivity, Specificity = confusion_matrix_CI(tn, fp, fn, tp)
        print(f'Modifed Test Accuracy: {ACC}')
        print("PPV:",PPV,"NPV:",NPV,"Sensitivity:",Sensitivity,"Specificity:",Specificity)
    else:
        for i in range(label_num):
            tp = full_cm[i, i]
            fp = full_cm[:, i].sum() - tp
            fn = full_cm[i, :].sum() - tp
            tn = full_cm.sum() - (fp + fn + tp)
            # 取小數點到第二位
            ACC, PPV, NPV, Sensitivity, Specificity = confusion_matrix_CI(tn, fp, fn, tp)
            print(f"Type: {class_type} {classes[i]}")
            print(f'Modifed Test Accuracy: {ACC}')
            print("PPV:",PPV,"NPV:",NPV,"Sensitivity:",Sensitivity,"Specificity:",Specificity)
    
    del test_ds
    del test_data
    gc.collect()

    # # Grad cam (close every times)
    # if args.test:
    #     if class_type == 'all':
    #         test_df = test_df.groupby('inj_solid').apply(lambda x: x.sample(min(len(x), 15))).reset_index(drop=True)
    #     elif class_type == 'liver':
    #         test_df = test_df.groupby('liv_inj_tmp').apply(lambda x: x.sample(min(len(x), 15))).reset_index(drop=True)
    #     elif class_type == 'spleen':
    #         test_df = test_df.groupby('spl_inj_tmp').apply(lambda x: x.sample(min(len(x), 15))).reset_index(drop=True)
    #     elif class_type == 'kidney':
    #         test_df = test_df.groupby('kid_inj_tmp').apply(lambda x: x.sample(min(len(x), 15))).reset_index(drop=True)

    #     test_data_dicts = data_progress_all(test_df, 'test_data_dict', class_type)
        

    # for i in range(math.ceil(len(test_data_dicts)/split_num)):
    #     # if input not str, all need to transfer to str
    #     print(f'--------Fold {i}--------',flush= True)
    #     grad_cam_run = subprocess.run(["python3","/tf/jacky831006/classification_torch/All_structure/ABD_selected_grad_cam_torch_split.py", 
    #                                 "-W", load_weight, "-D", test_df_path, "-C", cfgpath, "-F", str(i), "-T", class_type,
    #                                 "-S", str(split_num), "-H", heatmap_type, "-G", cam_type, "-O", file_name, "-L", str(args.label), "-K" ,str(k),
    #                                 "-TS", str(args.test)], 
    #                                 stdout=subprocess.PIPE, universal_newlines=True)
    #     print(grad_cam_run.stdout)
    # print(f'Fold {fold} is finish!')