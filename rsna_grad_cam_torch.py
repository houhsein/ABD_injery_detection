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
    test_mul_fpn,
    plot_confusion_matrix,
    confusion_matrix_CI,
    plot_multi_class_roc,
    multi_label_progress
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
    RepeatChanneld
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
    return parser

def data_progress_all(file, dicts, attention_mask = False):
    dicts = []
    dir = "/SSD/TotalSegmentator/rsna_selected_crop_bbox"
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
        row['healthy'] = 0
        if row["liver_healthy"] ==1 and row["spleen_healthy"] ==1 and row["kidney_healthy"] ==1:
            row['healthy']=1
        # Edit organ healthy label
        label = np.array([row["kidney_healthy"],row["kidney_low"],row["kidney_high"],
                          row["liver_healthy"],row["liver_low"],row["liver_high"],
                          row["spleen_healthy"],row["spleen_low"],row["spleen_high"],
                          row['healthy']])
        if attention_mask:
            dicts.append({"image_liv": image_liv, "image_spl": image_spl, 
                        "image_kid_r": image_kid_r, "image_kid_l": image_kid_l, 
                        "mask_liv": mask_liv, "mask_spl": mask_spl,
                        "mask_kid_r": mask_kid_r, "mask_kid_l": mask_kid_l,
                        "label": label})
        else:
            dicts.append({"image_liv": image_liv, "image_spl": image_spl, 
                        "image_kid_r": image_kid_r, "image_kid_l": image_kid_l, 
                        "label": label})

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


parser = get_parser()
args = parser.parse_args()
class_type = args.class_type
if args.file.endswith('ini'):
    cfgpath = f'/tf/jacky831006/ABD_classification/config/{class_type}/{args.file}'
else:
    cfgpath = f'/tf/jacky831006/ABD_classification/config/{class_type}/{args.file}.ini'


conf = configparser.ConfigParser()
conf.read(cfgpath)

# Augmentation
size = eval(conf.get('Augmentation','size'))
num_samples = conf.getint('Augmentation','num_sample')

# Data_setting
architecture = conf.get('Data_Setting','architecture')
if architecture == 'efficientnet':
    structure_num = conf.get('Data_Setting', 'structure_num')
gpu_num = conf.get('Data_Setting','gpu')
seed = conf.getint('Data_Setting','seed')
cross_kfold = conf.getint('Data_Setting','cross_kfold')
# normal_structure = conf.getboolean('Data_Setting','normal_structure')
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
# img_type = conf.get('Data_Setting','img_type')
# loss_type = conf.get('Data_Setting','loss')
fpn_type = conf.get('Data_Setting','fpn_type')
use_amp = conf.getboolean('Data_Setting','use_amp')
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
pos_data = All_data[All_data['any_injury']==1]
neg_data = All_data[All_data['any_injury']==0].sample(n=300, random_state=seed)
All_data = pd.concat([pos_data, neg_data])
no_seg_kid = pd.read_csv("/SSD/rsna-2023/nosegmentation_kid.csv")
no_seg = pd.read_csv("/SSD/rsna-2023/nosegmentation.csv")
All_data = All_data[~All_data['file_paths'].isin(no_seg_kid['file_paths'])]
All_data = All_data[~All_data['file_paths'].isin(no_seg['file_paths'])]

df_all = All_data
if not attention_mask:
    test_transforms = Compose([
            LoadImaged(keys=["image_liv","image_spl","image_kid_r","image_kid_l"]),
            EnsureChannelFirstd(keys=["image_liv","image_spl","image_kid_r","image_kid_l"]),
            ScaleIntensityRanged(
            # keys=["image"], a_min=-57, a_max=164, b_min=0.0, b_max=1.0, clip=True,
                keys=["image_liv","image_spl","image_kid_r","image_kid_l"], a_min=-50, a_max=250, b_min=0.0, b_max=1.0, clip=True,
            ),
            Spacingd(keys=["image_liv","image_spl","image_kid_r","image_kid_l"], pixdim=(1.5, 1.5, 2.0), mode=("bilinear")),
            Orientationd(keys=["image_liv","image_spl","image_kid_r","image_kid_l"], axcodes="RAS"),
            CropForegroundd(keys=["image_liv"], source_key="image_liv"),
            CropForegroundd(keys=["image_spl"], source_key="image_spl"),
            CropForegroundd(keys=["image_kid_r"], source_key="image_kid_r"),
            CropForegroundd(keys=["image_kid_l"], source_key="image_kid_l"),
            Resized(keys=["image_liv","image_spl"], spatial_size = size, mode=("trilinear")),
            Resized(keys=["image_kid_r","image_kid_l"], spatial_size = (size[0],size[1],size[2]//2), mode=("trilinear"))                
        ])
else:
    test_transforms = Compose([
                LoadImaged(keys=["image_liv","image_spl","image_kid_r","image_kid_l","mask_liv","mask_spl","mask_kid_r","mask_kid_l"]),
                EnsureChannelFirstd(keys=["image_liv","image_spl","image_kid_r","image_kid_l","mask_liv","mask_spl","mask_kid_r","mask_kid_l"]),
                ScaleIntensityRanged(
                # keys=["image"], a_min=-57, a_max=164, b_min=0.0, b_max=1.0, clip=True,
                    keys=["image_liv","image_spl","image_kid_r","image_kid_l"], 
                    a_min=-50, a_max=250, b_min=0.0, b_max=1.0, clip=True
                ),
                Spacingd(keys=["image_liv","image_spl","image_kid_r","image_kid_l","mask_liv","mask_spl","mask_kid_r","mask_kid_l"], 
                        pixdim=(1.5, 1.5, 2.0), mode=("bilinear")),
                Orientationd(keys=["image_liv","image_spl","image_kid_r","image_kid_l","mask_liv","mask_spl","mask_kid_r","mask_kid_l"], axcodes="RAS"),
                CropForegroundd(keys=["image_liv","mask_liv"], source_key="image_liv"),
                CropForegroundd(keys=["image_spl","mask_spl"], source_key="image_spl"),
                CropForegroundd(keys=["image_kid_r","mask_kid_r"], source_key="image_kid_r"),
                CropForegroundd(keys=["image_kid_l","mask_kid_l"], source_key="image_kid_l"),
                Resized(keys=["image_liv","image_spl","mask_liv","mask_spl"], spatial_size = size, mode=("trilinear")),
                Resized(keys=["image_kid_r","image_kid_l","mask_kid_r","mask_kid_l"], spatial_size = (size[0],size[1],size[2]//2), mode=("trilinear"))                
            ])


for k in range(len(data_file_name)):
    # cross_validation fold number
    fold = k
    file_name = cfgpath.split("/")[-1][:-4]

    if not args.label:
        file_name = f'{file_name}_predicted'
    if args.test:
        file_name = f'{file_name}_test'

    if len(data_file_name)==1:
        dir_path = f'/tf/jacky831006/ABD_classification/grad_cam_image/{class_type}/{file_name}/'
    else:
        dir_path = f'/tf/jacky831006/ABD_classification/grad_cam_image/{class_type}/{file_name}/{fold}'
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)

    load_weight = f'/tf/jacky831006/ABD_classification/training_checkpoints/{class_type}/{data_file_name[k]}/{data_acc[k]}.pth'

    print(f'Fold:{fold}, file:{data_file_name}, acc:{data_acc}')

    print("Collecting:", datetime.now(), flush=True)

    _, _, test_df = train_valid_test_split(df_all, ratio = data_split_ratio, seed = seed)

    test_df = test_df

    # test_data_dicts = data_progress_all(test_df, 'test_data_dict', class_type)
    
    test_data_dicts = data_progress_all(test_df, 'test_data_dict', attention_mask)

    test_ds = CacheDataset(data=test_data_dicts, transform=test_transforms, cache_rate=1, num_workers=dataloader_num_workers)
    test_data = DataLoader(test_ds, batch_size=testing_batch_size, num_workers=dataloader_num_workers)

    device = torch.device("cuda",0)

if architecture == 'densenet':
        if fpn_type == 'label_concat':
            model = DenseNet3D_FPN.DenseNet3D_FPN(n_input_channels=1, num_init_features=size[0], dropout=0.2, class_num=3, fpn_type=fpn_type)
        elif fpn_type == 'split':
            model = DenseNet3D_FPN.DenseNet3D_FPN(n_input_channels=1, num_init_features=size[0], dropout=0.2, class_num=3, fpn_type=fpn_type)
        elif fpn_type == 'feature_concat':
            model = DenseNet3D_FPN.DenseNet3D_FPN(n_input_channels=1, num_init_features=size[0], dropout=0.2, class_num=3, fpn_type=fpn_type)
    elif architecture == 'efficientnet':
        if fpn_type == 'label_concat':
            # model = EfficientNet3D_BiFPN(size=size, structure_num=structure_num, class_num=3, dropout=0.2, fpn_type=fpn_type)
            model = EfficientNet3D_FPN(size=size, structure_num=structure_num, class_num=3, fpn_type=fpn_type, depth_coefficient=depth_coefficient, normalize=False)
        elif fpn_type == 'split':
            model = EfficientNet3D_BiFPN(size=size, structure_num=structure_num, class_num=3, dropout=0.2, fpn_type=fpn_type, depth_coefficient=depth_coefficient)
        elif fpn_type == 'feature_concat':
            if attention_mask:
                model = EfficientNet3D_FPN(size=size, structure_num=structure_num, class_num=3, fpn_type=fpn_type, depth_coefficient=depth_coefficient, normalize=False, in_channels=2)
            else:
                model = EfficientNet3D_FPN(size=size, structure_num=structure_num, class_num=3, fpn_type=fpn_type, depth_coefficient=depth_coefficient, normalize=False)
            # model = EfficientNet3D_BiFPN(size=size, structure_num=structure_num, class_num=3, dropout=0.2, fpn_type=fpn_type, depth_coefficient=depth_coefficient)
    elif architecture == 'resnet':
        if fpn_type == 'label_concat':
            model = Resnet3D_3_input(size=size, num_classes=3, device=device)
    
    if gpu_num == 'all':
        model = nn.DataParallel(model).to(device)
    else:
        model.to(device)

    model.load_state_dict(torch.load(load_weight))

    # testing predicet
    y_pre = test_mul_fpn(model, test_data, device)
    # y_pre
    # "kidney_healthy","kidney_low","kidney_high,
    # "liver_healthy","liver_low","liver_high",
    # "spleen_healthy","spleen_low","spleen_high",'healthy'

    # # kidney將左右側資料複製並標註region為left, right
    # if class_type == 'kidney':
    #     test_df = kidney_df_progress(test_df)

    # ROC curve figure
    # if class_type == 'all':
    #     y_label = test_df['inj_solid'].values
    # elif class_type == 'liver':
    #     y_label = test_df['liv_inj_tmp'].values
    # elif class_type == 'spleen':
    #     y_label = test_df['spl_inj_tmp'].values
    # elif class_type == 'kidney':
    #     y_label = test_df['kid_inj_tmp'].values
    # if class_type == 'all':
    #     y_label = test_df['any_injury'].values

    # optimal_th = plot_roc(y_pre, y_label, dir_path, f'{file_name}_{fold}')

    # Data distributions
    # pos_list = []
    # neg_list = []
    # for i in zip(y_label, y_pre):
    #     if i[0] == 1:
    #         pos_list.append(i[1][1])
    #     else:
    #         neg_list.append(i[1][1])

    # plot_dis(pos_list, neg_list, dir_path, f'{file_name}_{fold}')
    # if 'cutoff' in locals():
    #     print(f'cutoff value:{cutoff}')
    #     print('Original cutoff')
    # else:
    #     print(f'cutoff value:{optimal_th}')
    index_ranges = {"kid": (0, 3), "liv": (3, 6), "spl": (6, 9)}
    y_pre_n = multi_label_progress(y_pre, index_ranges)

    selected_columns = test_df.iloc[:,6:15]
    values_list = selected_columns.values
    y_list = multi_label_progress(values_list, index_ranges)
    # if class_type == 'all':
    #     y_list = list(test_df.any_injury)
    # elif class_type == 'liver':
    #     y_list = list(test_df.liv_inj_tmp)
    # elif class_type == 'spleen':
    #     y_list = list(test_df.spl_inj_tmp)
    # elif  class_type == 'kidney': 
    #     y_list = list(test_df.kid_inj_tmp)

    # write csv
    test_df['pre_label'] = y_pre_n
    test_df['ori_pre'] = y_pre.tolist()
    test_df = test_df[test_df.columns.drop(list(test_df.filter(regex='Unnamed')))]
    test_df.to_csv(f"{dir_path}/{file_name}_{fold}.csv",index = False)
    test_df_path = f"{dir_path}/{file_name}_{fold}.csv"
    # liver don't have grade (not yet)
    # df_plot(test_df,dir_path,file_name,fold)

    # ROC curve
    y_pre_ori = y_pre.reshape(-1,3,3).tolist()
    pre_index_ranges = {"kid": 0, "liv": 1, "spl": 2}
    optimal_th_list = []
    for cls_type in ['kid','liv','spl']:
        index = pre_index_ranges[cls_type]
        y_list_tmp = [v[index] for v in y_list]
        y_pre_ori_tmp = [v[index] for v in y_pre_ori]
        optimal_th_list_tmp = plot_multi_class_roc(y_pre_ori_tmp, y_list_tmp, 3, cls_type, dir_path, file_name)
        optimal_th_list.append(optimal_th_list_tmp)

    y_pre_modified = multi_label_progress(y_pre, index_ranges, optimal_th_list)
    # confusion matrix
    n_classes = len(pre_index_ranges)
    classes = ['Healthy', 'Low', 'High']
    for cls_type in ['kid','liv','spl']:
        index = pre_index_ranges[cls_type]
        y_list_tmp = [v[index] for v in y_list]
        y_pre_n_tmp = [v[index] for v in y_pre_modified]
        
        cm = confusion_matrix(y_list_tmp, y_pre_n_tmp)
        full_cm = np.zeros((n_classes, n_classes), dtype=int)

        if cm.shape[0] == 1:
            full_cm[0, 0] = cm
        elif cm.shape[0] == 2:
            full_cm[0:2, 0:2] = cm
        else:
            full_cm = cm

        plot_confusion_matrix(full_cm, classes=classes, title=f'{cls_type} Confusion matrix')
        plt.savefig(f"{dir_path}/{file_name}_{cls_type}_{fold}.png")
        plt.close()
        # One-vs-All
        for i in range(n_classes):
            tp = full_cm[i, i]
            fp = full_cm[:, i].sum() - tp
            fn = full_cm[i, :].sum() - tp
            tn = full_cm.sum() - (fp + fn + tp)
            # 取小數點到第二位
            ACC, PPV, NPV, Sensitivity, Specificity = confusion_matrix_CI(tn, fp, fn, tp)
            print(f"Type: {cls_type} {classes[i]}")
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