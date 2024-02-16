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
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, roc_curve, auc
import cv2
import matplotlib.pyplot as plt
import pandas as pd 
import random
import csv
import nibabel as nib
import matplotlib.pyplot as plt
import sys
# 路徑要根據你的docker路徑來設定
sys.path.append("/tf/data/jacky831006/ABD_classification/model/")
from efficientnet_3d.model_3d import EfficientNet3D
from resnet_3d import resnet_3d
# 此架構參考這篇
# https://github.com/fei-aiart/NAS-Lung
sys.path.append("/tf/data/jacky831006/ABD_classification/model/NAS-Lung/") 
from models.cnn_res import ConvRes
from models.net_sphere import AngleLoss
import utils.config as config
import configparser
import gc
import math
import subprocess
import json
#from utils.training_torch_utils import train, validation, plot_loss_metric
from utils.training_torch_utils import FocalLoss, ImgAggd, AttentionModel, AttentionModel_new
from utils.grad_cam_torch_utils import test, plot_confusion_matrix, plot_roc, plot_dis, zipDir, confusion_matrix_CI

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
    parser.add_argument('-k', '--class_type', help=" The class of data. (liver, kidney, spleen, all) ", type=str)
    parser.add_argument('-f', '--file', help=" The config file name. ", type=str)
    parser.add_argument('-s', '--select', help=" The selection of data file number. ", type=str2num_or_false, default=False)
    parser.add_argument('-c', '--cam_type', help=" The CAM type (LayerCAM(L) or GradCAM(G)). ", type=str)
#    parser.add_argument('-g','--gt', help="  The test file use ground truth (Default is false)", default=False, type=bool)
    parser.add_argument('-l','--label', help="  The Cam map show as label", action='store_const', const=True, default=False)
    parser.add_argument('-t','--test', help="  Gradcam test for selection (10 pos, 10 neg) ", action='store_const', const=True, default=False)
    return parser

def data_progress_all(file, dicts, class_type):
    dicts = []
    for index, row in file.iterrows():
        if row['any_injury']==1:
            dir_label = 'pos'
        else:
            dir_label = 'neg'    
        # outputname = str(row['chartNo']) + str(row['examdate'])
        # outputname = outputname.replace('.0','')
        # 有些資料會補齊examdate，但原先是NA
        # test_image = f'/tf/jacky831006/TotalSegmentator/ABD_selected_crop_bbox/{dir_label}/liv/{outputname}.nii.gz'
        # if not os.path.exists(test_image):
        #     outputname = str(row['chartNo']) + str(np.NaN)

        if class_type=='all':
            label = int(row['any_injury'])
            image = row['file_paths']
            dicts.append({'image':image, 'label':label})
        # elif class_type=='liver':
        #     # label = int(row['liver_inj_no'])
        #     label = 0 if row['liv_inj'] == 0 else 1
        #     image = f'/tf/jacky831006/TotalSegmentator/ABD_selected_crop_bbox/{dir_label}/liv/{outputname}.nii.gz'
        #     dicts.append({'image':image, 'label':label})
        # elif class_type=='spleen':
        #     #label = int(row['spleen_inj_no'])
        #     label = 0 if row['spl_inj'] == 0 else 1
        #     image = f'/tf/jacky831006/TotalSegmentator/ABD_selected_crop_bbox/{dir_label}/spl/{outputname}.nii.gz'
        #     dicts.append({'image':image, 'label':label})
        # elif class_type=='kidney':
        #     # 目前kidney都以單邊受傷為主
        #     # label = int(row['kidney_inj_no'])
        #     if row['kid_inj_lt'] == row['kid_inj_rt'] == 0:
        #         label = 0
        #         image_l = f'/tf/jacky831006/TotalSegmentator/ABD_selected_crop_bbox/{dir_label}/kid/{outputname}_l.nii.gz'
        #         image_r = f'/tf/jacky831006/TotalSegmentator/ABD_selected_crop_bbox/{dir_label}/kid/{outputname}_r.nii.gz'
        #         if os.path.exists(image_r):
        #             dicts.append({'image':image_r, 'label':label})
        #         if os.path.exists(image_l):
        #             dicts.append({'image':image_l, 'label':label})
        #     if row['kid_inj_rt'] != 0:
        #         label = 1
        #         image = f'/tf/jacky831006/TotalSegmentator/ABD_selected_crop_bbox/{dir_label}/kid/{outputname}_r.nii.gz'
        #         if os.path.exists(image):
        #             dicts.append({'image':image, 'label':label})
        #         else:
        #             print(f'Positive: {outputname} is no seg, check it!')
        #     elif row['kid_inj_lt'] != 0:
        #         label = 1
        #         image = f'/tf/jacky831006/TotalSegmentator/ABD_selected_crop_bbox/{dir_label}/kid/{outputname}_l.nii.gz'
        #         if os.path.exists(image):
        #             dicts.append({'image':image, 'label':label})
        #         else:
        #             print(f'Positive: {outputname} is no seg, check it!')
            
    return dicts    

def data_progress_liver(file, dicts, img_type):
        dicts = []
        for index, row in file.iterrows():
            if row['liver_injury'] == 0:
                label_type = 'neg'
                label = 0
            else:
                label_type = 'pos'
                label = 1
                
            if img_type=='whole':
                if label_type == 'neg':
                    image = row['path'].replace('/data/','/tf/')
                else:
                    if type(row['softPath']) != str:
                        image = f"/tf/jacky831006/liver_seg_all/1/{row['chartNo']}/{row['nifti_name']}"
                    else:
                        image = row['source'].replace('/data/','/tf/')
            elif img_type=='cropping_normal':
                #image = f"/tf/jacky831006/TotalSegmentator/liv_neg_crop_no_dilation_fill_0/{row['chartNo']}.nii.gz"
                image = f"/tf/jacky831006/TotalSegmentator/liv_{label_type}_crop_no_dilation/{row['chartNo']}.nii.gz"
            elif img_type=='cropping_convex':
                image = f"/tf/jacky831006/TotalSegmentator/liv_{label_type}_crop_dilation_new/{row['chartNo']}.nii.gz"
            elif img_type=='cropping_dilation':
                image = f"/tf/jacky831006/TotalSegmentator/liv_{label_type}_crop_dilation/{row['chartNo']}.nii.gz"

            bbox = f"/tf/jacky831006/TotalSegmentator/liv_{label_type}_crop_bbox/{row['chartNo']}.nii.gz"

            dicts.append({'image':image, 'bbox':bbox, 'label':label})
        return dicts

def data_progress_kidney(file, dicts, img_type):
        dicts = []
        for index, row in file.iterrows():
            # kidney_injury_new 實際有無外傷
            if row['kidney_injury_new'] == '0':
                label = 0
            else:
                label = 1
            # kidney_injury 資料表的標記，segmentation照此產生
            if row['kidney_injury'] == 0:
                label_type = 'neg'
            else:
                label_type = 'pos'

            outname = str(row['chartNo']) + str(row['examdate'])
            outname = outname.replace('.0','')

            # kindey通常只有一邊受傷，只要有一邊受傷whole image就算positive
            if img_type=='whole':
                image = row['path']
                if label_type == 'neg':
                    label = 0
                else:
                    label = 1
                # if label_type == 'neg':
                #     image = f"/tf/jacky831006/ABD_data/kid_neg_dl/{row['chartNo']}/{row['source'].split('/')[-1]}"
                # else:
                #     if 'storage' in row['source']:
                #         if row['nifti_name']==row['nifti_name']:
                #             image = f"/tf/jacky831006/ABD_data/kid_pos_dl/{row['chartNo']}/{row['nifti_name']}"
                #         else:
                #             image = f"/tf/jacky831006/ABD_data/kid_pos_dl/{row['chartNo']}/venous_phase.nii.gz"
                #     else:
                #         image = row['source'].replace('/data/','/tf/')
            elif img_type=='cropping_normal':
                # 若Right_check為NA，則表示label為Right 
                if row['Right_check'] != row['Right_check']:
                #image = f"/tf/jacky831006/TotalSegmentator/liv_neg_crop_no_dilation_fill_0/{row['chartNo']}.nii.gz"
                    image = f"/tf/jacky831006/TotalSegmentator/kid_{label_type}_crop_no_dilation/{outname}_r.nii.gz"
                else:
                    image = f"/tf/jacky831006/TotalSegmentator/kid_{label_type}_crop_no_dilation/{outname}_l.nii.gz"
                
            elif img_type=='cropping_convex':
                if row['Right_check'] != row['Right_check']:
                #image = f"/tf/jacky831006/TotalSegmentator/liv_neg_crop_no_dilation_fill_0/{row['chartNo']}.nii.gz"
                    image = f"/tf/jacky831006/TotalSegmentator/kid_{label_type}_crop_dilation_new/{outname}_r.nii.gz"
                else:
                    image = f"/tf/jacky831006/TotalSegmentator/kid_{label_type}_crop_dilation_new/{outname}_l.nii.gz"
       
            elif img_type=='cropping_dilation':
                if row['Right_check'] != row['Right_check']:
                #image = f"/tf/jacky831006/TotalSegmentator/liv_neg_crop_no_dilation_fill_0/{row['chartNo']}.nii.gz"
                    image = f"/tf/jacky831006/TotalSegmentator/kid_{label_type}_crop_dilation/{outname}_r.nii.gz"
                else:
                    image = f"/tf/jacky831006/TotalSegmentator/kid_{label_type}_crop_dilation/{outname}_l.nii.gz"

            elif img_type=='bbox':
                if row['Right_check'] != row['Right_check']:
                #image = f"/tf/jacky831006/TotalSegmentator/liv_neg_crop_no_dilation_fill_0/{row['chartNo']}.nii.gz"
                    image = f"/tf/jacky831006/TotalSegmentator/kid_{label_type}_crop_bbox/{outname}_r.nii.gz"
                else:
                    image = f"/tf/jacky831006/TotalSegmentator/kid_{label_type}_crop_bbox/{outname}_l.nii.gz" 
            # Two image in attional mask or two channel        
            if bbox:
                if row['Right_check'] != row['Right_check']:
                #image = f"/tf/jacky831006/TotalSegmentator/liv_neg_crop_no_dilation_fill_0/{row['chartNo']}.nii.gz"
                    bbox_img = f"/tf/jacky831006/TotalSegmentator/kid_{label_type}_crop_bbox/{outname}_r.nii.gz"
                else:
                    bbox_img = f"/tf/jacky831006/TotalSegmentator/kid_{label_type}_crop_bbox/{outname}_l.nii.gz" 
                dicts.append({'image':image, 'bbox':bbox_img, 'label':label})
            else:
                dicts.append({'image':image, 'label':label})
        return dicts

def inj_check(row):
    kid_inj_tmp = 0 if row['kid_inj_rt'] == row['kid_inj_lt'] == 0 else 1
    liv_inj_tmp = 0 if row['liv_inj'] == 0 else 1
    spl_inj_tmp = 0 if row['spl_inj'] == 0 else 1
    return pd.Series([kid_inj_tmp, liv_inj_tmp, spl_inj_tmp])

def convert_date(x):
    if pd.isna(x):  # Check if the value is NaN
        return x  # If it's NaN, return it as-is
    else:
        return pd.to_datetime(int(x), format='%Y%m%d')

def train_valid_test_split(df, ratio=(0.7,0.1,0.2), seed=0, test_fix=None):
    df[['kid_inj_tmp', 'liv_inj_tmp', 'spl_inj_tmp']] = df.apply(inj_check, axis=1)
    # set key for df
    for liver_val in [0, 1]:
        for kidney_val in [0, 1]:
            for spleen_val in [0, 1]:
                condition = (df['liv_inj_tmp'] == liver_val) & (df['kid_inj_tmp'] == kidney_val) & (df['spl_inj_tmp'] == spleen_val)
                group_key = f'liver_{liver_val}_kidney_{kidney_val}_spleen_{spleen_val}'
                df.loc[condition,"group_key"] = group_key
                
    df = df.reset_index()
    if test_fix is None:
        train_df = df.groupby('group_key', group_keys=False).sample(frac=ratio[0],random_state=seed)
        df_sel = df.drop(train_df.index.to_list())
        valid_df = df_sel.groupby('group_key', group_keys=False).sample(frac=ratio[1]/(ratio[1]+ratio[2]),random_state=seed)
        test_df  = df_sel.drop(valid_df.index.to_list())
    else:
        test_df = df[df.year == test_fix]
        df_sel = df[df.year != test_fix]
        train_df = df_sel.groupby('group_key', group_keys=False).sample(frac=ratio[0],random_state=seed)
        valid_df = df_sel.drop(train_df.index.to_list())
    return train_df, valid_df, test_df  

# 因為kidney有兩顆，需要先將kidney依左右側資料複製並標註region為left, right
def kidney_df_progress(df):
    df_new = pd.DataFrame()
    for index, row in df.iterrows():
        if row['inj_solid']==1:
            dir_label = 'pos'
        else:
            dir_label = 'neg'    
        outputname = str(row['chartNo']) + str(row['examdate'])
        outputname = outputname.replace('.0','') 
        test_image = f'/tf/jacky831006/TotalSegmentator/ABD_selected_crop_bbox/{dir_label}/liv/{outputname}.nii.gz'
        if not os.path.exists(test_image):
            outputname = str(row['chartNo']) + str(np.NaN)
        right_path = f"/tf/jacky831006/TotalSegmentator/ABD_selected_crop_bbox/{dir_label}/kid/{outputname}_r.nii.gz"
        left_path = f"/tf/jacky831006/TotalSegmentator/ABD_selected_crop_bbox/{dir_label}/kid/{outputname}_l.nii.gz"
        # 检查kid_inj_tmp是否为0
        if row['kid_inj_tmp'] == 0:
            if os.path.isfile(right_path) and os.path.isfile(left_path):
                # 若两个文件都存在，复制该行，標記left
                new_row_right = row.copy()
                new_row_right['region'] = 'right'
                new_row_left = row.copy()
                new_row_left['region'] = 'left'
                df_add = pd.DataFrame([new_row_right,new_row_left])
                df_new = pd.concat([df_new, df_add], ignore_index=True)
            elif os.path.isfile(right_path):
                new_row_right = pd.DataFrame(row).transpose()
                new_row_right['region'] = 'right'
                df_new = pd.concat([df_new, new_row_right],ignore_index=True)
            elif os.path.isfile(left_path):
                new_row_left = pd.DataFrame(row).transpose()
                new_row_left['region'] = 'left'
                df_new = pd.concat([df_new, new_row_left],ignore_index=True)
        elif row['kid_inj_tmp'] == 1:
            if row['kid_inj_rt'] != 0 and os.path.isfile(right_path):
                new_row_right = pd.DataFrame(row).transpose()
                new_row_right['region'] = 'right'
                df_new = pd.concat([df_new, new_row_right],ignore_index=True)
            elif row['kid_inj_lt'] != 0 and os.path.isfile(left_path):
                new_row_left = pd.DataFrame(row).transpose()
                new_row_left['region'] = 'left'
                df_new = pd.concat([df_new, new_row_left],ignore_index=True)
    return df_new

parser = get_parser()
args = parser.parse_args()
class_type = args.class_type
if args.file.endswith('ini'):
    cfgpath = f'/tf/data/jacky831006/classification_torch/config/{class_type}/{args.file}'
else:
    cfgpath = f'/tf/data/jacky831006/classification_torch/config/{class_type}/{args.file}.ini'


conf = configparser.ConfigParser()
conf.read(cfgpath)

# Augmentation
size = eval(conf.get('Augmentation','size'))

# Data_setting
architecture = conf.get('Data_Setting','architecture')
if architecture == 'efficientnet':
    structure_num = conf.get('Data_Setting', 'structure_num')
gpu_num = conf.getint('Data_Setting','gpu')
seed = conf.getint('Data_Setting','seed')
cross_kfold = conf.getint('Data_Setting','cross_kfold')
normal_structure = conf.getboolean('Data_Setting','normal_structure')
data_split_ratio = eval(conf.get('Data_Setting','data_split_ratio'))
imbalance_data_ratio = conf.getint('Data_Setting','imbalance_data_ratio')
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
img_type = conf.get('Data_Setting','img_type')
loss_type = conf.get('Data_Setting','loss')
bbox = conf.getboolean('Data_Setting','bbox')
attention_mask = conf.getboolean('Data_Setting','attention_mask')
# HU range: ex 0,100
img_hu = eval(conf.get('Data_Setting','img_hu'))

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
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_num)

# Old data
# # Liver data
# liv_pos = pd.read_csv('/tf/jacky831006/dicom2nifti/liv_pos_20220926.csv')
# All_file = pd.read_csv('/tf/jacky831006/dicom2nifti/ABD_venous_all_20221107.csv')
# liv_neg = All_file[All_file.liver_injury == 0][~All_file.path.isna()][All_file.injury_site_abd_liver.isna()].drop_duplicates(subset=['chartNo'],keep=False)
# rm_list= [21724048, 20466159, 20790497, 2275168, 2562827, 2629999,2678201,2912353,3544319,9708664,20321975,20491950,20582178,21188917,21196422,21224579
#             ,21528321,21528512,21602632,21608978,21626301,21636464,21661413,21718028,21721776,21727756]
# liv_neg = liv_neg[~(liv_neg.chartNo.isin(rm_list))]
# liv_pos['liver_injury'] = 1

# # Kidney data
# kidney_pos = pd.read_csv('/tf/jacky831006/ABD_data/kid_pos_20230216_label.csv')
# # 2043008120100412  right 應該破損太嚴重到沒有seg
# kidney_pos = kidney_pos.drop(99)
# # MONAI 1.0版有問題 先刪除確認是此資料有問題
# kidney_pos = kidney_pos[~(kidney_pos.chartNo == 2512539)]

# # 將kid pos 自己判斷都是零的刪除
# kidney_pos = kidney_pos[~((kidney_pos.Right_check == '0') & (kidney_pos.Left_check == '0'))]
# kidney_pos['kidney_injury'] = 1
# #kidney_neg = pd.read_csv('/tf/jacky831006/dicom2nifti/kid_neg_20230202.csv')
# kidney_neg = pd.read_csv('/tf/jacky831006/ABD_data/kid_neg_20230221.csv')
# # kidney_neg have one same chartno but different examdate data pass it 
# kidney_neg = kidney_neg.drop_duplicates(subset=['chartNo'],keep='first')
# kidney_neg['Right_check'] = '0'
# kidney_neg['Left_check'] = '0'


# All_data = pd.read_csv('/tf/jacky831006/ABD_classification/ABD_venous_all_20230709_for_label_new.csv')
# #　rm_list = [21410269,3687455,21816625,21410022]
# rm_list = [21410269,21816625,21410022]
# All_data = All_data[~All_data.chartNo.isin(rm_list)]
# All_data.loc[:, ['kid_inj_lt','kid_inj_rt','liv_inj','spl_inj']] = All_data.loc[:, ['kid_inj_lt','kid_inj_rt','liv_inj','spl_inj']].fillna(0)
# All_data = All_data.dropna(subset=['label'])
# All_data = All_data[All_data.label != 'exclude']
# # 將2016年的資料取出當test
# All_data['TRDx_ER_arrival_time_tmp'] = pd.to_datetime(All_data['TRDx_ER_arrival_time'])
# All_data['TRDx_ER_arrival_time_tmp'] = All_data['TRDx_ER_arrival_time_tmp'].dt.strftime('%Y%m%d')
# All_data['examdate_tmp'] = All_data['examdate'].apply(convert_date)
# All_data['examdate_tmp'] = All_data['examdate_tmp'].fillna(All_data['TRDx_ER_arrival_time_tmp'])
# All_data['year'] = All_data['examdate_tmp'].dt.year
rsna_data = pd.read_csv('/tf/data/jacky831006/rsna-2023/rsna_train_new.csv')
rsna_label = pd.read_csv('/tf/SSD/rsna-2023/train.csv')

rsna_all = pd.merge(rsna_data.loc[:,['chartNo','file_paths']],rsna_label,left_on='chartNo',right_on='patient_id')
rsna_all['file_paths'] = rsna_all['file_paths'].apply(lambda x: '/tf' + str(x))

df_all = rsna_all
test_df = rsna_all

if bbox:
    test_transforms = Compose([
                LoadImaged(keys=["image", "bbox"]),
                EnsureChannelFirstd(keys=["image", "bbox"]),
                ImgAggd(keys=["image","bbox"], Hu_range=img_hu),
                    ScaleIntensityRanged(
                # keys=["image"], a_min=-57, a_max=164, b_min=0.0, b_max=1.0, clip=True,
                    keys=["image"], a_min=-50, a_max=250, b_min=0.0, b_max=1.0, clip=True,
                ),
                Spacingd(keys=["image"], pixdim=(1.5, 1.5, 2.0), mode=("bilinear")),
                Orientationd(keys=["image"], axcodes="RAS"),
                CropForegroundd(keys=["image"], source_key="image"),
                Resized(keys=['image'], spatial_size = size, mode=("trilinear"))
                
            ])
elif attention_mask:
    test_transforms = Compose([
                LoadImaged(keys=["image", "bbox"]),
                EnsureChannelFirstd(keys=["image", "bbox"]),
                ImgAggd(keys=["image","bbox"], Hu_range=img_hu, Concat=False),
                 ScaleIntensityRanged(
                # keys=["image"], a_min=-57, a_max=164, b_min=0.0, b_max=1.0, clip=True,
                    keys=["image","bbox"], a_min=-50, a_max=250, b_min=0.0, b_max=1.0, clip=True,
                ),
                Spacingd(keys=["image","bbox"], pixdim=(1.5, 1.5, 2.0), mode="bilinear"),
                Orientationd(keys=["image","bbox"], axcodes="RAS"),
                CropForegroundd(keys=["image","bbox"], source_key="image"),
                Resized(keys=['image',"bbox"], spatial_size = size, mode="trilinear")
               
            ])
else:
    test_transforms = Compose([
                LoadImaged(keys=["image"]),
                EnsureChannelFirstd(keys=["image"]),
                ScaleIntensityRanged(
                # keys=["image"], a_min=-57, a_max=164, b_min=0.0, b_max=1.0, clip=True,
                    keys=["image"], a_min=-50, a_max=250, b_min=0.0, b_max=1.0, clip=True,
                ),
                Spacingd(keys=["image"], pixdim=(1.5, 1.5, 2.0), mode=("bilinear")),
                Orientationd(keys=["image"], axcodes="RAS"),
                CropForegroundd(keys=["image"], source_key="image"),
                Resized(keys=['image'], spatial_size = size, mode=("trilinear"))
                
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
        dir_path = f'/tf/data/jacky831006/ABD_classification/grad_cam_image/{class_type}/{file_name}/'
    else:
        dir_path = f'/tf/data/jacky831006/ABD_classification/grad_cam_image/{class_type}/{file_name}/{fold}'
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)

    load_weight = f'/tf/data/jacky831006/classification_torch/training_checkpoints/{class_type}/{data_file_name[k]}/{data_acc[k]}.pth'

    print(f'Fold:{fold}, file:{data_file_name}, acc:{data_acc}')

    print("Collecting:", datetime.now(), flush=True)

    # _, _, test_df = train_valid_test_split(df_all, ratio = data_split_ratio, seed = seed, test_fix = 2016)

    # test_data_dicts = data_progress_all(test_df, 'test_data_dict', class_type)
    
    test_data_dicts = data_progress_all(test_df, 'test_data_dict', class_type)

    test_ds = CacheDataset(data=test_data_dicts, transform=test_transforms, cache_rate=1, num_workers=dataloader_num_workers)
    test_data = DataLoader(test_ds, batch_size=testing_batch_size, num_workers=dataloader_num_workers)

    device = torch.device("cuda",0)

    if architecture == 'densenet':
        if normal_structure:
            # Normal DenseNet121
            if bbox:
                model = densenet.densenet121(spatial_dims=3, in_channels=2, out_channels=2).to(device)
            else:
                model = densenet.densenet121(spatial_dims=3, in_channels=1, out_channels=2).to(device)
        else:
            if bbox:
            # Delete last dense block
                model = densenet.DenseNet(spatial_dims=3, in_channels=2, out_channels=2, block_config=(6, 12, 40)).to(device)
            else:
                model = densenet.DenseNet(spatial_dims=3, in_channels=1, out_channels=2, block_config=(6, 12, 40)).to(device)
    
    elif architecture == 'resnet':
        if bbox:
            model = resnet_3d.generate_model(101,normal=normal_structure,n_input_channels=2).to(device)
        else:
            model = resnet_3d.generate_model(101,normal=normal_structure).to(device)

    elif architecture == 'efficientnet':
        if bbox:
            model = EfficientNet3D.from_name(f"efficientnet-{structure_num}", in_channels=2, num_classes=2, image_size=size, normal=normal_structure).to(device)
        else:
            model = EfficientNet3D.from_name(f"efficientnet-{structure_num}", in_channels=1, num_classes=2, image_size=size, normal=normal_structure).to(device)

    elif architecture == 'CBAM':
        if size[0] == size[1] == size[2]:
            if bbox:
                model = ConvRes(size[0], [[64, 64, 64], [128, 128, 256], [256, 256, 256, 512]], input_channel=2, normal=normal_structure).to(device)
            else:
                model = ConvRes(size[0], [[64, 64, 64], [128, 128, 256], [256, 256, 256, 512]], normal=normal_structure).to(device)
        else:
            raise RuntimeError("CBAM model need same size in x,y,z axis")

    # Mask attention block in CNN
    if attention_mask:
        dense = densenet.DenseNet(spatial_dims=3, in_channels=1, out_channels=2, block_config=(6, 12, 20)).to(device)
        model = AttentionModel_new(2, size, model, dense, architecture).to(device)
        #model = AttentionModel(2, size, model, architecture).to(device)
    
    model.load_state_dict(torch.load(load_weight))

    # testing predicet
    y_pre = test(model, test_data, device)

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
    if class_type == 'all':
        y_label = test_df['any_injury'].values
        

    optimal_th = plot_roc(y_pre, y_label, dir_path, f'{file_name}_{fold}')

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

    # Select cutoff value by roc curve
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

    if class_type == 'all':
        y_list = list(test_df.any_injury)
    # elif class_type == 'liver':
    #     y_list = list(test_df.liv_inj_tmp)
    # elif class_type == 'spleen':
    #     y_list = list(test_df.spl_inj_tmp)
    # elif  class_type == 'kidney':
    #     y_list = list(test_df.kid_inj_tmp)

    # write csv
    test_df['pre_label']=np.array(y_pre_n)
    test_df['ori_pre']=list(y_pre)
    test_df = test_df[test_df.columns.drop(list(test_df.filter(regex='Unnamed')))]
    test_df.to_csv(f"{dir_path}/{file_name}_{fold}.csv",index = False)
    test_df_path = f"{dir_path}/{file_name}_{fold}.csv"
    # liver don't have grade (not yet)
    # df_plot(test_df,dir_path,file_name,fold)
    
    # confusion matrix
    result = confusion_matrix(y_list, y_pre_n)
    (tn, fp, fn, tp)=confusion_matrix(y_list, y_pre_n).ravel()
    plot_confusion_matrix(result, classes=[0, 1], title='Confusion matrix')
    plt.savefig(f"{dir_path}/{file_name}_{fold}.png")
    plt.close()
    #plt.show()
    # 取小數點到第二位
    ACC, PPV, NPV, Sensitivity, Specificity = confusion_matrix_CI(tn, fp, fn, tp)
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