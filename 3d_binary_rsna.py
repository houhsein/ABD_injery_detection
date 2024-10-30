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
# from torch.utils.tensorboard import SummaryWriter
# from torchsummary import summary
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
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
from efficientnet_3d.Efficient3D_BIFPN import EfficientNet_FPN, UNetEfficientFPN, UNetEfficient
from resnet_3d import resnet_3d
from resnet_3d.resnet_3d_new import resnet101, ResNetWithClassifier
from SuPreM.model.Universal_model import SwinUNETRClassifier
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
from utils.training_torch_utils import train, validation, train_seg, validation_seg, plot_loss_metric, FocalLoss, CombinedLoss, ImgAggd, AttentionModel, AttentionModel_new, Dulicated_new
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

def get_parser():
    parser = argparse.ArgumentParser(description='Abdomen injury classification & segmentation')
    parser.add_argument('-f', '--file', help=" The config file name. ", type=str)
    parser.add_argument('-c', '--class_type', help=" The class of data. (liver, kidney, spleen, all) ", type=str)
    parser.add_argument('-lb', '--label_type', help=" The label of data. (binary, multilabel, multiclass) ", type=str)
    parser.add_argument('-s', '--seg', help=" The model, which includes segmentation or not. ", action='store_true')
    parser.add_argument('-t', '--test', help=" Verify using fifty data samples", action='store_true')
    return parser
# 對應不同資料有不同路徑與處理
# 目前以data_progress_all為主
# 依照不同 class_type 來區分資料
# 目前都是以 bbox 為主
# 不同器官的mask又有區分
# whole : 整張影像
# cropping_normal : Totalsegmatator出來的結果，沒有做其他處理
# cropping_convex : 對Totalsegmatator出來的結果進行dilation與convex處理
# cropping_dilation : 對Totalsegmatator出來的結果進行dilation
# bbox : 對Totalsegmatator出來的結果進行dilation並且轉換成bounding box

def get_label(row, class_type, label_type):
    if label_type == 'binary':
        return 0 if row[f'{class_type}_healthy'] == 1 else 1
    else:
        return np.array([row[f"{class_type}_healthy"], row[f"{class_type}_low"], row[f"{class_type}_high"]])

def data_progress_all(file, dicts, class_type, label_type, image_type, attention_mask=False, seg=False):
    dicts = []
    # Seg: Using Segmentation with whole image
    # attention_mask: Using attentional mask by concate in channel 
    if seg:
        # Crop z axis by abd organ
        dir = '/Data/TotalSegmentator/rsna_select_z_whole_image'
        mask_dir = "/Data/TotalSegmentator/rsna_gaussian_mask"
    else:
        if image_type == 'bbox':
            dir = "/SSD/TotalSegmentator/rsna_selected_crop_bbox"
        elif image_type == 'cropping_convex':
            dir = "/Data/TotalSegmentator/rsna_selected_crop_dilation_new"
        elif image_type == 'gaussian_filter_channel_connected':
            dir = "/Data/TotalSegmentator/rsna_selected_crop_gaussian_channel"
        mask_dir = "/SSD/rsna-2023/train_images_new"

    for index, row in file.iterrows():
        # dirs = os.path.dirname(row['file_paths'])
        output = os.path.basename(row['file_paths'])[:-7]
        ID = str(row['patient_id'])
        Slice_ID = row['file_paths'].split('/')[-2]
        
        image_seg =  os.path.join(dir,output)+".nii.gz" # Seg image all class in one dir
        label = get_label(row, class_type, label_type)

        if class_type=='liver':
            image_liv = os.path.join(dir,"liv",output)+".nii.gz"
            if attention_mask:
                mask = os.path.join(mask_dir,ID,Slice_ID,"liver.nii.gz")
                dicts.append({'image':image_liv, 'mask':mask, 'label':label})
            elif seg:
                mask = os.path.join(mask_dir,"liv",output)+".nii.gz"
                dicts.append({'image':image_seg, 'seg':mask, 'label':label})
            else:
                dicts.append({'image':image_liv, 'label':label})
        elif class_type=='spleen':
            image_spl = os.path.join(dir,"spl",output)+".nii.gz"
            if attention_mask:
                mask = os.path.join(mask_dir,ID,Slice_ID,"spleen.nii.gz")
                dicts.append({'image':image_spl, 'mask':mask, 'label':label})
            elif seg:
                mask = os.path.join(mask_dir,"spl",output)+".nii.gz"
                dicts.append({'image':image_seg, 'seg':mask, 'label':label})
            else:
                dicts.append({'image':image_spl, 'label':label})
        elif class_type=='kidney':
            # 目前kidney都以單邊受傷為主
            # Negative資料可能會沒有kidney 要做判斷
            # label = int(row['kidney_inj_no'])
            image_kid_r = os.path.join(dir,"kid",output)+"_r.nii.gz"
            image_kid_l = os.path.join(dir,"kid",output)+"_l.nii.gz"
            if attention_mask:
                mask_r = os.path.join(mask_dir,ID,Slice_ID,"kidney_right.nii.gz")
                mask_l = os.path.join(mask_dir,ID,Slice_ID,"kidney_left.nii.gz")
                dicts.append({'image_r':image_kid_r,'image_l':image_kid_l,'mask_r':mask_r,'mask_l':mask_l,'label':label})
            elif seg:
                mask_r = os.path.join(mask_dir,"kid",output)+"_r.nii.gz"
                mask_l = os.path.join(mask_dir,"kid",output)+"_l.nii.gz"
                dicts.append({'image':image_seg, 'seg_r':mask_r, 'seg_l':mask_l, 'label':label})
            else:
                dicts.append({'image_r':image_kid_r,'image_l':image_kid_l,'label':label})
            
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
                    # image = row['path'].replace('/data/','/tf/').replace('phase_check','ABD_data').replace('classification_torch','ABD_data').replace('classification_negative_data','spleen_negative_data')
                    # if 'ABD_data' not in image:
                    #     image = image.replace('/jacky831006/','/jacky831006/ABD_data/')
                    image = row['path'].replace('/data/jacky831006','/Data/')    
                else:
                    if type(row['softPath']) != str:
                        image = f"/Data/ABD_data/liver_seg_all/1/{row['chartNo']}/{row['nifti_name']}"
                    else:
                        image = row['source'].replace('/data/','/tf/').replace('phase_check','ABD_data')
                        print(f'Error {image}')
                        if 'ABD_data' not in image:
                            image = image.replace('/jacky831006/','/jacky831006/ABD_data/')       
            elif img_type=='cropping_normal':
                #image = f"/tf/jacky831006/TotalSegmentator/liv_neg_crop_no_dilation_fill_0/{row['chartNo']}.nii.gz"
                image = f"/Data/TotalSegmentator/liv_{label_type}_crop_no_dilation/{row['chartNo']}.nii.gz"
            elif img_type=='cropping_convex':
                image = f"/Data/TotalSegmentator/liv_{label_type}_crop_dilation_new/{row['chartNo']}.nii.gz"
            elif img_type=='cropping_dilation':
                image = f"/Data/TotalSegmentator/liv_{label_type}_crop_dilation/{row['chartNo']}.nii.gz"
            elif img_type=='bbox':
                image = f"/Data/TotalSegmentator/liv_{label_type}_crop_bbox/{row['chartNo']}.nii.gz" 
            
            # Two image in attional mask or two channel        
            if bbox:
                bbox_img = f"/Data/TotalSegmentator/liv_{label_type}_crop_bbox/{row['chartNo']}.nii.gz"
                dicts.append({'image':image, 'bbox':bbox_img, 'label':label})
            else:
                dicts.append({'image':image, 'label':label})
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
                    image = f"/Data/TotalSegmentator/kid_{label_type}_crop_no_dilation/{outname}_r.nii.gz"
                else:
                    image = f"/Data/TotalSegmentator/kid_{label_type}_crop_no_dilation/{outname}_l.nii.gz"
                
            elif img_type=='cropping_convex':
                if row['Right_check'] != row['Right_check']:
                #image = f"/tf/jacky831006/TotalSegmentator/liv_neg_crop_no_dilation_fill_0/{row['chartNo']}.nii.gz"
                    image = f"/Data/TotalSegmentator/kid_{label_type}_crop_dilation_new/{outname}_r.nii.gz"
                else:
                    image = f"/Data/TotalSegmentator/kid_{label_type}_crop_dilation_new/{outname}_l.nii.gz"
       
            elif img_type=='cropping_dilation':
                if row['Right_check'] != row['Right_check']:
                #image = f"/tf/jacky831006/TotalSegmentator/liv_neg_crop_no_dilation_fill_0/{row['chartNo']}.nii.gz"
                    image = f"/Data/TotalSegmentator/kid_{label_type}_crop_dilation/{outname}_r.nii.gz"
                else:
                    image = f"/Data/TotalSegmentator/kid_{label_type}_crop_dilation/{outname}_l.nii.gz"

            elif img_type=='bbox':
                if row['Right_check'] != row['Right_check']:
                #image = f"/tf/jacky831006/TotalSegmentator/liv_neg_crop_no_dilation_fill_0/{row['chartNo']}.nii.gz"
                    image = f"/Data/TotalSegmentator/kid_{label_type}_crop_bbox/{outname}_r.nii.gz"
                else:
                    image = f"/Data/TotalSegmentator/kid_{label_type}_crop_bbox/{outname}_l.nii.gz" 
            # Two image in attional mask or two channel        
            if bbox:
                if row['Right_check'] != row['Right_check']:
                #image = f"/tf/jacky831006/TotalSegmentator/liv_neg_crop_no_dilation_fill_0/{row['chartNo']}.nii.gz"
                    bbox_img = f"/Data/TotalSegmentator/kid_{label_type}_crop_bbox/{outname}_r.nii.gz"
                else:
                    bbox_img = f"/Data/TotalSegmentator/kid_{label_type}_crop_bbox/{outname}_l.nii.gz" 
                dicts.append({'image':image, 'bbox':bbox_img, 'label':label})
            else:
                dicts.append({'image':image, 'label':label})
        return dicts


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
    if test_fix:
        valid_df = df.drop(train_df.index.to_list())

        return train_df, valid_df
    else:
        df_sel = df.drop(train_df.index.to_list())
        valid_df = df_sel.groupby("group_key", group_keys=False).sample(
                frac=(ratio[1]/(ratio[2]+ratio[1])), random_state=seed)
        test_df = df_sel.drop(valid_df.index.to_list())
    
        return train_df, valid_df, test_df

def get_transforms(keys, size, prob, sigma_range, magnitude_range, translate_range, rotate_range, scale_range, valid=False):
    # Seg: Using Segmentation with whole image
    # mask: Using attentional mask by concate in channel
    # only kidney have right, left side
    if 'image_r' in keys and 'image_l' in keys:
        # right, left side concate in z axis, so z axis size divide by 2
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
    elif 'seg_r' in keys:
        CropForegroundd_list = [
        CropForegroundd(keys=['image','seg_r','seg_l'], source_key='image')
        ]
    elif 'seg' in keys:
        CropForegroundd_list = [
        CropForegroundd(keys=['image','seg'], source_key='image')
        ]
    else:
        CropForegroundd_list = [CropForegroundd(keys=[key], source_key=key) for key in keys]
        
    common_transforms = [
            LoadImaged(keys=keys),
            EnsureChannelFirstd(keys=keys),
            # ImgAggd(keys=["image","bbox"], Hu_range=img_hu),
            ScaleIntensityRanged(
                # keys=["image"], a_min=-57, a_max=164, b_min=0.0, b_max=1.0, clip=True,
                # keys=other_key, a_min=-50, a_max=250, b_min=0.0, b_max=1.0, clip=True,
                keys=other_key, a_min=-50, a_max=250, b_min=0.0, b_max=255.0, clip=True,
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

# 進行完整一次預測
def run_once(times=0):
    # reset config parameter
    config.initialize()

    if test_fix:
        global test_df
        train_df, valid_df = train_valid_test_split(df_all, ratio = (0.8, 0.2), seed = seed, test_fix = True)
    else:
        train_df, valid_df, test_df = train_valid_test_split(df_all, ratio = data_split_ratio, seed = seed)

    train_data_dicts = data_progress_all(train_df, 'train_data_dict', class_type, label_type, image_type, attention_mask, seg)
    valid_data_dicts = data_progress_all(valid_df, 'valid_data_dict', class_type, label_type, image_type, attention_mask, seg)
    test_data_dicts  = data_progress_all(test_df, 'test_data_dict',   class_type, label_type, image_type, attention_mask, seg)
    #with open('/tf/jacky831006/ABD_data/train.pickle', 'wb') as f:
    #    pickle.dump(train_data_dicts, f)

    set_determinism(seed=0)
    train_ds = CacheDataset(data=train_data_dicts, transform=train_transforms, cache_rate=1, num_workers=dataloader_num_workers)
    train_loader = DataLoader(train_ds, batch_size=traning_batch_size, shuffle=True, num_workers=dataloader_num_workers)
    valid_ds = CacheDataset(data=valid_data_dicts, transform=valid_transforms, cache_rate=1, num_workers=dataloader_num_workers)
    val_loader = DataLoader(valid_ds, batch_size=valid_batch_size, num_workers=dataloader_num_workers)

    device = torch.device("cuda",0)

    # Model setting
    # normal_structure :
    # True 為沒有修改原始架構(深度較深，最後的影像解析度較低)
    # False 則為修改原始架構(深度較淺，最後的影像解析度較高)
    # bbox 則代表input除了原始的影像外，還加入bounding box影像藉由channel增維
    if label_type == 'binary':
        label_num = 2
    elif label_type == 'multicalss':
        label_num = 3
    elif label_type == 'multicalss':
        label_num = 9 # (kid,spl,liv) X (_healty,_low,_high)

    if attention_mask:
        input_channel = 2
    else:
        input_channel = 1

    if architecture == 'densenet':
        if normal_structure:
            # Normal DenseNet121
            model = densenet.densenet121(spatial_dims=3, in_channels=input_channel, out_channels=label_num).to(device)
        else:
            model = densenet.DenseNet(spatial_dims=3, in_channels=input_channel, out_channels=label_num, block_config=(6, 12, 40)).to(device)
           
    elif architecture == 'resnet':
        # pretrain resnet input z, x, y need to translate
        base_model = resnet101(sample_input_W=size[0], sample_input_H=size[1], sample_input_D=size[2], num_input=input_channel, num_seg_classes=label_num, shortcut_type='B').to(device)
        # model = resnet_3d.generate_model(101,normal=normal_structure,n_classes=label_num).to(device)
        model = ResNetWithClassifier(base_model, num_classes=label_num).to(device)
        load_weight_path = '/tf/jacky831006/ABD_classification/pretrain_weight/resnet_101.pth'
        net_dict = model.state_dict()
        pretrain = torch.load(load_weight_path)
        pretrain_dict = {new_key: v for k, v in pretrain['state_dict'].items() if (new_key := k.replace('module.', 'base_model.')) in net_dict.keys()}
        net_dict.update(pretrain_dict)
        model.load_state_dict(net_dict)

    elif architecture == 'efficientnet':
        model = EfficientNet3D.from_name(f"efficientnet-{structure_num}", in_channels=input_channel, num_classes=label_num, image_size=size, normal=normal_structure).to(device)
    
    elif architecture == 'CBAM':
        if size[0] == size[1] == size[2]:
            model = ConvRes(size[0], [[64, 64, 64], [128, 128, 256], [256, 256, 256, 512]], input_channel=input_channel, num_classes=label_num ,normal=normal_structure).to(device)
        else:
            raise RuntimeError("CBAM model need same size in x,y,z axis")
    
    elif architecture == 'efficientnet_fpn':
        model = EfficientNet_FPN(size=size, structure_num=structure_num, class_num=label_num, fpn_type=fpn_type, in_channels=input_channel).to(device)
    
    elif architecture == 'swintransformer':
        # pretrain swintransformer input z, x, y need to translate
        model = SwinUNETRClassifier(img_size=size,
            in_channels=input_channel,
            out_channels=label_num,
            encoding='word_embedding'
            ).to(device)
        #Load pre-trained weights
        checkpoint_path = '/tf/jacky831006/ABD_classification/pretrain_weight/supervised_suprem_swinunetr_2100.pth'
        net_dict = model.state_dict()
        checkpoint = torch.load(checkpoint_path)
        load_dict = checkpoint['net']

        pretrain_dict = {new_key: v for k, v in checkpoint['net'].items() if (new_key := k.replace('module.', '')) in net_dict.keys() and "organ_embedding" not in new_key}
        net_dict.update(pretrain_dict)
        model.load_state_dict(net_dict)
    elif seg and architecture == 'efficientnet_fpn_unet':
        model = UNetEfficientFPN(size=size, structure_num=structure_num, class_num=label_num, fpn_type=fpn_type, normal=False).to(device)
    
    elif seg and architecture == 'efficientnet_unet':
        model = UNetEfficient(size=size, structure_num=structure_num, class_num=label_num, normal=normal_structure).to(device)
    # Mask attention block in CNN
    # Old version
    # if attention_mask:
    #     dense = densenet.DenseNet(spatial_dims=3, in_channels=1, out_channels=2, block_config=(6, 12, 20)).to(device)
    #     model = AttentionModel_new(2, size, model, dense, architecture).to(device)
        #model = AttentionModel(2, size, model, architecture).to(device)
    
    # Imbalance loss
    # 根據不同資料的比例做推測
    # 目前kidney都以單邊受傷為主

    if class_type=='liver':
        #grouped = df_all.groupby('liver_inj_no')
        grouped = df_all.groupby('liver_healthy')
    elif class_type=='spleen':
        #grouped = df_all.groupby('spleen_inj_no')
        grouped = df_all.groupby('spleen_healthy')
    elif class_type=='kidney': 
        #grouped = df_all.groupby('kidney_inj_no')
        grouped = df_all.groupby('kidney_healthy')

    # 根據資料比例來給予不同weight
    group_dict = {0: 0, 1: 0}
    for name, group in grouped:
        # 只有kidney的negative情況會需要考慮兩側
        # if class_type=='kidney' and name == 0:
        #     pass
        # else:
        group_dict[name] = len(group) * num_samples

    if label_type == 'binary':
        # weights = torch.tensor([1/group_dict[1], 1/group_dict[0]]).to(device)
        weights = torch.tensor([1/(group_dict[1] + 1e-6), 1/(group_dict[0] + 1e-6)]).to(device)
    elif label_type ==  'multiclass':
        class_weights = [1.0, 2.0, 4.0]
        weights = torch.FloatTensor(class_weights).to(device)
    elif label_type == 'multilabel':
        class_weights = [1.0, 2.0, 4.0, 1.0, 2.0, 4.0, 1.0, 2.0, 4.0]
        weights = torch.FloatTensor(class_weights).to(device)
    print(f'\npos:{group_dict[0]}, neg:{group_dict[1]}')
    # CBAM有自己的loss function
    if architecture == 'CBAM' and not normal_structure:
        loss_function = AngleLoss(weight=weights)
    elif loss_type == 'crossentropy':
        loss_function = torch.nn.CrossEntropyLoss(weight=weights)
    elif loss_type == 'focalloss':
        if label_type == 'binary' or label_type == 'multiclass':
            loss_function = FocalLoss(gamma=2, alpha=0.25, use_softmax=True, weight=weights)
        elif label_type == 'multilabel':
            loss_function = FocalLoss(gamma=2, alpha=0.25, use_softmax=False, weight=weights)
    elif loss_type == 'focal & amse':
        loss_function = CombinedLoss(alpha=0.5, beta=0.5, seg='amse', cls='focal', weight=weights, label_type=label_type)
    elif loss_type == 'ce & amse':
        loss_function = CombinedLoss(alpha=0.5, beta=0.5, seg='amse', cls='ce', weight=weights, label_type=label_type)
    # Grid search
    if len(init_lr) == 1:
        optimizer = torch.optim.Adam(model.parameters(), init_lr[0])
    else:
        optimizer = torch.optim.Adam(model.parameters(), init_lr[times])
        
    if lr_decay_epoch == 0:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=lr_decay_rate, patience=epochs, verbose =True)
        #scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=epochs,gamma=lr_decay_rate, verbose=True )
    else:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=lr_decay_rate, patience=lr_decay_epoch, verbose =True)

    now = datetime.utcnow().strftime("%Y%m%d%H%M%S")               
    root_logdir = f"/tf/jacky831006/ABD_classification/tfboard/{class_type}/{label_type}"     
    # logdir = "{}/run-{}/".format(root_logdir, now) 

    # tfboard file path
    # 創一個主目錄 之後在train內的sumamaryWriter都會以主目錄創下面路徑
    # writer = SummaryWriter(logdir)
    # if not os.path.isdir(logdir):
    #     os.makedirs(logdir)
    # check_point path
    check_path = f'/tf/jacky831006/ABD_classification/training_checkpoints/{class_type}/{label_type}/{now}'
    if not os.path.isdir(check_path):
        os.makedirs(check_path)
    print(f'\n Weight location:{check_path}',flush = True)
    if cross_kfold == 1:
        print(f'\n Processing begining',flush = True)
    else:
        print(f'\n Processing fold #{times}',flush = True)


    data_num = len(train_ds)
    #test_model = train(model, device, data_num, epochs, optimizer, loss_function, train_loader, \
    #                    val_loader, early_stop, init_lr, lr_decay_rate, lr_decay_epoch, check_path)
    
    if seg:
        test_model = train_seg(model, device, data_num, epochs, optimizer, loss_function, train_loader, \
                        val_loader, early_stop, scheduler, check_path)
    else:
        test_model = train(model, device, data_num, epochs, optimizer, loss_function, train_loader, \
                        val_loader, early_stop, scheduler, check_path)
                    
    # plot train loss and metric 
    plot_loss_metric(config.epoch_loss_values, config.metric_values, check_path)
    # remove dataloader to free memory
    del train_ds
    del train_loader
    del valid_ds
    del val_loader
    gc.collect()

    # Avoid ram out of memory
    test_ds = CacheDataset(data=test_data_dicts, transform=valid_transforms, cache_rate=1, num_workers=dataloader_num_workers)
    test_loader = DataLoader(test_ds, batch_size=testing_batch_size, num_workers=dataloader_num_workers)
    # validation is same as testing
    print(f'\nBest accuracy:{config.best_metric}')
    if config.best_metric != 0:
        load_weight = f'{check_path}/{config.best_metric}.pth'
        model.load_state_dict(torch.load(load_weight))

    # record paramter
    accuracy_list.append(config.best_metric)
    file_list.append(now)
    epoch_list.append(config.best_metric_epoch)

    if seg:
        test_acc = validation_seg(model, test_loader, device)
    else:
        test_acc = validation(model, test_loader, device)

    test_accuracy_list.append(test_acc)
    del test_ds
    del test_loader
    gc.collect()

    print(f'\n Best f1 score:{config.best_metric}, Best test f1 score:{test_acc}')


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    class_type = args.class_type
    label_type = args.label_type
    seg = args.seg
    test_check = args.test
    # 讀檔路徑，之後可自己微調
    if args.file.endswith('ini'):
        cfgpath = f'/tf/jacky831006/ABD_classification/config/{class_type}/{label_type}/{args.file}'
    else:
        cfgpath = f'/tf/jacky831006/ABD_classification/config/{class_type}/{label_type}/{args.file}.ini'
    

    conf = configparser.ConfigParser()
    conf.read(cfgpath)

    # Augmentation
    num_samples = conf.getint('Augmentation','num_sample')
    size = eval(conf.get('Augmentation','size'))
    prob = conf.getfloat('Rand3DElasticd','prob')
    sigma_range = eval(conf.get('Rand3DElasticd','sigma_range'))
    magnitude_range = eval(conf.get('Rand3DElasticd','magnitude_range'))
    translate_range = eval(conf.get('Rand3DElasticd','translate_range'))
    rotate_range = eval(conf.get('Rand3DElasticd','rotate_range'))
    scale_range = eval(conf.get('Rand3DElasticd','scale_range'))

    # Data_setting
    architecture = conf.get('Data_Setting','architecture')
    if 'efficientnet' in architecture:
        structure_num = conf.get('Data_Setting', 'structure_num')
    if 'efficientnet_fpn' in architecture :
        fpn_type = conf.get('Data_Setting', 'fpn_type')
    gpu_num = conf.getint('Data_Setting','gpu')
    seed = conf.getint('Data_Setting','seed')
    cross_kfold = conf.getint('Data_Setting','cross_kfold')
    normal_structure = conf.getboolean('Data_Setting','normal_structure')
    data_split_ratio = eval(conf.get('Data_Setting','data_split_ratio'))
    # imbalance_data_ratio = conf.getint('Data_Setting','imbalance_data_ratio')
    epochs = conf.getint('Data_Setting','epochs')
    # early_stop = 0 means None
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
    # cropping_convex, bbox
    image_type = conf.get('Data_Setting','image_type', fallback=None)
    loss_type = conf.get('Data_Setting','loss')
    # bbox = conf.getboolean('Data_Setting','bbox')
    attention_mask = conf.getboolean('Data_Setting','attention_mask', fallback=False)
    test_fix = conf.getboolean('Data_Setting','test_fix')
    # HU range: ex 0,100
    # img_hu = eval(conf.get('Data_Setting','img_hu'))

    # Setting cuda environment
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_num)
    # Old data
    # liver_pos = pd.read_csv('/tf/jacky831006/dicom2nifti/liv_pos_20220926.csv')
    # All_file = pd.read_csv('/tf/jacky831006/dicom2nifti/ABD_venous_all_20221107.csv')
    # liver_neg = All_file[All_file.liver_injury == 0][~All_file.path.isna()][All_file.injury_site_abd_liver.isna()].drop_duplicates(subset=['chartNo'],keep=False)
    # # 沒有腹部影像
    # rm_list= [21724048, 20466159, 20790497, 2275168, 2562827, 2629999,2678201,2912353,3544319,9708664,20321975,20491950,20582178,21188917,21196422,21224579
    #             ,21528321,21528512,21602632,21608978,21626301,21636464,21661413,21718028,21721776,21727756]
    # liver_neg = liver_neg[~(liver_neg.chartNo.isin(rm_list))]
    # liver_pos['liver_injury'] = 1

    # #kidney_pos = pd.read_csv('/tf/jacky831006/dicom2nifti/kid_pos_20220919.csv')
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
    # Data progressing
    All_data = pd.read_csv("/SSD/rsna-2023/rsna_train_new_v2.csv")
    test_df = pd.read_csv('/tf/jacky831006/ABD_classification/rsna_test_20240531.csv')
    pos_data = All_data[All_data['any_injury']==1]
    # neg_data = All_data[All_data['any_injury']==0].sample(n=len(pos_data), random_state=seed)
    neg_data = All_data[All_data['any_injury']==0]
    neg_data = neg_data.sample(n=int(len(neg_data)*0.5), random_state=seed)
    All_data = pd.concat([pos_data, neg_data])
    no_seg_kid = pd.read_csv("/SSD/rsna-2023/nosegmentation_kid.csv")
    no_seg = pd.read_csv("/SSD/rsna-2023/nosegmentation.csv")
    All_data = All_data[~All_data['file_paths'].isin(no_seg_kid['file_paths'])]
    All_data = All_data[~All_data['file_paths'].isin(no_seg['file_paths'])]

    if test_fix:
        All_data = All_data[~All_data.file_paths.isin(test_df.file_paths.values)]

    if test_check:
        df_all = All_data[:50]
    else:
        df_all = All_data
    
    if attention_mask:
        if class_type =='kidney':
            keys = ["image_r","image_l","mask_r","mask_l"]
        else:
            keys = ["image","mask"]
    elif seg:
        if class_type =='kidney':
            keys = ["image","seg_r","seg_l"]
        else:
            keys = ["image","seg"]
    else:
        if class_type =='kidney':
            keys = ["image_r","image_l"]
        else:
            keys = ["image"]
        
    train_transforms = get_transforms(keys, size, prob, sigma_range, magnitude_range, 
                                translate_range, rotate_range, scale_range)
    valid_transforms = get_transforms(keys, size, prob, sigma_range, magnitude_range, 
                                translate_range, rotate_range, scale_range, valid=True)

    # Training by cross validation
    accuracy_list = []
    test_accuracy_list = []
    file_list = []
    epoch_list = []

    # if cross_kfold*data_split_ratio[2] != 1 and cross_kfold!=1:
    #     raise RuntimeError("Kfold number is not match test data ratio")

    first_start_time = time.time()
    # kfold 
    if cross_kfold != 1:
        for k in range(cross_kfold):
            run_once(k)
    # grid search
    elif len(init_lr) != 1:
        for k in range(len(init_lr)):
            run_once(k)
    else:
        run_once()
    
    if cross_kfold != 1:
        print(f'\n Mean accuracy:{sum(accuracy_list)/len(accuracy_list)}')

    final_end_time = time.time()
    hours, rem = divmod(final_end_time-first_start_time, 3600)
    minutes, seconds = divmod(rem, 60)
    all_time = "All time:{:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds)
    print(all_time)
    # write some output information in ori ini
    conf['Data output'] = {}
    conf['Data output']['Running time'] = all_time
    conf['Data output']['Data file name'] = str(file_list)
    # ini write in type need str type
    conf['Data output']['Best accuracy'] = str(accuracy_list)
    conf['Data output']['Best Test accuracy'] = str(test_accuracy_list)
    conf['Data output']['Best epoch'] = str(epoch_list)

    with open(cfgpath, 'w') as f:
        conf.write(f)