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
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, roc_curve, auc
from scipy.ndimage import zoom
import cv2
import matplotlib.pyplot as plt
import pandas as pd 
import random
import csv
import nibabel as nib
import matplotlib.pyplot as plt
import sys
# 路徑要根據你的docker路徑來設定
sys.path.append("/tf/jacky831006/ABD_classification/model/")
from efficientnet_3d.model_3d import EfficientNet3D
from resnet_3d import resnet_3d
# 此架構參考這篇
# https://github.com/fei-aiart/NAS-Lung
sys.path.append("/tf/jacky831006/ABD_classification/model/NAS-Lung/") 
from models.cnn_res import ConvRes
from models.net_sphere import AngleLoss

from tqdm.auto import tqdm
import concurrent.futures
import utils.config as config
import configparser
import gc
import math
import json
from utils.training_torch_utils import FocalLoss, ImgAggd, AttentionModel, AttentionModel_new
from utils.grad_cam_torch_utils import plot_heatmap_detail, plot_heatmap_one_picture, plot_vedio, get_last_conv_name, GradCAM, LayerCAM

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

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('布林值應為：True/False, Yes/No, T/F, Y/N, 1/0')

def get_parser():
    parser = argparse.ArgumentParser(description='ABD classification')
    parser.add_argument('-W', '--input_weight', help=" Input data weight ", type=str)
    parser.add_argument('-D', '--data_file', help=" Data file ", type=str)
    parser.add_argument('-T', '--class_type', help=" Class type(Liver, Kidney) ", type=str)
    parser.add_argument('-C','--config', help=" Config file ", type=str)
    parser.add_argument('-F','--fold', help=" Fold of grad-cam ", type=int)
    parser.add_argument('-S','--split', help=" split number of data ", type=int)
    parser.add_argument('-H','--heatmap_type', help=" heatmap type ", type=str)
    parser.add_argument('-G', '--cam_type', help=" The CAM type (LayerCAM(L) or GradCAM(G)). ", type=str)
    parser.add_argument('-O','--output', help=" output path ", type=str)
    parser.add_argument('-L','--label', help=" CAM map show as label ", type=str)
    parser.add_argument('-K','--kfold', help=" The fold number ", type=str)
    parser.add_argument('-TS','--test', help=" Gradcam test for selection (15 pos, 15 neg) ", type=str2bool)
    
    return parser

def data_progress_all_in_split(file, dicts, class_type):
    # 差別在kidney部分因為df已經把左右側拆開
    dicts = []
    for index, row in file.iterrows():
        if row['inj_solid']==1:
            dir_label = 'pos'
        else:
            dir_label = 'neg'    
        outputname = str(row['chartNo']) + str(row['examdate'])
        outputname = outputname.replace('.0','')
        # 有些資料會補齊examdate，但原先是NA
        test_image = f'/tf/jacky831006/TotalSegmentator/ABD_selected_crop_bbox/{dir_label}/liv/{outputname}.nii.gz'
        if not os.path.exists(test_image):
            outputname = str(row['chartNo']) + str(np.NaN)

        if class_type=='all':
            label = int(row['inj_solid'])
            image = row['path']
            dicts.append({'image':image, 'label':label})
        elif class_type=='liver':
            # label = int(row['liver_inj_no'])
            label = 0 if row['liv_inj'] == 0 else 1
            image = f'/tf/jacky831006/TotalSegmentator/ABD_selected_crop_bbox/{dir_label}/liv/{outputname}.nii.gz'
            dicts.append({'image':image, 'label':label})
        elif class_type=='spleen':
            #label = int(row['spleen_inj_no'])
            label = 0 if row['spl_inj'] == 0 else 1
            image = f'/tf/jacky831006/TotalSegmentator/ABD_selected_crop_bbox/{dir_label}/spl/{outputname}.nii.gz'
            dicts.append({'image':image, 'label':label})
        elif class_type=='kidney':
            # 目前kidney都以單邊受傷為主
            # label = int(row['kidney_inj_no'])
            region = row['region']
            label = 0 if row['kid_inj_tmp'] == 0 else 1
            if region =='right':
                image = f'/tf/jacky831006/TotalSegmentator/ABD_selected_crop_bbox/{dir_label}/kid/{outputname}_r.nii.gz'
            elif region =='left':
                image = f'/tf/jacky831006/TotalSegmentator/ABD_selected_crop_bbox/{dir_label}/kid/{outputname}_l.nii.gz'
            dicts.append({'image':image, 'label':label})
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

            if img_type=='whole':
                image = row['path']
                '''
                if whole_label_type == 'neg':
                    image = f"/tf/jacky831006/ABD_data/kid_neg_dl/{row['chartNo']}/{row['source'].split('/')[-1]}"
                else:
                    if 'storage' in row['source']:
                        if row['nifti_name']==row['nifti_name']:
                            image = f"/tf/jacky831006/ABD_data/kid_pos_dl/{row['chartNo']}/{row['nifti_name']}"
                        else:
                            image = f"/tf/jacky831006/ABD_data/kid_pos_dl/{row['chartNo']}/venous_phase.nii.gz"
                    else:
                        image = row['source'].replace('/data/','/tf/')
                '''
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

            if row['Right_check'] != row['Right_check']:
            #image = f"/tf/jacky831006/TotalSegmentator/liv_neg_crop_no_dilation_fill_0/{row['chartNo']}.nii.gz"
                bbox = f"/tf/jacky831006/TotalSegmentator/kid_{label_type}_crop_bbox/{outname}_r.nii.gz"
            else:
                bbox = f"/tf/jacky831006/TotalSegmentator/kid_{label_type}_crop_bbox/{outname}_l.nii.gz" 
    
            dicts.append({'image':image,'bbox':bbox, 'label':label})
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

def process_plot_detail(j, heatmap_total, image, final_path):
    plot_heatmap_detail(heatmap_total[:,:,j],image[:,:,j],f"{final_path}/{j:03}.png")
    

def process_plot_multiprocess(heatmap_total, image, final_path, num_cores = 10):
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_cores) as executor: #創建線程池
        futures = [executor.submit(process_plot_detail, j, heatmap_total, image, final_path) for j in range(image.shape[-1])] #將函數和相應的參數提交給線程池執行
        for _ in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc='GradCam plot progressing'): # 確認函數迭代已完成的任務並用tqdm進行進度調顯示
            pass
    plot_vedio(final_path)

def CAM_plot(model, test_data, test_df, class_type, output_file_name, size, device, first, detail, cam_type, architecture, Label, attention_mask, kfold):
    channel_first = True
    input_shape = size
    # cropping img output size 固定 128,128,64, whole img output size 固定 300,300,64
    if class_type == 'all':
        output_shape = 300,300,64
    else:
        output_shape = 128,128,64

    # #file_name = "DenseNet_crop_blockless_5_2"
    dir_path = f'/tf/jacky831006/ABD_classification/grad_cam_image/{class_type}/{output_file_name}/{cam_type}_{kfold}'
    if not os.path.isdir(dir_path):
        os.mkdir(dir_path)


    pos_path = f"{dir_path}/POS"
    pos_total = f"{dir_path}/POS_total"
    neg_path = f"{dir_path}/NEG"
    neg_total = f"{dir_path}/NEG_total"
    # LayerCam挑選的層數
    layer_dic = {
        'efficientnet':[],
        'resnet':['layer1.2.conv3','layer2.3.conv3','layer3.25.conv3'],
        'densenet':['features.denseblock1.denselayer6.layers.conv2','features.denseblock2.denselayer12.layers.conv2',
                    'features.denseblock3.denselayer40.layers.conv2'],
        'CBAM':['layers.3.conv3.0','layers.8.conv3.0','layers.14.conv3.0']
        }
    if architecture == 'efficientnet' and cam_type == "LayerCAM":
        inputs = torch.rand(1, 1, 128, 128, 64).to(device)
        _, endindex = model.extract_endpoints(inputs)
        layer_name = [f"_blocks.{i-1}._project_conv" for i in endindex.values()] + ['_conv_head']
        layer_dic['efficientnet'] = layer_name

    # 依照切分後的資料順序，依次運算
    k = first

    for testdata in test_data:
        test_images = testdata['image'].to(device)
        if class_type == 'all':
            test_images = F.interpolate(test_images, size=input_shape, mode='trilinear', align_corners=False)
        test_labels = testdata['label'].to(device)
        if attention_mask:
            test_bboxs = testdata['bbox'].to(device)
        #out_images = testdata['ori_image'].to(device)
        file_name = testdata['image_meta_dict']['filename_or_obj']
        '''
        GradCam need reset at every epoch
        '''
        if cam_type == "GradCAM":
            # two mask layer name is multiply
            if attention_mask:
                layer_name = 'multiply'
            elif architecture == 'CBAM':
                layer_name = get_last_conv_name(model)[-2]
            else:
                layer_name = get_last_conv_name(model)[-1]

            grad_cam = GradCAM(model, layer_name, device)          
        elif cam_type == "LayerCAM":
            layer_name = layer_dic[architecture]
            grad_cam = LayerCAM(model, layer_name, device)

        start_time = time.time()
        # label true means show the map as label, false means show the map as predicted
        if Label == 'True':
            if attention_mask:
                result_list = grad_cam(test_images, test_bboxs, index_sel=test_labels)
            else:     
                result_list = grad_cam(test_images, index_sel=test_labels)
        else:
            if attention_mask:
                result_list = grad_cam(test_images, test_bboxs)
            else:
                result_list = grad_cam(test_images)
        grad_cam.remove_handlers()
        end_time = time.time()
        print("Time taken for GradCAM: ", end_time - start_time)
        start_time = time.time()

        for i in range(len(result_list)):
            print(f"Read file in line {k}",flush = True)
            if class_type == 'kidney':
                file_path = f'{str(test_df.chartNo[k])}_{str(test_df.region[k])}'
            else:
                file_path = test_df.chartNo[k]
            #if not os.path.isdir(f"{test_path}/{file_path}"):
            #    os.makedirs(f"{test_path}/{file_path}")
            if class_type == 'all':
                pos_label = True if test_df['inj_solid'][k] == 1 else False
            elif class_type == 'liver':
                pos_label = True if test_df['liv_inj_tmp'][k] == 1 else False
            elif class_type == 'spleen':
                pos_label = True if test_df['spl_inj_tmp'][k] == 1 else False
            elif class_type == 'kidney':
                pos_label = True if test_df['kid_inj_tmp'][k] == 1 else False

            if pos_label:
                final_path = f"{pos_path}/{file_path}" 
                total_final_path = f"{pos_total}/{file_path}_total" 
            else:
                final_path = f"{neg_path}/{file_path}"
                total_final_path = f"{neg_total}/{file_path}_total" 
                #print("Label is negative, pass it !")
                #k += 1
                #continue 
            
            if not os.path.isdir(final_path):
                os.makedirs(final_path)
            if not os.path.isdir(total_final_path):
                os.makedirs(total_final_path)    

            if channel_first:
                if class_type == 'all':
                    test_images = testdata['image'].to(device)
                    image = test_images[i,0,:,:,:].cpu().detach().numpy()
                else:
                    image = test_images[i,0,:,:,:].cpu().detach().numpy()
            #    image = out_images[i,0,:,:,:].cpu().detach().numpy()   
            else:
                if class_type == 'all':
                    test_images = testdata['image'].to(device)
                    image = test_images[i,:,:,:,0].cpu().detach().numpy()
                else:
                    image = test_images[i,:,:,:,0].cpu().detach().numpy()
            #    image = out_images[i,:,:,:,0].cpu().detach().numpy()
            heatmap_total = result_list[i]
            
            # image and heatmap resize (z aixs didn't chage)
            # inuput size sometimes not match output_size(128,128,64)
            if class_type != 'all':
                image = zoom(image, (output_shape[0]/input_shape[0], output_shape[1]/input_shape[1], output_shape[2]/input_shape[2]))
            heatmap_total = zoom(heatmap_total,(output_shape[0]/input_shape[0], output_shape[1]/input_shape[1], output_shape[2]/input_shape[2]))

            if detail == 'detail':
                #print('detail is true',flush=True)
                process_plot_multiprocess(heatmap_total, image, final_path)
                for j in range(image.shape[-1]):
                    plot_heatmap_detail(heatmap_total[:,:,j],image[:,:,j],f"{final_path}/{j:03}.png")
                    plot_vedio(final_path)
            elif detail == 'one_picture':
                #print('detail is false',flush=True)
                plot_heatmap_one_picture(heatmap_total,image,f'{total_final_path}/total_view.png')
            elif detail == 'all':
                plot_heatmap_one_picture(heatmap_total,image,f'{total_final_path}/total_view.png')
                #process_plot_multiprocess(heatmap_total, image, final_path)
                for j in range(image.shape[-1]):
                    plot_heatmap_detail(heatmap_total[:,:,j],image[:,:,j],f"{final_path}/{j:03}.png")
                    plot_vedio(final_path)
            k += 1
            print(f'{file_path} is already done!',flush = True)
        end_time = time.time()
        print("Time taken for plotting: ", end_time - start_time)

# def main():
parser = get_parser()
args = parser.parse_args()

class_type = args.class_type
load_weight = args.input_weight
data_file = args.data_file
output_file = args.output
split_num = args.split
fold = args.fold
heatmap_type = args.heatmap_type
cam_type = args.cam_type
Label = args.label
cfgpath = args.config
kfold = args.kfold
test = args.test

conf = configparser.ConfigParser()
conf.read(cfgpath)
#print(conf.sections())
# Augmentation
size = eval(conf.get('Augmentation','size'))

# Data_setting (necessary)
architecture = conf.get('Data_Setting','architecture')
if architecture == 'efficientnet':
    structure_num = conf.get('Data_Setting', 'structure_num')
gpu_num = conf.getint('Data_Setting','gpu')
normal_structure = conf.getboolean('Data_Setting','normal_structure')
dataloader_num_workers = conf.getint('Data_Setting','dataloader_num_workers')
img_type = conf.get('Data_Setting','img_type')
bbox = conf.getboolean('Data_Setting','bbox')
attention_mask = conf.getboolean('Data_Setting','attention_mask')
# HU range: ex 0,100
img_hu = eval(conf.get('Data_Setting','img_hu'))
#optimizer = conf.get('Data_Setting','optimizer')

# set parameter 
if class_type == 'all':
    output_shape = 300,300,64
input_shape = size
batch_size = 8
test_df = pd.read_csv(data_file)
if test:
    if class_type == 'all':
        test_df = test_df.groupby('inj_solid').apply(lambda x: x.sample(min(len(x), 15), random_state=1)).reset_index(drop=True)
    elif class_type == 'liver':
        test_df = test_df.groupby('liv_inj_tmp').apply(lambda x: x.sample(min(len(x), 15), random_state=1)).reset_index(drop=True)
    elif class_type == 'spleen':
        test_df = test_df.groupby('spl_inj_tmp').apply(lambda x: x.sample(min(len(x), 15), random_state=1)).reset_index(drop=True)
    elif class_type == 'kidney':
        test_df = test_df.groupby('kid_inj_tmp').apply(lambda x: x.sample(min(len(x), 15), random_state=1)).reset_index(drop=True)
            
test_data_dicts  = data_progress_all_in_split(test_df, 'test_data_dict', class_type)
# select data fold
test_data_dicts_sel = test_data_dicts[split_num*fold:split_num*(fold+1)]

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
# Whole image就先讀取大size在內部再resize，避免解析度太低
elif class_type == 'all':
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
                Resized(keys=['image'], spatial_size = output_shape, mode=("trilinear"))
                
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

print("Collecting:", datetime.now(), flush=True)

if cam_type == 'LayerCAM':
    batch_size = 1
test_ds = CacheDataset(data=test_data_dicts_sel, transform=test_transforms, cache_rate=1, num_workers=dataloader_num_workers)
test_data = DataLoader(test_ds, batch_size=batch_size, num_workers=dataloader_num_workers)

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

# Plot grad cam
CAM_plot(model, test_data, test_df, class_type, output_file, size, device, split_num*fold, heatmap_type, cam_type, architecture, Label, attention_mask, kfold)



# if __name__ == '__main__':
#     main()