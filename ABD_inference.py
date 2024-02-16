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
sys.path.append("/tf/jacky831006/ABD_classification/model/")
from efficientnet_3d.model_3d import EfficientNet3D
from resnet_3d import resnet_3d
# 此架構參考這篇
# https://github.com/fei-aiart/NAS-Lung
sys.path.append("/tf/jacky831006/ABD_classification/model/NAS-Lung/") 
from models.cnn_res import ConvRes
from models.net_sphere import AngleLoss
import configparser
import gc
import math
import subprocess
import json
#from utils.training_torch_utils import train, validation, plot_loss_metric
from utils.training_torch_utils import FocalLoss, ImgAggd, AttentionModel, AttentionModel_new
from utils.grad_cam_torch_utils import inference, plot_confusion_matrix, plot_roc, plot_dis, zipDir, confusion_matrix_CI, plot_heatmap_detail, plot_heatmap_one_picture, plot_vedio, get_last_conv_name, GradCAM
from utils.bbox_trans import data_progress, process_image, process_image_multiprocess
from scipy.ndimage import zoom
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

def get_parser():
    parser = argparse.ArgumentParser(description='Oragan classification')
    # parser.add_argument('-k', '--class_type', help=" The class of data. (liver, kidney, spleen, all of them) ", type=str, default='all of them')
    parser.add_argument('-i', '--input', help=" Input dictionary path ", type=str, required=True)
    # parser.add_argument('-c', '--cam_type', help=" The CAM type (LayerCAM(L) or GradCAM(G)). ", type=str)
    parser.add_argument('-o', '--output', help=" Output dictionary path ", type=str, required=True)
    return parser

def model_run(model, bbox_output, device, output_dict, class_type):
    start_time = time.time()

    # Data list
    test_data_dicts = []
    for i in bbox_output:
        if class_type == 'kidney':
            test_data_dicts.append({'image':f'{i}/{class_type}_left.nii.gz'})
            test_data_dicts.append({'image':f'{i}/{class_type}_right.nii.gz'})
        else:
            test_data_dicts.append({'image':f'{i}/{class_type}.nii.gz'})
    test_ds = CacheDataset(data=test_data_dicts, transform=test_transforms, cache_rate=1, num_workers=dataloader_num_workers)
    test_data = DataLoader(test_ds, batch_size=16, num_workers=dataloader_num_workers)
    model.load_state_dict(torch.load(eval(f'{class_type}_load_weight')))
    y_pre = inference(model, test_data, device)
    y_pre_n = []
    for i in range(y_pre.shape[0]):
        if y_pre[i][1] < eval(f'{class_type}_cutoff'):
            y_pre_n.append(0)
        else:
            y_pre_n.append(1)
    if class_type == 'kidney':
        result = [(y_pre[i], y_pre[i + 1]) for i in range(0, len(y_pre), 2)]
    else:
        result = list((y_pre, y_pre_n))
    output_dict[f'{class_type}'] = result
    end_time = time.time()
    elapsed_time = end_time - start_time
    hours, minutes, seconds = convert_seconds(elapsed_time)
    print(f"{class_type}_prediction: {hours}hrs,{minutes}mins,{seconds}secs")

    return output_dict

# def process_plot_detail(j, heatmap_total, image, final_path):
#     plot_heatmap_detail(heatmap_total[:,:,j],image[:,:,j],f"{final_path}/{j:03}.png")

# def process_plot_multiprocess(heatmap_total, image, final_path, num_cores = 10):
#     with concurrent.futures.ThreadPoolExecutor(max_workers=num_cores) as executor: #創建線程池
#         futures = [executor.submit(process_plot_detail, j, heatmap_total, image, final_path) for j in range(image.shape[-1])] #將函數和相應的參數提交給線程池執行
#         for _ in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc='GradCam plot progressing'): # 確認函數迭代已完成的任務並用tqdm進行進度調顯示
#             pass
#     plot_vedio(final_path)

def CAM_plot(model, bbox_output, device, class_type):
    start_time = time.time()
    input_shape = 64,64,64
    output_shape = 128,128,64
    file_name = 'test'

    test_data_dicts = []
    for i in bbox_output:
        if class_type == 'kidney':
            test_data_dicts.append({'image':f'{i}/{class_type}_left.nii.gz'})
            test_data_dicts.append({'image':f'{i}/{class_type}_right.nii.gz'})
        else:
            test_data_dicts.append({'image':f'{i}/{class_type}.nii.gz'})
    test_ds = CacheDataset(data=test_data_dicts, transform=test_transforms, cache_rate=1, num_workers=dataloader_num_workers)
    test_data = DataLoader(test_ds, batch_size=16, num_workers=dataloader_num_workers)
    model.load_state_dict(torch.load(eval(f'{class_type}_load_weight')))

    for testdata in test_data:
        test_images = testdata['image'].to(device)
        testdata['image_meta_dict']['filename_or_obj']
        #out_images = testdata['ori_image'].to(device)
        '''
        GradCam need reset at every epoch
        '''
        layer_name = get_last_conv_name(model)[-2]
        grad_cam = GradCAM(model, layer_name, device)          
        result_list = grad_cam(test_images)
        grad_cam.remove_handlers()
        file_path_list = [os.path.basename(i).replace('.nii.gz','') for i in testdata['image_meta_dict']['filename_or_obj']]

        for i in range(len(result_list)):

            final_path = f'/tf/jacky831006/ABD_classification/gradcam/{file_name}/{file_path_list[i]}'            
            if not os.path.isdir(final_path):
                os.makedirs(final_path)
            image = test_images[i,0,:,:,:].cpu().detach().numpy() 
            heatmap_total = result_list[i]
            
            # image and heatmap resize (z aixs didn't chage)
            # inuput size sometimes not match output_size(128,128,64)

            heatmap_total = zoom(heatmap_total,(output_shape[0]/input_shape[0], output_shape[1]/input_shape[1], output_shape[2]/input_shape[2]))

            for j in range(image.shape[-1]):
                plot_heatmap_detail(heatmap_total[:,:,j],image[:,:,j],f"{final_path}/{j:03}.png")
                plot_vedio(final_path)
           
        end_time = time.time()
        elapsed_time = end_time - start_time
        hours, minutes, seconds = convert_seconds(elapsed_time)
        print(f"Time taken for plotting: {hours}hrs,{minutes}mins,{seconds}secs")


def convert_seconds(total_seconds):
    # 將總秒數轉換為小時、分鐘和秒
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60

    return hours, minutes, seconds

if __name__ == '__main__':
    total_start_time = time.time()
    parser = get_parser()
    args = parser.parse_args()
    # class_type = args.class_type
    input_dir = args.input
    output_path = args.output

    size = 64,64,64
    normal_structure = False
    dataloader_num_workers = 10
    os.environ["CUDA_VISIBLE_DEVICES"] = str(0)
        
    kidney_load_weight = '/tf/jacky831006/classification_torch/training_checkpoints/kidney/20230720093805/0.9596662030598053.pth'
    liver_load_weight = '/tf/jacky831006/classification_torch/training_checkpoints/liver/20230725141541/0.8666666666666667.pth'
    spleen_load_weight = '/tf/jacky831006/classification_torch/training_checkpoints/spleen/20230720013056/0.9209183673469388.pth'
    kidney_cutoff = 0.008680093102157116
    liver_cutoff = 0.501065194606781
    spleen_cutoff = 0.08149196207523346


    # if args.cam_type not in ['L','G','GradCAM','LayerCAM']:
    #     raise ValueError("Input error! Only GradCAM(G) and LayerCAM(L) type")
    # elif args.cam_type == 'L':
    #     cam_type = 'LayerCAM'
    # elif args.cam_type == 'G':
    #     cam_type = 'GradCAM'
    # else:
    #     cam_type = args.cam_type

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

    # DICOM2NIFTI
    start_time = time.time()
    dicom_path_list = [os.path.join(input_dir, d) for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))]
    output_path_list = [os.path.join(input_dir, d, 'NIFTI') for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))]

    # 經過TotalSeg
    for dicom_path, output_dir in zip(dicom_path_list, output_path_list):
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)
        command = ["dcm2niix", 
                        "-z",  'y', 
                        "-f", '%j', 
                        "-o", output_dir, dicom_path
                        ]
        dcm2nifti  = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding="utf-8")
    end_time = time.time()
    elapsed_time = end_time - start_time
    hours, minutes, seconds = convert_seconds(elapsed_time)
    print(f"DICOM to Nifti progress: {hours}hrs,{minutes}mins,{seconds}secs")
    file_path = []
    for d in output_path_list:
        for f in os.listdir(d):
            if f.endswith('.nii.gz'):
                file_path.append(os.path.join(d,f))
    # file_path = [os.path.join(input_dir,f) for f in os.listdir(input_dir) if f.endswith('.nii.gz')]
    # 經過TotalSeg
    start_time = time.time()
    for nifti_path in file_path:
        output_path = nifti_path.replace('.nii.gz','')
        #output_path = '/'.join(nifti_path.split('/')[:-1])
        segment = subprocess.run(["TotalSegmentator", "-i",  nifti_path, "-o", output_path, "--fast"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding="utf-8")
    end_time = time.time()
    elapsed_time = end_time - start_time
    hours, minutes, seconds = convert_seconds(elapsed_time)
    print(f"Segmentation generation: {hours}hrs,{minutes}mins,{seconds}secs")

    # 經過Bounding box轉換 (output:name_bbox)
    start_time = time.time()
    for class_type in ['kidney', 'liver', 'spleen']:
        inference_data_dict = data_progress(file_path, 'inference_data_dict', class_type)
        process_image_multiprocess(inference_data_dict, class_type)
    end_time = time.time()
    elapsed_time = end_time - start_time
    hours, minutes, seconds = convert_seconds(elapsed_time)
    print('\n')
    print(f"BBox transfer: {hours}hrs,{minutes}mins,{seconds}secs")

    # Model
    device = torch.device("cuda", 0)
    model = ConvRes(size[0], [[64, 64, 64], [128, 128, 256], [256, 256, 256, 512]], normal=normal_structure).to(device)

    # Inference
    output_dict = {}
    bbox_output = [i.replace('.nii.gz','_bbox') for i in file_path]
    for class_type in ['kidney', 'liver', 'spleen']:
        output_dict = model_run(model, bbox_output, device, output_dict, class_type)
    print(output_dict)  
    # file_name_list = [os.path.basename(i) for i in file_path]
    # rows = []
    # for i in range(len(file_name_list)):
    #     y_pre = tuple(output_dict[key][0][i] for key in output_dict)
    #     y_pre_n = tuple(output_dict[key][1][i] for key in output_dict)
    #     rows.append([file_name_list[i], y_pre, y_pre_n])

    # csv_output = pd.DataFrame(rows, columns=['file_name','ori_pre','pre_label'])

    # csv_output.to_csv(f'{output_path}.abd_injury_report.csv')
    
    # Visulization (Gradcam)
    for class_type in ['kidney', 'liver', 'spleen']:
        CAM_plot(model, bbox_output, device, class_type)

    total_end_time = time.time()
    elapsed_time = total_end_time - total_start_time
    hours, minutes, seconds = convert_seconds(elapsed_time)
    print(f"All progressing time: {hours}hrs,{minutes}mins,{seconds}secs")