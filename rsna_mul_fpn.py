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
from efficientnet_3d.Efficient3D_BIFPN import EfficientNet3D_BiFPN, EfficientNet3D_FPN
from resnet_3d.resnet_3d_new import Resnet3D_3_input
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
    plot_loss_metric,
    FocalLoss
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

def get_parser():
    parser = argparse.ArgumentParser(description='spleen classification')
    parser.add_argument('-f', '--file', help=" The config file name. ", type=str)
    parser.add_argument('-c', '--class_type', help=" The class of data. (liver, kidney, spleen, all) ", type=str)
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

# 這個部分也要看你docker路徑去改對應路徑
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

# 日期判斷並轉換
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
    if len(ratio) == 2:
        valid_df = df_sel
        test_df = None
    else:
        valid_df = df_sel.groupby("group_key", group_keys=False).sample(
                frac=(ratio[1]/(ratio[2]+ratio[1])), random_state=seed)
        test_df = df_sel.drop(valid_df.index.to_list())
    
    return train_df, valid_df, test_df

# 進行完整一次預測
def run_once(times=0):
    # reset config parameter
    config.initialize()

    train_df, valid_df, test_df = train_valid_test_split(df_all, ratio = data_split_ratio, seed = seed)

    # train_df.to_csv("/tf/yilian618/ABD_classification/rsna_total_train.csv", index=False) 
    # valid_df.to_csv("/tf/yilian618/ABD_classification/rsna_total_valid.csv", index=False) 
    # test_df.to_csv("/tf/yilian618/ABD_classification/rsna_total_test.csv", index=False) 

    # test
    # train_df = train_df
    # valid_df = train_df
    # test_df = test_df

    train_data_dicts = data_progress_all(train_df, 'train_data_dict', attention_mask)
    valid_data_dicts = data_progress_all(valid_df, 'valid_data_dict', attention_mask)
    if test_df:
        test_data_dicts  = data_progress_all(test_df, 'test_data_dict', attention_mask)
    #with open('/tf/jacky831006/ABD_data/train.pickle', 'wb') as f:
    #    pickle.dump(train_data_dicts, f)

    set_determinism(seed=0)
    train_ds = CacheDataset(data=train_data_dicts, transform=train_transforms, cache_rate=1, num_workers=dataloader_num_workers)
    train_loader = DataLoader(train_ds, batch_size=traning_batch_size, shuffle=True, num_workers=dataloader_num_workers, pin_memory=True)
    valid_ds = CacheDataset(data=valid_data_dicts, transform=valid_transforms, cache_rate=1, num_workers=dataloader_num_workers)
    val_loader = DataLoader(valid_ds, batch_size=valid_batch_size, num_workers=dataloader_num_workers, pin_memory=True)
    
    if gpu_num == 'all':
        device = torch.device("cuda")
        if torch.cuda.device_count() > 1:
            # 如果有多个 GPU 可用，则使用所有 GPU
            print("Let's use", torch.cuda.device_count(), "GPUs!")
    else:
        device = torch.device("cuda",0)
        print("Only one")

    # Model setting
    # DenseBlock = DenseNet3D_FPN._DenseBlock
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
            if fpn == 'bifpn':
                model = EfficientNet3D_BiFPN(size=size, structure_num=structure_num, class_num=3, dropout=0.2, fpn_type=fpn_type, depth_coefficient=depth_coefficient, 
                                            in_channels=2 if attention_mask else 1)
            else:
                model = EfficientNet3D_FPN(size=size, structure_num=structure_num, class_num=3, fpn_type=fpn_type, depth_coefficient=depth_coefficient, 
                                            normalize=False, in_channels=2 if attention_mask else 1)
        elif fpn_type == 'split':
            model = EfficientNet3D_BiFPN(size=size, structure_num=structure_num, class_num=3, dropout=0.2, fpn_type=fpn_type, depth_coefficient=depth_coefficient)
        elif fpn_type == 'feature_concat':
            if fpn == 'bifpn':
                model = EfficientNet3D_BiFPN(size=size, structure_num=structure_num, class_num=3, dropout=0.2, fpn_type=fpn_type, depth_coefficient=depth_coefficient, 
                                            in_channels=2 if attention_mask else 1)
            else:
                model = EfficientNet3D_FPN(size=size, structure_num=structure_num, class_num=3, fpn_type=fpn_type, depth_coefficient=depth_coefficient, 
                                        normalize=False, in_channels=2 if attention_mask else 1)
            # model = EfficientNet3D_BiFPN(size=size, structure_num=structure_num, class_num=3, dropout=0.2, fpn_type=fpn_type, depth_coefficient=depth_coefficient)
    elif architecture == 'resnet':
        if fpn_type == 'label_concat':
            model = Resnet3D_3_input(size=size, num_classes=3, device=device)


    # for name, module in model.densenet3d.features.named_children():
    #     if isinstance(module, DenseBlock):  # 假设DenseBlock是Dense Blocks的类名
    #         for layer_name, layer in module.named_children():
    #             if hasattr(layer, 'conv1'):
    #                 prune.ln_structured(layer.conv1, name='weight', amount=0.2, n=1, dim=0)
    #                 prune.remove(layer.conv1, 'weight')

    if gpu_num == 'all':
        model = nn.DataParallel(model).to(device)
    else:
        if architecture == 'resnet':
            model.to(device)
            # model = model.to(device)
            # model = nn.DataParallel(model, device_ids=None)
            # net_dict = model.state_dict()
            # load_weight = '/tf/jacky831006/ABD_classification/pretrain_weight/resnet_101.pth'
            # pretrain = torch.load(load_weight)
            # pretrain_dict = {k: v for k, v in pretrain['state_dict'].items() if k in net_dict.keys()}
            
            # net_dict.update(pretrain_dict)
            # model.load_state_dict(net_dict)

            # new_parameters = [] 
            # for pname, p in model.named_parameters():
            #     for layer_name in opt.new_layer_names:
            #         if pname.find(layer_name) >= 0:
            #             new_parameters.append(p)
            #             break

            # new_parameters_id = list(map(id, new_parameters))
            # base_parameters = list(filter(lambda p: id(p) not in new_parameters_id, model.parameters()))
            # parameters = {'base_parameters': base_parameters, 
            #             'new_parameters': new_parameters}
        else:
            model.to(device)

    # all class split in healthy, low, high
    weights = [1.0, 2.0, 4.0]
    class_weights = torch.FloatTensor(weights).to(device)
    # concat,individual
    # else:
    #     weights = [1.0, 2.0, 4.0, 1.0, 2.0, 4.0, 1.0, 2.0, 4.0]
    #     class_weights = torch.FloatTensor(weights).to(device)
    if loss_type == 'crossentropy':
        loss_function = nn.CrossEntropyLoss(weight=class_weights)
    elif loss_type == 'focalloss':
        loss_function = FocalLoss(alpha=0.25, gamma=2, use_softmax=True, weight=class_weights)
    # else:
    #     loss_function = nn.BCEWithLogitsLoss(reduction='mean')

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
    root_logdir = f"/tf/jacky831006/ABD_classification/tfboard/{class_type}"     
    logdir = "{}/run-{}/".format(root_logdir, now) 

    # tfboard file path
    # 創一個主目錄 之後在train內的sumamaryWriter都會以主目錄創下面路徑
    writer = SummaryWriter(logdir)
    if not os.path.isdir(logdir):
        os.makedirs(logdir)
    # check_point path
    check_path = f'/tf/jacky831006/ABD_classification/training_checkpoints/{class_type}/{now}'
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
    
   
    test_model = train_mul_fpn(model, device, data_num, epochs, optimizer, loss_function, train_loader, \
                        val_loader, early_stop, scheduler, check_path, fpn_type, eval_score, use_amp, attention_mask)
                    
    # plot train loss and metric 
    plot_loss_metric(config.epoch_loss_values, config.metric_values, check_path)
    # remove dataloader to free memory
    del train_ds
    del train_loader
    del valid_ds
    del val_loader
    gc.collect()
    if test_df:
        # Avoid ram out of memory
        test_ds = CacheDataset(data=test_data_dicts, transform=valid_transforms, cache_rate=1, num_workers=dataloader_num_workers)
        test_loader = DataLoader(test_ds, batch_size=testing_batch_size, num_workers=dataloader_num_workers)
        # validation is same as testing
        print(f'Best accuracy:{config.best_metric}')
        if config.best_metric != 0:
            load_weight = f'{check_path}/{config.best_metric}.pth'
            model.load_state_dict(torch.load(load_weight))

        # record paramter
        accuracy_list.append(config.best_metric)
        file_list.append(now)
        epoch_list.append(config.best_metric_epoch)

        test_acc = valid_mul_fpn(model, test_loader, device, eval_score, attention_mask)
        test_accuracy_list.append(test_acc)
        del test_ds
        del test_loader
        gc.collect()

        print(f'\n Best accuracy:{config.best_metric}, Best test accuracy:{test_acc}')


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    class_type = args.class_type
    # 讀檔路徑，之後可自己微調
    if args.file.endswith('ini'):
        cfgpath = f'/tf/jacky831006/ABD_classification/config/{class_type}/{args.file}'
    else:
        cfgpath = f'/tf/jacky831006/ABD_classification/config/{class_type}/{args.file}.ini'
    

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
    if architecture == 'efficientnet':
        structure_num = conf.get('Data_Setting', 'structure_num')
        depth_coefficient = conf.getfloat('Data_Setting', 'depth_coefficient')
        fpn = conf.get('Data_Setting','fpn')
    gpu_num = conf.get('Data_Setting','gpu')
    seed = conf.getint('Data_Setting','seed')
    cross_kfold = conf.getint('Data_Setting','cross_kfold')
    # normal_structure = conf.getboolean('Data_Setting','normal_structure')
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
    # whole, cropping_normal, cropping_convex, cropping_dilation
    # img_type = conf.get('Data_Setting','img_type')
    loss_type = conf.get('Data_Setting','loss')
    fpn_type = conf.get('Data_Setting','fpn_type')
    eval_score = conf.get('Data_Setting','eval_score')
    use_amp = conf.getboolean('Data_Setting','use_amp')
    attention_mask = conf.getboolean('Data_Setting','attention_mask')

    # bbox = conf.getboolean('Data_Setting','bbox')
    # HU range: ex 0,100
    # img_hu = eval(conf.get('Data_Setting','img_hu'))

    # Setting cuda environment
    if gpu_num != 'all':
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_num
    else:
        print('GPU ok')
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
    # torch.autograd.set_detect_anomaly(True)
    # Data progressing
    All_data = pd.read_csv("/SSD/rsna-2023/rsna_train_new_v2.csv")
    pos_data = All_data[All_data['any_injury']==1]
    # neg_data = All_data[All_data['any_injury']==0].sample(n=len(pos_data), random_state=seed)
    neg_data = All_data[All_data['any_injury']==0].sample(n=300, random_state=seed)
    All_data = pd.concat([pos_data, neg_data])
    no_seg_kid = pd.read_csv("/SSD/rsna-2023/nosegmentation_kid.csv")
    no_seg = pd.read_csv("/SSD/rsna-2023/nosegmentation.csv")
    All_data = All_data[~All_data['file_paths'].isin(no_seg_kid['file_paths'])]
    All_data = All_data[~All_data['file_paths'].isin(no_seg['file_paths'])]

    df_all = All_data
    # if bbox and attention_mask:
        # raise ValueError("Only one of 'bbox' and 'attention_mask' can be selected as True.")
    if not attention_mask:
        train_transforms = Compose([
                LoadImaged(keys=["image_liv","image_spl","image_kid_r","image_kid_l"]),
                EnsureChannelFirstd(keys=["image_liv","image_spl","image_kid_r","image_kid_l"]),
                #RepeatChanneld(keys=["image","label"], repeats = num_sample),
                ScaleIntensityRanged(
                    #keys=["image"], a_min=-57, a_max=164, b_min=0.0, b_max=1.0, clip=True,
                    keys=["image_liv","image_spl","image_kid_r","image_kid_l"], a_min=-50, a_max=250, b_min=0.0, b_max=1.0, clip=True,
                ),
                #Dulicated_new(keys=["image"], num_samples=num_samples, pos_sel=True),
                Spacingd(keys=["image_liv","image_spl","image_kid_r","image_kid_l"], pixdim=(1.5, 1.5, 2.0), mode=("bilinear")),
                Orientationd(keys=["image_liv","image_spl","image_kid_r","image_kid_l"], axcodes="RAS"),
                CropForegroundd(keys=["image_liv"], source_key="image_liv"),
                CropForegroundd(keys=["image_spl"], source_key="image_spl"),
                CropForegroundd(keys=["image_kid_r"], source_key="image_kid_r"),
                CropForegroundd(keys=["image_kid_l"], source_key="image_kid_l"),
                Resized(keys=["image_liv","image_spl"], spatial_size = size, mode=("trilinear")),
                Rand3DElasticd(
                    keys=["image_liv","image_spl","image_kid_r","image_kid_l"],
                    mode=("bilinear"),
                    prob=prob,
                    sigma_range=sigma_range,
                    magnitude_range=magnitude_range,
                    spatial_size=size,
                    translate_range=translate_range,
                    rotate_range=rotate_range,
                    scale_range=scale_range,
                    padding_mode="border"),
                Resized(keys=["image_kid_r","image_kid_l"], spatial_size = (size[0],size[1],size[2]//2), mode=("trilinear")),
            ])
        valid_transforms = Compose([
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
        train_transforms = Compose([
                LoadImaged(keys=["image_liv","image_spl","image_kid_r","image_kid_l",
                                "mask_liv","mask_spl","mask_kid_r","mask_kid_l"]),
                EnsureChannelFirstd(keys=["image_liv","image_spl","image_kid_r","image_kid_l",
                                        "mask_liv","mask_spl","mask_kid_r","mask_kid_l"]),
                #RepeatChanneld(keys=["image","label"], repeats = num_sample),
                ScaleIntensityRanged(
                    #keys=["image"], a_min=-57, a_max=164, b_min=0.0, b_max=1.0, clip=True,
                    keys=["image_liv","image_spl","image_kid_r","image_kid_l"], 
                        a_min=-50, a_max=250, b_min=0.0, b_max=1.0, clip=True,
                ),
                #Dulicated_new(keys=["image"], num_samples=num_samples, pos_sel=True),
                Spacingd(keys=["image_liv","image_spl","image_kid_r","image_kid_l","mask_liv","mask_spl","mask_kid_r","mask_kid_l"], 
                        pixdim=(1.5, 1.5, 2.0), mode=("bilinear")),
                Orientationd(keys=["image_liv","image_spl","image_kid_r","image_kid_l","mask_liv","mask_spl","mask_kid_r","mask_kid_l"], axcodes="RAS"),
                CropForegroundd(keys=["image_liv","mask_liv"], source_key="image_liv"),
                CropForegroundd(keys=["image_spl","mask_spl"], source_key="image_spl"),
                CropForegroundd(keys=["image_kid_r","mask_kid_r"], source_key="image_kid_r"),
                CropForegroundd(keys=["image_kid_l","mask_kid_l"], source_key="image_kid_l"),
                Resized(keys=["image_liv","image_spl","mask_liv","mask_spl"], spatial_size = size, mode=("trilinear")),
                Rand3DElasticd(
                    keys=["image_liv","image_spl","image_kid_r","image_kid_l","mask_liv","mask_spl","mask_kid_r","mask_kid_l"],
                    mode=("bilinear"),
                    prob=prob,
                    sigma_range=sigma_range,
                    magnitude_range=magnitude_range,
                    spatial_size=size,
                    translate_range=translate_range,
                    rotate_range=rotate_range,
                    scale_range=scale_range,
                    padding_mode="border"),
                Resized(keys=["image_kid_r","image_kid_l","mask_kid_r","mask_kid_l"], spatial_size = (size[0],size[1],size[2]//2), mode=("trilinear")),
            ])

        valid_transforms = Compose([
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