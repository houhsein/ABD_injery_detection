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
import matplotlib.pyplot as plt
import pandas as pd
import cv2
import csv
import nibabel as nib
import matplotlib.pyplot as plt
import sys

# 路徑要根據你的docker路徑來設定
sys.path.append("/tf/yilian618/ABD_classification/model/")
from efficientnet_3d.model_3d import EfficientNet3D
from resnet_3d import resnet_3d
from DenseNet3D_FPN import DenseNet3D_FPN,DenseNet3D_FPN_seg

# 此架構參考這篇
# https://github.com/fei-aiart/NAS-Lung
sys.path.append("/tf/yilian618/ABD_classification/model/NAS-Lung/")
from models.cnn_res import ConvRes
from models.net_sphere import AngleLoss

import utils.config as config
import configparser
import gc
import math
import json
from utils.training_torch_utils import (
    train_rsna,
    train_rsna_seg,
    validation_rsna,
    validation_rsna_seg,
    plot_loss_metric,
    FocalLoss,
    ImgAggd,
    AttentionModel,
    AttentionModel_new,
    Dulicated_new,
)
import pickle

# Data augmnetation module (based on MONAI)
from monai.networks.nets import UNet, densenet, SENet, ViT
from monai.apps import download_and_extract
from monai.data import CacheDataset, DataLoader, Dataset
from monai.config import print_config
from monai.utils import first, set_determinism
from monai.losses import DiceLoss
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
)
import functools

# let all of print can be flush = ture
print = functools.partial(print, flush=True)


def get_parser():
    parser = argparse.ArgumentParser(description="spleen classification")
    parser.add_argument("-f", "--file", help=" The config file name. ", type=str)
    parser.add_argument(
        "-c",
        "--class_type",
        help=" The class of data. (liver, kidney, spleen, all) ",
        type=str,
    )
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
def data_progress_all(file, dicts):
    dicts = []
#     dir = "/SSD/TotalSegmentator/rsna_selected_crop_bbox"
    for index, row in file.iterrows():
        dirs = os.path.dirname(row['file_paths'])
        image_whole = row['file_paths']
        image_liv = os.path.join(dirs,"liver.nii.gz")
        image_spl = os.path.join(dirs,"spleen.nii.gz")
        image_kid_r = os.path.join(dirs,"kidney_right.nii.gz")
        image_kid_l = os.path.join(dirs,"kidney_left.nii.gz")
        row['healthy']=0
        if row["liver_healthy"] ==1 and row["spleen_healthy"] ==1 and row["kidney_healthy"] ==1:
            row['healthy']=1
        label = np.array([row["liver_low"],row["liver_high"],row["spleen_low"],row["spleen_high"],row["kidney_low"],row["kidney_high"],row['healthy']])

        dicts.append({"image_whole": image_whole, "image_liv": image_liv, "image_spl": image_spl, "image_kid_r": image_kid_r, "image_kid_l": image_kid_l, "label": label})

    return dicts



# 判斷是否為injury
# def inj_check(row):
#     kid_inj_tmp = 0 if row["kid_inj_rt"] == row["kid_inj_lt"] == 0 else 1
#     liv_inj_tmp = 0 if row["liv_inj"] == 0 else 1
#     spl_inj_tmp = 0 if row["spl_inj"] == 0 else 1
#     return pd.Series([kid_inj_tmp, liv_inj_tmp, spl_inj_tmp])


# 日期判斷並轉換
def convert_date(x):
    if pd.isna(x):  # Check if the value is NaN
        return x  # If it's NaN, return it as-is
    else:
        return pd.to_datetime(int(x), format="%Y%m%d")


# 將positive進行複製
def duplicate(df, col_name, num_sample, pos_sel=True):
    if pos_sel:
        df_inj_tmp = df[df[col_name] == 1]
    else:
        df_inj_tmp = df

    # 進行重複
    df_inj_tmp_duplicated = pd.concat([df_inj_tmp] * num_sample, ignore_index=True)

    # 將原始的df和複製後的df結合
    df_new = pd.concat([df, df_inj_tmp_duplicated], ignore_index=True)

    return df_new


# 依據不同positive情況進行資料切分
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


class CustomLoss(nn.Module):
    def __init__(self, weight=1):
        super(CustomLoss, self).__init__()
        #self.bce = nn.BCEWithLogitsLoss(pos_weight=torch.as_tensor([2.318]).cuda()).cuda()
        self.bce = nn.BCELoss()
        self.dice = DiceLoss(to_onehot_y=False, softmax=True)
        self.weight=weight

    def forward(self, outputs, targets, masks_outputs,masks_targets):

        loss1 = self.bce(outputs, targets)
        loss2 = self.dice(masks_outputs, masks_targets)
        loss = loss1 + (loss2 * self.weight) 

        return loss


# 進行完整一次預測
def run_once(times=0):
    # reset config parameter
    config.initialize()
#     print(loss_weight)

    #拆分資料
    train_df, valid_df, test_df = train_valid_test_split(
        df_all, ratio=data_split_ratio, seed=seed, test_fix=2016
    )

    train_df.to_csv("/tf/yilian618/ABD_classification/rsna_total_train.csv", index=False) 
    valid_df.to_csv("/tf/yilian618/ABD_classification/rsna_total_valid.csv", index=False) 
    test_df.to_csv("/tf/yilian618/ABD_classification/rsna_total_test.csv", index=False) 

#     train_df = train_df[:50]
#     valid_df = train_df[:20]
#     test_df = test_df[:20]
    #處理成dict格式 包含影像路徑與label
    train_data_dicts = data_progress_all(train_df, "train_data_dict")
    valid_data_dicts = data_progress_all(valid_df, "valid_data_dict")
    test_data_dicts = data_progress_all(test_df, "test_data_dict")

    set_determinism(seed=0)

    train_ds = CacheDataset(
        data=train_data_dicts,
        transform=train_transforms,
        cache_rate=1,
        num_workers=dataloader_num_workers,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=traning_batch_size,
        shuffle=True,
        num_workers=dataloader_num_workers,
    )
    valid_ds = CacheDataset(
        data=valid_data_dicts,
        transform=valid_transforms,
        cache_rate=1,
        num_workers=dataloader_num_workers,
    )
    val_loader = DataLoader(
        valid_ds, batch_size=valid_batch_size, num_workers=dataloader_num_workers
    )

    device = torch.device("cuda", 0)
#     CUDA_VISIBLE_DEVICES=1
#     device = torch.device("cuda")

    # Model setting
    model = DenseNet3D_FPN_seg.DenseNet3D_FPN_seg(n_input_channels=1,dropout=0.2,class_mum=7)
#     model = nn.DataParallel(model)
    model.to(device)


    # Imbalance loss
    # 根據不同資料的比例做推測
    # 目前kidney都以單邊受傷為主
    if class_type == "all":
        grouped = df_all.groupby("inj_solid")
    elif class_type == "liver":
        # grouped = df_all.groupby('liver_inj_no')
        grouped = df_all.groupby("liv_inj_tmp")
    elif class_type == "spleen":
        # grouped = df_all.groupby('spleen_inj_no')
        grouped = df_all.groupby("spl_inj_tmp")
    elif class_type == "kidney":
        # grouped = df_all.groupby('kidney_inj_no')
        grouped = df_all.groupby("kid_inj_tmp")






    # Grid search
    if len(init_lr) == 1:
        optimizer = torch.optim.Adam(model.parameters(), init_lr[0])
    else:
        optimizer = torch.optim.Adam(model.parameters(), init_lr[times])

    if len(loss_weight) == 1:
        weight = loss_weight[0]
    else:
        weight = loss_weight[times]

    if lr_decay_epoch == 0:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=lr_decay_rate, patience=epochs, verbose=True
        )
        # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=epochs,gamma=lr_decay_rate, verbose=True )
    else:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=lr_decay_rate,
            patience=lr_decay_epoch,
            verbose=True,
        )
    loss_function = CustomLoss(weight)
    now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    root_logdir = f"/tf/yilian618/classification_torch/tfboard/{class_type}"
    logdir = "{}/run-{}/".format(root_logdir, now)

    # tfboard file path
    # 創一個主目錄 之後在train內的sumamaryWriter都會以主目錄創下面路徑
    writer = SummaryWriter(logdir)
    if not os.path.isdir(logdir):
        os.makedirs(logdir)
    # check_point path
    check_path = (
        f"/tf/yilian618/classification_torch/training_checkpoints/{class_type}/{now}"
    )
    if not os.path.isdir(check_path):
        os.makedirs(check_path)
    print(f"\n Weight location:{check_path}", flush=True)
    if cross_kfold == 1:
        print(f"\n Processing begining", flush=True)
    else:
        print(f"\n Processing fold #{times}", flush=True)

    data_num = len(train_ds)

    test_model = train_rsna_seg(
        model,
        device,
        data_num,
        epochs,
        optimizer,
        loss_function,
        train_loader,
        val_loader,
        early_stop,
        scheduler,
        check_path,
    )

    # plot train loss and metric
    plot_loss_metric(config.epoch_loss_values, config.metric_values, check_path)
    # remove dataloader to free memory
    del train_ds
    del train_loader
    del valid_ds
    del val_loader
    gc.collect()

    # Avoid ram out of memory
    test_ds = CacheDataset(
        data=test_data_dicts,
        transform=valid_transforms,
        cache_rate=1,
        num_workers=dataloader_num_workers,
    )
    test_loader = DataLoader(
        test_ds, batch_size=testing_batch_size, num_workers=dataloader_num_workers
    )
    # validation is same as testing
    print(f"Best accuracy:{config.best_metric}")
    if config.best_metric != 0:
        load_weight = f"{check_path}/{config.best_metric}.pth"
        model.load_state_dict(torch.load(load_weight))

    # record paramter
    accuracy_list.append(config.best_metric)
    file_list.append(now)
    epoch_list.append(config.best_metric_epoch)

    test_acc = validation_rsna_seg(model, test_loader, device)
    test_accuracy_list.append(test_acc)
    del test_ds
    del test_loader
    gc.collect()

    print(f"\n Best accuracy:{config.best_metric}, Best test accuracy:{test_acc}")


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    class_type = args.class_type
    # 讀檔路徑，之後可自己微調
    if args.file.endswith("ini"):
        cfgpath = f"/tf/yilian618/ABD_classification/config/{class_type}/{args.file}"
    else:
        cfgpath = (
            f"/tf/yilian618/ABD_classification/config/{class_type}/{args.file}.ini"
        )

    conf = configparser.ConfigParser()
    conf.read(cfgpath)

    # Augmentation
    num_samples = conf.getint("Augmentation", "num_sample")
    size = eval(conf.get("Augmentation", "size"))
    prob = conf.getfloat("Rand3DElasticd", "prob")
    sigma_range = eval(conf.get("Rand3DElasticd", "sigma_range"))
    magnitude_range = eval(conf.get("Rand3DElasticd", "magnitude_range"))
    translate_range = eval(conf.get("Rand3DElasticd", "translate_range"))
    rotate_range = eval(conf.get("Rand3DElasticd", "rotate_range"))
    scale_range = eval(conf.get("Rand3DElasticd", "scale_range"))

    # Data_setting
    architecture = conf.get("Data_Setting", "architecture")
    if architecture == "efficientnet":
        structure_num = conf.get("Data_Setting", "structure_num")
    gpu_num = conf.getint("Data_Setting", "gpu")
    seed = conf.getint("Data_Setting", "seed")
    cross_kfold = conf.getint("Data_Setting", "cross_kfold")
    normal_structure = conf.getboolean("Data_Setting", "normal_structure")
    data_split_ratio = eval(conf.get("Data_Setting", "data_split_ratio"))
    # imbalance_data_ratio = conf.getint('Data_Setting','imbalance_data_ratio')
    epochs = conf.getint("Data_Setting", "epochs")
    # early_stop = 0 means None
    early_stop = conf.getint("Data_Setting", "early_stop")
    traning_batch_size = conf.getint("Data_Setting", "traning_batch_size")
    valid_batch_size = conf.getint("Data_Setting", "valid_batch_size")
    testing_batch_size = conf.getint("Data_Setting", "testing_batch_size")
    dataloader_num_workers = conf.getint("Data_Setting", "dataloader_num_workers")
    # init_lr = conf.getfloat('Data_Setting','init_lr')
    init_lr = json.loads(conf.get("Data_Setting", "init_lr"))
    loss_weight = json.loads(conf.get("Data_Setting", "loss_weight"))
    # optimizer = conf.get('Data_Setting','optimizer')
    lr_decay_rate = conf.getfloat("Data_Setting", "lr_decay_rate")
    lr_decay_epoch = conf.getint("Data_Setting", "lr_decay_epoch")
    # whole, cropping_normal, cropping_convex, cropping_dilation
    img_type = conf.get("Data_Setting", "img_type")
    loss_type = conf.get("Data_Setting", "loss")
    bbox = conf.getboolean("Data_Setting", "bbox")
    attention_mask = conf.getboolean("Data_Setting", "attention_mask")
    # HU range: ex 0,100
    img_hu = eval(conf.get("Data_Setting", "img_hu"))

    # Setting cuda environment
#     os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_num)
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

    # Data progressing
    All_data = pd.read_csv("/tf/yilian618/rsna_train_new_v2.csv")
    pos_data = All_data[All_data['any_injury']==1]
    neg_data = All_data[All_data['any_injury']==0].sample(n=300, random_state=seed)
    All_data = pd.concat([pos_data, neg_data])
    no_seg_kid = pd.read_csv("/tf/yilian618/nosegmentation_kid.csv")
    no_seg = pd.read_csv("/tf/yilian618/nosegmentation.csv")
    All_data = All_data[~All_data['file_paths'].isin(no_seg_kid['file_paths'])]
    All_data = All_data[~All_data['file_paths'].isin(no_seg['file_paths'])]

    df_all = All_data

    if bbox and attention_mask:
        raise ValueError(
            "Only one of 'bbox' and 'attention_mask' can be selected as True."
        )

    train_transforms = Compose(
            [
                LoadImaged(keys=["image_whole","image_liv","image_spl","image_kid_r","image_kid_l"]),
                EnsureChannelFirstd(keys=["image_whole","image_liv","image_spl","image_kid_r","image_kid_l"]),
                # RepeatChanneld(keys=["image","label"], repeats = num_sample),
                ScaleIntensityRanged(
                    # keys=["image"], a_min=-57, a_max=164, b_min=0.0, b_max=1.0, clip=True,
                    keys=["image_whole","image_liv","image_spl","image_kid_r","image_kid_l"],
                    a_min=-50,
                    a_max=250,
                    b_min=0.0,
                    b_max=1.0,
                    clip=True,
                ),
                # Dulicated_new(keys=["image"], num_samples=num_samples, pos_sel=True),
                Spacingd(keys=["image_whole","image_liv","image_spl","image_kid_r","image_kid_l"], pixdim=(1.5, 1.5, 2.0), mode=("bilinear")),
                Orientationd(keys=["image_whole","image_liv","image_spl","image_kid_r","image_kid_l"], axcodes="RAS"),
                CropForegroundd(keys=["image_whole","image_liv","image_spl","image_kid_r","image_kid_l"], source_key="image_whole"),
#                 CropForegroundd(keys=["image_spl"], source_key="image_spl"),
#                 CropForegroundd(keys=["image_kid_r"], source_key="image_kid_r"),
#                 CropForegroundd(keys=["image_kid_l"], source_key="image_kid_l"),
                Resized(keys=["image_whole","image_liv","image_spl","image_kid_r","image_kid_l"], spatial_size=size, mode=("trilinear")),
                Rand3DElasticd(
                    keys=["image_whole","image_liv","image_spl","image_kid_r","image_kid_l"],
                    mode=("bilinear"),
                    prob=prob,
                    sigma_range=sigma_range,
                    magnitude_range=magnitude_range,
                    spatial_size=size,
                    translate_range=translate_range,
                    rotate_range=rotate_range,
                    scale_range=scale_range,
                    padding_mode="border",
                ),
            ]
        )
    valid_transforms = Compose(
        [
            LoadImaged(keys=["image_whole","image_liv","image_spl","image_kid_r","image_kid_l"]),
            EnsureChannelFirstd(keys=["image_whole","image_liv","image_spl","image_kid_r","image_kid_l"]),
            ScaleIntensityRanged(
                # keys=["image"], a_min=-57, a_max=164, b_min=0.0, b_max=1.0, clip=True,
                keys=["image_whole","image_liv","image_spl","image_kid_r","image_kid_l"],
                a_min=-50,
                a_max=250,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
            Spacingd(keys=["image_whole","image_liv","image_spl","image_kid_r","image_kid_l"], pixdim=(1.5, 1.5, 2.0), mode=("bilinear")),
            Orientationd(keys=["image_whole","image_liv","image_spl","image_kid_r","image_kid_l"], axcodes="RAS"),
            CropForegroundd(keys=["image_whole","image_liv","image_spl","image_kid_r","image_kid_l"], source_key="image_whole"),
#             CropForegroundd(keys=["image_spl"], source_key="image_spl"),
#             CropForegroundd(keys=["image_kid_r"], source_key="image_kid_r"),
#             CropForegroundd(keys=["image_kid_l"], source_key="image_kid_l"),
            Resized(keys=["image_whole","image_liv","image_spl","image_kid_r","image_kid_l"], spatial_size=size, mode=("trilinear")),
                ]
    )
    # Training by cross validation
    accuracy_list = []
    test_accuracy_list = []
    file_list = []
    epoch_list = []

    if cross_kfold * data_split_ratio[2] != 1 and cross_kfold != 1:
        raise RuntimeError("Kfold number is not match test data ratio")

    first_start_time = time.time()

    # kfold
    if cross_kfold != 1:
        for k in range(cross_kfold):
            run_once(k)
    # grid search
    elif len(init_lr) != 1:
        for k in range(len(init_lr)):
            run_once(k)

    elif len(loss_weight) != 1:
        for k in range(len(loss_weight)):
#             print(loss_weight)
            run_once(k)
    else:
        run_once()

    if cross_kfold != 1:
        print(f"\n Mean accuracy:{sum(accuracy_list)/len(accuracy_list)}")

    final_end_time = time.time()
    hours, rem = divmod(final_end_time - first_start_time, 3600)
    minutes, seconds = divmod(rem, 60)
    all_time = "All time:{:0>2}:{:0>2}:{:05.2f}".format(
        int(hours), int(minutes), seconds
    )
    print(all_time)
    # write some output information in ori ini
    conf["Data output"] = {}
    conf["Data output"]["Running time"] = all_time
    conf["Data output"]["Data file name"] = str(file_list)
    # ini write in type need str type
    conf["Data output"]["Best accuracy"] = str(accuracy_list)
    conf["Data output"]["Best Test accuracy"] = str(test_accuracy_list)
    conf["Data output"]["Best epoch"] = str(epoch_list)

    with open(cfgpath, "w") as f:
        conf.write(f)