import os
import time
import sys
from datetime import datetime
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.autograd import Variable
from copy import deepcopy
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
import random
import numpy as np
import utils.config as config
import matplotlib.pyplot as plt
import os, psutil
import functools
from skimage.measure import label as sk_label
from skimage.measure import regionprops as sk_regions
from skimage.transform import resize
from tqdm import tqdm, trange
from sklearn.metrics import accuracy_score, log_loss
# Based on MONAI 1.1
from monai.transforms.transform import MapTransform
from monai.utils import ensure_tuple_rep
from monai.config import KeysCollection
from typing import Optional
from monai.utils.enums import PostFix
# let all of print can be flush = ture
print = functools.partial(print, flush=True)

#-------- Dataloder --------
# Based on MONAI 0.4.0
# After augmnetation with resize, crop spleen area and than transofermer 
class BoxCrop(object):
    '''
    Croping image by bounding box label after augmentation 
    input: keys=["image", "label"]
    label:
    [[x1,y1,x2,y2,z1,z2,class]...]
    image:
    [1,x,y,z]
    output dictionary add 
        im_info: [x,y,z,scale_x_y,scale_z]
        num_box: 1 (All is one in our data)
    '''
    def __init__(self,keys):
        self.keys = keys

    def __call__(self, data):
        d = dict(data)
        image = d['image']
        label = d['label']
        # only one label
        if type(label) == type(np.array([])):
            label_list = label.tolist()
        else:
        # more than one label
        # select the first label      
            label_list = eval(label)[0]
        if label_list[1]>=label_list[3] or label_list[0]>=label_list[2] or label_list[4]>=label_list[5]:
            raise RuntimeError(f"{d['image_meta_dict']['filename_or_obj']} bounding box error")
                #print(f"{d['image_meta_dict']['filename_or_obj']} bounding box error ")
        out_image = image[0, int(label_list[1]):int(label_list[3]), int(label_list[0]):int(label_list[2]), int(label_list[4]):int(label_list[5])]
        d['image'] = np.expand_dims(out_image,axis=0)
        if len(label_list) == 7:
            d['label'] = label_list[6]
        else:
            d['label'] = 'None'
        #print(d['image'].shape)
        return d

# Dulicated dataset by num_samples
class Dulicated(object):
    '''
    Dulicated data for augmnetation
    '''
    def __init__(self,
                 keys,
                 num_samples: int = 1):
        self.keys = keys
        self.num_samples = num_samples

    def __call__(self, data):
        d = dict(data)
        image = d['image']
        label = d['label']
        results: List[Dict[Hashable, np.ndarray]] = [dict(data) for _ in range(self.num_samples)]
            
        for key in data.keys():            
            for i in range(self.num_samples):
                results[i][key] = data[key]
        return results
        #return d

# Based on MONAI 1.1
# DEFAULT_POST_FIX = PostFix.meta()

class ImgAggd(MapTransform):
    def __init__(
        self,
        keys: KeysCollection,
        # Selection of bounding box  image Hu range 
        Hu_range: tuple or None,
        # bbox & image concat to one image or not
        Concat: bool = True,
        meta_keys: Optional[KeysCollection] = None,
        allow_missing_keys: bool = False,
        # meta_key_postfix: str = DEFAULT_POST_FIX,
        # overwriting: bool = False, 
    ) -> None:
        """
        Args:
            keys: keys of the corresponding items to be transformed.
                See also: :py:class:`monai.transforms.compose.MapTransform`
            meta_keys: explicitly indicate the key to store the corresponding metadata dictionary.
                the metadata is a dictionary object which contains: filename, original_shape, etc.
                it can be a sequence of string, map to the `keys`.
                if None, will try to construct meta_keys by `key_{meta_key_postfix}`.
            meta_key_postfix: if meta_keys is None, use `key_{postfix}` to store the metadata of the nifti image,
                default is `meta_dict`. The metadata is a dictionary object.
                For example, load nifti file for `image`, store the metadata into `image_meta_dict`.
        """
        super().__init__(keys, allow_missing_keys)
        self.meta_keys = ensure_tuple_rep(meta_keys, len(self.keys))
        self.Hu_range = Hu_range
        self.Concat = Concat
        # self.meta_key_postfix = ensure_tuple_rep(meta_key_postfix, len(self.keys))

    def img_sel(self, image, a_min, a_max):
        img = image.clone()
        img = torch.where((a_min<=img) & (img<=a_max), img, -1000)
        return img

    def __call__(self, data):
        """
        Raises:
            KeyError: When not ``self.overwriting`` and key already exists in ``data``.
        """
        d = dict(data)
        image = d['image']
        bbox  = d['bbox']
        # None則代表不篩選
        if self.Hu_range:
            Hu_min = self.Hu_range[0]
            Hu_max = self.Hu_range[1]
            bbox = self.img_sel(bbox, Hu_min, Hu_max)
        if self.Concat:    
            new_img = torch.cat( (image,bbox), 0 )
            d['image'] = new_img
            # 將bbox變zero tensor 減少記憶體使用
            d['bbox'] = torch.zeros(1)
        else:
            # 只做Hu selection
            #TODO 看要不要將segmentation 轉成mask
            d['bbox'] = bbox
        return d

# Dulicated dataset by num_samples
class Dulicated_new(MapTransform):
    '''
    Dulicated data for augmnetation
    '''
    def __init__(
        self,
        keys: KeysCollection,
        meta_keys: Optional[KeysCollection] = None,
        allow_missing_keys: bool = False,
        num_samples: int = 1,
        # Only duplicated positive data 
        pos_sel: bool = True
        # meta_key_postfix: str = DEFAULT_POST_FIX,
        # overwriting: bool = False, 
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.meta_keys = ensure_tuple_rep(meta_keys, len(self.keys))
        self.num_samples = num_samples
        self.pos_sel = pos_sel

    def __call__(self, data):
        d = dict(data)
        image = d['image']
        label = d['label']
        # Initialize an empty list
        results: List[Dict[Hashable, torch.Tensor]] = []
        if self.pos_sel:
            if label == 1:
                results = [dict(data) for _ in range(self.num_samples)]
                # deep copy all the unmodified data
                for i in range(self.num_samples):
                    for key in set(data.keys()).difference(set(self.keys)):
                        results[i][key] = deepcopy(data[key])
            else:
                results.append(deepcopy(data))
        else:
            results = [dict(data) for _ in range(self.num_samples)]
            # deep copy all the unmodified data
            for i in range(self.num_samples):
                for key in set(data.keys()).difference(set(self.keys)):
                    results[i][key] = deepcopy(data[key])
        return results

class Kid_concatd(MapTransform):
    def __init__(
        self,
        keys: KeysCollection,
        meta_keys: Optional[KeysCollection] = None,
        allow_missing_keys: bool = False,
        # meta_key_postfix: str = DEFAULT_POST_FIX,
        # overwriting: bool = False, 
    ) -> None:
        """
        Args:
            keys: keys of the corresponding items to be transformed.
                See also: :py:class:`monai.transforms.compose.MapTransform`
            meta_keys: explicitly indicate the key to store the corresponding metadata dictionary.
                the metadata is a dictionary object which contains: filename, original_shape, etc.
                it can be a sequence of string, map to the `keys`.
                if None, will try to construct meta_keys by `key_{meta_key_postfix}`.
            meta_key_postfix: if meta_keys is None, use `key_{postfix}` to store the metadata of the nifti image,
                default is `meta_dict`. The metadata is a dictionary object.
                For example, load nifti file for `image`, store the metadata into `image_meta_dict`.
        """
        super().__init__(keys, allow_missing_keys)
        self.meta_keys = ensure_tuple_rep(meta_keys, len(self.keys))
        # self.meta_key_postfix = ensure_tuple_rep(meta_key_postfix, len(self.keys))
    def __call__(self, data):
        """
        Raises:
            KeyError: When not ``self.overwriting`` and key already exists in ``data``.
        """
        d = dict(data)
        image = d['image']
        image2  = d['image2']
        new_img = torch.where(image != -1000, image, image2 )
        d['image'] = new_img
        # 將bbox變zero tensor 減少記憶體使用
        d['image2'] = torch.zeros(1)
        return d
# True label

class Annotate(object):
    '''
    transform mask to bounding box label after augmentation
    check the image shape to know scale_x_y, scale_z 
    input: keys=["image", "label"]
    output dictionary add 
        im_info: [x,y,z,scale_x_y,scale_z]
        num_box: 1 (All is one in our data)
    '''
    def __init__(self,keys):
        self.keys = keys

    def __call__(self, data):
        d = dict(data)
        #image = d[self.keys[0]]
        #label = d[self.keys[1]]
        image = d['image']
        label = d['label']
        label = label.squeeze(0)
        annotations = np.zeros((1, 7))
        annotation = mask2boundingbox(label)
        if annotation == 0:
            annotation = annotations
            raise ValueError('Dataloader data no annotations')
            #print("Dataloader data no annotations")
        else:
            # add class label
            cls = d['class']
            annotation = np.array(annotation)
            annotation = np.append(annotation, cls)
            #annotation = np.expand_dims(annotation,0)
        #print(annotation.shape)
        #print(image.shape)
        d['label'] = annotation
        return d

def mask2boundingbox(label):
    if torch.is_tensor(label):
        label = label.numpy()   
    sk_mask = sk_label(label) 
    regions = sk_regions(label.astype(np.uint8))
    #global top, left, low, bottom, right, height 
    #print(regions)
    # check regions is empty
    if not regions:
        return 0

    for region in regions:
        # print('[INFO]bbox: ', region.bbox)
        # region.bbox (x1,y1,z1,x2,y2,z2)
        # top, left, low, bottom, right, height = region.bbox
        y1, x1, z1, y2, x2, z2 = region.bbox
   # return left, top, right, bottom, low, height
    return x1, y1, x2, y2, z1, z2

#-------- Running setting -------- 
'''
def adjust_learning_rate_by_step(optimizer, epoch, init_lr, decay_rate=.5 ,lr_decay_epoch=40):
    #Sets the learning rate to initial LR decayed by e^(-0.1*epochs)
    lr = init_lr * (decay_rate ** (epoch // lr_decay_epoch))
    for param_group in optimizer.param_groups:
        #param_group['lr'] =  param_group['lr'] * math.exp(-decay_rate*epoch)
        param_group['lr'] = lr
        #lr = init_lr * (0.1**(epoch // lr_decay_epoch))
    #print('LR is set to {}'.format(param_group['lr']))
    return optimizer , lr

def adjust_learning_rate(optimizer, epoch, init_lr, decay_rate=.5):
    #Sets the learning rate to initial LR decayed by e^(-0.1*epochs)
    lr = init_lr * decay_rate 
    for param_group in optimizer.param_groups:
        #param_group['lr'] =  param_group['lr'] * math.exp(-decay_rate*epoch)
        param_group['lr'] = lr
        #lr = init_lr * (0.1**(epoch // lr_decay_epoch))
    #print('LR is set to {}'.format(param_group['lr']))
    return optimizer , lr
'''
# RSNA Weighted Mean score
# Editting for list not df
def normalize_probabilities_to_one(df: pd.DataFrame, group_columns: list) -> pd.DataFrame:
    # Normalize the sum of each row's probabilities to 100%.
    # 0.75, 0.75 => 0.5, 0.5
    # 0.1, 0.1 => 0.5, 0.5
    row_totals = df[group_columns].sum(axis=1)
    if row_totals.min() == 0:
        raise ParticipantVisibleError('All rows must contain at least one non-zero prediction')
    for col in group_columns:
        df[col] /= row_totals
    return df

def create_training_solution(y_train):
    sol_train = y_train.copy()
    
    # bowel healthy|injury sample weight = 1|2
    #sol_train['bowel_weight'] = np.where(sol_train['bowel_injury'] == 1, 2, 1)
    
    # extravasation healthy/injury sample weight = 1|6
    #sol_train['extravasation_weight'] = np.where(sol_train['extravasation_injury'] == 1, 6, 1)
    
    # kidney healthy|low|high sample weight = 1|2|4
    sol_train['kidney_weight'] = np.where(sol_train['kidney_low'] == 1, 2, np.where(sol_train['kidney_high'] == 1, 4, 1))
    
    # liver healthy|low|high sample weight = 1|2|4
    sol_train['liver_weight'] = np.where(sol_train['liver_low'] == 1, 2, np.where(sol_train['liver_high'] == 1, 4, 1))
    
    # spleen healthy|low|high sample weight = 1|2|4
    sol_train['spleen_weight'] = np.where(sol_train['spleen_low'] == 1, 2, np.where(sol_train['spleen_high'] == 1, 4, 1))
    
    # any healthy|injury sample weight = 1|6
    sol_train['any_injury_weight'] = np.where(sol_train['any_injury'] == 1, 6, 1)
    return sol_train

def rsna_score_cal(solution: pd.DataFrame, submission: pd.DataFrame) -> float:
    '''
    Pseudocode:
    1. For every label group (liver, bowel, etc):
        - Normalize the sum of each row's probabilities to 100%.
        - Calculate the sample weighted log loss.
    2. Derive a new any_injury label by taking the max of 1 - p(healthy) for each label group
    3. Calculate the sample weighted log loss for the new label group
    4. Return the average of all of the label group log losses as the final score.
    '''

    # # Run basic QC checks on the inputs
    # if not pandas.api.types.is_numeric_dtype(submission.values):
    #     raise ParticipantVisibleError('All submission values must be numeric')

    # if not np.isfinite(submission.values).all():
    #     raise ParticipantVisibleError('All submission values must be finite')

    # if solution.min().min() < 0:
    #     raise ParticipantVisibleError('All labels must be at least zero')
    # if submission.min().min() < 0:
    #     raise ParticipantVisibleError('All predictions must be at least zero')

    # Calculate the label group log losses

    triple_level_targets = ['kidney', 'liver', 'spleen']
    all_target_categories = triple_level_targets

    # smaple weight
    scale_by_2 = ['kidney_low','liver_low','spleen_low']
    scale_by_4 = ['kidney_high','liver_high','spleen_high']
    sf_2 = 2
    sf_4 = 4
    sf_6 = 6
    submission[scale_by_2] *= sf_2
    submission[scale_by_4] *= sf_4

    # log loss weight
    solution = create_training_solution(solution)
    label_group_losses = []
    for category in all_target_categories:
        col_group = [f'{category}_healthy', f'{category}_low', f'{category}_high']

        solution = normalize_probabilities_to_one(solution, col_group)

        for col in col_group:
            if col not in submission.columns:
                raise ParticipantVisibleError(f'Missing submission column {col}')

        submission = normalize_probabilities_to_one(submission, col_group)
        label_group_losses.append(
            log_loss(
                y_true=solution[col_group].values,
                y_pred=submission[col_group].values,
                sample_weight=solution[f'{category}_weight'].values
            )
        )
    # Derive a new any_injury label by taking the max of 1 - p(healthy) for each label group
    healthy_cols = [x + '_healthy' for x in all_target_categories]
    any_injury_labels = (1 - solution[healthy_cols]).max(axis=1)
    # any_injury is sf_6
    any_injury_predictions = (1 - submission[healthy_cols]).max(axis=1)
    any_injury_loss = log_loss(
        y_true=any_injury_labels.values,
        y_pred=any_injury_predictions.values,
        sample_weight=solution['any_injury_weight'].values
    )
    label_group_losses.append(any_injury_loss)
    return np.mean(label_group_losses)

def train(model, device, data_num, epochs, optimizer, loss_function, train_loader, valid_loader, early_stop, scheduler, check_path):
    # Let ini config file can be writted
    #global best_metric
    #global best_metric_epoch
    #val_interval = 2
    best_metric = -1
    best_metric_epoch = -1
    trigger_times = 0
    if early_stop == 0:
        early_stop = None
    #epoch_loss_values = list()
    
    writer = SummaryWriter()
    for epoch in range(epochs):
        print("-" * 10)
        print(f"epoch {epoch + 1}/{epochs}")
        # record ram memory used
        process = psutil.Process(os.getpid())
        print(f'RAM used:{process.memory_info().rss/ 1024 ** 3} GB')
        model.train()
        epoch_loss = 0
        step = 0
        first_start_time = time.time()
        for batch_data in train_loader:
            step += 1
            if batch_data['label'].dim() == 1:
                labels = batch_data['label'].long().to(device)
            else:
                labels = batch_data['label'].float().to(device)
            inputs = batch_data['image'].to(device)
            if "bbox" in batch_data:
                bboxs = batch_data['bbox'].to(device)

            optimizer.zero_grad()
            #inputs, labels = Variable(inputs), Variable(labels)
            if "bbox" in batch_data:
                outputs = model(bboxs, inputs)
            else:
                outputs = model(inputs)
            # print(f'outputs:{outputs.size()}')
            # print(f'labels:{labels.size()}')
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            epoch_len = data_num // train_loader.batch_size
            print(f"{step}/{epoch_len}, train_loss: {loss.item():.4f}")
            writer.add_scalar("train_loss", loss.item(), epoch_len * epoch + step)
        epoch_loss /= step
        config.epoch_loss_values.append(epoch_loss)
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")
        final_end_time = time.time()
        hours, rem = divmod(final_end_time-first_start_time, 3600)
        minutes, seconds = divmod(rem, 60)
        print(f'one epoch runtime:{int(minutes)}:{seconds}')
        # Early stopping & save best weights by using validation
        metric = validation(model, valid_loader, device)
        scheduler.step(metric)

        # checkpoint setting
        if metric > best_metric:
            # reset trigger_times
            trigger_times = 0
            best_metric = metric
            best_metric_epoch = epoch + 1
            torch.save(model.state_dict(), f"{check_path}/{best_metric}.pth")
            print('trigger times:', trigger_times)
            print("saved new best metric model")
        else:
            trigger_times += 1
            print('trigger times:', trigger_times)
            # Save last 3 epoch weight
            if early_stop and early_stop - trigger_times <= 3 or epochs - epoch <= 3:
                torch.save(model.state_dict(), f"{check_path}/{metric}_last.pth")
                print("save last metric model")
        print(
            "current epoch: {} current accuracy: {:.4f} best accuracy: {:.4f} at epoch {}".format(
                epoch + 1, metric, best_metric, best_metric_epoch
            )
        )
        writer.add_scalar("val_accuracy", metric, epoch + 1)

        # early stop 
        if early_stop and trigger_times >= early_stop:
            print('Early stopping!\nStart to test process.')
            break
        
    print(f"train completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}")
    config.best_metric = best_metric
    config.best_metric_epoch = best_metric_epoch
    writer.close()
    #print(f'training_torch best_metric:{best_metric}',flush =True)
    #print(f'training_torch config.best_metric:{config.best_metric}',flush =True)
    return model

def train_mul_fpn(model, device, data_num, epochs, optimizer, loss_function, train_loader, 
    valid_loader, early_stop, scheduler, check_path, fpn_type='concat', eval_score='total_score'):
    # Let ini config file can be writted
    # fpn loss 分成 concat, softmax, individual
    best_metric_epoch = -1
    trigger_times = 0
    num_correct = 0
    if early_stop == 0:
        early_stop = None
    #epoch_loss_values = list()
    # 混合精度學習
    scaler = torch.cuda.amp.GradScaler()
    writer = SummaryWriter()
    for epoch in range(epochs):
        print("-" * 10)
        print(f"epoch {epoch + 1}/{epochs}")
        # record ram memory used
        process = psutil.Process(os.getpid())
        print(f'RAM used:{process.memory_info().rss/ 1024 ** 3} GB')
        model.train()
        epoch_loss = 0
        step = 0
        total_train_acc = {"kid": 0, "liv": 0, "spl": 0}
        index_ranges = {"kid": (0, 3), "liv": (3, 6), "spl": (6, 9)}
        first_start_time = time.time()
        for batch_data in train_loader:
            step += 1
            # kidney_healthy,kidney_low,kidney_high,liver_healthy,liver_low,liver_high,spleen_healthy,spleen_low,spleen_high,healthy
            labels = batch_data['label'].float().to(device)
            input_liv, input_spl, input_kid_r, input_kid_l = batch_data['image_liv'].to(device), batch_data['image_spl'].to(device), \
                                                            batch_data['image_kid_r'].to(device), batch_data['image_kid_l'].to(device)                              
            input_kid = torch.cat((input_kid_r,input_kid_l), dim=-1)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                # FPN layer concate 分所有結果
                if fpn_type == 'concat':
                    outputs = model(input_liv, input_spl, input_kid)
                    # outputs = F.sigmoid(outputs)
                    loss = loss_function(outputs, labels[:-1])
                    # train accuracy
                    outputs = F.softmax(outputs, dim=1)
                    binary_predictions = (outputs > 0.5).float()
                    for prediction, ground_truth in zip(binary_predictions, labels[:,:-1]):
                        for part, (start_idx, end_idx) in index_ranges.items():
                            accuracy = calculate_multi_label_accuracy(prediction[start_idx:end_idx], ground_truth[start_idx:end_idx])
                            total_train_acc[part] += accuracy
                        num_correct += 1
                # FPN layer concate 分不同器官的結果
                elif fpn_type == 'split':
                    out_liv, out_spl, out_kid = model(input_liv, input_spl, input_kid)
                    # label分開
                    label_kid = labels[:,0:3]
                    label_spl = labels[:,3:6]
                    label_liv = labels[:,6:9]
                    loss_l = loss_function(out_liv, label_liv)
                    loss_s = loss_function(out_spl, label_spl)
                    loss_k = loss_function(out_kid, label_kid)
                    loss = loss_l + loss_s + loss_k
                    # train accuracy
                    outputs = torch.cat((out_kid,out_liv,out_spl), dim=1)
                    outputs = F.softmax(outputs, dim=1)
                    binary_predictions = (outputs > 0.5).float()
                    for prediction, ground_truth in zip(binary_predictions, labels[:,:-1]):
                        for part, (start_idx, end_idx) in index_ranges.items():
                            accuracy = calculate_multi_label_accuracy(prediction[start_idx:end_idx], ground_truth[start_idx:end_idx])
                            total_train_acc[part] += accuracy
                        num_correct += 1
                # 各FPN layer分開運算loss再相加
                elif fpn_type == 'individual':
                    fpn_layer2, fpn_layer3, fpn_layer4 = model(input_liv, input_spl, input_kid)
                    loss_2 = loss_function(fpn_layer2, labels)
                    loss_3 = loss_function(fpn_layer3, labels)
                    loss_4 = loss_function(fpn_layer4, labels)
                    loss = loss_2 + loss_3 + loss_4
                    # train accuracy
                    outputs = torch.cat((fpn_layer2, fpn_layer3, fpn_layer4), dim=1)
                    outputs = F.softmax(outputs, dim=1)
                    binary_predictions = (outputs > 0.5).float()
                    for prediction, ground_truth in zip(binary_predictions, labels[:,:-1]):
                        for part, (start_idx, end_idx) in index_ranges.items():
                            accuracy = calculate_multi_label_accuracy(prediction[start_idx:end_idx], ground_truth[start_idx:end_idx])
                            total_train_acc[part] += accuracy
                        num_correct += 1
            # loss.backward()
            # optimizer.step()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += loss.item()
            epoch_len = data_num // train_loader.batch_size
            print(f"{step}/{epoch_len}, train_loss: {loss.item():.4f}")
            writer.add_scalar("train_loss", loss.item(), epoch_len * epoch + step)
        epoch_loss /= step
        metric = {part: acc / num_correct for part, acc in total_train_acc.items()}
        config.epoch_loss_values.append(epoch_loss)
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")
        print(f'epoch {epoch + 1} average accuracy: {sum(metric.values()) / len(metric):.4f}')
        final_end_time = time.time()
        hours, rem = divmod(final_end_time-first_start_time, 3600)
        minutes, seconds = divmod(rem, 60)
        print(f'one epoch runtime:{int(minutes)}:{seconds}')
        # Early stopping & save best weights by using validation
        metric = valid_mul_fpn(model, valid_loader, device, eval_score)
        scheduler.step(metric)
        if eval_score == 'rsna_score':
            comparison_operator = lambda a, b: a < b
            best_metric = 1000
        else:
            comparison_operator = lambda a, b: a > b
            best_metric = -1
        # checkpoint setting
        if comparison_operator(metric, best_metric):
            # reset trigger_times
            trigger_times = 0
            best_metric = metric
            best_metric_epoch = epoch + 1
            torch.save(model.state_dict(), f"{check_path}/{best_metric}.pth")
            print('trigger times:', trigger_times)
            print("saved new best metric model")
        else:
            trigger_times += 1
            print('trigger times:', trigger_times)
            # Save last 3 epoch weight
            if early_stop and early_stop - trigger_times <= 3 or epochs - epoch <= 3:
                torch.save(model.state_dict(), f"{check_path}/{metric}_last.pth")
                print("save last metric model")
        print(
            "current epoch: {} current accuracy: {:.4f} best accuracy: {:.4f} at epoch {}".format(
                epoch + 1, metric, best_metric, best_metric_epoch
            )
        )
        writer.add_scalar("val_accuracy", metric, epoch + 1)

        # early stop 
        if early_stop and trigger_times >= early_stop:
            print('Early stopping!\nStart to test process.')
            break
        
    print(f"train completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}")
    config.best_metric = best_metric
    config.best_metric_epoch = best_metric_epoch
    writer.close()
    #print(f'training_torch best_metric:{best_metric}',flush =True)
    #print(f'training_torch config.best_metric:{config.best_metric}',flush =True)
    return model


def train_mul(model, device, data_num, epochs, optimizer, loss_function, train_loader, valid_loader, early_stop, scheduler, check_path, output_log):
    # Let ini config file can be writted
    #global best_metric
    #global best_metric_epoch
    #val_interval = 2
    best_metric = -1
    best_metric_epoch = -1
    trigger_times = 0
    if early_stop == 0:
        early_stop = None
    #epoch_loss_values = list()
    
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    try:
        with open(f'{output_log}.log', 'w') as f:
            # Redirect stdout to the file
            sys.stdout = f
            sys.stderr = f
            writer = SummaryWriter()
            for epoch in trange(epochs, desc="Epochs"):
            # for epoch in range(epochs):
                # print("-" * 10)
                # print(f"epoch {epoch + 1}/{epochs}")
                # record ram memory used
                # process = psutil.Process(os.getpid())
                # print(f'RAM used:{process.memory_info().rss/ 1024 ** 3} GB')
                model.train()
                epoch_loss = 0
                step = 0
                first_start_time = time.time()
                progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}", file=sys.stdout)
                for batch_data in progress_bar:
                    step += 1
                    inputs, labels = batch_data['image'].to(device), batch_data['label'].long().to(device)
                    if "bbox" in batch_data:
                        bboxs = batch_data['bbox'].to(device)

                    optimizer.zero_grad()
                    #inputs, labels = Variable(inputs), Variable(labels)
                    if "bbox" in batch_data:
                        outputs = model(bboxs, inputs)
                    else:
                        outputs = model(inputs)
                    # print(f'outputs:{outputs}')
                    # print(f'labels:{labels.size()}')
                    loss = loss_function(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()
                    epoch_len = data_num // train_loader.batch_size
                    print(f"{step}/{epoch_len}, train_loss: {loss.item():.4f}")
                    progress_bar.set_description(f"Epoch {epoch+1} (Loss: {loss.item():.4f})")
                    writer.add_scalar("train_loss", loss.item(), epoch_len * epoch + step)
                epoch_loss /= step
                config.epoch_loss_values.append(epoch_loss)
                # print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")
                final_end_time = time.time()
                hours, rem = divmod(final_end_time-first_start_time, 3600)
                minutes, seconds = divmod(rem, 60)
                # print(f'one epoch runtime:{int(minutes)}:{seconds}')
                # Early stopping & save best weights by using validation
                metric = validation(model, valid_loader, device)
                scheduler.step(metric)

                # checkpoint setting
                if metric > best_metric:
                    # reset trigger_times
                    trigger_times = 0
                    best_metric = metric
                    best_metric_epoch = epoch + 1
                    torch.save(model.state_dict(), f"{check_path}/{best_metric}.pth")
                    print('trigger times:', trigger_times)
                    print("saved new best metric model")
                else:
                    trigger_times += 1
                    print('trigger times:', trigger_times)
                    # Save last 3 epoch weight
                    if early_stop and early_stop - trigger_times <= 3 or epochs - epoch <= 3:
                        torch.save(model.state_dict(), f"{check_path}/{metric}_last.pth")
                        print("save last metric model")
                print(
                    "current epoch: {} current accuracy: {:.4f} best accuracy: {:.4f} at epoch {}".format(
                        epoch + 1, metric, best_metric, best_metric_epoch
                    )
                )
                writer.add_scalar("val_accuracy", metric, epoch + 1)

                # early stop 
                if early_stop and trigger_times >= early_stop:
                    print('Early stopping!\nStart to test process.')
                    break
                
            print(f"train completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}")
            config.best_metric = best_metric
            config.best_metric_epoch = best_metric_epoch
            writer.close()
    finally:
        # Restore the original stdout
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        return model
    #print(f'training_torch best_metric:{best_metric}',flush =True)
    #print(f'training_torch config.best_metric:{config.best_metric}',flush =True)
    return model

class AngleLoss_predict(nn.Module):
    def __init__(self, gamma=0):
        super(AngleLoss_predict, self).__init__()
        self.gamma = gamma
        self.it = 1
        self.LambdaMin = 5.0
        self.LambdaMax = 1500.0
        self.lamb = 1500.0

    def forward(self, input):
        cos_theta, phi_theta = input
        cos_theta = cos_theta.as_tensor()
        phi_theta = phi_theta.as_tensor()
        #cos_theta = torch.tensor(cos_theta,  requires_grad=True)
        #phi_theta = torch.tensor(phi_theta,  requires_grad=True)
        #target = target.view(-1, 1)  # size=(B,1)
        index = cos_theta.data * 0.0  # size=(B, Classnum)
        # index = index.scatter(1, target.data.view(-1, 1).long(), 1)
        #index = index.byte()
        index = index.bool()  
        index = Variable(index)
        # index = Variable(torch.randn(1,2)).byte()

        self.lamb = max(self.LambdaMin, self.LambdaMax / (1 + 0.1 * self.it))
        output = cos_theta * 1.0  # size=(B,Classnum)
        output1 = output.clone()
        # output1[index1] = output[index] - cos_theta[index] * (1.0 + 0) / (1 + self.lamb)
        # output1[index1] = output[index] + phi_theta[index] * (1.0 + 0) / (1 + self.lamb)
        output[index] = output1[index]- cos_theta[index] * (1.0 + 0) / (1 + self.lamb)+ phi_theta[index] * (1.0 + 0) / (1 + self.lamb)
        return(output)

def calculate_multi_label_accuracy(predictions, ground_truths):
    # Convert predictions to the same type as ground truths for comparison
    predictions = predictions.type_as(ground_truths)

    # Calculate correct predictions
    correct_predictions = (predictions == ground_truths).sum()

    # Total number of predictions
    total_predictions = ground_truths.numel()

    # Calculate accuracy
    accuracy = correct_predictions.float() / total_predictions

    return accuracy.item()
    
def valid_mul_fpn(model, val_loader, device, eval_score='total_acc'):
    #metric_values = list()
    model.eval()
    with torch.no_grad():
        num_correct = 0
        metric_count = 0
        total_labels = 0
        total_score = 0.0
        total_acc = {"kid": 0, "liv": 0, "spl": 0}
        index_ranges = {"kid": (0, 3), "liv": (3, 6), "spl": (6, 9)}
        column_names_sol = ['kidney_healthy', 'kidney_low', 'kidney_high',
                        'liver_healthy', 'liver_low', 'liver_high',
                        'spleen_healthy', 'spleen_low', 'spleen_high','any_injury']
        # submit any_injury col is generate in rsna_score_cal
        column_names_sub = ['kidney_healthy', 'kidney_low', 'kidney_high',
                        'liver_healthy', 'liver_low', 'liver_high',
                        'spleen_healthy', 'spleen_low', 'spleen_high']
        rsna_solution_df = pd.DataFrame(columns=column_names_sol)
        rsna_submission_df = pd.DataFrame(columns=column_names_sub)
        for val_data in val_loader:
            input_liv, input_spl, input_kid_r, input_kid_l = val_data['image_liv'].to(device), val_data['image_spl'].to(device), \
                                                val_data['image_kid_r'].to(device), val_data['image_kid_l'].to(device)
            val_labels = val_data['label'].to(device)
            input_kid = torch.cat((input_kid_r,input_kid_l), dim=-1)
            val_outputs = model(input_liv, input_spl, input_kid)
            # 先前沒有經過softmax，因此valid記得做，避免後續預測有問題
            
            # 確認output是否為器官預測分開的結果
            if isinstance(val_outputs, tuple):
                val_outputs = [F.softmax(tensor, dim=1) for tensor in val_outputs]
                # kidney, liver, spleen
                val_outputs = torch.cat((val_outputs[2], val_outputs[0], val_outputs[1]), dim=1)
            else:
                val_outputs = F.softmax(val_outputs, dim=1)
            # RSNA score 
            # log loss需要全部一起看，先建立df
            sol_tmp = pd.DataFrame(val_labels.cpu().numpy(), columns=column_names_sol)
            sub_tmp = pd.DataFrame(val_outputs.cpu().numpy(), columns=column_names_sub)
            rsna_solution_df = pd.concat([rsna_solution_df, sol_tmp], ignore_index=True)
            rsna_submission_df = pd.concat([rsna_submission_df, sub_tmp], ignore_index=True)
            # 根據每個標籤預測正確的比例
            binary_predictions = (val_outputs > 0.5).float()
            for prediction, ground_truth in zip(binary_predictions, val_labels[:,:-1]):
                for part, (start_idx, end_idx) in index_ranges.items():
                    accuracy = calculate_multi_label_accuracy(prediction[start_idx:end_idx], ground_truth[start_idx:end_idx])
                    total_acc[part] += accuracy
                num_correct += 1
            # 計算標籤完全正確的比例
            val_labels = val_labels[:,:-1].cpu().numpy()
            binary_predictions = binary_predictions.cpu().numpy()
            score = accuracy_score(val_labels, binary_predictions)
            total_score += score
            total_labels += 1
        # transfer df to numeric
        rsna_solution_df[column_names_sol] = rsna_solution_df[column_names_sol].apply(pd.to_numeric)
        rsna_submission_df[column_names_sub] = rsna_submission_df[column_names_sub].apply(pd.to_numeric)
        rsna_score = rsna_score_cal(rsna_solution_df, rsna_submission_df)
        metric = {part: acc / num_correct for part, acc in total_acc.items()}
        score = total_score / total_labels
        config.metric_values.append(sum(metric.values()) / len(metric))

        
        print(f'validation kid acc:{metric["kid"]}, liv acc:{metric["liv"]}, spl acc:{metric["spl"]}',flush =True)
        print(f'validation total acc:{score}',flush =True)
        print(f'validation rsna:{rsna_score}',flush =True)
        #print(f'validation metric:{config.metric_values}',flush =True)
    if eval_score == 'total_acc':
        return score
    elif eval_score == 'acc':
        return sum(metric.values()) / len(metric)
    elif eval_score =='rsna_score':
        return rsna_score


def validation(model, val_loader, device):
    #metric_values = list()
    model.eval()
    with torch.no_grad():
        num_correct = 0.0
        metric_count = 0
        total_labels = 0
        num_correct_labels = 0.0
        for val_data in val_loader:
            val_images, val_labels = val_data['image'].to(device), val_data['label'].to(device)
            if "bbox" in val_data:
                bboxs = val_data['bbox'].to(device)
                val_outputs = model(bboxs, val_images)
            else:
                val_outputs = model(val_images)
            # print(val_outputs.size())
            # base on AngleLoss
            if isinstance(val_outputs, tuple):
                val_outputs = AngleLoss_predict()(val_outputs)
            if val_outputs.size(1) == 2:
                value = torch.eq(val_outputs.argmax(dim=1), val_labels)
                num_correct += value.sum().item()
            else:
                # 将预测分数转换为二进制标签，例如，通过应用阈值
                binary_predictions = (val_outputs > 0.5).float()
                correct_predictions = torch.eq(binary_predictions, val_labels)
                # 计算每个样本的所有标签是否都被正确预测
                all_correct_per_sample = torch.all(correct_predictions, dim=1)
                # 计算正确分类的标签数
                num_correct_labels = correct_predictions.sum().item()
                total_labels = torch.numel(val_labels)
                # 计算正确分类的样本数
                num_correct += all_correct_per_sample.sum().item()
            metric_count += val_outputs.size(0)
        # if 'total_labels' in locals():
        if total_labels !=0:
            label_accuracy = num_correct_labels / total_labels
            print(f'validation num_correct_labels:{label_accuracy}',flush =True)
        metric = num_correct / metric_count
        config.metric_values.append(metric)
        #print(f'validation metric:{config.metric_values}',flush =True)
    return metric

def plot_loss_metric(epoch_loss_values, metric_values,save_path):
    plt.figure("train", (12, 6))
    plt.subplot(1, 2, 1)
    plt.title("Epoch Average Loss")
    x = [i + 1 for i in range(len(epoch_loss_values))]
    y = epoch_loss_values
    plt.xlabel("epoch")
    plt.plot(x, y)
    plt.subplot(1, 2, 2)
    plt.title("Val Accuracy")
    x = [i + 1 for i in range(len(metric_values))]
    y = metric_values
    plt.xlabel("epoch")
    plt.plot(x, y)
    plt.savefig(f'{save_path}/train_loss_metric.png')

def kfold_split(file, kfold, seed, type, fold):
    if type == 'pos':
        d = {}
        file_list = ['file']
        file_list.extend([f'pos_split_df_{i}' for i in range(kfold)])
        d['file'] = file
        for i in range(kfold):
            d[f'test_pos_df_{i}'] = d[file_list[i]].groupby(["gender","age_range","spleen_injury_class"],group_keys=False).apply(lambda x: x.sample(frac=1/(kfold-i),random_state=1))
            d[f'pos_split_df_{i}'] = d[file_list[i]].drop(d[f'test_pos_df_{i}'].index.to_list())
        output_file = d[f'test_pos_df_{fold}']

    elif type == 'neg':
        file_list = [f'neg_split_df_{i}' for i in range(kfold)]
        file_list = np.array_split(file.sample(frac=1,random_state=seed), kfold)
        output_file = file_list[fold]
        
    return output_file

def Data_progressing(pos_file, neg_file, box_df, imbalance_data_ratio, data_split_ratio, seed, fold, save_file = False, cropping = True):
    # Pos data progress
    for index, row in pos_file.iterrows():
        if row['OIS']==row['OIS']:
            pos_file.loc[index,'spleen_injury_grade'] = row['OIS']
        else:
            pos_file.loc[index,'spleen_injury_grade'] = row['R_check']

    new_col= 'age_range'
    new_col_2 = 'spleen_injury_class'
    bins = [0,30,100]
    bins_2 = [0,2,5]
    label_2 = ['OIS 1,2','OIS 3,4,5']
    pos_file[new_col] = pd.cut(x=pos_file.age, bins=bins)
    pos_file[new_col_2] = pd.cut(x=pos_file.spleen_injury_grade, bins=bins_2, labels=label_2)

    # positive need select column and split in kfold 
    test_pos_df = kfold_split(pos_file, int(1/data_split_ratio[2]), seed, 'pos', fold)
    train_pos_file = pos_file.drop(test_pos_df.index.to_list())
    valid_pos_df = train_pos_file.groupby(['gender','age_range','spleen_injury_class'],group_keys=False).apply(lambda x: x.sample(frac=data_split_ratio[1]/(1-data_split_ratio[2]),random_state=seed))
    train_pos_df = train_pos_file.drop(valid_pos_df.index.to_list())
    
    # negative only need split in kfold 
    neg_sel_df = neg_file.sample(n=len(pos_file),random_state=seed)
    test_neg_df =  kfold_split(neg_sel_df, int(1/data_split_ratio[2]), seed, 'neg', fold)
    train_neg_file = neg_file.drop(test_neg_df.index.to_list())
    valid_neg_df = train_neg_file.sample(n=len(valid_pos_df),random_state=seed)
    train_neg_df = train_neg_file.drop(valid_neg_df.index.to_list()).sample(n=len(train_pos_df)*imbalance_data_ratio,random_state=seed)

    train_df = pd.concat([train_neg_df,train_pos_df])
    valid_df = pd.concat([valid_neg_df,valid_pos_df])
    test_df = pd.concat([test_neg_df,test_pos_df])

    train_data = box_df[box_df.Path.isin(train_df.source.to_list())]
    valid_data = box_df[box_df.Path.isin(valid_df.source.to_list())]
    test_data = box_df[box_df.Path.isin(test_df.source.to_list())]

    train_df['spleen_injury'] = np.array([0 if i else 1 for i in train_df.spleen_injury_class.isna().tolist()])
    valid_df['spleen_injury'] = np.array([0 if i else 1 for i in valid_df.spleen_injury_class.isna().tolist()])
    test_df['spleen_injury'] = np.array([0 if i else 1 for i in test_df.spleen_injury_class.isna().tolist()])

    if save_file:
        test_df_output = pd.merge(test_data.loc[:,['ID','Path','BBox','Posibility']],test_df,left_on='Path',right_on='source',suffixes = ['','_x'])
        valid_df_output = pd.merge(test_data.loc[:,['ID','Path','BBox','Posibility']],test_df,left_on='Path',right_on='source',suffixes = ['','_x'])
        test_df_output = test_df_output.drop(['ID_x'],axis=1)
        valid_df_output = valid_df_output.drop(['ID_x'],axis=1)
        test_df_output = test_df_output.loc[:,test_df_output.columns[~test_df_output.columns.str.contains('Unnamed')]]
        valid_df_output = valid_df_output.loc[:,valid_df_output.columns[~valid_df_output.columns.str.contains('Unnamed')]]
        valid_df_output.to_csv(f'{save_file}/fold{fold}_valid.csv',index = False)
        test_df_output.to_csv(f'{save_file}/fold{fold}_test.csv',index = False)

    if cropping:
        train_data_dicts = []
        for index,row in train_data.iterrows():
            image = row['Path']
            label = row['BBox']
            train_data_dicts.append({'image':image,'label':label})
        valid_data_dicts = []
        for index,row in valid_data.iterrows():
            image = row['Path']
            label = row['BBox']
            valid_data_dicts.append({'image':image,'label':label})
        test_data_dicts = []
        for index,row in test_data.iterrows():
            image = row['Path']
            label = row['BBox']
            test_data_dicts.append({'image':image,'label':label})
    else:
        train_data_dicts =[
            {"image": image_name, "label": label_name}
            for image_name, label_name in zip([i for i in train_df.source.tolist()], [i for i in train_df.spleen_injury.tolist()] )
        ]
        valid_data_dicts =[
            {"image": image_name, "label": label_name}
            for image_name, label_name in zip([i for i in valid_df_output.source.tolist()], [i for i in valid_df_output.spleen_injury.tolist()] )
        ]
        test_data_dicts =[
            {"image": image_name, "label": label_name}
            for image_name, label_name in zip([i for i in test_df_output.source.tolist()], [i for i in test_df_output.spleen_injury.tolist()] )
        ]

    
    return train_data_dicts, valid_data_dicts, test_data_dicts

class FocalLoss(nn.Module):
    def __init__(self, class_num, alpha=None, gamma=2, size_average=True, weight=None):
        """
        focal_loss损失函数, -α(1-yi)**γ *ce_loss(xi,yi)      
        步骤详细的实现了 focal_loss损失函数.
        :param alpha:   阿尔法α,类别权重. 当α是列表时,为各类别权重,当α为常数时,类别权重为[α, 1-α, 1-α, ....],常用于 目标检测算法中抑制背景类 , retainnet中设置为0.25
        :param gamma:   伽马γ,难易样本调节参数. retainnet中设置为2
        :param num_classes:     类别数量
        :param size_average:    损失计算方式,默认取均值
        """
        super(FocalLoss, self).__init__()
        # if alpha is None:  # alpha 是平衡因子
        #     self.alpha = Variable(torch.ones(class_num, 1))
        # else:
        #     if isinstance(alpha, list):
        #         self.alpha = torch.Tensor(alpha)
        #     else:
        #         self.alpha = torch.zeros(class_num)
        #         self.alpha[0] += alpha
        #         self.alpha[1:] += (1-alpha)
        self.alpha = alpha
        self.gamma = gamma  # 指数
        self.class_num = class_num  # 类别数目
        self.size_average = size_average  # 返回的loss是否需要mean一下
        self.weight = weight
        self.use_softmax = use_softmax
        
    def forward(self, preds, labels):
        """
        focal_loss损失计算        
        :param preds:   预测类别. size:[B,N,C] or [B,C]  分别对应与检测与分类任务, B 批次, N检测框数, C类别数        
        :param labels:  实际类别. size:[B,N] or [B]
        :return:
        """        
        # assert preds.dim()==2 and labels.dim()==1        
        preds = preds.view(-1,preds.size(-1))        
        if self.use_softmax:
            loss = softmax_focal_loss(preds, labels, self.gamma, self.alpha)
        else:
            loss = sigmoid_focal_loss(preds, labels, self.gamma, self.alpha)

        if self.weight is not None:
            class_weight: Optional[torch.Tensor] = None
            num_of_classes = labels.shape[1]
            # 對損失進行加權
            if isinstance(self.weight, (float, int)):
                class_weight = torch.as_tensor([self.weight] * num_of_classes)
            else:
                class_weight = torch.as_tensor(self.weight)
                        # apply class_weight to loss
            class_weight = class_weight.to(loss)
            broadcast_dims = [-1] + [1] * len(labels.shape[2:])
            class_weight = class_weight.view(broadcast_dims)
            loss = class_weight * loss

        if self.size_average:        
            loss = loss.mean()        
        else:
            loss = loss.mean(dim=list(range(2, len(labels.shape))))            
            loss = loss.sum()        
        
        return loss
    
    def softmax_focal_loss(preds: torch.Tensor, labels: torch.Tensor, gamma: float = 2.0, alpha: Optional[float] = None):
        input_ls = preds.log_softmax(1)
        loss= -(1 - input_ls.exp()).pow(gamma) * input_ls * labels

        if alpha is not None:
            # (1-alpha) for the background class and alpha for the other classes
            alpha_fac = torch.tensor([1 - alpha] + [alpha] * (labels.shape[1] - 1)).to(loss)
            broadcast_dims = [-1] + [1] * len(labels.shape[2:])
            alpha_fac = alpha_fac.view(broadcast_dims)
            loss = alpha_fac * loss

        return loss

    def sigmoid_focal_loss(preds: torch.Tensor, labels: torch.Tensor, gamma: float = 2.0, alpha: Optional[float] = None):
        max_val = (-preds).clamp(min=0)
        loss= preds - preds * labels + max_val + ((-max_val).exp() + (-preds - max_val).exp()).log()
        invprobs = F.logsigmoid(-preds * (labels * 2 - 1))  # reduced chance of overflow
        loss = (invprobs * gamma).exp() * loss

        if alpha is not None:
            # alpha if t==1; (1-alpha) if t==0
            alpha_factor = target * alpha + (1 - target) * (1 - alpha)
            loss = alpha_factor * loss

        return loss

class ChannelAttentionModule(nn.Module):
    def __init__(self, channel, ratio=4):
        super(ChannelAttentionModule, self).__init__()
        #使用自适应池化缩减map的大小，保持通道不变
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)
        self.shared_MLP = nn.Sequential(
            nn.Conv3d(channel, channel // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv3d(channel // ratio, channel, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avgout = self.shared_MLP(self.avg_pool(x))
        maxout = self.shared_MLP(self.max_pool(x))
        return self.sigmoid(avgout + maxout)
    
class SpatialAttentionModule(nn.Module):
    def __init__(self):
        super(SpatialAttentionModule, self).__init__()
        self.conv3d = nn.Conv3d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        #map尺寸不变，缩减通道
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avgout, maxout], dim=1)
        out = self.sigmoid(self.conv3d(out))
        return out

class CBAM(nn.Module):
    def __init__(self, channel, ratio = 4):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttentionModule(channel, ratio)
        self.spatial_attention = SpatialAttentionModule()
        
    def forward(self, x):
        out = self.channel_attention(x) * x
        out = self.spatial_attention(out) * out
        return out

class Multiply(nn.Module):
    def forward(self, x, y):
        return x * y

class AttentionModel(nn.Module):
    def __init__(self, num_classes, size, model, model_name):
        super(AttentionModel, self).__init__()
        
        # Load the pre-trained EfficientNet model
        # efficient 1280
        # densenet 1536
        # cbam 512
        self.classification = model
        self.multiply = Multiply()

        if model_name == 'densenet':
            block_num = 1536
        elif model_name == 'efficientnet':
            block_num = 1280
        elif model_name == 'CBAM':
            block_num = 512

        # Add a linear layer for classification
        self.fc = nn.Linear(block_num, num_classes)
        
        # Add a convolutional layer and a linear layer for the attention mask
        # 這邊padding可以改 讓size先變小，block_num 就可以變大
        self.att_conv = nn.Conv3d(1, 16, kernel_size=3, padding=1)
        self.max_pool = nn.MaxPool3d(kernel_size=(8,8,8), stride=(8,8,8))
        self.last_convd = nn.Conv3d(16, block_num, kernel_size=3, padding=1)
        self.max_pool_last = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        
        # CBAM 
        self.cbam = CBAM(16)

        # Initialize att_features as None
        self.att_features = None
        
    def forward(self, x, mask):
        # Pass the input image through the EfficientNet model
        features = self.classification.extract_features(x) # num, 8, 8, 4
        # Compute the attention mask using the convolutional and linear layers
        att_mask = F.relu(self.att_conv(mask))
        #att_mask = self.cbam(att_mask) * att_mask
        att_mask = self.cbam(att_mask)
        att_mask = self.max_pool(att_mask) #　16, 16, 16, 8
        att_mask = F.relu(self.last_convd(att_mask))
        att_mask = self.max_pool_last(att_mask)

        # att_mask = F.relu(self.att_conv(mask))
        # att_mask = self.max_pool(att_mask)
        # att_mask = att_mask.view(att_mask.size(0), -1)
        # att_mask = F.relu(self.att_fc(att_mask))
        # att_mask = att_mask.view(att_mask.size(0), self.block_num, 1, 1, 1)
        
        # Apply the attention mask to the features using element-wise multiplication
        self.att_features = self.multiply(features, att_mask)
        #self.att_features = torch.cat((features, att_mask), dim= 0 )

        
        # Compute the logits for the classification task using the modified features
        logits = self.fc(self.att_features.mean([2, 3, 4]))
        
        return logits


class AttentionModel_new(nn.Module):
    def __init__(self, num_classes, size, model, model_att, model_name):
        super(AttentionModel_new, self).__init__()
        # densenet_feature_extractor = torch.nn.Sequential(*list(dense.children())[:-1])
        # Load the pre-trained EfficientNet model
        # efficient 1280
        # densenet 1536
        # cbam 512
        self.classification = model
        # 添加自定義Multiply模組
        self.multiply = Multiply()

        if model_name == 'densenet':
            block_num = 1536
        elif model_name == 'efficientnet':
            block_num = 1280
        elif model_name == 'CBAM':
            block_num = 512
            
        # Add a linear layer for classification
        self.fc = nn.Linear(block_num, num_classes)
        
        # Add a convolutional layer and a linear layer for the attention mask
#         self.att_conv = nn.Conv3d(1, 16, kernel_size=3, padding=1)
#         self.max_pool = nn.MaxPool3d(kernel_size=(8,8,8), stride=(8,8,8))
        self.last_convd = nn.Conv3d(896, block_num, kernel_size=3, padding=1)
        # CBAM 
        #self.cbam = CBAM(16)
        # Initialize att_features as None
        self.att_features = None
        self.att_block = nn.Sequential(*list(model_att.children())[:-1])
        
    def forward(self, x, mask):
        # Pass the input image through the EfficientNet model
        features = self.classification.extract_features(x)
        # Attention block
        att_mask = self.att_block(mask)
        att_mask = F.relu(self.last_convd(att_mask))
        
        # 使用Multiply模組計算att_features
        self.att_features = self.multiply(features, att_mask)
        # Compute the logits for the classification task using the modified features
        logits = self.fc(self.att_features.mean([2, 3, 4]))
        
        return logits