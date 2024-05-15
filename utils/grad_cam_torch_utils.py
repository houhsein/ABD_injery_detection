from skimage.transform import resize
from scipy.ndimage import zoom
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable
import torch.nn.functional as F
from utils.training_torch_utils import(
    calculate_multi_label_accuracy,
    rsna_score_cal
)
from sklearn.metrics import roc_curve, auc, accuracy_score, log_loss
from torch import nn
import os
from sklearn import metrics
from sklearn.utils import resample  
import scipy 
import pandas as pd
import zipfile
from scipy.stats import norm

'''
參考https://github.com/yizt/Grad-CAM.pytorch/tree/master/detection
'''
class Backup(object):
    '''
    Backup ori image for gradcam output
    '''
    def __init__(self, keys):
        self.keys = keys

    def __call__(self, data):
        d = dict(data)
        image = d['image']
        d['ori_image'] = image
        return d


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

# test prediction
def test(model, testLoader, device):
    pre_first = True
    model.eval()
    with torch.no_grad():
        num_correct = 0.0
        metric_count = 0
        for testdata in testLoader:
            test_images, test_labels = testdata['image'].to(device), testdata['label'].to(device)
            if "bbox" in testdata:
                bboxs = testdata['bbox'].to(device)
                output = model(bboxs, test_images)
            else:
                output = model(test_images)
            if isinstance(output, tuple):
                output = AngleLoss_predict()(output)
            # output 並非0-1之間 故進行轉換
            pre = nn.functional.softmax(output,dim=1).cpu().detach().numpy()
            # 比較output與label 對的話則返回 true 錯則 false
            value = torch.eq(output.argmax(dim=1), test_labels)
            metric_count += len(value)
            num_correct += value.sum().item()
            if pre_first:
                pre_first = None
                predict_values = pre
            else:
                predict_values = np.concatenate((predict_values,pre),axis=0)
        metric = num_correct / metric_count
        print("Test Accuracy: {}".format(num_correct / metric_count))
        return (predict_values)

def test_mul_fpn(model, testLoader, device):
    pre_first = True
    model.eval()
    with torch.no_grad():
        num_acc = 0
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

        for testdata in testLoader:
            input_liv, input_spl, input_kid_r, input_kid_l = testdata['image_liv'].to(device), testdata['image_spl'].to(device), \
                                                testdata['image_kid_r'].to(device), testdata['image_kid_l'].to(device)
            test_labels = testdata['label'].to(device)
            input_kid = torch.cat((input_kid_r,input_kid_l), dim=-1)
            output = model(input_liv, input_spl, input_kid)
            outputs = [F.softmax(tensor, dim=1) for tensor in output]
            predict_list = []
            for tensor in outputs:
                _, max_indices = torch.max(tensor, dim=1)
                output_tmp = torch.zeros_like(tensor)
                for i in range(tensor.size(0)):  # 遍历所有行
                    output_tmp[i, max_indices[i]] = 1
                predict_list.append(output_tmp)
            # kidney, liver, spleen
            predict = torch.cat((predict_list[2], predict_list[0], predict_list[1]), dim=1)
            outputs = torch.cat((outputs[2], outputs[0], outputs[1]), dim=1).cpu().detach().numpy()
            sol_tmp = pd.DataFrame(test_labels.cpu().numpy(), columns=column_names_sol)
            sub_tmp = pd.DataFrame(outputs, columns=column_names_sub)
            rsna_solution_df = pd.concat([rsna_solution_df, sol_tmp], ignore_index=True)
            rsna_submission_df = pd.concat([rsna_submission_df, sub_tmp], ignore_index=True)
            # 根據每個標籤預測正確的比例
            for prediction, ground_truth in zip(predict, test_labels[:,:-1]):
                for part, (start_idx, end_idx) in index_ranges.items():
                    accuracy = calculate_multi_label_accuracy(prediction[start_idx:end_idx], ground_truth[start_idx:end_idx])
                    total_acc[part] += accuracy
                num_acc += 1
            # 計算標籤完全正確的比例
            test_labels = test_labels[:,:-1].cpu().numpy()
            predict = predict.cpu().numpy()
            score = accuracy_score(test_labels, predict)
            total_score += score
            total_labels += 1
            if pre_first:
                pre_first = None
                predict_values = outputs
            else:
                predict_values = np.concatenate((predict_values, outputs),axis=0)
        # transfer df to numeric
        rsna_solution_df[column_names_sol] = rsna_solution_df[column_names_sol].apply(pd.to_numeric)
        rsna_submission_df[column_names_sub] = rsna_submission_df[column_names_sub].apply(pd.to_numeric)
        rsna_score = rsna_score_cal(rsna_solution_df, rsna_submission_df)
        metric = {part: acc / num_acc for part, acc in total_acc.items()}
        score = total_score / total_labels
        
        print(f'Test kid acc:{metric["kid"]}, liv acc:{metric["liv"]}, spl acc:{metric["spl"]}',flush =True)
        print(f'Test total acc:{score}',flush =True)
        print(f'Test rsna:{rsna_score}',flush =True)

        return (predict_values)

# inference 
def inference(model, testLoader, device):
    pre_first = True
    model.eval()
    with torch.no_grad():
        num_correct = 0.0
        metric_count = 0
        for testdata in testLoader:
            test_images = testdata['image'].to(device)
            if "bbox" in testdata:
                bboxs = testdata['bbox'].to(device)
                output = model(bboxs, test_images)
            else:
                output = model(test_images)
            if isinstance(output, tuple):
                output = AngleLoss_predict()(output)
            # output 並非0-1之間 故進行轉換
            pre = nn.functional.softmax(output, dim=1).cpu().detach().numpy()
            if pre_first:
                pre_first = None
                predict_values = pre
            else:
                predict_values = np.concatenate((predict_values,pre),axis=0)
        return (predict_values)

# plot confusion_plot
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range (cm.shape[0]):
        for j in range (cm.shape[1]):
            plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 fontsize =14,
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

# CI of confusion matrix
def wilson_binomial_confidence_interval(s, n, round_num=2, confidence_level=.95):
    '''
    Computes the binomial confidence interval of the probability of a success s, 
    based on the sample of n observations. The normal approximation is used,
    appropriate when n is equal to or greater than 30 observations.
    The confidence level is between 0 and 1, with default 0.95.
    Returns [p_estimate, interval_range, lower_bound, upper_bound].
    For reference, see Section 5.2 of Tom Mitchel's "Machine Learning" book.
    '''

    p_estimate = (1.0 * s) / n
    z = norm.interval(confidence_level)[1]
    
    wilson_p = (p_estimate + z**2/(2*n)) / (1 + z**2/n)
    
    wilson_interval_range = (z * np.sqrt( (p_estimate * (1-p_estimate))/n + z**2/(4*n**2) ) ) / (1 + z**2/n)
    
    interval_range =  z * np.sqrt( (p_estimate * (1-p_estimate))/n )
    output_p = f'%.{round_num}f'%(s/n)
    output_d = f'%.{round_num}f'%(wilson_p - wilson_interval_range)
    output_u = f'%.{round_num}f'%(wilson_p + wilson_interval_range)
    #return p_estimate, interval_range, p_estimate - interval_range, p_estimate + interval_range
    return f'{output_p}({output_d}-{output_u})'

def confusion_matrix_CI(tn, fp, fn, tp, round_num=2):
    acc = wilson_binomial_confidence_interval(tn+tp,tn+fp+fn+tp,round_num)
    PPV = wilson_binomial_confidence_interval(tp,tp+fp,round_num)
    NPV = wilson_binomial_confidence_interval(tn,fn+tn,round_num)
    Sensitivity = wilson_binomial_confidence_interval(tp,tp+fn,round_num)
    Specificity = wilson_binomial_confidence_interval(tn,fp+tn,round_num)
    return acc, PPV, NPV, Sensitivity, Specificity 

def Find_Optimal_Cutoff(target, predicted):
    """ Find the optimal probability cutoff point for a classification model related to event rate
    Parameters
    ----------
    target : Matrix with dependent or target data, where rows are observations

    predicted : Matrix with predicted data, where rows are observations

    Returns
    -------     
    list type, with optimal cutoff value
        
    """
    fpr, tpr, threshold = roc_curve(target, predicted)
    Youden_index = np.argmax(tpr - fpr)
    optimal_threshold = threshold[Youden_index]
    point = [fpr[Youden_index],tpr[Youden_index]]
    
    return optimal_threshold, point

def get_roc_CI(y_true, y_score):
#     roc_curves, auc_scores = zip(*Parallel(n_jobs=4)(delayed(bootstrap_func)(i, y_true, y_score) for i in range(1000)))
    roc_curves, auc_scores, aupr_scores = [], [], []
    for j in range(1000):
        yte_true_b, yte_pred_b = resample(y_true, y_score, replace=True, random_state=j)
        roc_curve = metrics.roc_curve(yte_true_b, yte_pred_b)
        auc_score = metrics.roc_auc_score(yte_true_b, yte_pred_b)
        aupr_score = metrics.auc(*metrics.precision_recall_curve(yte_true_b, yte_pred_b)[1::-1])

        roc_curves.append(roc_curve)
        auc_scores.append(auc_score)
        aupr_scores.append(aupr_score)

    #print('Test AUC: {:.3f}'.format(metrics.roc_auc_score(y_true, y_score)))
    #print('Test AUC: ({:.3f}, {:.3f}) percentile 95% CI'.format(np.percentile(auc_scores, 2.5), np.percentile(auc_scores, 97.5))) 
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    for fpr, tpr, _ in roc_curves:
        #print(scipy.interp(mean_fpr, fpr, tpr))
        tprs.append(scipy.interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        aucs.append(metrics.auc(fpr, tpr))
            
    mean_tpr = np.mean(tprs, axis=0)

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + 1.96 * std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - 1.96 * std_tpr, 0)
    return roc_curves, auc_scores, mean_fpr, tprs_lower, tprs_upper


def find_import_label(lst, tmp_pre):
    # 從high , low ,healthy依序確認，有過閾值就為標記
    # 若全部都沒過，則選最大值
    for i in range(len(lst) - 1, -1, -1):
        if lst[i] == 1:
            return i
    return np.argmax(tmp_pre) 

def multi_label_progress(arr, index_ranges, optimal_th_list=False):
    out_lst = []
    for i in range(arr.shape[0]):
        one_list = []
        for cls_type in ['kid','liv','spl']:
            if optimal_th_list:
                tmp_pre = arr[i][index_ranges[cls_type][0]:index_ranges[cls_type][1]]
                tmp_th = optimal_th_list[index_ranges[cls_type][0]//3]
                result = [1 if a > b else 0 for a, b in zip(tmp_pre, tmp_th)]
                one_list.append(find_import_label(result,tmp_pre))
            else:
                tmp_pre = arr[i][index_ranges[cls_type][0]:index_ranges[cls_type][1]]
                one_list.append(np.argmax(tmp_pre))
        out_lst.append(one_list)
    return out_lst

def plot_multi_class_roc(y_pre, y_label, n_classes, cls_type, dir_path, file_name):
    fig = plt.figure(figsize=(6, 6))
    lw = 2
    y_label = np.array(y_label)
    y_pre = np.array(y_pre)
    optimal_th_list = []
    # 計算每個類別的 ROC 曲線和 AUC
    for i in range(n_classes):
        # 將當前類別視為正類，其他所有類別視為負類
        y_true_binary = np.where(y_label == i, 1, 0)
        y_pre_binary = y_pre[:,i]
        fpr, tpr, _ = roc_curve(y_true_binary, y_pre_binary)
        roc_auc = auc(fpr, tpr)
        optimal_th, optimal_point = Find_Optimal_Cutoff(y_true_binary, y_pre_binary)
        optimal_th_list.append(optimal_th)
        roc_curves, auc_scores, mean_fpr, tprs_lower, tprs_upper = get_roc_CI(y_true_binary, y_pre_binary)

        # plt.plot(fpr, tpr, label=f'Class {i} (area = {roc_auc:.2f})')
        conf_int = ' ({:.3f}-{:.3f})'.format(np.percentile(auc_scores, 2.5), np.percentile(auc_scores, 97.5))
        plot_label = f'Class {i} AUC:{roc_auc:.3f}\n95% CI, {conf_int}'
        plt.plot(fpr, tpr, lw=lw, label=plot_label)
        plt.plot(optimal_point[0], optimal_point[1], marker = 'o', color='r')
        plt.text(optimal_point[0], optimal_point[1], f'Class {i} Threshold:{optimal_th:.3f}')
        plt.fill_between(mean_fpr, tprs_lower, tprs_upper, alpha=.1, color='b')
        
    ticks = np.linspace(0, 1, 11)
    plt.xticks(ticks)
    plt.yticks(ticks)
    plt.plot([0, 1], [0, 1], color='gray', lw=lw, linestyle='--')
    plt.xlim([-0.05, 1.0])
    plt.ylim([0.0, 1.05])
    plt.grid()
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC-AUC of {cls_type} for each class')
    plt.legend(loc="lower right")
    fig.savefig(f"{dir_path}/{file_name}_{cls_type}_roc.png")
    plt.close()
    # plt.show()

    return optimal_th_list

def plot_roc(y_pre, y_label, dir_path, file_name):
    scores = list()
    # 正樣本的數值輸出
    for i in range(y_pre.shape[0]):
        scores.append(y_pre[i][1])
    scores = np.array(scores)
    y_label = y_label.astype(int)
    fpr, tpr, _ = roc_curve(y_label, scores)
    roc_auc = auc(fpr, tpr)
    optimal_th, optimal_point = Find_Optimal_Cutoff(y_label, scores)
    roc_curves, auc_scores, mean_fpr, tprs_lower, tprs_upper = get_roc_CI(y_label, scores)
    fig = plt.figure(figsize=(6, 6))
    lw = 2
    conf_int = ' ({:.3f}-{:.3f})'.format(np.percentile(auc_scores, 2.5), np.percentile(auc_scores, 97.5))
    test = f'AUC:{roc_auc:.3f}\n95% CI, {conf_int}'
    plt.plot(fpr, tpr, lw=lw, color='k', label=test)
    plt.plot([0, 1], [0, 1], color='gray', lw=lw, linestyle='--')
    plt.plot(optimal_point[0], optimal_point[1], marker = 'o', color='r')
    plt.text(optimal_point[0], optimal_point[1], f'Threshold:{optimal_th:.3f}')
    ticks = np.linspace(0, 1, 11)
    plt.xticks(ticks)
    plt.yticks(ticks)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, alpha=.1, color='b')
    plt.xlim([-0.05, 1.0])
    plt.ylim([0.0, 1.05])
    plt.grid() # 網格
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC-AUC')
    plt.legend(loc="lower right")
    fig.savefig(f"{dir_path}/{file_name}_roc.png")
    plt.close()

    return optimal_th 

# Plot spleen injury grade 
def df_plot(df, dir_path, file_name, fold):
    correct_file = df[df['spleen_injury']==df['pre_label']]
    error_file = df[df['spleen_injury']!=df['pre_label']]
    error_file = error_file.groupby('spleen_injury_class')['spleen_injury_class'].size().reset_index(name='Error number')
    correct_file = correct_file.groupby('spleen_injury_class')['spleen_injury_class'].count().reset_index(name='Correct number')
    out_file=pd.merge(correct_file, error_file,on='spleen_injury_class').set_index('spleen_injury_class')
    plt.figure()            # 視窗名稱
    ax = plt.axes(frame_on=False)# 不要額外框線
    ax.xaxis.set_visible(False)  # 隱藏X軸刻度線
    ax.yaxis.set_visible(False)  # 隱藏Y軸刻度線
    pd.plotting.table(ax, out_file, loc='center') #將mytable投射到ax上，且放置於ax的中間
    #plt.show()
    plt.savefig(f'{dir_path}/{file_name}_{fold}_table.png')
    plt.close()

def plot_dis(pos_list, neg_list, dir_path, file_name):      
    plt.hist(pos_list, alpha=.5, label='Pos')
    plt.hist(neg_list, alpha=.5, label='Neg')
    plt.title("Data distributions")
    plt.legend(loc="lower right")
    plt.savefig(f"{dir_path}/{file_name}_dis.png")
    plt.close()

class GradCAM(object):
    """
    1: 网络不更新梯度,输入需要梯度更新
    2: 使用目标类别的得分做反向传播
    """

    def __init__(self, net, layer_name, device):
        self.net = net
        self.layer_name = layer_name
        self.feature = None
        self.gradient = None
        self.device = device
        # 使模型能夠計算gradient
        self.net.eval()
        # 儲存feature和gradient的list(hook資料型態)
        self.handlers = []
        # 將 feature與gradient 取出
        self._register_hook()

    def _get_features_hook(self, module, input, output):
        self.feature = output

    def _get_grads_hook(self, module, input_grad, output_grad):
        """
        :param input_grad: tuple,  input_grad[0]: None
                                   input_grad[1]: weight
                                   input_grad[2]: bias
        :param output_grad:tuple,  长度为1
        :return:
        """
        self.gradient = output_grad[0]

    # 利用hook取指定層的feature和gradient
    def _register_hook(self):
        if self.layer_name == 'multiply':
            self.handlers.append(self.net.multiply.register_forward_hook(self._get_features_hook))
            self.handlers.append(self.net.multiply.register_backward_hook(self._get_grads_hook))
        else:
            for (name, module) in self.net.named_modules():
                if name == self.layer_name:
                    #print("OK")
                    # forward取feature
                    self.handlers.append(module.register_forward_hook(self._get_features_hook))
                    # backward取gradient
                    self.handlers.append(module.register_backward_hook(self._get_grads_hook))
                #else:
                    #print("Nothing to do")
    # 每次計算完後就把結果刪除，避免memory不夠
    def remove_handlers(self):
        for handle in self.handlers:
            handle.remove()

    def __call__(self, inputs, bbox=False, index_sel=None, normalize=True):
        """
        :param inputs: {"image": [C,H,W], "height": height, "width": width}
        :param index_sel: list for label
        :return:  a list of each batch heatmap
        """
        # 將模型gradinet歸零 (應該可以不用)
        self.net.zero_grad()
        # inference取得output
        if torch.is_tensor(bbox):
            output = self.net(inputs, bbox)
        else:
            output = self.net(inputs) # [1,num_classes]
        if isinstance(output, tuple):
            output = AngleLoss_predict()(output)

        # 針對output準備取得feature與gradient    
        heatmap_list = []
        #print(f"output shape: {output.shape}")
        for i in range(output.shape[0]):
            # 如果沒有指定取得的標籤，預設則使用預測的標籤
            if  index_sel == None:
                index = np.argmax(output[i,:].cpu().data.numpy())
                print(f"predict:{index}")  
            else:
                index = int(index_sel[i])
                print(f"predict:{index}")
          
            # backward
            target = output[i][index]
            # 將backward的結果保留，才能取得gradient
            target.backward(retain_graph=True)
            #取得gradient和feature
            gradient = self.gradient[i].cpu().detach().data.numpy()  # [C,H,W,D]
            weight = np.mean(gradient, axis=(1, 2, 3))  # [C]

            feature = self.feature[i].cpu().detach().data.numpy()  # [C,H,W,D]
            
            # 計算grad cam方法
            # np.newaxis 增加維度的方法
            cam = feature * weight[:, np.newaxis, np.newaxis, np.newaxis]  # [C,H,W,D] feature map 與 weight相乘
            cam = np.sum(cam, axis=0)  # [H,W,D] 
            cam = np.maximum(cam, 0)  # ReLU
            # normalize 
            if normalize:
                heatmap = (cam - cam.min()) / (cam.max() - cam.min())
            else:
                heatmap = cam
            heatmap = resize(heatmap,inputs.shape[2:5])
            heatmap_list.append(heatmap)

        return heatmap_list

class LayerCAM(object):
    """
    1: 网络不更新梯度,输入需要梯度更新
    2: 使用目标类别的得分做反向传播
    """

    def __init__(self, net, layer_list, device):
        self.net = net
        self.layer_list = layer_list
        self.feature = []
        self.gradient = []
        self.device = device
        self.net.eval()
        self.handlers = []
        # 將 feature與gradient 取出
        self._register_hook()

    def _get_features_hook(self, module, input, output):
        feature = output
        self.feature.append(feature.cpu().detach())

    def _get_grads_hook(self, module, input_grad, output_grad):
        """
        :param input_grad: tuple,  input_grad[0]: None
                                   input_grad[1]: weight
                                   input_grad[2]: bias
        :param output_grad:tuple,  长度为1
        :return:
        """
        gradient = output_grad[0]
        self.gradient.append(gradient.cpu().detach())
        

    def _register_hook(self):
        # transfer the only one layer to list
        if type(self.layer_list) != list:
            self.layer_list = [self.layer_list]
        for (name, module) in self.net.named_modules():
            if name in self.layer_list:
                #print(f'{name} is select!')
                self.handlers.append(module.register_forward_hook(self._get_features_hook))
                self.handlers.append(module.register_backward_hook(self._get_grads_hook))

    def remove_handlers(self):
        for handle in self.handlers:
            handle.remove()

    def __call__(self, inputs, index_sel=None):
        """
        :param inputs: {"image": [C,H,W], "height": height, "width": width}
        :param index_sel: 第几个边框
        :return:  a list of each batch heatmap
        """
        # 將 feature與gradient取出
        #self._register_hook()
        # forward
        # with torch.no_grad():
        self.net.zero_grad()
        output = self.net(inputs) # [1,num_classes]
        if isinstance(output, tuple):
            output = AngleLoss_predict()(output)

        heatmap_list = [] # batch size, gradcam

        # only for batch one
        if  index_sel == None:
            index = np.argmax(output[0,:].cpu().data.numpy())
            print(f"predict:{index}")  
        else:
            index = int(index_sel[0])
        # backward
        target = output[0][index]
        target.backward(retain_graph=True)
        gradient_list = [ j[0].cpu().data.numpy() for j in self.gradient]  # [C,H,W,D]
        feature_list = [ j[0].cpu().data.numpy() for j in self.feature ]  # [C,H,W,D]
        cam_per_target_layer = []
        # Loop over the saliency image from every layer
        
        for k in range(len(self.layer_list)):
            if k < len(gradient_list):
            # Don't know why gradient is accumulate in each batch
                layer_grads = gradient_list[len(self.layer_list)-1-k]
            if k < len(feature_list):
                layer_features = feature_list[k]

            # gradcam 與 layercam的差異"
            weight = np.maximum(layer_grads, 0)
            cam = layer_features * weight  # [C,H,W,D] feature map 與 weight相乘
            cam = np.sum(cam, axis=0)  # [H,W,D] 
            cam = np.maximum(cam, 0)  # ReLU
            # 若要融合各個cam則需要這樣處理 tanh(2*cam/max(cam)) paper上寫的 
            cam = np.tanh((2*cam)/np.max(cam))
            # normalize 
            heatmap = (cam - cam.min()) / (cam.max() - cam.min()) #[8,8,8]
            heatmap = resize(heatmap,inputs.shape[2:5]) #[128,128,128]
            cam_per_target_layer.append(heatmap[:, None, :])
        #  aggregate mutiple layer cam 
        cam_per_target_layer = np.concatenate(cam_per_target_layer, axis=1)
        cam_per_target_layer = np.maximum(cam_per_target_layer, 0) 
        result = np.mean(cam_per_target_layer, axis=1)
        heatmap_list.append(result)
            
        return heatmap_list
    
def plot_heatmap_detail(heatmap, img, save_path):
    
    fig, ax = plt.subplots(1, 2, figsize = (10,20))
    plt.axis('off') 
    # 水平翻轉跟順時鐘旋轉 (原本為RAS)
    img = cv2.flip(img, 1)
    heatmap = cv2.flip(heatmap, 1)
    img = cv2.rotate(img, cv2.cv2.ROTATE_90_CLOCKWISE) 
    heatmap = cv2.rotate(heatmap, cv2.cv2.ROTATE_90_CLOCKWISE) 
    heatmap = np.uint8(255 * heatmap)
    
    # 以 0.6 透明度繪製原始影像
    ax[0].imshow(img, cmap ='bone')
    ax[0].set_axis_off()
    # 以 0.4 透明度繪製熱力圖
    ax[1].imshow(img, cmap ='bone')
    ax[1].imshow(heatmap, cmap ='jet', alpha=0.4)
    ax[1].set_axis_off()
    #plt.title(pred_class_name)
    plt.savefig(save_path, bbox_inches='tight', pad_inches = 0)
    #plt.show()
    plt.close()

def plot_heatmap_one_picture(heatmap, img, save_path, fig_size=(5,100)):
    
    fig, ax  = plt.subplots(heatmap.shape[2],2, figsize = fig_size, constrained_layout=True)
    fig.subplots_adjust(hspace=0, wspace=0)
    for i in range(heatmap.shape[2]):
        # 水平翻轉跟順時鐘旋轉 (原本為RAS)
        img_show = cv2.flip(img[:,:,i], 1)
        heatmap_show = cv2.flip(heatmap[:,:,i], 1)
        img_show=cv2.rotate(img_show, cv2.cv2.ROTATE_90_CLOCKWISE) 
        heatmap_show=cv2.rotate(heatmap_show, cv2.cv2.ROTATE_90_CLOCKWISE) 
        heatmap_show = np.uint8(255 * heatmap_show)
        ax[i,0].imshow(img_show,cmap ='bone')
        ax[i,0].set_axis_off()
        ax[i,1].imshow(img_show,cmap ='bone')
        ax[i,1].imshow(heatmap_show,cmap ='jet', alpha=0.4)
        ax[i,1].set_axis_off()
    fig.savefig(save_path)
    #plt.show()
    plt.close()

def plot_vedio(path):
    #path = '/data/jacky831006/classification_torch/grad_cam_image/all_test_config_2_new/AIS12/CGMHTR03273'
    img = cv2.imread(f'{path}/000.png')
    size = (img.shape[1],img.shape[0])
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    vedio_path_list = path.split('/')
    vedio_path_list.insert(-1,'video')
    video = cv2.VideoWriter(f'{"/".join(vedio_path_list)}.avi', fourcc, 20, size) # 檔名, 編碼格式, 偵數, 影片大小(圖片大小)

    dir_path = '/'.join(vedio_path_list[:-1])
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)
        
    files = os.listdir(path)
    files.sort()
    for i in files:
        file_name = f'{path}/{i}'
        img = cv2.imread(file_name)
        video.write(img)

def get_last_conv_name(net):
    """
    获取网络的最后一个卷积层的名字
    :param net:
    :return:
    """
    #layer_name = None
    layer_name_list = []
    for name, m in net.named_modules():
        #print(name)
        #print(m)
        if isinstance(m, nn.Conv3d):
            layer_name_list.append(name)
            #layer_name = name
    return layer_name_list

# zip file with absolute path
def zipDir(dirPath, zipPath):
    
    zipf = zipfile.ZipFile(zipPath , mode='w')
    lenDirPath = len(dirPath)
    for root, _ , files in os.walk(dirPath):
        for file in files:
            filePath = os.path.join(root, file)
            zipf.write(filePath , filePath[lenDirPath :] )
    zipf.close()