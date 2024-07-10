import torch
import numpy as np
import pandas as pd
import math
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, RandomSampler
from torchvision import transforms, utils
import openml
import wandb
from enum import Flag, auto
from itertools import product
from collections import Counter
from torchmetrics import Accuracy
import sklearn.model_selection
from sklearn.preprocessing import LabelEncoder
from imblearn.datasets import make_imbalance
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler, EditedNearestNeighbours, RepeatedEditedNearestNeighbours, AllKNN
import warnings
import numpy as np
from dotenv import load_dotenv
import os
import delu
from typing import Dict, Tuple
from tqdm import tqdm
from rtdl_revisiting_models import MLP, ResNet, FTTransformer
import argparse
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import roc_auc_score, f1_score, auc, precision_recall_curve, brier_score_loss, average_precision_score, precision_score, recall_score, confusion_matrix, roc_curve
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer, log_loss
from bayes_opt import BayesianOptimization, UtilityFunction
from xgboost import XGBClassifier
from util import load_data
class ClassImbPrep(Flag):
    NONE = auto()
    CLASS_WEIGHT = auto()
    RANDOM_OVER_SAMPLER = auto()
    SMOTE = auto()
    ADASYN = auto()
    RANDOM_UNDER_SAMPLER = auto()
    EDITED_NEAREST_NEIGHBOURS = auto()
    REPEATED_EDITED_NEAREST_NEIGHBOURS = auto()
    ALL_KNN = auto()
    BATCH_BALANCED_OVER = auto()
    BATCH_BALANCED_UNDER = auto()
    BATCH_STRACTIFIED = auto()
    
ADJUST_BATCH = ClassImbPrep.BATCH_BALANCED_OVER | ClassImbPrep.BATCH_BALANCED_UNDER | ClassImbPrep.BATCH_STRACTIFIED

warnings.filterwarnings("ignore")
load_dotenv()
openml.config.apikey = os.getenv("OPENML_API_KEY")

parser = argparse.ArgumentParser(description='select dataset')
parser.add_argument('--datatype', type=str, choices=['control', 'rwdata'], default='control', help='dataset type: "debug" or "control" or "rwdata". Default is "control".')
parser.add_argument('--device', type=int, help='cuda device number.')
parser.add_argument('--append', type=str, choices=['True', 'False'], default='False', help='add new dataset and add result to existing file')
import sys
# sys.argv = ['data_meta_features', '--datatype', 'control', '--device', '4']

args = parser.parse_args()

def highlight_max(s):
    is_max = s == s[4:].max()
    return ['background-color: #6aa84f' if v else '' for v in is_max]

def weighted_log_loss(y_true, y_pred):
    if class_imb_prep==ClassImbPrep.CLASS_WEIGHT:
        class_weight_dict = dict(zip(class_labels, class_weights))
    else:
        class_weight_dict = {0: 1, 1: 1}  # Adjust weights based on class imbalance
        
    # Create a dictionary to map class labels to their corresponding weights
    # Compute weighted log loss
    sample_weights = np.array([class_weight_dict[label] for label in y_true])
    return log_loss(y_true, y_pred, sample_weight=sample_weights)

def xgb_hp_tune(eta, max_depth, reg_alpha):
    eta = eta
    max_depth = int(max_depth)
    reg_alpha = reg_alpha
    scorer = make_scorer(weighted_log_loss, greater_is_better=False)

    skf = sklearn.model_selection.StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=random_state)
    clf = XGBClassifier(max_depth=max_depth, eta=eta, reg_alpha = reg_alpha, gpu_id=gpu_id)
    return np.mean(cross_val_score(clf, X_train_xgb, y_train_xgb, cv=skf, scoring=scorer))

class ClassImbDataset(Dataset):
    """Class Imbalance Dataset from OpenML"""
    def __init__(self, data, labels, device, class_imb_prep=ClassImbPrep.NONE):
        self.data = data.fillna(0)
        self.labels = labels
        self.class_imb = class_imb_prep
        self.device = device
        self.weight = torch.Tensor(compute_class_weight(class_weight="balanced", classes=np.unique(labels), y=labels))
        samples_weight = np.array([self.weight[t] for t in self.labels])
        samples_weight = torch.from_numpy(samples_weight)
        self.samples_weight = samples_weight.double()/100.
        if self.class_imb == ClassImbPrep.RANDOM_OVER_SAMPLER:
            sampler = RandomOverSampler(random_state=random_state)
        elif self.class_imb == ClassImbPrep.SMOTE:
            sampler = SMOTE(random_state=random_state)
        elif self.class_imb == ClassImbPrep.ADASYN:
            sampler = ADASYN(random_state=random_state)
        elif self.class_imb == ClassImbPrep.RANDOM_UNDER_SAMPLER:
            sampler = RandomUnderSampler(random_state=random_state)
        elif self.class_imb == ClassImbPrep.EDITED_NEAREST_NEIGHBOURS:
            sampler = EditedNearestNeighbours()
        elif self.class_imb == ClassImbPrep.REPEATED_EDITED_NEAREST_NEIGHBOURS:
            sampler = RepeatedEditedNearestNeighbours()
        elif self.class_imb == ClassImbPrep.ALL_KNN:
            sampler = AllKNN()
        elif self.class_imb in [ClassImbPrep.NONE, ClassImbPrep.CLASS_WEIGHT] or self.class_imb in ADJUST_BATCH:
            pass
        else:
            raise ValueError("Invalid class_imb value. Supported values are 'ros', 'smote', 'adasyn', 'rus', 'enn', 'renn', and 'allknn'.")
        if self.class_imb not in [ClassImbPrep.NONE, ClassImbPrep.CLASS_WEIGHT] and self.class_imb not in ADJUST_BATCH:
            self.data, self.labels = sampler.fit_resample(self.data, self.labels)
        if device.type != 'cpu':
            self.data = torch.as_tensor(self.data.astype(np.float32).values, device=self.device)
            # self.labels = torch.as_tensor(self.labels, device=self.device)
            
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.data[index], self.labels[index]
    
    def get_class_weight(self):
        if self.class_imb == ClassImbPrep.CLASS_WEIGHT:
            return self.weight
        else:
            return None
        
    def get_sample_weight(self):
        if self.class_imb == ClassImbPrep.BATCH_STRACTIFIED:
            weight = 1. / len(self.labels.detach().cpu().numpy())
            return torch.ones(len(self.data)) * weight
        else:
            return self.samples_weight
        
    def get_sample_len(self):
        if self.class_imb not in ADJUST_BATCH:
            return len(self.data)
            # raise ValueError("Class Imbalance Preprocessing is not Adjusting Batch.")
        elif self.class_imb == ClassImbPrep.BATCH_BALANCED_OVER:
            return int(len(self.data) * 1.5)
        else:
            return len(self.data)
        
    def get_data_label(self):
        return self.data, self.labels
    
    def get_feature_len(self):
        return self.data.shape[1]

architecture = 'xgboost'
task_type = "multiclass"
n_classes = None
random_state = 42
gpu_id = args.device
num_folds = 3
init_points = 5
n_iter = 10
device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")

if args.datatype == 'control':
    imb_ratio_list = [0.01, 0.05, 0.1, 0.5, 1]
    dataset_list = [151, 1120]
    
    class_imb_prep_list = [
        ClassImbPrep.NONE,
        ClassImbPrep.CLASS_WEIGHT,
        ClassImbPrep.RANDOM_OVER_SAMPLER,
        ClassImbPrep.SMOTE,
        ClassImbPrep.ADASYN,
        ClassImbPrep.RANDOM_UNDER_SAMPLER,
        ClassImbPrep.EDITED_NEAREST_NEIGHBOURS,
        ClassImbPrep.REPEATED_EDITED_NEAREST_NEIGHBOURS,
        ClassImbPrep.ALL_KNN
        ]
elif args.datatype == 'rwdata':
    dataset_list = ['GiveMeSomeCredit','IntrusionDetection0']
    
    class_imb_prep_list = [
        ClassImbPrep.NONE,
        ClassImbPrep.CLASS_WEIGHT,
        ClassImbPrep.RANDOM_OVER_SAMPLER,
        ClassImbPrep.SMOTE,
        ClassImbPrep.ADASYN,
        ClassImbPrep.RANDOM_UNDER_SAMPLER,
        ClassImbPrep.EDITED_NEAREST_NEIGHBOURS,
        ClassImbPrep.REPEATED_EDITED_NEAREST_NEIGHBOURS,
        ClassImbPrep.ALL_KNN
        ]
    
if args.datatype == 'control':
    iter_list = product(dataset_list, imb_ratio_list)
elif args.datatype == 'rwdata':
    iter_list = dataset_list

columns=[
    'dataset', 
    'imb_ratio',
    'len_neg',
    'len_pos',
    *class_imb_prep_list]

columns_param=[
    'dataset', 
    'imb_ratio',
    *class_imb_prep_list]
if args.append == False:
    df_eval_mean_min = pd.DataFrame(columns=columns)
    df_eval_mean_maj = pd.DataFrame(columns=columns)
    df_eval_mean_macro = pd.DataFrame(columns=columns)
    df_eval_std_min = pd.DataFrame(columns=columns)
    df_eval_std_maj = pd.DataFrame(columns=columns)
    df_eval_std_macro = pd.DataFrame(columns=columns)
    df_best_eta= pd.DataFrame(columns=columns_param)
    df_best_max_depth= pd.DataFrame(columns=columns_param)
    df_best_reg_alpha= pd.DataFrame(columns=columns_param)
else:
    sav_path = f'/ceph/ejoo/result/{architecture}_{args.datatype}_eval_f1.xlsx'

    df_eval_mean_min = pd.DataFrame(columns=columns)
    df_eval_mean_maj = pd.DataFrame(columns=columns)
    df_eval_mean_macro = pd.DataFrame(columns=columns)
    df_eval_std_min = pd.DataFrame(columns=columns)
    df_eval_std_maj = pd.DataFrame(columns=columns)
    df_eval_std_macro = pd.DataFrame(columns=columns)
    

    df_best_eta= pd.read_excel(sav_path, sheet_name='eta')
    df_best_max_depth= pd.read_excel(sav_path, sheet_name='max_depth')
    df_best_reg_alpha= pd.read_excel(sav_path, sheet_name='reg_alpha')
    
for idx, elem in enumerate(iter_list):
    if args.datatype == 'control':
        dataset_id, imb_ratio = elem
        pjt_name = 'cotrol' + architecture
        
    elif args.datatype == 'rwdata':
        dataset_id = elem
        pjt_name = 'real_world_' + architecture
    
    X, Y, _ , _= load_data(dataset_id)

    if args.datatype == 'control':
        len_neg, len_pos = Counter(Y)[0], Counter(Y)[1]
        testset_size = int(len_neg * 0.2)
    elif args.datatype == 'rwdata':
        len_neg, len_pos = Counter(Y)[0], Counter(Y)[1]
        len_min = min(len_neg, len_pos)
        testset_size = int(len_min * 0.2)
    idx_neg = np.where(Y==0)
    idx_pos = np.where(Y==1)

    data_neg = X.iloc[idx_neg]
    label_neg = Y[idx_neg]
    data_pos = X.iloc[idx_pos]
    label_pos = Y[idx_pos]

    X_train_neg, X_test_neg, y_train_neg, y_test_neg = sklearn.model_selection.train_test_split(data_neg, label_neg, test_size=testset_size, random_state=42, shuffle=True)
    X_train_pos, X_test_pos, y_train_pos, y_test_pos = sklearn.model_selection.train_test_split(data_pos, label_pos, test_size=testset_size, random_state=42, shuffle=True)
    X_train = pd.concat([X_train_neg, X_train_pos], ignore_index=True)
    y_train = np.concatenate((y_train_neg, y_train_pos), axis=0)

    X_test = pd.concat([X_test_neg, X_test_pos], ignore_index=True)
    y_test = np.concatenate((y_test_neg, y_test_pos), axis=0)

    test_dataset = ClassImbDataset(X_test, y_test, device)
    accuracy = Accuracy(task="binary",num_classes=2,top_k=1).to(device)
    
    len_neg, len_pos = Counter(y_train)[0], Counter(y_train)[1]
    if args.datatype == 'control':
            min_label=1
            maj_label=0
            if imb_ratio==1:
                min_len = min(len_neg, len_pos)
                X_imb, Y_imb = make_imbalance(X_train, y_train,
                                sampling_strategy={0: min_len, 1: min_len},
                                random_state=random_state)
                len_neg = min_len
                len_pos = min_len
            elif len_neg*imb_ratio<=len_pos:
                X_imb, Y_imb = make_imbalance(X_train, y_train,
                                sampling_strategy={0: len_neg, 1: int(len_neg*imb_ratio)},
                                random_state=random_state)
                len_pos = int(len_neg*imb_ratio)
            else:
                X_imb, Y_imb = make_imbalance(X_train, y_train,
                                sampling_strategy={0: int(len_pos*1/imb_ratio), 1: len_pos},
                                random_state=random_state)
    
    elif args.datatype == 'rwdata':
        if len_neg < len_pos:
            min_label = 0
            maj_label = 1
            imb_ratio = len_neg / len_pos
        else:
            min_label = 1
            maj_label = 0
            imb_ratio = len_pos / len_neg
            
        X_imb = X_train
        Y_imb = y_train

    row_mean_min = {'dataset': dataset_id, 'imb_ratio': imb_ratio, 'len_neg':len_neg, 'len_pos':len_pos}
    row_mean_maj = {'dataset': dataset_id, 'imb_ratio': imb_ratio, 'len_neg':len_neg, 'len_pos':len_pos}
    row_mean_macro = {'dataset': dataset_id, 'imb_ratio': imb_ratio, 'len_neg':len_neg, 'len_pos':len_pos}
    row_std_min = {'dataset': dataset_id, 'imb_ratio': imb_ratio, 'len_neg':len_neg, 'len_pos':len_pos}
    row_std_maj = {'dataset': dataset_id, 'imb_ratio': imb_ratio, 'len_neg':len_neg, 'len_pos':len_pos}
    row_std_macro = {'dataset': dataset_id, 'imb_ratio': imb_ratio, 'len_neg':len_neg, 'len_pos':len_pos}
    row_eta = {'dataset': dataset_id, 'imb_ratio': imb_ratio}
    row_max_depth = {'dataset': dataset_id, 'imb_ratio': imb_ratio}
    row_reg_alpha = {'dataset': dataset_id, 'imb_ratio': imb_ratio}
    for class_imb_prep in class_imb_prep_list:
        if args.datatype == 'control':
            sav_name = f'{architecture}_{dataset_id}_{imb_ratio}_{class_imb_prep}'
            if imb_ratio == 1 and class_imb_prep != ClassImbPrep.NONE:
                break
        elif args.datatype == 'rwdata':
            sav_name = f'{architecture}_{dataset_id}_{class_imb_prep}'
        f1_min_list = []
        f1_maj_list = []
        f1_macro_list = []
        for i in range(3):
            train_dataset = ClassImbDataset(X_imb.iloc[train_index], Y_imb[train_index], device, categorical_indicator, class_imb_prep, architecture)
            val_dataset = ClassImbDataset(X_imb.iloc[test_idx], Y_imb[test_idx], device, categorical_indicator, ClassImbPrep.NONE, architecture)
            # dataset_sav_path = f'/ceph/ejoo/dataset/{dataset_id}/'
            # if args.datatype == 'control':
            #     name_subset = f'{dataset_id}_{imb_ratio}_{class_imb_prep}_{i}'
            # elif args.datatype == 'rwdata':
            #     name_subset = f'{dataset_id}_{class_imb_prep}_{i}'
            # torch.save(train_dataset, f'{sav_path}train_dataset_{name_subset}.pt')
            # torch.save(val_dataset, f'{sav_path}val_dataset_{name_subset}.pt')
            # train_dataset = torch.load(f'{dataset_sav_path}train_dataset_{name_subset}.pt')
            # val_dataset = torch.load(f'{dataset_sav_path}val_dataset_{name_subset}.pt')
            X_train_xgb, y_train_xgb = train_dataset.get_data_label()
            X_train_xgb = X_train_xgb.detach().cpu().numpy()
            y_train_xgb = y_train_xgb.detach().cpu().numpy()
            class_labels = np.unique(y_train_xgb)
            if class_imb_prep==ClassImbPrep.CLASS_WEIGHT:
                class_weights = compute_class_weight(class_weight='balanced', classes=class_labels, y=y_train_xgb)
                class_weight_dict = dict(zip(class_labels, class_weights))
            
            sample_weights_train = None
            if class_imb_prep == ClassImbPrep.CLASS_WEIGHT:
                sample_weights_train = np.array([class_weight_dict[label] for label in y_train_xgb])
            
            sav_path = f'/ceph/ejoo/result/{architecture}_{args.datatype}_eval_f1.xlsx'
            df_best_eta = pd.read_excel(sav_path, sheet_name='eta')
            df_best_max_depth = pd.read_excel(sav_path, sheet_name='max_depth')
            df_best_reg_alpha = pd.read_excel(sav_path, sheet_name='reg_alpha')
            eta = df_best_eta.loc[idx, str(class_imb_prep)]
            max_depth = int(df_best_max_depth.loc[idx, str(class_imb_prep)])
            reg_alpha = df_best_reg_alpha.loc[idx, str(class_imb_prep)]
            clf = XGBClassifier(eta=eta, max_depth=max_depth, reg_alpha=reg_alpha, gpu_id=gpu_id)
            clf.fit(X_train_xgb, y_train_xgb, sample_weight=sample_weights_train)
            y_pred_test = clf.predict(X_test)
            f1_minority = f1_score(y_test, y_pred_test, pos_label=min_label)
            f1_majority = f1_score(y_test, y_pred_test, pos_label=maj_label)
            f1_macro = f1_score(y_test, y_pred_test, average='macro')

            f1_min_list.append(f1_minority)
            f1_maj_list.append(f1_majority)
            f1_macro_list.append(f1_macro)

        mean_f1_minority = np.mean(f1_min_list)
        std_f1_minority = np.std(f1_min_list)
        mean_f1_majority = np.mean(f1_maj_list)
        std_f1_majority = np.std(f1_maj_list)
        mean_f1_macro = np.mean(f1_macro_list)
        std_f1_macro = np.std(f1_macro_list)

        row_mean_min[class_imb_prep] = mean_f1_minority
        row_mean_maj[class_imb_prep] = mean_f1_majority
        row_mean_macro[class_imb_prep] = mean_f1_macro
        
        row_std_min[class_imb_prep] = std_f1_minority
        row_std_maj[class_imb_prep] = std_f1_majority
        row_std_macro[class_imb_prep] = std_f1_macro
    df_eval_mean_min = pd.concat([df_eval_mean_min, pd.DataFrame(row_mean_min, index=[0])], ignore_index=True)
    df_eval_mean_maj = pd.concat([df_eval_mean_maj, pd.DataFrame(row_mean_maj, index=[0])], ignore_index=True)
    df_eval_mean_macro = pd.concat([df_eval_mean_macro, pd.DataFrame(row_mean_macro, index=[0])], ignore_index=True)
    
    df_eval_std_min = pd.concat([df_eval_std_min, pd.DataFrame(row_std_min, index=[0])], ignore_index=True)
    df_eval_std_maj = pd.concat([df_eval_std_maj, pd.DataFrame(row_std_maj, index=[0])], ignore_index=True)
    df_eval_std_macro = pd.concat([df_eval_std_macro, pd.DataFrame(row_std_maj, index=[0])], ignore_index=True)
    
df_eval_min_style = df_eval_mean_min.style.apply(highlight_max, axis=1)
df_eval_maj_style = df_eval_mean_maj.style.apply(highlight_max, axis=1)
df_eval_macro_style = df_eval_mean_macro.style.apply(highlight_max, axis=1)

sav_path = f'/ceph/ejoo/result/raw/{architecture}_{args.datatype}_eval_f1_cv'
df_eval_min_style.to_excel(f'{sav_path}_min.xlsx', engine='openpyxl', index=False)
df_eval_maj_style.to_excel(f'{sav_path}_maj.xlsx', engine='openpyxl', index=False)
df_eval_macro_style.to_excel(f'{sav_path}_macro.xlsx', engine='openpyxl', index=False)

sav_path = f'/ceph/ejoo/result/{architecture}_{args.datatype}_eval_f1_cv.xlsx'
with pd.ExcelWriter(sav_path, engine='openpyxl') as writer:
    df_eval_min_style.to_excel(writer, sheet_name=f'mean_f1_minority', index=False)
    df_eval_maj_style.to_excel(writer, sheet_name=f'mean_f1_majority', index=False)
    df_eval_macro_style.to_excel(writer, sheet_name=f'mean_f1_macro', index=False)
    
    df_eval_std_min.to_excel(writer, sheet_name=f'std_f1_minority', index=False)
    df_eval_std_maj.to_excel(writer, sheet_name=f'std_f1_majority', index=False)
    df_eval_std_macro.to_excel(writer, sheet_name=f'std_f1_macro', index=False)
    
    df_best_eta.to_excel(writer, sheet_name=f'eta', index=False)
    df_best_max_depth.to_excel(writer, sheet_name=f'max_depth', index=False)
    df_best_reg_alpha.to_excel(writer, sheet_name=f'reg_alpha', index=False)