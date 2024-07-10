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
from util import load_data
def highlight_max(s):
    is_max = s == s[4:].max()
    return ['background-color: #6aa84f' if v else '' for v in is_max]
# second max #d9ead3
# worse than None #f4cccc
# worse than None & worst perform recovery method #e06666
# better than None & worst performance #f6b26b
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
random_state = 42

parser = argparse.ArgumentParser(description='evaluation for class imbalance, deep learning, tabular data experiments.')
parser.add_argument('--architecture', type=str, choices=['resnet', 'ft_transformer'], default='resnet', help='The architecture to be evaluated: "resnet" or "ft_transformer". Default is "resnet".')
parser.add_argument('--datatype', type=str, choices=['control', 'rwdata'], default='control', help='dataset type: "debug" or "control" or "rwdata". Default is "control".')
parser.add_argument('--device', type=int, help='cuda device number.')
parser.add_argument('--append', type=str, choices=['True', 'False'], default='False', help='add new dataset and add result to existing file')

args = parser.parse_args()

delu.random.seed(0)
best_loss = {}


def apply_model(model, *args_apply):
    # if isinstance(model, (MLP, ResNet)):
    if args.architecture == "resnet":
        return model(args_apply[0].float()).squeeze(-1)
    elif args.architecture == "ft_transformer":
        if len(args_apply)==1:
            return model(args_apply[0], None).squeeze(-1)
        else:
            return model(args_apply[0], args_apply[1]).squeeze(-1)
    else:
        raise RuntimeError(f"Unknown model type: {type(model)}")

def predict_y(y_prob):
    y_prob = torch.special.expit(y_prob)            
    y_pred = y_prob.argmax(1)
    return y_pred

@torch.no_grad()
def evaluate(model, dataset, accuracy, loss_fn, device):
    *args_eval, y_true = dataset.get_data_label()
    # for arg in args_eval:
    #     arg = torch.as_tensor(arg, device=device)
    y_true = torch.as_tensor(y_true, device=device)
    y_prob = apply_model(model, *args_eval)
    y_pred = predict_y(y_prob)
    acc = accuracy(y_pred, y_true)
    loss = loss_fn(y_prob, y_true)

    y_prob = torch.softmax(y_prob, dim=1)
    y_prob = y_prob[:,1]

    y_prob = y_prob.detach().cpu().numpy()
    y_pred = y_pred.detach().cpu().numpy()
    y_true = y_true.detach().cpu().numpy()
    
    return acc, loss, y_prob, y_pred, y_true

class ClassImbDataset(Dataset):
    """Class Imbalance Dataset from OpenML"""
    def __init__(self, data, labels, device, categorical_indicator, class_imb_prep=ClassImbPrep.NONE, model='resnet'):
        self.data = data.fillna(0)
        self.labels = labels
        self.class_imb = class_imb_prep
        self.device = device
        self.categorical_indicator = categorical_indicator
        self.model = model
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
        if self.model != "ft_transformer":
            self.n_cont_features = self.data.shape[1]
            self.cat_cardinalities = None
        else:
            self.X_cont: np.ndarray = np.array(self.data.iloc[:,list(~np.array(self.categorical_indicator))])
            self.X_cont = self.X_cont.astype(np.float32)
            self.n_cont_features = self.X_cont.shape[1]
            self.X_cat = None
            if any(self.categorical_indicator):
                df = self.data.iloc[:, self.categorical_indicator]
                for col in df.columns:
                    df.loc[:, col] = LabelEncoder().fit_transform(df[col])
                self.X_cat: np.ndarray = np.array(df)
            self.cat_cardinalities = [
                # NOTE: uncomment the two lines below to add two categorical features.
                # 4,  # Allowed values: [0, 1, 2, 3].
                # 7,  # Allowed values: [0, 1, 2, 3, 4, 5, 6].
            ]
            for idx, col in enumerate(self.categorical_indicator):
                if col == True:
                    self.cat_cardinalities.append(len(self.data.iloc[:, idx].unique()))
        if device.type != 'cpu':
            if self.model == 'ft_transformer':
                self.X_cont = torch.as_tensor(self.X_cont.astype(np.float32), device=self.device)
                if self.X_cat is not None:
                    self.X_cat = torch.as_tensor(self.X_cat, device=self.device)
                self.labels = torch.as_tensor(self.labels, device=self.device)
            else:
                self.data = torch.as_tensor(self.data.astype(np.float32).values, device=self.device)
                self.labels = torch.as_tensor(self.labels, device=self.device)
            
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        if self.model == 'ft_transformer':
            if self.X_cat is not None:
                return self.X_cont[index], self.X_cat[index], self.labels[index]
            else:
                return self.X_cont[index], self.labels[index]
        else:
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
        if self.model == 'ft_transformer':
            return self.X_cont, self.X_cat, self.labels
        else:
            return self.data, self.labels
    
    def get_feature_len(self):
        return self.n_cont_features
architecture = args.architecture
task_type = "multiclass"
n_classes = None
random_state = 42
gpu_id = args.device
num_folds = 3
hp_cnt = 15
sav_folder="save_models"
device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")

if args.datatype == 'control':
    imb_ratio_list = [0.01, 0.05, 0.1, 0.5, 1]
    dataset_list = [151, 1120]
    iter_list = product(dataset_list, imb_ratio_list)
    class_imb_prep_list = [
        ClassImbPrep.NONE,
        ClassImbPrep.CLASS_WEIGHT,
        ClassImbPrep.RANDOM_OVER_SAMPLER,
        ClassImbPrep.SMOTE,
        ClassImbPrep.ADASYN,
        ClassImbPrep.RANDOM_UNDER_SAMPLER,
        ClassImbPrep.EDITED_NEAREST_NEIGHBOURS,
        ClassImbPrep.REPEATED_EDITED_NEAREST_NEIGHBOURS,
        ClassImbPrep.ALL_KNN,
        ClassImbPrep.BATCH_BALANCED_OVER,
        ClassImbPrep.BATCH_BALANCED_UNDER,
        ClassImbPrep.BATCH_STRACTIFIED
        ]
elif args.datatype == 'rwdata':
    dataset_list = ['GiveMeSomeCredit', 'IntrusionDetection0']
    
    iter_list = dataset_list
    class_imb_prep_list = [
        ClassImbPrep.NONE,
        ClassImbPrep.CLASS_WEIGHT,
        ClassImbPrep.RANDOM_OVER_SAMPLER,
        ClassImbPrep.SMOTE,
        ClassImbPrep.ADASYN,
        ClassImbPrep.RANDOM_UNDER_SAMPLER,
        ClassImbPrep.EDITED_NEAREST_NEIGHBOURS,
        ClassImbPrep.REPEATED_EDITED_NEAREST_NEIGHBOURS,
        ClassImbPrep.ALL_KNN,
        ClassImbPrep.BATCH_BALANCED_OVER,
        ClassImbPrep.BATCH_BALANCED_UNDER,
        ClassImbPrep.BATCH_STRACTIFIED
        ]
columns=[
    'dataset', 
    'imb_ratio',
    'len_neg',
    'len_pos',
    *class_imb_prep_list]
dataset_name = []
ratio_name = []
dataset_name = []
df_eval_min = pd.DataFrame(columns=columns)
df_eval_maj = pd.DataFrame(columns=columns)
df_eval_macro = pd.DataFrame(columns=columns)

df_eval_min_std = pd.DataFrame(columns=columns)
df_eval_maj_std = pd.DataFrame(columns=columns)
df_eval_macro_std = pd.DataFrame(columns=columns)
print(args.append)
if args.append == 'False':
    df_eval_min = pd.DataFrame(columns=columns)
    df_eval_maj = pd.DataFrame(columns=columns)
    df_eval_macro = pd.DataFrame(columns=columns)

    df_eval_min_std = pd.DataFrame(columns=columns)
    df_eval_maj_std = pd.DataFrame(columns=columns)
    df_eval_macro_std = pd.DataFrame(columns=columns)
else:
    sav_path = f'/ceph/ejoo/result/{architecture}_{args.datatype}_eval_f1_and_std.xlsx'

    df_eval_min = pd.read_excel(sav_path, sheet_name='min_f1')
    df_eval_maj = pd.read_excel(sav_path, sheet_name='maj_f1')
    df_eval_macro= pd.read_excel(sav_path, sheet_name='macro_f1')
    
    df_eval_min_std= pd.read_excel(sav_path, sheet_name='min_f1_std')
    df_eval_maj_std= pd.read_excel(sav_path, sheet_name='maj_f1_std')
    df_eval_macro_std= pd.read_excel(sav_path, sheet_name='macro_f1_std')
    
for elem in iter_list:
    if args.datatype == 'control':
        dataset_id, imb_ratio = elem
    elif args.datatype == 'rwdata':
        dataset_id = elem
    X, Y, categorical_indicator, cat_cardinalities = load_data(dataset_id)

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

    test_dataset = ClassImbDataset(X_test, y_test, device, categorical_indicator, ClassImbPrep.NONE, architecture)
    accuracy = Accuracy(task="binary",num_classes=2,top_k=1).to(device)
    config = dict(
            lr=0.0008504042203355388, weight_decay=0.0006332430789856007, dropout1=0.5
        )
    len_neg, len_pos = Counter(y_train)[0], Counter(y_train)[1]
    if args.datatype == 'control':
        if imb_ratio==1:
            min_len = min(len_neg, len_pos)
            len_neg = min_len
            len_pos = min_len
        elif len_neg*imb_ratio<=len_pos:
            len_pos = int(len_neg*imb_ratio)
        else:
            len_neg = int(len_pos*1/imb_ratio)
    elif args.datatype == 'rwdata':
        if len_neg < len_pos:
            min_label = 0
            maj_label = 1
            imb_ratio = len_neg / len_pos
        else:
            min_label = 1
            maj_label = 0
            imb_ratio = len_pos / len_neg
    row_min = {'dataset': dataset_id, 'imb_ratio': imb_ratio, 'len_neg':len_neg, 'len_pos':len_pos}
    row_maj = {'dataset': dataset_id, 'imb_ratio': imb_ratio, 'len_neg':len_neg, 'len_pos':len_pos}
    row_macro = {'dataset': dataset_id, 'imb_ratio': imb_ratio, 'len_neg':len_neg, 'len_pos':len_pos}
    
    row_min_std = {'dataset': dataset_id, 'imb_ratio': imb_ratio, 'len_neg':len_neg, 'len_pos':len_pos}
    row_maj_std = {'dataset': dataset_id, 'imb_ratio': imb_ratio, 'len_neg':len_neg, 'len_pos':len_pos}
    row_macro_std = {'dataset': dataset_id, 'imb_ratio': imb_ratio, 'len_neg':len_neg, 'len_pos':len_pos}
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
        for i in range(num_folds):
            model = torch.load(f"/ceph/ejoo/{sav_folder}/{sav_name}_model{i}.pt").to(device)
            # print(model)
            loss_fn = torch.nn.CrossEntropyLoss()
            model.eval()
            test_score, test_loss, y_prob_test, y_pred_test, y_true_test = evaluate(model, test_dataset, accuracy, loss_fn, device)
        
            f1_minority = f1_score(y_true_test, y_pred_test, pos_label=1)
            f1_min_list.append(f1_minority)
            f1_majority = f1_score(y_true_test, y_pred_test, pos_label=0)
            f1_maj_list.append(f1_majority)
            f1_macro = f1_score(y_true_test, y_pred_test, average='macro')
            f1_macro_list.append(f1_macro)
        
        row_min[class_imb_prep] = np.mean(f1_min_list)
        row_maj[class_imb_prep] = np.mean(f1_maj_list)
        row_macro[class_imb_prep] = np.mean(f1_macro_list)
        print(dataset_id, imb_ratio, class_imb_prep, row_min[class_imb_prep])
        row_min_std[class_imb_prep] = np.std(f1_min_list)
        row_maj_std[class_imb_prep] = np.std(f1_maj_list)
        row_macro_std[class_imb_prep] = np.std(f1_macro_list)
        
    df_eval_min = pd.concat([df_eval_min, pd.DataFrame(row_min, index=[0])], ignore_index=True)
    df_eval_maj = pd.concat([df_eval_maj, pd.DataFrame(row_maj, index=[0])], ignore_index=True)
    df_eval_macro = pd.concat([df_eval_macro, pd.DataFrame(row_macro, index=[0])], ignore_index=True)
    
    df_eval_min_std = pd.concat([df_eval_min_std, pd.DataFrame(row_min_std, index=[0])], ignore_index=True)
    df_eval_maj_std = pd.concat([df_eval_maj_std, pd.DataFrame(row_maj_std, index=[0])], ignore_index=True)
    df_eval_macro_std = pd.concat([df_eval_macro_std, pd.DataFrame(row_macro_std, index=[0])], ignore_index=True)

# Apply the function to each row
df_eval_min_style = df_eval_min.style.apply(highlight_max, axis=1)
df_eval_maj_style = df_eval_maj.style.apply(highlight_max, axis=1)
df_eval_macro_style = df_eval_macro.style.apply(highlight_max, axis=1)

# Save styled DataFrame as Excel file
sav_path = f'/ceph/ejoo/result/raw/{architecture}_{args.datatype}_eval_f1'
df_eval_min_style.to_excel(f'{sav_path}_min.xlsx', engine='openpyxl', index=False)
df_eval_maj_style.to_excel(f'{sav_path}_maj.xlsx', engine='openpyxl', index=False)
df_eval_macro_style.to_excel(f'{sav_path}_macro.xlsx', engine='openpyxl', index=False)

sav_path = f'/ceph/ejoo/result/{architecture}_{args.datatype}_eval_f1_and_std.xlsx'
with pd.ExcelWriter(sav_path, engine='openpyxl') as writer:
    df_eval_min_style.to_excel(writer, sheet_name=f'min_f1', index=False)
    df_eval_maj_style.to_excel(writer, sheet_name=f'maj_f1', index=False)
    df_eval_macro_style.to_excel(writer, sheet_name=f'macro_f1', index=False)
    df_eval_min_std.to_excel(writer, sheet_name=f'min_f1_std', index=False)
    df_eval_maj_std.to_excel(writer, sheet_name=f'maj_f1_std', index=False)
    df_eval_macro_std.to_excel(writer, sheet_name=f'macro_f1_std', index=False)
