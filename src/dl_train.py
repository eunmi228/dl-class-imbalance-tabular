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
import sys
parser = argparse.ArgumentParser(description='evaluation for class imbalance, deep learning, tabular data experiments.')
parser.add_argument('--architecture', type=str, choices=['resnet', 'ft_transformer'], default='resnet', help='The architecture to be evaluated: "resnet" or "ft_transformer". Default is "resnet".')
parser.add_argument('--datatype', type=str, choices=['control', 'rwdata'], default='control', help='dataset type: "debug" or "control" or "rwdata". Default is "control".')
parser.add_argument('--device', type=int, help='cuda device number.')
# sys.argv = ['working_space', '--architecture', 'ft_transformer', '--datatype', 'rwdata', '--device', '4']
args = parser.parse_args()

delu.random.seed(0)
best_loss = {}

def prepare_model(device, config, architecture, n_features, cat_cardinalities=None):
    d_out = 2

    # NOTE: uncomment to train MLP
    if architecture == "mlp":
        model = MLP(
                    d_in=n_features,
                    d_out=d_out,
                    n_blocks=2,
                    d_block=384,
                    dropout=config.dropout1,
                ).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    # NOTE: uncomment to train ResNet
    if architecture == "resnet":
        model = ResNet(
                    d_in=n_features,
                    d_out=d_out,
                    n_blocks=config.n_blocks,
                    d_block=config.d_block,
                    d_hidden=None,
                    d_hidden_multiplier=2.0,
                    dropout1=config.dropout1,
                    dropout2=0,
                ).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    # NOTE: uncomment to train ResNet
    if architecture == "ft_transformer":
        model = FTTransformer(
                n_cont_features=n_features,
                cat_cardinalities=cat_cardinalities,
                d_out=d_out,
                **FTTransformer.get_default_kwargs(n_blocks=config.n_blocks),
            ).to(device)
        
        optimizer = torch.optim.AdamW(
            # Instead of model.parameters(),
            model.make_parameter_groups(),
            lr=config.lr, weight_decay=config.weight_decay
        )
    return model, optimizer

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
    for arg in args_eval:
        arg = torch.as_tensor(arg, device=device)
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
        label_encoder = LabelEncoder()
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
            print(f'self.data shape: {self.data.shape}, self.labels shape: {self.labels.shape}')
            self.data, self.labels = sampler.fit_resample(self.data, self.labels)
            print("sampling is done")
            print(f'self.data shape: {self.data.shape}, self.labels shape: {self.labels.shape}')
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
                    df.loc[:, col] = label_encoder.fit_transform(df[col])
                self.X_cat: np.ndarray = np.array(df)
            self.cat_cardinalities = [
                # NOTE: uncomment the two lines below to add two categorical features.
                # 4,  # Allowed values: [0, 1, 2, 3].
                # 7,  # Allowed values: [0, 1, 2, 3, 4, 5, 6].
            ]
            for idx, col in enumerate(self.categorical_indicator):
                if col == True:
                    self.cat_cardinalities.append(len(self.data.iloc[:, idx].unique()))
        # if device.type != 'cpu':
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
random_state = 42
gpu_id = args.device
num_folds = 3
hp_cnt = 15
sav_folder="save_models"
device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

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
        ClassImbPrep.ALL_KNN,
        ClassImbPrep.BATCH_BALANCED_OVER,
        ClassImbPrep.BATCH_BALANCED_UNDER,
        ClassImbPrep.BATCH_STRACTIFIED
        ]
elif args.datatype == 'rwdata':
    dataset_list = ['GiveMeSomeCredit', 'IntrusionDetection0']
    
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
if args.datatype == 'control':
    iter_list = product(dataset_list, imb_ratio_list, class_imb_prep_list)
elif args.datatype == 'rwdata':
    iter_list = product(dataset_list, class_imb_prep_list)
    
def main():
    run = wandb.init(name=sav_name)
    
    skf = sklearn.model_selection.StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=random_state)
    
    # sum of all fold models' val_loss for one hp config
    val_loss_sum = 0
    
    model_list = []
    for i, (train_index, test_idx) in enumerate(skf.split(all_idx, Y_imb)):
        train_dataset = ClassImbDataset(X_imb.iloc[train_index], Y_imb[train_index], device, categorical_indicator, class_imb_prep, architecture)
        val_dataset = ClassImbDataset(X_imb.iloc[test_idx], Y_imb[test_idx], device, categorical_indicator, ClassImbPrep.NONE, architecture)
        class_weight = train_dataset.get_class_weight()
        if class_weight is not None:
            class_weight = class_weight.to(device)
        n_features = train_dataset.get_feature_len()
        model, optimizer = prepare_model(device, wandb.config, architecture, n_features, cat_cardinalities)

        n_epochs = 1_000
        patience = 10
        batch_size = 256
        
        if class_imb_prep in ADJUST_BATCH:
            samples_weight = train_dataset.get_sample_weight()
            samples_len = train_dataset.get_sample_len()
            sampler = WeightedRandomSampler(
                weights=samples_weight, num_samples=samples_len, replacement=True
            )
            train_dataloader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, shuffle=False, drop_last=True, num_workers=0)
        else:
            samples_len = train_dataset.get_sample_len()
            sampler = RandomSampler(train_dataset, replacement=True, num_samples=samples_len)
            train_dataloader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, shuffle=False, drop_last=True, num_workers=0)
            
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        timer = delu.tools.Timer()
        early_stopping = delu.tools.EarlyStopping(patience, mode="min")
        

        print(f"Device: {device.type.upper()}")
        print("-" * 88 + "\n")
        timer.run()
        loss_fn = torch.nn.CrossEntropyLoss(weight=class_weight)
        for epoch in range(n_epochs):
            for *args_train, y_batch in train_dataloader:
                model.train()
                optimizer.zero_grad()
                y_prob = apply_model(model, *args_train)
                y_true = torch.as_tensor(y_batch, device=device)
                y_pred = predict_y(y_prob)
                train_score = accuracy(y_pred, y_true)
                train_loss = loss_fn(y_prob, y_true)
                train_loss.backward()
                optimizer.step()
            
            # evaluate with validation data
            model.eval()

            val_score, val_loss, y_prob_val, y_pred_val, y_true_val = evaluate(model, val_dataset, accuracy, loss_fn, device)
            f1_minority = f1_score(y_true_val, y_pred_val, average='binary', pos_label=1)
            
            print(f"epoch: {epoch}, val_loss: {val_loss}, f1_minority: {f1_minority}, val_acc: {val_score}")
            wandb.log(
                        {
                            "epoch": epoch,
                            "train_accuracy": train_score, 
                            "train_loss": train_loss, 
                            "val_accuracy": val_score, 
                            "val_loss": val_loss,
                            "f1_minority": f1_minority
                        }
                    )
            early_stopping.update(val_loss)
            if early_stopping.should_stop():
                # add val_loss when early stop
                val_loss_sum += val_loss
                print(f"loss sum until model{i}: ", val_loss_sum)
                model_list.append(model)
                break
        # end train one fold model
    # end cross validation (end train all models)
    
    # check if current hp config models' val sum is less than previous best loss for this dataset, imb_ratio, class imb prep configuration.
    if val_loss_sum < best_loss[sav_name]:
        best_loss[sav_name] = val_loss_sum
        print("best val_loss_sum: ", best_loss[sav_name])
        # Save the model parameters for highest val loss among hp tuning search
        for i, model_elem in enumerate(model_list):
            torch.save(model_elem, f"/ceph/ejoo/{sav_folder}/{sav_name}_model{i}.pt")

for elem in iter_list:
    if args.datatype == 'control':
        dataset_id, imb_ratio, class_imb_prep = elem
        pjt_name = 'control' + architecture
        if imb_ratio == 1 and class_imb_prep != ClassImbPrep.NONE:
            continue
    elif args.datatype == 'rwdata':
        dataset_id, class_imb_prep = elem
        pjt_name = 'real_world_' + architecture
    
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
    all_idx = np.arange(len(Y_imb))

    if args.datatype == 'control':
        sav_name = f'{architecture}_{dataset_id}_{imb_ratio}_{class_imb_prep}'
    elif args.datatype == 'rwdata':
        sav_name = f'{architecture}_{dataset_id}_{class_imb_prep}'
    
    best_loss[sav_name] = math.inf
    if architecture == 'resnet':
        sweep_configuration = {
        "method": "bayes",
        "metric": {"goal": "minimize", "name": "val_loss"},
        "parameters": {
            "lr": {
                "max": 0.001, "min": 0.0001}, # 0.001 ~ 0.0001
            "weight_decay": {
                "max": 0.0001, "min": 1e-5},
            "n_blocks": {"min": 2, "max": 6},
            "d_block": {"values": [64,128,256]},
            # "d_hidden_multiplier": {"values":[1.0,1.5,2.0]},
            "dropout1": {"min": 0.2, "max": 0.5},
            # "dropout2": {"min": 0.2, "max": 0.5}
            },
        }
    elif architecture == 'ft_transformer':
        sweep_configuration = {
        "method": "bayes",
        "metric": {"goal": "minimize", "name": "val_loss"},
        "parameters": {
            "lr": {
                "max": 0.001, "min": 0.00001},
            "weight_decay": {
                "max": 0.001, "min": 1e-5},
            "n_blocks": {"values": [2,3,4]}
            },
        }
    sweep_id = wandb.sweep(sweep=sweep_configuration, project=pjt_name)

    wandb.agent(sweep_id, function=main, count=hp_cnt)
    
