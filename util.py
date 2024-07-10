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

def load_data(dataset_id):
    label_encoder = LabelEncoder()
    
    if isinstance(dataset_id, int):
        dataset = openml.datasets.get_dataset(dataset_id)
        X, Y, categorical_indicator, attribute_names = dataset.get_data(
            dataset_format="dataframe", target=dataset.default_target_attribute
        )
    elif dataset_id == 'GiveMeSomeCredit':
        dataset = pd.read_csv(f'/ceph/ejoo/dataset/{dataset_id}/credit_combine.csv')
        X = dataset.drop(columns=['SeriousDlqin2yrs'])
        Y = np.array(dataset['SeriousDlqin2yrs'])
        categorical_indicator = [False]*len(X.columns)
    if dataset_id == 'IntrusionDetection0':
        dataset = pd.read_csv(f'/ceph/ejoo/dataset/IntrusionDetection/0.csv')
        X = dataset.drop(columns=['Label'])
        Y = np.array(dataset['Label'])
        categorical_columns = ['protocolName', 'appName', 'direction']
        categorical_indicator = np.isin(X.columns, categorical_columns)
    n_cont_features = X.shape[1]
    cat_cardinalities = [
                # NOTE: uncomment the two lines below to add two categorical features.
                # 4,  # Allowed values: [0, 1, 2, 3].
                # 7,  # Allowed values: [0, 1, 2, 3, 4, 5, 6].
            ]
    for idx, col in enumerate(categorical_indicator):
        if col == True:
            cat_cardinalities.append(len(X.iloc[:, idx].unique()))
    data_cat = X.iloc[:, categorical_indicator]
    for col in data_cat.columns:
        X.loc[:, col] = label_encoder.fit_transform(data_cat[col])
    Y = label_encoder.fit_transform(Y)
        
    return X, Y, categorical_indicator, cat_cardinalities