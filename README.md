# Deep Learning with Tabular Data: Handling Class Imbalance

This repository contains the code, datasets, and documentation for the thesis titled "Investigating the Impact of Class Imbalance Handling Methods on Deep Learning with Tabular Data." The research focuses on comparing various class imbalance handling methods and their impact on the performance of deep learning models applied to tabular data.

## Contents

- `src/`: Contains the source code for the experiments.
  
   ### Files and Descriptions
  1. **dl_eval.py**:
      This script is used for evaluating deep learning models on various datasets with different class imbalance handling methods. It includes the implementation of model evaluation metrics and procedures.
  2. **dl_train.py**:
      This script is responsible for training deep learning models such as ResNet and FT-Transformer on tabular datasets. It handles the setup of data loaders, model preparation, and training loop, including early stopping and logging.
  3. **util.py**:
      This utility script contains helper functions for data loading and preprocessing, which are used across the training and evaluation scripts. It includes functions for loading datasets, transforming features, and splitting data into training and test sets.
  4. **xgb_train_eval_cv.py**:
      This script is designed for training and evaluating XGBoost models using cross-validation. It fetches the best hyperparameter settings achieved from `xgb_train_eval.py` and evaluates the model's performance under various class imbalance scenarios using these hyperparameters.
  5. **xgb_train_eval.py**:
   This script focuses on training and evaluating XGBoost models. It performs hyperparameter tuning using Bayesian optimization to find the best settings, which are then used by `xgb_train_eval_cv.py` for cross-validation.

- `docs/`: Contains the thesis report.

## Thesis Summary

The thesis investigates the impact of various class imbalance handling methods on the performance of deep learning models when applied to imbalanced tabular data. The research involves a comparative analysis between XGBoost for traditional machine learning approaches, and ResNet and FT-Transformer for deep learning approaches under different class imbalance scenarios.

### Key Points:
- **Data Collection and Preprocessing**: Sourcing datasets from finance and cybersecurity domains, handling missing values, and transforming features.
- **Experimental Setup**: Implementing and evaluating different class imbalance handling methods like SMOTE, ADASYN, and class weighting.
- **Comparative Analysis**: Comparing the performance of traditional ML models (XGBoost) with DL models (ResNet, FT-Transformer) on imbalanced datasets.
- **Results**: Determining the effectiveness of each method in improving the performance of DL models on imbalanced data and identifying scenarios where DL models can outperform traditional ML models.

## How to Run

1. **Training Deep Learning Models**:
   ```bash
   python dl_train.py --architecture resnet --datatype control --device 0

2. **Evaluating Deep Learning Models**:
   ```bash
   python dl_eval.py --architecture resnet --datatype control --device 0
3. **Training XGBoost Models**:
   ```bash
   python xgb_train_eval.py --datatype control --device 0
4. **Training XGBoost Models with Cross-Validation**:
   ```bash
   python xgb_train_eval_cv.py --datatype control --device 0

## Class Imbalance Handling Methods

The following class imbalance handling methods are compared in this study:

- Class Weighting
- SMOTE
- ADASYN
- Noise Reduction Methods (ENN, RENN, All KNN)
- Random Over-Sampling (ROS)
- Random Under-Sampling (RUS)
- Balanced Batch Over/Under Sampling

## Models

The study evaluates the performance of the following models:

- FT-Transformer
- ResNet
- XGBoost

## Datasets

The following datasets are used in this study:

1. **Electricity**:
   - **Description**: The Electricity dataset contains electricity usage data with features representing various attributes of the usage patterns.
   - **Link**: [Electricity Dataset on OpenML](https://www.openml.org/d/151)

2. **MagicTelescope**:
   - **Description**: The MagicTelescope dataset includes gamma telescope data to differentiate between high-energy gamma particles and background cosmic rays.
   - **Link**: [MagicTelescope Dataset on OpenML](https://www.openml.org/d/1120)

3. **Give Me Some Credit**:
   - **Description**: The Give Me Some Credit dataset contains financial history and demographic data of borrowers for credit risk prediction.
   - **Link**: [Give Me Some Credit Dataset on Kaggle](https://www.kaggle.com/c/GiveMeSomeCredit)

4. **Intrusion Detection Evaluation Dataset (ISCXIDS2012)**:
   - **Description**: The ISCXIDS2012 dataset includes network traffic data capturing various types of network behavior, including normal activities and cyber-attacks.
   - **Link**: [ISCXIDS2012 Dataset](https://www.unb.ca/cic/datasets/ids.html)

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments
The research presented in this repository was conducted as part of a Master's thesis at the University of Mannheim.
