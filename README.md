# Deep Learning with Tabular Data: Handling Class Imbalance

This repository contains the code, datasets, and documentation for the thesis titled "Investigating the Impact of Class Imbalance Handling Methods on Deep Learning with Tabular Data." The research focuses on comparing various class imbalance handling methods and their impact on the performance of deep learning models applied to tabular data.

## Contents

- `src/`: Contains the source code for the experiments.
- `docs/`: Contains the thesis report.

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
