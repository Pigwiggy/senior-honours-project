# Script for running basic models on the dataset (non-flow matching)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import h5py
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, roc_curve

raw_df = pd.read_hdf('data_files/new_Input_NonResonant_yy_25th_January2026.h5', key='VBF_Polarisation_Tree')


# Select features and target variable
mask1 = raw_df['HiggsM'] < 140000
mask2 = raw_df['HiggsM'] > 110000
pole_data = raw_df[mask1 & mask2].copy()
tail_data = raw_df[~(mask1 & mask2)].copy()


# Use DNN_score, M_jj, Zepp, DPhi_jj, Eta_jj, Njets as feature and HiggsM as target

features = ['DNN_score', 'M_jj', 'Zepp', 'DPhi_jj', 'Eta_jj', 'Njets']
X_train = tail_data[features].values
y_train = tail_data['is_signal'].values          # assuming 0 = background, 1 = signal
w_train = tail_data['finalWeight'].values


# Train
model = HistGradientBoostingClassifier(
    max_iter=200,          # like n_estimators
    learning_rate=0.05,
    max_depth=6,           # typical for HEP
    random_state=42
)

model.fit(X_train, y_train, sample_weight=w_train)

# Evaluate
y_pred_proba = model.predict_proba(X_val)[:, 1]   # prob of being signal
auc = roc_auc_score(y_val, y_pred_proba, sample_weight=w_val)
print(f"Validation AUC: {auc:.4f}")

# Compare to DNN_score
dnn_val = df.loc[X_val.index, 'DNN_score'].values   # adjust indexing as needed
print(f"Pearson correlation with DNN_score: {np.corrcoef(y_pred_proba, dnn_val)[0,1]:.3f}")
"""
model = LinearRegression()
X_train = tail_data[features]
y_train = tail_data['HiggsM']
model.fit(X_train, y_train)

X_pole = pole_data[features]
pred_background = model.predict(X_pole)
"""

print(raw_df.columns)

