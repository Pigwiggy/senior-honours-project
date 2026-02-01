# For visualisation purposes only

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import h5py
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression



raw_df = pd.read_hdf('data_files/new_Input_NonResonant_yy_25th_January2026.h5', key='VBF_Polarisation_Tree')

print(raw_df.head())

mask1 = raw_df['HiggsM'] < 127000
mask2 = raw_df['HiggsM'] > 123000
pole_data = raw_df[mask1 & mask2].copy()
tail_data_1 = raw_df[~(mask1)].copy()
tail_data_2 = raw_df[~(mask2)].copy()
print(pole_data['DNN_score'].mean(), tail_data_1['DNN_score'].mean(), tail_data_2['DNN_score'].mean())

print(tail_data_1.shape, tail_data_2.shape, pole_data.shape)

print(raw_df["OO1"].describe())

plt.hist(raw_df['HiggsM'], bins=50, color='skyblue', edgecolor='black')
plt.xlabel('HiggsM')
plt.ylabel('Frequency')     
# plt.show(block=True)
output_path = "graphs/higgsM_hist.png"
plt.savefig(output_path, dpi=300, bbox_inches="tight")
plt.close()
print(f"Saved plot to {output_path}")

"""
# Line graph with one data point every 1000 rows
sampled_df = raw_df.iloc[::1000].reset_index(drop=True)
plt.figure(figsize=(14,7))
plt.scatter(sampled_df['HiggsM'], sampled_df['DNN_score'], marker='o', color='darkblue', label=f'Sampled Data (n={len(sampled_df)})')
plt.xlabel('HiggsM')
plt.ylabel('DNN_score')
plt.title('DNN_score vs HiggsM (Sampled: 1 point per 1000 rows)')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
"""