# For visualisation purposes only, outputs in "graphs" folder

import numpy as np
import pandas as pd
import matplotlib
# matplotlib.use("Agg")
import matplotlib.pyplot as plt
import h5py
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


raw_df = pd.read_hdf('data_files/new_Input_NonResonant_yy_25th_January2026.h5', key='VBF_Polarisation_Tree')
print(raw_df.columns)


# Scatter plots of HiggsM vs other variables
# Check graphs/higgsM_vs_others/ for output
# Very useful to visualise each feature across HiggsM spectrum, good for determining feature engineering steps
grouped_df = raw_df.groupby(pd.cut(raw_df['HiggsM'], bins=5000), observed=True).mean(numeric_only=True)
grouped_df.index.name = 'HiggsM_interval'
grouped_df = grouped_df.reset_index()
print(grouped_df)

for target_variable in raw_df.columns:
    if target_variable == 'HiggsM':
        continue
    plt.figure(figsize=(14,7))
    plt.scatter(grouped_df['HiggsM_interval'].apply(lambda x: x.mid), grouped_df[f'{target_variable}'], marker='o', s=5, color='darkblue')
    plt.xlabel('HiggsM')
    plt.ylabel(f'{target_variable}')
    plt.title(f'{target_variable} vs HiggsM')
    plt.grid(alpha=0.3)
    output_path = f"graphs/higgsM_vs_others/{target_variable}_scatter_plot.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved plot to {output_path}")


"""
# Histogram plots for each variable
# Not particularly useful except for HiggsM
# Check graphs/frequency_histograms/ for output
target_variable = 'DNN_score'
for target_variable in raw_df.columns:

    mask1 = raw_df['HiggsM'] < 127000
    mask2 = raw_df['HiggsM'] > 123000
    pole_data = raw_df[mask1 & mask2].copy()
    tail_data_1 = raw_df[~(mask1)].copy()
    tail_data_2 = raw_df[~(mask2)].copy()
    print(pole_data[target_variable].mean(), tail_data_1[target_variable].mean(), tail_data_2[target_variable].mean())

    plt.hist(raw_df[f'{target_variable}'], bins=80, color='skyblue', edgecolor='black')
    plt.xlabel(f'{target_variable}')
    plt.ylabel('Frequency')     
    output_path = f"graphs/frequency_histograms/{target_variable}_hist.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved plot to {output_path}")
"""


"""
# Line graph with one data point every 1000 rows
sampled_df = raw_df.iloc[::100].reset_index(drop=True)
plt.figure(figsize=(14,7))
plt.scatter(sampled_df['HiggsM'], sampled_df[f'{target_variable}'], marker='o', s=10, color='darkblue', label=f'Sampled Data (n={len(sampled_df)})')
plt.xlabel('HiggsM')
plt.ylabel(f'{target_variable}')
plt.title(f'{target_variable} vs HiggsM (Sampled: 1 point per 1000 rows)')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
"""