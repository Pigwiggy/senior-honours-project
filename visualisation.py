# For visualisation purposes only

import numpy as np
import pandas as pd
import matplotlib
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

# Make histograms of HiggsM
"""
plt.figure(figsize=(10,6))
plt.hist(raw_df['HiggsM'], bins=100, alpha=0.7, color='blue', label='HiggsM Distribution')
plt.xlabel('Higgs Mass (HiggsM)')
plt.ylabel('Frequency')
plt.title('Histogram of Higgs Mass (HiggsM)')
plt.legend()
plt.grid()
plt.show()  
"""