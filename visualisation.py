# For visualisation purposes only

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import h5py
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression



raw_df = pd.read_hdf('data_files/new_Input_NonResonant_yy_25th_January2026.h5', key='VBF_Polarisation_Tree')

# Make histograms of HiggsM

plt.figure(figsize=(10,6))
plt.hist(raw_df['HiggsM'], bins=100, alpha=0.7, color='blue', label='HiggsM Distribution')
plt.xlabel('Higgs Mass (HiggsM)')
plt.ylabel('Frequency')
plt.title('Histogram of Higgs Mass (HiggsM)')
plt.legend()
plt.grid()
plt.show()  