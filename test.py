import sys
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import scipy
import h5py

print("Hello World!")
print(sys.prefix)

"""
with h5py.File('data_files/new_Input_NonResonant_yy_25th_January2026.h5', 'r') as f:
    print(list(f.keys()))
"""

raw_df = pd.read_hdf('data_files/new_Input_NonResonant_yy_25th_January2026.h5', key='VBF_Polarisation_Tree')
print(raw_df.shape)
print(raw_df.head())

HiggsM_df = raw_df['HiggsM']
print(HiggsM_df.shape)
print(HiggsM_df.head()) 

print("School laptop test")