import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import h5py
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

raw_df = pd.read_hdf('data_files/new_Input_NonResonant_yy_25th_January2026.h5', key='VBF_Polarisation_Tree')
print(raw_df.head())

# Select features and target variable
print(raw_df["HiggsM"].mean())
print(raw_df["HiggsM"].min())
print(raw_df["HiggsM"].max())


