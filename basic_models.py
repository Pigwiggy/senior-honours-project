# Script for running basic models on the dataset (non-flow matching)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import h5py
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

raw_df = pd.read_hdf('data_files/new_Input_NonResonant_yy_25th_January2026.h5', key='VBF_Polarisation_Tree')


### Polynomial Regression Model with singular feature (HiggsM) ###

# Define bounds and number of bins model
lower_bound = 115000
upper_bound = 135000
num_bins = 275

# Creating frequency distribution for HiggsM
freq = pd.cut(raw_df['HiggsM'], bins=num_bins)
freq_counts = freq.value_counts().sort_index()
higgsM_values = [interval.mid for interval in freq_counts.index]
frequencies = freq_counts.values.tolist()

# Create DataFrame for ML Model
model_df = pd.DataFrame({'HiggsM': higgsM_values, 'Frequency': frequencies})
mask1 = model_df['HiggsM'] > upper_bound
mask2 = model_df['HiggsM'] < lower_bound
model_df_tails = model_df[mask1 | mask2]

# Fit a polynomial regression model
X = model_df_tails[['HiggsM']]
y = model_df_tails['Frequency']
poly = PolynomialFeatures(degree = 3)
X_poly = poly.fit_transform(X)
lin_reg = LinearRegression()
lin_reg.fit(X_poly, y)

# Predict frequencies using the trained model
model_df_pole = model_df[~(mask1 | mask2)]
X_pole = model_df_pole[['HiggsM']]
X_pole_poly = poly.transform(X_pole)
y_pole_pred = lin_reg.predict(X_pole_poly)

# Evaluate model performance with MSE
y_pole_true = model_df_pole['Frequency'].values
mse = mean_squared_error(y_pole_true, y_pole_pred)
mse_normilized = mse / np.var(y_pole_true)
print(f"Mean Squared Error (normalized) on pole region: {mse_normilized}")  

plt.scatter(X,y)
plt.scatter(X_pole, y_pole_pred, color='red')
# plt.show()  






"""
# Select features and target variable
mask1 = raw_df['HiggsM'] < upper_bound
mask2 = raw_df['HiggsM'] > lower_bound
# tail_data_1 = raw_df[~(mask1)].copy()
# tail_data_2 = raw_df[~(mask2)].copy()
pole_data = raw_df[mask1 & mask2].copy()
tail_data = raw_df[~(mask1 & mask2)].copy()

features = ['HiggsM']
freq, bins = pd.cut(tail_data[features[0]], bins=intervals, retbins=True)
frequency_counts = freq.value_counts().sort_index()
"""



