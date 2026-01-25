import sys
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import scipy
import h5py

print("Hello World")
print(sys.prefix)
print(np.__version__)

with h5py.File('data_files/new_Input_NonResonant_yy_25th_January2026.h5', 'r') as f:
    print(list(f.keys()))


