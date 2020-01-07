"""
A script loading .py file and makes modifications such as interpolation, adding a column, making changes to index etc.,
then saving it to a new file.
"""
from function_for_smoothing import *
from sklearn.preprocessing import MinMaxScaler, RobustScaler
import pandas as pd
import pickle
import numpy as np
import random
import scipy
from itertools import combinations
from tkinter import *
from functions_for_orginization import *
folderName = 'data'
# fileNameInterp = 'RnD_data_interpolated_new.p'
fileName = 'RnD_Data_5_1.p'
import os.path

with open(os.path.join(folderName, fileName), 'rb') as f:
    data = pickle.load(f)
# with open(os.path.join(folderName, fileNameInterp), 'rb') as j:
#     dataInterp = pickle.load(j)
# data = filter_data(data, processType='Tobramycin')  # Activate smoothing function on data
# regDataAfterInterp = data_interp_df(data)
for exp in data.keys():
    data[exp].index = data[exp].index * 60
    data[exp]['Time'] = data[exp].index


# file = open('RnD_Data_5_1.p', 'wb')
# pickle.dump(data, file)
# file.close()
q=2