#def RunSimulation():
import numpy as np
import math
import random
from read_data import read_data
import pickle
from functions_for_model import *
import itertools
from tkinter import *
import matplotlib.pyplot as plt
from itertools import combinations
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler
MeasStruct=1
Settings, InitialCond,Constants=set_initial_constants_and_settings()
xls = pd.ExcelFile("C:\simulation_upgrade\Data\model data file.xlsx")
sheetX = xls.parse(0)
W1=sheetX['normlized weight'][0:12]
test_normlized=sum(W1)
# if ~test_normlized==1
   # error('W did not normliezd')

Settings['W1']=W1
#  main_simulation(Settings)
# function main_simulation(Settings)
Settings['W1']=Settings['W1']/sum(Settings['W1']);
#rng('shuffle')
#%% Import initial condition and constants
#[InitialCond, Constants] = simulation_initialization();
Settings['vayuData']=[1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1]; #%for every experiment, 1 if VAYU data exists, 0 if not
Settings['vayuData']=Settings['vayuData'](Settings['SelectedExperimentsIdx']);
# Randomly devide the date to Trial and test
VayuDataTrial,RefExpNumTrial,VayuDataTest,RefExpNumTest=devide_data(Settings);

%% Create random matrix for parameters from all cases
load('C:\Users\Admin\VAYU Sense AG\VAYU ltd - Documents\R&D\algo 7.2\tobramycin modeling\simulation upgrade results\params_yps=0.15,Ki=40,Kps2=0.01,mupp=4.2.mat');
[paramMat,isExpVec,paramsNames]=mat_creater_random_co2_only(bestConfig,Settings.nConfig);
%[paramMat,isExpVec,paramsNames]=mat_creater_random(Settings.nConfig);
goldenScoreMat=[];  medianXScoreMatFiltr=[];
ImportedData=cell(length(RefExpNumTrial),1);

