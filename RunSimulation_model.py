#def RunSimulation():
import numpy as np
import math
import random
#from read_data import read_data
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
#Settings['vayuData']=[1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1]; #%for every experiment, 1 if VAYU data exists, 0 if not
#Settings['vayuData']=Settings['vayuData'](Settings['SelectedExperimentsIdx']);
# Randomly devide the date to Trial and test
experimentsOptions = ['0119A', '0119B', '0319A', '0319B', '0419A', '0419B', '0519A', '0519B', '0619A', '0619B', '0719A',
                      '0819A',
                      '0819B', '040217', '040117', '330615', '340615', '350615', '370615', '380615', '390615', '400615',
                      '410615',
                      '430615', '450815', '470815', '0019_REF', '0019_IC', '0119_REF', '0119_IC', '0219_REF', '0219_IC']
incyteExp = ['0419A', '0519A', '0519B', '0619A', '0619B', '0719A', '0819A', '0119_IC', '0219_IC']
exp2019 = ['0119A', '0119B', '0319A', '0319B', '0419A', '0419B', '0519A', '0519B', '0619A', '0619B', '0719A', '0819A',
           '0819B', '0019_REF', '0019_IC', '0119_REF', '0119_IC', '0219_REF', '0219_IC']
interpWantedData = load_data_df(0, experimentsOptions)

trainData, testData = devide_data(interpWantedData)
del interpWantedData
lenT=trainData[list(trainData.keys())[0]].shape[0]
paramMat,isExpVec,paramsNames=mat_creater_random_co2_only(Settings['nConfig'],0)
for iExp in np.arange (1,len(trainData)):
    for iBatch in np.arange(1,np.floor(Settings['nConfig']/Settings['nConfigPerIter'])):
        #%find relevant configurations
        firstConfig=(iBatch-1)*Settings['nConfigPerIter']+1 #%first configuration for this batch
        lastConfig=iBatch*Settings['nConfigPerIter']#;% last configuration for this batch
        params=loadParams(firstConfig,lastConfig,paramMat);
        # %% run simulation and find golden score
        goldenScoreVec[firstConfig:lastConfig],Xmedian[firstConfig:lastConfig]= find_golden_score(lenT,params,trainData[list(trainData.keys())[iExp]],Settings,InitialCond,Constants);

#VayuDataTrial,RefExpNumTrial,VayuDataTest,RefExpNumTest=devide_data(Settings);

#%% Create random matrix for parameters from all cases
#load('C:\Users\Admin\VAYU Sense AG\VAYU ltd - Documents\R&D\algo 7.2\tobramycin modeling\simulation upgrade results\params_yps=0.15,Ki=40,Kps2=0.01,mupp=4.2.mat');
#[paramMat,isExpVec,paramsNames]=mat_creater_random_co2_only(bestConfig,Settings.nConfig);
#%[paramMat,isExpVec,paramsNames]=mat_creater_random(Settings.nConfig);
#goldenScoreMat=[];  medianXScoreMatFiltr=[];
#ImportedData=cell(length(RefExpNumTrial),1);

