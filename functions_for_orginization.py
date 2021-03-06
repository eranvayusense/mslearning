from function_for_smoothing import *
from sklearn.preprocessing import MinMaxScaler, RobustScaler
import pandas as pd
import pickle
import numpy as np
import random
import scipy
from itertools import combinations
from tkinter import *
import os
import easygui as e
from pathlib import Path
path = Path(__file__).parent.absolute()
import sys
sys.path.append("C:\\Users\\admin\\VAYU Sense AG\\VAYU ltd - Documents\\R&D\\algo 7.2\\"
                "BiondVax\\biondvax model analysis")
from model_analysis_biondvax import *
sys.path.append(path)
def setPreferences():
    """
    Description:
    A function which runs GUI of selecting wanted preferences for the run.
    Output:
    pref- a dictionary containing all relevant preferences for
          next run (each preference is either a number or a string)
    """
    # Run GUI function to determine preferences
    pref = linear_model_GUI()

    # Decide fraction options to test according to input from GUI
    if pref['Fraction maximal value']-pref['Fraction minimal value'] < 0.4:
        pref['fracOptions'] = np.arange(pref['Fraction minimal value'], pref['Fraction maximal value'] + 0.05, 0.05)
    else:
        pref['fracOptions'] = np.arange(pref['Fraction minimal value'], pref['Fraction maximal value'] + 0.1, 0.1)

    # Add aditional preferences
    # pref['featuresDist'] = ['Time']
    pref['Combinations'] = create_combinations(pref)
    pref['variables'] = ['Product']
    return pref




def linear_model_GUI():
    def passForword1():
        global processType, preProcessing, relTrainSize, relDataModelingSize
        processType = processTypeGUI.get()
        preProcessing = preProcessingGUI.get()
        relTrainSize = relTrainSizeGUI.get()
        relDataModelingSize = relDataModelingSizeGUI.get()
        firstGUI.destroy()

    def passForword2():
        global varForModel, varForData, varForDist, fractionMinVal, fractionMaxVal, selectedNewFeatures
        indexOptModel = varModelListbox.curselection()
        selectedVarModel = []
        for idx in range(0, len(indexOptModel)):
            selectedVarModel.append(varModelListbox.get(indexOptModel[idx]))
        varForModel = selectedVarModel

        indexOptData = varDataListbox.curselection()
        selectedVarData = []
        for idx in range(0, len(indexOptData)):
            selectedVarData.append(varDataListbox.get(indexOptData[idx]))
        varForData = selectedVarData

        indexNewFeat = newFeatListbox.curselection()
        selectedNewFeatData = []
        for idx in range(0, len(indexNewFeat)):
            selectedNewFeatData.append(newFeatListbox.get(indexNewFeat[idx]))
        selectedNewFeatures = selectedNewFeatData

        indexOptDist = varDistListbox.curselection()
        selectedVarDist = []
        for idx in range(0, len(indexOptDist)):
            selectedVarDist.append(varDistListbox.get(indexOptDist[idx]))
        varForDist = selectedVarDist

        fractionMinVal = float(FracMinValEntry.get())
        fractionMaxVal = float(FracMaxValEntry.get())
        secondGUI.destroy()

    # Options vectors
    preprocessingOptions = ['No preprocessing', 'scaling (0-1)', 'Standardize (Robust scalar)']


    firstGUI = Tk()
    firstGUI.geometry('1100x700') #set the size for the GUI window

    titleLabel = Label(firstGUI, text='Preferences for linear model', font=('Helvetica', '17', 'bold'), fg='blue')
    titleLabel.grid(row=0, column=1, sticky=E, pady=20)

    # Select process name
    processLabel = Label(firstGUI, text='Process name:', font=('Helvetica', '10', 'bold'))
    processLabel.grid(row=1, column=1, sticky=E, pady=10, padx=10)
    processTypeGUI = StringVar(firstGUI)
    processTypeGUI.set("Tobramycin")  # default value
    methodDropDown = OptionMenu(firstGUI, processTypeGUI, "Tobramycin", "BiondVax")
    methodDropDown.grid(row=1, column=2, sticky=E, pady=10)

    # Select preprocessing technique
    preProcessingText = Label(firstGUI, text='choose preprocessing technique:', font=('Helvetica', '10', 'bold'))
    preProcessingText.grid(row=2, column=1, sticky=E, padx=10, pady=10)
    preProcessingGUI = StringVar(firstGUI)
    preProcessingGUI.set("Standardize (Robust scalar)")  # default value
    preProcessingDropDown = OptionMenu(firstGUI, preProcessingGUI, 'No preprocessing', 'scaling (0-1)', 'Standardize (Robust scalar)')
    preProcessingDropDown.grid(row=2, column=2, sticky=E, pady=10)

    # Select size for modeling process (train and test). the rest is validation
    relDataModelingSizeText = Label(firstGUI, text='Modeling process data relative size:', font=('Helvetica', '10', 'bold'))
    relDataModelingSizeText.grid(row=3, column=1, sticky=E, padx=10, pady=10)
    relDataModelingSizeGUI = StringVar(firstGUI)
    relDataModelingSizeGUI.set("0.8")  # default value
    relDataModelingSizeDropDown = OptionMenu(firstGUI, relDataModelingSizeGUI, '0.75', '0.8', '0.85', '0.9')
    relDataModelingSizeDropDown.grid(row=3, column=2, sticky=E, pady=10)

    # train relative size
    relTrainSizeText = Label(firstGUI, text='Training dataset relative size:', font=('Helvetica', '10', 'bold'))
    relTrainSizeText.grid(row=3, column=3, sticky=E, padx=10, pady=10)
    relTrainSizeGUI = StringVar(firstGUI)
    relTrainSizeGUI.set("0.75")  # default value
    relTrainSizeDropDown = OptionMenu(firstGUI, relTrainSizeGUI, '0.7', '0.75', '0.8')
    relTrainSizeDropDown.grid(row=3, column=4, sticky=E, pady=10)


    filterDataText = Label(firstGUI, text='Filter data?', font=('Helvetica', '10', 'bold'))
    filterDataText.grid(row=4, column=1, sticky=E, pady=20, padx=10)
    isFilterData = IntVar(value=1)
    Checkbutton(firstGUI, variable=isFilterData).grid(row=4, column=2, sticky=W, pady=20)

    interDataText = Label(firstGUI, text='Load interpolated data?', font=('Helvetica', '10', 'bold'))
    interDataText.grid(row=4, column=3, sticky=E, pady=20, padx=10)
    isLoadInterpolated = IntVar(value=1)
    Checkbutton(firstGUI, variable=isLoadInterpolated).grid(row=4, column=4, sticky=W, pady=20)

    runParallelText = Label(firstGUI, text='Run parallel computing?', font=('Helvetica', '10', 'bold'))
    runParallelText.grid(row=5, column=1, sticky=E, pady=20, padx=10)
    isRunParallel = IntVar(value=1)
    Checkbutton(firstGUI, variable=isRunParallel).grid(row=5, column=2, sticky=W, pady=20)

    #Continue to next GUI window
    button = Button(text="continue", command=passForword1, width=20, height=1, font=('Helvetica', '17'))
    button.place(rely=0.95, relx=0.5, anchor=CENTER)
    firstGUI.mainloop()

    # get boolean values
    isFilterData = isFilterData.get()
    isLoadInterpolated = isLoadInterpolated.get()
    isRunParallel = isRunParallel.get()
    # Orginize needed data for next GUI
    data = load_data(processType, isFilterData, isLoadInterpolated=isLoadInterpolated)  # load data for variable options
    expList = list(data.keys())
    varOpt = []
    varInExp = pd.DataFrame(index=expList)
    for exp in expList:
        for var in list(data[exp].columns):
            if var not in varOpt:
                varOpt.append(var)
                varInExp[var] = 0
                varInExp[var][exp] = 1
            else:
                varInExp[var][exp] = 1


    #set modeled variables needs to be changed for every new process
    if processType == 'Tobramycin':
        modeledVariables = ['Incyte', 'Tobramycin', 'Kanamycin', 'pH_x', 'pH_y', 'DO', 'Dextrose[percent]',
                       'Ammonia[percent]', 'CO2']
    elif processType == 'BiondVax':
        modeledVariables = ['OD', 'DO', 'pH']
    else:
        modeledVariables = varOpt
    # varOpt = list(data[expList[5]].columns)
    # Import sophisticated features for modeling
    newFeaturesDict = set_new_features(processType, modeledVariables)
    newFeatures = list(newFeaturesDict.keys())

    secondGUI = Tk()
    secondGUI.geometry('1100x700') #set the size for the GUI window

    titleLabel = Label(secondGUI, text='Preferences for ' + processType + ' model', font=('Helvetica', '17', 'bold'), fg='blue')
    titleLabel.grid(row=0, column=1, sticky=E, pady=20)

    # Variables list selection
    varModelText = Label(secondGUI, text='Variables for model:', font=('Helvetica', '10', 'bold'))
    varModelText.grid(row=1, column=1, sticky=E, pady=10, padx=20)
    varModelListbox = Listbox(secondGUI, selectmode='multiple', exportselection=False)
    varModelListbox.grid(row=1, column=2, pady=10)
    for item in varOpt:
        if item in modeledVariables:
            varModelListbox.insert(END, item)
    varModelListbox.selection_set(1)
    # varModelListbox.selection_set(2)
    varModelListbox.selection_set(3)
    varModelListbox.selection_set(7)

    varDataText = Label(secondGUI, text='Variables for data:', font=('Helvetica', '10', 'bold'))
    varDataText.grid(row=1, column=3, sticky=E, pady=10, padx=20)
    varDataListbox = Listbox(secondGUI, selectmode='multiple', exportselection=False)
    varDataListbox.grid(row=1, column=4, pady=10)
    for item in varOpt:
        if item not in modeledVariables:
            varDataListbox.insert(END, item)
    # varDataListbox.selection_set(1)
    # varDataListbox.selection_set(2)
    varDataListbox.selection_set(3)

    newFeatText = Label(secondGUI, text='New features:', font=('Helvetica', '10', 'bold'))
    newFeatText.grid(row=2, column=1, sticky=E, pady=10, padx=20)
    newFeatListbox = Listbox(secondGUI, selectmode='multiple', exportselection=False)
    newFeatListbox.grid(row=2, column=2, pady=10)
    for item in newFeatures:
        newFeatListbox.insert(END, item)
    # varDataListbox.selection_set(1)
    # varDataListbox.selection_set(2)
    newFeatListbox.selection_set(0)

    varDistText = Label(secondGUI, text='Variables for distance:', font=('Helvetica', '10', 'bold'))
    varDistText.grid(row=2, column=3, sticky=E, pady=10, padx=20)
    varDistListbox = Listbox(secondGUI, selectmode='multiple', exportselection=False)
    varDistListbox.grid(row=2, column=4, pady=10)
    for item in varOpt:
        if item not in modeledVariables:
            varDistListbox.insert(END, item)
    # varDataListbox.selection_set(1)
    # varDataListbox.selection_set(2)
    varDistListbox.selection_set(8)


    fracMinText = Label(secondGUI, text='Minimal fraction for group:', font=('Helvetica', '10', 'bold'))
    fracMinText.grid(row=3, column=1, sticky=E, pady=10, padx=20)
    FracMinValEntry = Entry(secondGUI)
    FracMinValEntry.grid(row=3, column=2, pady=10)
    FracMinValEntry.delete(0, END)
    FracMinValEntry.insert(0, "0.3")

    fracMaxText = Label(secondGUI, text='Maximal fraction for group:', font=('Helvetica', '10', 'bold'))
    fracMaxText.grid(row=3, column=3, sticky=E, pady=10, padx=20)
    FracMaxValEntry = Entry(secondGUI)
    FracMaxValEntry.grid(row=3, column=4, pady=10)
    FracMaxValEntry.delete(0, END)
    FracMaxValEntry.insert(0, "0.3")

    button = Button(text="Continue", command=passForword2, width=20, height=1, font=('Helvetica', '17'))
    button.place(rely=0.95, relx=0.5, anchor=CENTER)

    secondGUI.mainloop()

    relVarList = list(set(varForModel + varForData + varForDist))
    relVarInExp = varInExp[relVarList]#create 0/1 dataframe only for selected variables
    relExp = list(relVarInExp[relVarInExp.sum(axis=1) == len(relVarList)].index)
    random.shuffle(relExp)# Shuffle experiments order to avoid unwanted dependencies
    modelingRelExp = relExp[0:int(float(relDataModelingSize) * len(relExp))]
    validationRelExp = relExp[int(float(relDataModelingSize) * len(relExp)):]
    newFeaturesDict = {feat: newFeaturesDict[feat] for feat in selectedNewFeatures}


    pref = {'Variables': varForModel,'Data variables': varForData, 'New features': selectedNewFeatures,
            'New features dict': newFeaturesDict, 'Process Type': processType, 'preProcessing type': preProcessing,
            'Rel train size': relTrainSize, 'Rel modeling size': relDataModelingSize, 'Is filter data': isFilterData,
            'Fraction minimal value': fractionMinVal, 'Fraction maximal value': fractionMaxVal,
            'Relevant experiments': relExp, 'allVariables': varOpt, 'isLoadInterpolated': isLoadInterpolated,
            'Modeling experiments': modelingRelExp, 'Validation experiments': validationRelExp,
            'Is run parallel': isRunParallel, 'featuresDist': varForDist}
    pref['numCVOpt'] = len(modelingRelExp) #number of options for different CV
    pref['CVTrain'] = modelingRelExp[0:int(float(relTrainSize) * len(modelingRelExp))]# Chosen experiments for train, in first CV division
    pref['CVTest'] = modelingRelExp[int(float(relTrainSize) * len(modelingRelExp)):]# Chosen experiments for test, in first CV division
    return pref
    # list(data.columns)

def set_new_features(processType, modeledVariables):
    if processType == 'Tobramycin':
        newFeatures = {'DO * Incyte': ['DO', 'Incyte'], 'S * Incyte': ['Dextrose[percent]', 'Incyte'],
                       'A * Incyte': ['Ammonia[percent]', 'Incyte'], 'pH * Incyte': ['pH_x', 'Incyte']}
        for var in modeledVariables:
            newFeatures[var + '_del'] = [var]
    if processType == 'BiondVax':
        newFeatures = {'DO * OD': ['DO', 'OD'], 'pH * OD': ['pH', 'OD']}
        # newFeatures = ['DO * X', 'S * X', 'A * X', 'pH * X'] + [var + '_del' for var in modeledVariables]
    return newFeatures


def load_data(process, isFilterData, relExp='All', isLoadInterpolated=1):
    """
    Description:
    A function which loads data from .py files, and then filters, interpolates (if data is
    not already interpolated) and pre processes the data according to user selection
    Inputs:
      1. process- Which process is modeled ('Tobramycin', 'BiondVax' etc.)
      2. preProcessing- selected type of preprocessing
      3. isFilterData- 1 if filter data is wanted, 0 if not
      4. relExp(difault is 'All')- list containing experiments where all selected
                                 variables for model were measured. if 'All', all
                                 experiments are used.
    Outputs :
      1. interpData- a dictionary containing dataframes for relevant experiments. Each data
                   frame represents one experiment after linear interpulation where rows
                   are time and columns are measured variables.
      2. interpDataPP- the same format as data, after pre process protocol according to
                     selected type.
    """
    #  load interpolated of un-interpolated data and filter if wanted.
    folderName = 'data'
    if process == "Tobramycin":
        if isLoadInterpolated:
            fileName = 'RnD_Data_5_1_interp.p'
        else:
            fileName = 'RnD_Data_5_1.p'
        with open(os.path.join(folderName, fileName), 'rb') as f:
            data = pickle.load(f)
        if isFilterData &  isLoadInterpolated == 0:
            data = filter_data(data, processType=process)  # Activate smoothing function on data

    elif process == "BiondVax":
        unRelExp = ['276', 'Failed_270_BV_SS20L_161005', 'Failed_TF_SU30L_190717', '271__BV_SS20L_160525',
                    'CY_SS5L_160203A', 'CY_SS5L_160203B', 'CY_SS5L_160203C', 'CY_SS5L_160203D']
        data = load_and_orginize_biondvax_data(isLoadInterpolated)
        for exp in unRelExp:
            if exp in data.keys():
                del data[exp]
        # fileName = 'BiondVaxDataEran.p'
        # with open(os.path.join(folderName, fileName), 'rb') as f:
        #     data = pickle.load(f)
        # if isFilterData:
        #     data = filter_data(data, processType=process)  # Activate smoothing function on data

    # screen out unrelevant experiments
    if relExp=='All':
        relExp = data.keys()
    data = {wantedExp: data[wantedExp] for wantedExp in relExp}

    interpData = data

    # Conduct pre-processing according to wanted type
    # interpDataPP, scale_params= pre_process_function(interpData, preProcessing)


    return interpData



def pre_process_function(data, preProcessing):
    """
        Description:
        A function which conducts pre processing to the data, according to the selected
        type by the user
        Inputs :
          1. data (similar to interpData in "load_data")- a dictionary containing dataframes for
                       relevant experiments. Each data frame represents one experiment after linear
                       interpo lation where rows are time and columns are measured variables.
          2. preProcessing- Type of preprocessing technique.
        Outputs :
          1. dataPP- same format as data, after preprocessing technique activated.
    """
    dataPP = {}
    dataCombined = pd.DataFrame()
    numOfMeasVec = []
    for exp in data.keys():
        dataCombined = dataCombined.append(data[exp], ignore_index=True, sort=False)
        numOfMeasVec.append(len(data[exp]))
    if preProcessing == 'Standardize (Robust scalar)':
        varNames = list(dataCombined.columns)
        scal=RobustScaler()
        dataProcessed = scal.fit_transform(dataCombined)
        dataPPCombined = pd.DataFrame(dataProcessed, columns=varNames)
    if preProcessing == 'scaling (0-1)':
        varNames = list(dataCombined.columns)
        scal=MinMaxScaler()
        dataProcessed = scal.fit_transform(dataCombined)
        dataPPCombined = pd.DataFrame(dataProcessed, columns=varNames)
    if preProcessing == 'No preprocessing':
        varNames = list(dataCombined.columns)
        dataPPCombined = data
    idx = 0
    for expIdx, expName in enumerate(data.keys()):
        dataPP[expName] = dataPPCombined.iloc[idx:idx + numOfMeasVec[expIdx]]
        idx += numOfMeasVec[expIdx]
        if 'TimeMeas' in varNames:
           demo=1
        else:
             varNames.append('TimeMeas')
        dataPP[expName].reset_index(drop=True, inplace=True)
        dataPP[expName].loc[:, 'TimeMeas'] = data[expName].index

    scale1 = pd.DataFrame()
    for var in varNames[:-1]:
        if preProcessing == 'scaling (0-1)':
            scale1[var] = [dataCombined[var].min(), dataCombined[var].max()]
        else:
            scale1[var] = [dataCombined[var].median(), scipy.stats.iqr(dataCombined[var])]
    return dataPP, scale1

def data_interp_df(allData):
    """
    Description:
    A function which interpolates the data.
    Inputs:
      1. data- a dictionary containing dataframe for each experiment
    Outputs:
      1. interpDataOrginized- same structure as data, after interpolation
      with one minute interval
    """
    interpData = dict()  # empty dictionary
    interpDataOrginized = dict()
    for exp in allData.keys():
        interpData[exp] = allData[exp]
        # interpData[exp].index = interpData[exp].index * 60

        # for time in range(0, int(interpData[exp].index[-1])):
        #     abs_sub = abs(time - interpData[exp].index.to_numpy())
        #     min_idx = np.array([np.where(abs_sub == np.amin(abs_sub))[0][0]])
        #     if time == 0:
        #         nextRow = interpData[exp].loc[interpData[exp].index[min_idx]]
        #         nextRow.rename(index={nextRow.index[0]: time}, inplace=True)
        #         interpDataOrginized[exp] = nextRow
        #     else:
        #
        #         nextRow = interpData[exp].loc[interpData[exp].index[min_idx]]
        #         nextRow.rename(index={nextRow.index[0]: time}, inplace=True)
        #         interpDataOrginized[exp] = interpDataOrginized[exp].append(nextRow)
        try:
            for time in range(0, int(interpData[exp].index[-1])+1):
                if time == 0:
                    nextRow = pd.DataFrame(interpData[exp].loc[interpData[exp].index[time]]).T
                    nextRow.rename(index={nextRow.index[0]: time}, inplace=True)
                    interpDataOrginized[exp] = nextRow

                elif time in interpData[exp].index:
                    nextRow = pd.DataFrame(interpData[exp].loc[time]).T
                    nextRow.rename(index={nextRow.index[0]: time}, inplace=True)
                    interpDataOrginized[exp] = interpDataOrginized[exp].append(nextRow)
                else:
                    nextRowNP = np.empty((interpData[exp].shape[1], 1))
                    nextRowNP[:] = np.nan
                    nextRow = pd.DataFrame(nextRowNP).T
                    nextRow.columns = list(interpData[exp].columns)
                    nextRow.rename(index={nextRow.index[0]: time}, inplace=True)
                    interpDataOrginized[exp] = interpDataOrginized[exp].append(nextRow)
            interpDataOrginized[exp] = interpDataOrginized[exp].interpolate()
        except:
            q=2

    return interpDataOrginized

def append_natural_var(interpDataFull, interpDataPP, pref):
    for exp in interpDataFull.keys():
        for var in pref['Variables']:
            interpDataPP[exp][var + '_natural'] = interpDataFull[exp][var]
            interpDataPP[exp][var + '_del' + '_natural'] = interpDataFull[exp][var + '_del']
    return interpDataPP

def define_model_valid(interpDataPP, pref):
    interpDataPPModeling = {exp: interpDataPP[exp] for exp in pref['Modeling experiments']}
    interpDataPPValid = {exp: interpDataPP[exp] for exp in pref['Validation experiments']}
    return interpDataPPModeling, interpDataPPValid


def meas_creator(interpDataPP, pref):
    """
    Description:
    A function which takes interpolated data and creates a set of measurements for each
    experiment, with interval of 1 hour.
    Inputs:
      1. interpDataPP- A dictionary containing data frames for relevant experiments. Each data frame represents one
                      experiment after linear interpolation and pre-processing where rows are time and columns
                       are measured variables.
      2. pref- Preferences dictionary
    Outputs:
      1. dataMeasurementsPP- Same structure as interpDataPP, where each experiment holds mean data-frame for every
                           hour. This will be used to train the linear model of each modeled variable.
                            number of rows for every data frame is the length in hours of the relevant experiment.
    """
    dataMeasurementsPP = {}
    for exp in interpDataPP.keys():
        # Run over each hour of the experiment. inside each hour, compute the average of +- 30 min for each measurement
        dataMeasurementsPP[exp] = \
            pd.concat([pd.DataFrame([interpDataPP[exp].iloc[hour-30:hour+30].mean()],
                                    columns=interpDataPP[exp].columns)
                       for hour in range(30, len(interpDataPP[exp]) - 30, 60)],
                      ignore_index=True)

        # Create "dataDelta"- data frame containing the dx/dt, with units [x/hour],
        # where x are modeled variables
        # dataDelta = pd.concat(
        #     [pd.DataFrame([interpDataPP[exp].iloc[hour+30] - interpDataPP[exp].iloc[hour-30]],
        #                   columns=interpDataPP[exp].columns)
        #                for hour in range(30, len(interpDataPP[exp]) - 30, 60)], ignore_index=True)
        # dataDelta = dataDelta[pref['Variables']]
        # columnNames = []
        # for columns in pref['Variables']:
        #     columnNames.append(columns + '_del12')
        # dataDelta.column = columnNames
        # dataMeasurements[exp][dataDelta.column] = dataDelta
    return dataMeasurementsPP

def add_new_features(interpData, pref):

    for exp in interpData.keys():
        dataDelta = interpData[exp][pref['Variables']].diff() * 60 #  delta for wanted variables to model in hours
        dataDelta.iloc[0] = dataDelta.iloc[1]
        dataDelta.columns = [name + '_del' for name in list(interpData[exp][pref['Variables']].columns)]
        interpData[exp][dataDelta.columns] = dataDelta
        columnNames = interpData[exp].columns
        for newFeat in pref['New features']:
            if newFeat not in columnNames:
                interpData[exp][newFeat] = extract_feat(newFeat, interpData[exp], pref)
    return interpData

def extract_feat(newFeat, dataMeasurements, pref):
    # newFeatures = ['DO * Incyte', 'S * Incyte', 'A * Incyte', 'pH * Incyte']
    if not all(elem in pref['Variables'] for elem in pref['New features dict'][newFeat]):
        e.msgbox("Please make sure that new features are generated from selected variables! ", "Error!")
    else:
        # features for Tobramycin
        if newFeat == 'DO * Incyte':
            return dataMeasurements['DO'] * dataMeasurements['Incyte']
        elif newFeat == 'S * Incyte':
            return dataMeasurements['Dextrose[percent]'] * dataMeasurements['Incyte']
        elif newFeat == 'A * Incyte':
            return dataMeasurements['Ammonia[percent]'] * dataMeasurements['Incyte']
        elif newFeat == 'pH * Incyte':
            return dataMeasurements['pH_x'] * dataMeasurements['Incyte']

        #features for biondVax
        elif newFeat == 'DO * OD':
            return dataMeasurements['DO'] * dataMeasurements['OD']
        elif newFeat == 'pH * OD':
            return dataMeasurements['pH'] * dataMeasurements['OD']


def create_combinations(pref):
    """
    Description:
    A function which creates combinations of hyper parameters according to the preferences
    selected by the user.
    Inputs:
      1. pref- Preferences dictionary
    Outputs:
      1. Combinations- A dictionary, containing all Hyper parameters combinations to be executed for this modeled
      variable. Hyper parameters could be- variables of the linear eqations, distance input, fraction etc.
    """
    allFeatForModel = pref['Variables'] + pref['Data variables'] + pref['New features']
    featuresOptions = sum([list(map(list, combinations(allFeatForModel, i)))
                           for i in range(len(allFeatForModel) + 1)], [])
    featuresOptions = featuresOptions[1:]
    distOptions = sum([list(map(list, combinations(pref['featuresDist'], i)))
                           for i in range(len(pref['featuresDist']) + 1)], [])
    distOptions = distOptions[1:]
    Combinations = {}
    counter = 0
    for distComb in distOptions:
        for featOpt in featuresOptions:
            for fracOpt in pref['fracOptions']:
                Combinations[counter] = {'features': featOpt, 'frac': fracOpt, 'featuresDist': distComb}
                counter += 1
    return Combinations


def devide_data_comb(dataMeasurements,trainNames,testNames):
    """
    Description:
    A function which changes the trainData/testData combinations, in order to allow
    the cross validation concept.
    Inputs:
      1. dataMeasurements-  A dictionary containing dataframes for relevant experiments. Each experiment holds mean data-
                            frame for every hour. This will be used to train the linear model of each modeled variable.
                            number of rows for every data frame is the length in hours of the relevant experiment.
      2. trainNames- Names of experiments currently selected for train data.
      3. testNames - Names of experiments currently selected for test data.
    Outputs:
      1. trainData- Dictionary containing data frames of data for selected training experiments.
      2. testData- Dictionary containing data frames of data for selected test experiments.
      3. trainNames- Names of experiments currently selected for train data.
      4. testNames - Names of experiments currently selected for test data.
    """
    stepSize = 1 #How many experiments are moving from train to test every iteration
    trainNames.extend(testNames[:stepSize])
    testNames.extend(trainNames[:stepSize])
    trainNames = trainNames[stepSize:]
    testNames = testNames[stepSize:]
    trainData = {exp: dataMeasurements[exp] for exp in trainNames}
    testData = {exp: dataMeasurements[exp] for exp in testNames}

    return trainData, testData, trainNames, testNames



def combineData(trainData, testData, isTestCombine=True):
    """
    Description:
    A function which takes trainData and testData dictionaries, and creates for each of them
        a data frame where all the experiments are concatenated one after the other.
    Inputs:
      1. trainData- Dictionary containing data frames of data for selected training experiments.
      2. testData- Dictionary containing data frames of data for selected test experiments.
      3. isTestCombine - True if you want to concatenate both train and test, False if only
                         test is wanted.
    Outputs:
      1. trainDataCombined- Dataframe containing all data for selected training experiments,
                            concatenated one after the other.
      2. testDataCombined- Dataframe containing all data for selected test experiments,
                            concatenated one after the other.
    """
    trainDataCombined = pd.DataFrame()
    testDataCombined = pd.DataFrame()
    for exp in trainData.keys():
        trainDataCombined = trainDataCombined.append(trainData[exp], ignore_index=True, sort=False)
    if isTestCombine:
        for exp in testData.keys():
            testDataCombined = testDataCombined.append(testData[exp], ignore_index=True, sort=False)
    else:
        testDataCombined = testData
    return trainDataCombined, testDataCombined


def list_to_dict(modeledVarsList, expNames):
    modeledVars = {}
    for idx in range(0, len(modeledVarsList)):
        modeledVars[expNames[idx]] = modeledVarsList[idx]
    return modeledVars
