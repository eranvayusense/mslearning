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
    pref['featuresDist'] = ['Time']
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
        global varForModel, varForData, fractionMinVal, fractionMaxVal
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
    data, dataPP,scale_params  = load_data(processType, preProcessing, isFilterData, isLoadInterpolated = isLoadInterpolated)#load data for variable options
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
    else:
        modeledVariables = varOpt
    # varOpt = list(data[expList[5]].columns)


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


    fracMinText = Label(secondGUI, text='Minimal fraction for group:', font=('Helvetica', '10', 'bold'))
    fracMinText.grid(row=2, column=1, sticky=E, pady=10, padx=20)
    FracMinValEntry = Entry(secondGUI)
    FracMinValEntry.grid(row=2, column=2, pady=10)
    FracMinValEntry.delete(0, END)
    FracMinValEntry.insert(0, "0.3")

    fracMaxText = Label(secondGUI, text='Maximal fraction for group:', font=('Helvetica', '10', 'bold'))
    fracMaxText.grid(row=2, column=3, sticky=E, pady=10, padx=20)
    FracMaxValEntry = Entry(secondGUI)
    FracMaxValEntry.grid(row=2, column=4, pady=10)
    FracMaxValEntry.delete(0, END)
    FracMaxValEntry.insert(0, "0.3")

    button = Button(text="Continue", command=passForword2, width=20, height=1, font=('Helvetica', '17'))
    button.place(rely=0.95, relx=0.5, anchor=CENTER)

    secondGUI.mainloop()

    relVarInExp = varInExp[varForModel]#create 0/1 dataframe only for selected variables
    relExp = list(relVarInExp[relVarInExp.sum(axis=1) == len(varForModel)].index)
    random.shuffle(relExp)# Shuffle experiments order to avoid unwanted dependencies
    modelingRelExp = relExp[0:int(float(relDataModelingSize) * len(relExp))]
    validationRelExp = relExp[int(float(relDataModelingSize) * len(relExp)):]


    pref = {'Variables': varForModel,'Data variables': varForData, 'Process Type': processType,
            'preProcessing type': preProcessing, 'Rel train size': relTrainSize,
            'Rel modeling size': relDataModelingSize, 'Is filter data': isFilterData,
            'Fraction minimal value': fractionMinVal, 'Fraction maximal value': fractionMaxVal,
            'Relevant experiments': relExp, 'allVariables': varOpt, 'isLoadInterpolated': isLoadInterpolated,
            'Modeling experiments': modelingRelExp, 'Validation experiments': validationRelExp,
            'Is run parallel': isRunParallel}
    pref['numCVOpt'] = len(modelingRelExp) #number of options for different CV
    pref['CVTrain'] = modelingRelExp[0:int(float(relTrainSize) * len(modelingRelExp))]# Chosen experiments for train, in first CV division
    pref['CVTest'] = modelingRelExp[int(float(relTrainSize) * len(modelingRelExp)):]# Chosen experiments for test, in first CV division
    return pref
    # list(data.columns)


def load_data(process, preProcessing, isFilterData, relExp='All', isLoadInterpolated=1):
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
        fileName = 'RnD_data_new.p'
        with open(os.path.join(folderName, fileName), 'rb') as f:
            data = pickle.load(f)
        if isFilterData:
            data = filter_data(data, processType=process)  # Activate smoothing function on data

    # screen out unrelevant experiments
    if relExp=='All':
        relExp = data.keys()
    data = {wantedExp: data[wantedExp] for wantedExp in relExp}

    # If data is not interpolated, conduct interpolation
    if isLoadInterpolated:
        interpData = data
    else:
        interpData = data_interp_df(data)

    # Conduct pre-processing according to wanted type
    interpDataPP ,scale_params= pre_process_function(interpData, preProcessing)


    return interpData, interpDataPP,scale_params


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
    scale1= pd.DataFrame()
    for var in varNames[:-1]:
        try:
            if preProcessing == 'scaling (0-1)':
                scale1[var] = [dataCombined[var].min(),dataCombined[var].max()]
            else:
                scale1[var] = [dataCombined[var].median(), scipy.stats.iqr(dataCombined[var])]
        except:
            dd=1
    return dataPP,scale1

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
        try:
            interpDataOrginized[exp] =interpDataOrginized[exp].interpolate()
        except:
            q=2

    return interpDataOrginized

def meas_creator(interpDataPP, pref):
    """
    Description:
    A function which takes interpolated data and creates a set of measurements for each
    experiment, with interval of 1 hour.
    Inputs:
      1. interpDataPP- a dictionary containing dataframes for relevant experiments. Each data frame represents one
                      experiment after linear interpulation where rows are time and columns are measured variables.
      2. pref- preferences dictionary
    Outputs:
      1. dataMeasurements- same structure as interpDataPP, where each experiment holds mean data-frame for every hour.
                           This will be used to train the linear model of each modeled variable. number of rows for
                           every data frame is the length in hours of the relevant experiment.
    """
    dataMeasurements = {}
    for exp in interpDataPP.keys():
        # Run over each hour of the experiment. inside each hour, compute the average of +- 30 min for each measurement
        dataMeasurements[exp] = \
            pd.concat([pd.DataFrame([interpDataPP[exp].iloc[hour-30:hour+30].mean()],
                                    columns=interpDataPP[exp].columns)
                       for hour in range(30, len(interpDataPP[exp]) - 30, 60)],
                      ignore_index=True)

        # Create "dataDelta"- data frame containing the dx/dt, with units [x/hour],
        # where x are modeled variables
        dataDelta = pd.concat(
            [pd.DataFrame([interpDataPP[exp].iloc[hour+30] - interpDataPP[exp].iloc[hour-30]],
                          columns=interpDataPP[exp].columns)
                       for hour in range(30, len(interpDataPP[exp]) - 30, 60)], ignore_index=True)
        dataDelta = dataDelta[pref['Variables']]
        columnNames = []
        for columns in pref['Variables']:
            columnNames.append(columns + '_del')
        dataDelta.column = columnNames
        dataMeasurements[exp][dataDelta.column] = dataDelta

    return dataMeasurements


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
    allFeatForModel = pref['Variables'] + pref['Data variables']
    featuresOptions = sum([list(map(list, combinations(allFeatForModel, i)))
                           for i in range(len(allFeatForModel) + 1)], [])
    featuresOptions = featuresOptions[1:]
    Combinations = {}
    counter = 0
    for featOpt in featuresOptions:
        for fracOpt in pref['fracOptions']:
            Combinations[counter] = {'features': featOpt, 'frac': fracOpt, 'featuresDist': pref['featuresDist']}
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
