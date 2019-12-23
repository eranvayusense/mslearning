import pandas as pd
import numpy as np


def run_var_model(variable, paramComb, trainData, testData):
# Inputs:
#   1. variable-  Name of the variable to model.
#   2. paramComb- Dictionary containing variables for linear eqation, variables for distance and fraction value for
#                 current run.
#   3. trainData - Dictionary containing data frames of data for selected training experiments.
#   4. testData- Dictionary containing data frames of data for selected test experiments.
# Outputs:
#   1. errorVal- Sum for all mean errors, comparing data and modeled values for all test data,

    delVariableName = variable + '_del'  # name for delta value of variable, so that it is different from original name.

    # dataframe containing all non nan train measurements, for relevant features and the modeled variable
    allRelTrainData = pd.concat([trainData[paramComb['features']], trainData[paramComb['featuresDist']],
                            trainData[delVariableName]], axis=1).dropna()

    relTrainData = allRelTrainData[paramComb['features']].to_numpy()  # relevant features for linear equation.
    trainDistVar = allRelTrainData[paramComb['featuresDist']].to_numpy()  # relevant distance values for LLR algorithm.
    trainResults = allRelTrainData[delVariableName].to_numpy()  # train delta values of modeled variable

    # dataframe containing all non nan test measurements, for relevant features and the modeled variable
    allRelTestData = pd.concat([testData[paramComb['features']], testData[paramComb['featuresDist']],
                                testData[delVariableName]], axis=1).dropna()
    errorVal = 0  # Initialize error value
    for index, row in allRelTestData.iterrows():
        relTestData = row[paramComb['features']].to_numpy()
        testDistVar = row[paramComb['featuresDist']].to_numpy()
        testResult = row[delVariableName]
        zout1, wout = loess_nd_test_point(relTrainData, trainDistVar, trainResults,
                                      relTestData, testDistVar, frac=paramComb['frac'])

        errorVal += abs(zout1 - testResult)/(abs(zout1) + abs(testResult))
    return errorVal
            # np.mean(np.abs((z_smoot_test[:, idxFeat, idxFilt] - titterTestNP) / (titterTestNP + z_smoot_test[:, idxFeat, idxFilt])))


def loess_nd_test_point(trainModelVar, trainDistVar, trainResults, testModelVar, testdistVar,
                        frac=0.5, degree=1, sigz=None, npoints=None):
    """
# Inputs:
#   1. trainModelVar- Data frame for all linear modeling variables values in training stock. each row represents
#                     a measurements at specific time.
#   2. trainDistVar- Data frame for all distance variables values in training stock.
#   3. trainResults- Modeled variable actual values for each measurement
#   4. testModelVar- List of linear modeling variables representing one test measurement
#   5. testDistVar- List of distance variables representing one test measurement
#   6. testResult- Modeled variable actual values for each measurement
#   7. frac- Float value representing the fraction of points building a linear model for a given test measurement
#   8. degree- Integer representing the degree of the equation (degree=1 is linear equation)
#   9. rescale- binary value (True/False) for rescaling the data before creating the model
#   10. sigz- 1-sigma errors for the Z values. If this keyword is used the biweight fit is done assuming those errors.
#             If this keyword is *not* used, the biweight fit determines the errors in Z from the scatter of the
#             neighbouring points.
# Outputs:
#   1. zout- Modeled values for wanted variable to model.
#   2. wout- Weights of importance for algorithm according to distace
    """


    if frac == 0:
        return trainResults, np.ones_like(trainResults)

    if npoints is None:
        npoints = int(np.ceil(frac * trainModelVar[:, 0].size))

    distSumSqr = 0
    for varForDist in range(testdistVar.size):
        distVarSqr = (trainDistVar[:, varForDist] - testdistVar[varForDist]) ** 2
        distSumSqr += distVarSqr
    dist = np.sqrt(distSumSqr)
    w = np.argsort(dist)[:npoints]
    distWeights = (1 - (dist[w]/dist[w[-1]])**3)**3  # tricube function distance weights
    zfit = polyfit_nd(trainModelVar[w, :], trainResults[w], degree, weights=distWeights)

    # Use errors if those are known.
    bad = []
    for p in range(10):  # do at most 10 iterations

        if sigz is None:  # Errors are unknown
            aerr = np.abs(zfit - trainResults[w])  # Note ABS()
            mad = np.median(aerr)  # Characteristic scale
            uu = (aerr/(6*mad))**2  # For a Gaussian: sigma=1.4826*MAD
        else:  # Errors are assumed known
            uu = ((zfit - trainResults[w])/(4*sigz[w]))**2  # 4*sig ~ 6*mad

        uu = uu.clip(0, 1)
        biWeights = (1 - uu)**2
        totWeights = distWeights*biWeights
        zfit =polyfit_nd(trainModelVar[w,:], trainResults[w], degree, weights=totWeights)
        badOld = bad
        bad = np.where(biWeights < 0.34)[0] # 99% confidence outliers
        if np.array_equal(badOld, bad):
            break
    coeff, c_test = polyfit_nd_coeff(trainModelVar[w, :], trainResults[w], testModelVar, degree, weights=totWeights)

    zout = c_test.dot(coeff)# zfit[0]
    wout = biWeights[0]


    return zout, wout

def polyfit_nd(trainModelVar, trainResults, degree, sigz=None, weights=None):
    """
    Fit a bivariate polynomial of given DEGREE to a set of points
    (X, Y, Z), assuming errors SIGZ in the Z variable only.

    For example with DEGREE=1 this function fits a plane

       z = a + b*x + c*y

    while with DEGREE=2 the function fits a quadratic surface

       z = a + b*x + c*x^2 + d*y + e*x*y + f*y^2

    """
    if weights is None:
        if sigz is None:
            sw = 1.
        else:
            sw = 1./sigz
    else:
        sw = np.sqrt(weights)

    npol = trainModelVar.shape[1] + 1
    a = np.empty((trainModelVar.shape[0], npol))
    c = np.ones_like(a)
    # c_test= np.ones_like(a)
    # k = 0
    L1 = np.array([[0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0],
                   [0, 0, 1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0, 0, 0],
                   [0, 0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 0, 1]])
    # re=2**X.shape[1]
    for r1 in range(npol):# loop on config
        for col in range(trainModelVar.shape[1]):# loop on vars
            c[:, r1] = c[:, r1]*trainModelVar[:, col]**L1[r1, col] #x**j * y**i
            # c_test[:, r1] =c_test[:, r1]*X_test[:,col]**L1[r1,col]
        a[:, r1] = c[:, r1]*sw

    coeff = np.linalg.lstsq(a, trainResults*sw, rcond=None)[0]

    return c.dot(coeff)

def polyfit_nd_coeff(trainModelVar, trainResults, testModelVar, degree, sigz=None, weights=None):
    """
    Fit a bivariate polynomial of given DEGREE to a set of points
    (X, Y, Z), assuming errors SIGZ in the Z variable only.

    For example with DEGREE=1 this function fits a plane

       z = a + b*x + c*y

    while with DEGREE=2 the function fits a quadratic surface

       z = a + b*x + c*x^2 + d*y + e*x*y + f*y^2

    """
    if weights is None:
        if sigz is None:
            sw = 1.
        else:
            sw = 1./sigz
    else:
        sw = np.sqrt(weights)

    npol = trainModelVar.shape[1] + 1
    a = np.empty((trainModelVar.shape[0], npol))
    c = np.ones_like(a)
    c_test= np.ones_like(a)
    k = 0
    #X=np.column_stack((x, y));
    L1 = np.array([[0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0],
                   [0, 0, 1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0, 0, 0],
                   [0, 0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 0, 1]])
    #re=2**X.shape[1]
    for r1 in range(npol):  # loop on config
        for col in range(trainModelVar.shape[1]):# loop on vars
            c[:, r1] = c[:, r1]*trainModelVar[:, col]**L1[r1, col]  # x**j * y**i

            c_test[0, r1] = c_test[0, r1]*testModelVar[col]**L1[r1, col]
        a[:, r1] = c[:, r1]*sw

    coeff = np.linalg.lstsq(a, trainResults*sw, rcond=None)[0]

    return coeff, c_test[0, :]

def sort_param_comb(pref, resultsVec):
# Inputs:
#   1. pref- Preferences dictionary
#   2. resultsVec - vector containing results for all combinations
# Outputs:
#   1. sortedResultsVecIdx- Indexes of best results
#   2. sortedResultsVec- resultsVec after sort. first value represents the best result (based on best Hyper parameters
#                        combination).
#   3. bestParams- Dictionary containing best variables for linear eqation, variables for distance and fraction value
#                  for current run, selected according to their results.
    sortedResultsVecIdx = resultsVec.argsort()
    sortedResultsVec = resultsVec[sortedResultsVecIdx]
    bestParams = pref['Combinations'][sortedResultsVecIdx[0]]
    return sortedResultsVecIdx, sortedResultsVec, bestParams

def run_and_test_full_model(pref, results, modelingDataCombined, validationData):
# Disciption:
#   A function which simulates full rolling model, using the best linear model for each of the selected variables.
# Inputs:
#   1. pref- Preferences dictionary
#   2. results- Dictionary, containing a results dictionary for each variable
#   3. modelingDataCombined- Data frame with all relevant measurements from all experiments used for modeling(train and
#                       test), according to the set of best combinations.
#   4. testData- dictionary containing interpolated data frame for every validation experiment.
# Outputs:
#   1. modeledVars- Dictionary containing data frames for each test experiment, with all modeled variable values.

    # Initial conditions and constants
    modeledVars = {}


    for exp in validationData.keys():
        #  Create a Dictionary containing relevant data for linear modeling of each modeled variable.
        dataDict = {}
        for var in pref['Variables']:
            delVariableName = var + '_del'
            dataDict[var] = {}
            bestParams = results[var]['bestParams']
            # dataframe containing all non nan train measurements, for relevant features and the modeled variable
            allRelTrainDataInit = pd.concat([modelingDataCombined[bestParams['features']],
                                             modelingDataCombined[bestParams['featuresDist']],
                                             modelingDataCombined[delVariableName]], axis=1).dropna()

            allModelingData = allRelTrainDataInit.T.drop_duplicates().T  # Remove duplications from "allRelTrainDataInit" dataframe

            dataDict[var]['relTrainData'] = allModelingData[bestParams['features']].to_numpy()  # relevant features for linear equation.
            dataDict[var]['trainDistVar'] = allModelingData[bestParams['featuresDist']].to_numpy()  # relevant distance values for LLR algorithm.
            dataDict[var]['trainResults'] = allModelingData[delVariableName].to_numpy()  # train delta values of modeled variable


        Settings, Const, modeledVars[exp] = simulation_initialization(validationData[exp], pref)

        currModelStateInit = pd.concat([modeledVars[exp].iloc[0], validationData[exp][pref['Data variables']].iloc[0],
                                   validationData[exp][pref['featuresDist']].iloc[0]], axis=0)  #  vector containing current relevant modeled values and controlled parameters
        # Remove duplications from "allRelTrainDataInit" dataframe
        currModelNames = list(currModelStateInit.index)
        nameIdx = []
        nameVec = []
        for idx in range(len(currModelNames)):
            if currModelNames[idx] not in nameVec:
                nameIdx.append(idx)
                nameVec.append(currModelNames[idx])
        currModelState = currModelStateInit.iloc[nameIdx]

        t_end=modeledVars[exp].index[-1]
        # run model every time step for all modeled variables
        for t in range(0, t_end, Settings['DT']):  #  start from t=1[minutes]
            currModelState[pref['Data variables']] =\
                validationData[exp][pref['Data variables']].iloc[t]
            #  If featuresDist is a modeled parameter, we should not update it from data!!
            currModelState[pref['featuresDist']] = validationData[exp][pref['featuresDist']].iloc[t]
            for var in pref['Variables']:
                bestParams = results[var]['bestParams']
                relTestData = currModelState[bestParams['features']].to_numpy()
                testDistVar = currModelState[bestParams['featuresDist']].to_numpy()
                deltaVar, x = loess_nd_test_point\
                        (dataDict[var]['relTrainData'], dataDict[var]['trainDistVar'],
                         dataDict[var]['trainResults'], relTestData,
                         testDistVar, frac=bestParams['frac'])
                modeledVars[exp][var].iloc[t+1] = \
                    modeledVars[exp][var].iloc[t] + deltaVar/60
                currModelState[var] = modeledVars[exp][var].iloc[t+1]
    gold_mean = gold_hyper(pref, validationData, modeledVars)
    return modeledVars, gold_mean


def gold_hyper(pref,testData,modeledVars):
    # function gold_hyper calculate the goodness of predicted model (modeledVars) to test data (testData)
   # input:
    gold_var={}
    gold_mean_comul = 0
    for exp in testData.keys():
        #expLength = int(round(testData[exp]['TimeMeas'].iloc[-1])) + 1 #  Experiments length in minutes
        gold_var[exp] = pd.DataFrame()
        for var in pref['Variables']:
           # calculte goodness of prediction  for spesific exp and specific featchare
            gold_var[exp][var]=[sum(abs((testData[exp][var]-modeledVars[exp][var])/
                                        (abs(testData[exp][var])+abs(modeledVars[exp][var]))))/len(testData[exp][var])]
        gold_mean_comul = gold_mean_comul+gold_var[exp].mean(axis=1)  # assuming uniform weights for the featchers
    gold_mean = gold_mean_comul/len(testData.keys())
    return gold_mean



def simulation_initialization(expData, pref):
# Disciption:
#   A function which initializes vectores for each of the modeled variables, sets costant values and
#   determines setting for the rolling model protocol.
# Inputs:
#   1. expData- Data frame of measured values for specific experiment.
#   2. pref- Preferences dictionary
# Outputs:
#   1. Settings- Dictionary containing settings for rolling model protocol
#   2. Const- Dictionary containing Chemical, Physical and technical constants of the model
#   3. modeledVarsForExp- Data frame containing vectors at experiment's length of each modeled variables, filled with
#                         the initial values of the experiment.

    # Settings initialization
    Settings = dict()
    Settings['DT'] = 1  # [min]
    Settings['case'] = 1

    # Constants initialization
    Const = dict()
    Const['CO2_MOLAR_MASS'] = 44.01
    Const['DEXTROSE_MOLAR_MASS'] = 180.16
    Const['R_CONST'] = 0.082057
    Const['NUM_C_IN_DEXSTROSE'] = 6
    Const['DO_MAX'] = 7e-3
    Const['TOTAL_VOLUME'] = 150
    Const['AF_REF_VAL'] = 0.35
    Const['AMMONIA_FEEDING_CONCENTRATION'] = 0.25
    Const['DEXTROSE_FEEDING_CONCENTRATION'] = 0.5
    Const['LOW_DEX_VAL'] = 5
    Const['DEPRESSION_MAX_VAL'] = 1 / 3

    # initial conditions data frame initialization
    modeledVarsForExp = pd.DataFrame()
    expLength = int(round(expData['TimeMeas'].iloc[-1])) + 1 #  Experiments length in minutes
    # Create a dataframe for future modeled values, with all wanted modeled variables
    for var in pref['Variables']:
        modeledVarsForExp[var] = expData[var].iloc[0] * np.ones([expLength, ])
    # InitCond['X_0'] = 0.15*np.ones([])
    # InitCond['Vl_0'] = 80
    # InitCond['Vg_0'] = Const['TOTAL_VOLUME'] - InitCond['Vl_0']
    # InitCond['DO_0'] = Const['DO_MAX']
    # InitCond['P1_0'] = 0

    # return values
    return Settings, Const, modeledVarsForExp
