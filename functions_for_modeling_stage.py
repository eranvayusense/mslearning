import pandas as pd
import numpy as np
from functions_for_orginization import *
import time
def run_var_model(variable, paramComb, trainData, testData):
    """
    Description:
    A function which runs LLR modeling for a specific variable, and calculates the model's
    sum error for every Hyper parameters combination.
    Inputs:
      1. variable-  Name of the variable to model.
      2. paramComb- Dictionary containing variables for linear equation,
       variables for distance and fraction value for current run.
      3. trainData - Dictionary containing data frames of data for selected training
       experiments.
      4. testData- Dictionary containing data frames of data for selected test experiments.
    Outputs:
      1. errorVal- Sum for all mean errors, comparing data and modeled values for all test
                   data.
    """
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
    #tt=np.repeat(testData[paramComb['featuresDist']].to_numpy(), 4, axis=1)
    errorVal = 0  # Initialize error value
    tParallel=time.time()
    for index, row in allRelTestData.iterrows():
        relTestData = row[paramComb['features']].to_numpy()
        testDistVar = row[paramComb['featuresDist']].to_numpy()
        testResult = row[delVariableName]
        zout1, wout = loess_nd_test_point(relTrainData, trainDistVar, trainResults,
                                      relTestData, testDistVar, frac=paramComb['frac'])

        errorVal += abs(zout1 - testResult)/(abs(zout1) + abs(testResult))
    print('reg run: ' + str(time.time() - tParallel))
    return errorVal
            # np.mean(np.abs((z_smoot_test[:, idxFeat, idxFilt] - titterTestNP) / (titterTestNP + z_smoot_test[:, idxFeat, idxFilt])))
def run_var_model_mat(pref,variable, paramComb, trainData, testData,scale_params):
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

    allModelingData = allRelTrainData.T.drop_duplicates().T  # Remove duplications from "allRelTrainDataInit" dataframe

    relTrainData = allModelingData[paramComb['features']].to_numpy()  # relevant features for linear equation.
    trainDistVar = allModelingData[paramComb['featuresDist']].to_numpy()  # relevant distance values for LLR algorithm.
    trainResults = allModelingData[delVariableName].to_numpy()  # train delta values of modeled variable

    # dataframe containing all non nan test measurements, for relevant features and the modeled variable
    allRelTestDataInit = pd.concat([testData[paramComb['features']], testData[paramComb['featuresDist']],
                                testData[delVariableName],testData[variable]], axis=1).dropna()

    allRelTestData = allRelTestDataInit.T.drop_duplicates().T  # Remove duplications from "allRelTrainDataInit" dataframe

    #tParallel=time.time()
    test_dist={}; train_dist={}; distSumSqr = 0
    for varForDist in paramComb['featuresDist']:
        test_dist[varForDist] = np.repeat(testData[paramComb['featuresDist']].to_numpy().T,
                                          trainDistVar.size, axis=0)
        train_dist[varForDist] = np.repeat(trainDistVar,
                                           len(testData[paramComb['featuresDist']].to_numpy()),
                                           axis=1)
        try:
            distVarSqr = (test_dist[varForDist] - train_dist[varForDist]) ** 2
        except:
            q=2
        distSumSqr += distVarSqr
    npoints = int(np.ceil(paramComb['frac'] * relTrainData[:, 0].size))
    dist = np.sqrt(distSumSqr)
    w = np.argsort(dist,axis=0)[:npoints]
    errorVal = 0  # Initialize error value
   # print('mat1 run: ' + str(time.time() - tParallel))
    tParallel=time.time()
    zout1, wout,bias_re,bias_vel = loess_nd_test_point_mat(pref,allRelTestData, paramComb['features'], relTrainData, trainDistVar,
                                          trainResults, dist, w,scale_params[variable],variable,allRelTestData[variable], frac=paramComb['frac'])
    for index, row in allRelTestData.iterrows():
        #relTestData = row[paramComb['features']].to_numpy()
        #testDistVar = row[paramComb['featuresDist']].to_numpy()
        testResult = row[delVariableName]


        errorVal += abs(zout1[0,index] - testResult)/(abs(zout1[0,index]) + abs(testResult))
    #print('mat2 run: ' + str(time.time() - tParallel))
    return errorVal
            # np.mean(np.abs((z_smoot_test[:, idxFeat, idxFilt] - titterTestNP) / (titterTestNP + z_smoot_test[:, idxFeat, idxFilt])))

def loess_nd_test_point_mat(pref,allRelTestData,paramComb_fetchers,trainModelVar, trainDistVar, trainResults,
                       dist_all,w_all,scale_params,var,varT,frac=0.5, degree=1, sigz=None, npoints=None):
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
   # if npoints is None:
  #      npoints = int(np.ceil(frac * trainModelVar[:, 0].size))
    distSumSqr = 0
    # for varForDist in range(testdistVar.size):
    #     distVarSqr = (trainDistVar[:, varForDist] - testdistVar[varForDist]) ** 2
    #     distSumSqr += distVarSqr
    #dist = np.sqrt(testdistVar)
    #w = np.argsort(dist)[:npoints]
    bias_re=np.zeros([1,allRelTestData.shape[0]])
    zout=np.zeros([1,allRelTestData.shape[0]]) ; wout=np.zeros([1,allRelTestData.shape[0]])
    for index, row in allRelTestData.iterrows():
       # if len(allRelTestData[paramComb_fetchers[0]])>0*1e9:
        testModelVar = row[paramComb_fetchers].to_numpy()
       # else:
        #    testModelVar=allRelTestData.to_numpy()
        dist=dist_all[:, index]
        w=w_all[:, index]
        distWeights = (1 - (dist[w]/dist[w[-1]])**3)**3  # tricube function distance weights
        zfit = polyfit_nd(trainModelVar[w, :], trainResults[w], degree, weights=distWeights)

    # Use errors if those are known.
        bad = []
        for p in range(1):  # do at most 10 iterations       moti

            if sigz is None:  # Errors are unknown
                aerr = np.abs(zfit - trainResults[w])  # Note ABS()
                mad = np.median(aerr)  # Characteristic scale
                uu = (aerr/(6*mad))**2  # For a Gaussian: sigma=1.4826*MAD
            else:  # Errors are assumed known
                uu = ((zfit - trainResults[w])/(4*sigz[w]))**2  # 4*sig ~ 6*mad

            uu = uu.clip(0, 1)
            biWeights = (1 - uu)**2
            totWeights = distWeights*biWeights
            zfit =polyfit_nd(trainModelVar[w, :], trainResults[w], degree, weights=totWeights)
            badOld = bad
            bad = np.where(biWeights < 0.34)[0]  # 99% confidence outliers
            if np.array_equal(badOld, bad):
                break
        if len(allRelTestData[paramComb_fetchers[0]])>1:
           varTs=varT[index]
        else:
           varTs=varT

        coeff, c_test = polyfit_nd_coeff(pref,trainModelVar[w, :], trainResults[w], testModelVar,scale_params,var,varTs, degree, weights=totWeights)

        zout[0, index] = c_test.dot(coeff)# zfit[0]
        wout[0, index]= biWeights[0]
        bias_retio=(coeff[0]/zout[0, index])
        if bias_retio<1.4 and bias_retio>0.7:
            end1=int(len(w)*0.25)
            coeff, c_test = polyfit_nd_coeff(pref,trainModelVar[w[0:end1], :], trainResults[w[0:end1]], testModelVar,scale_params,var,varTs, degree, weights=totWeights[0:end1])
            zout[0, index] = c_test.dot(coeff)
            bias_retio=(coeff[0]/zout[0, index])
            if bias_retio<1.3 and bias_retio>0.7:
                 bias_re[0,index]=1


    return zout, wout,bias_re,bias_retio


def polyfit_nd(trainModelVar, trainResults, degree, sigz=None, weights=None):
    """
    Fit a bivariate polynomial of given DEGREE to a set of points
    (X, Y, Z), assuming errors SIGZ in the Z variable only.

    For example with DEGREE=1 this function fits a plane

       z = a + b*x + c*y

    while with DEGREE=2 the function fits a quadratic surface

       z = a + b*x + c*x^2 + d*y + e*x*y + f*y^2

    """
    from scipy.optimize import minimize
    from scipy.optimize import LinearConstraint
    if weights is None:
        if sigz is None:
            sw = 1.
        else:
            sw = 1./sigz
    else:
        sw = np.sqrt(weights)

    npol = trainModelVar.shape[1] + 1
   # a = np.empty((trainModelVar.shape[0], npol))
    a2 = np.empty((trainModelVar.shape[0], npol))
    #c = np.ones_like(a)
    c2 = np.ones_like(a2)
    # c_test= np.ones_like(a)
    # k = 0
    L1 = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]])
    # re=2**X.shape[1]

    # for r1 in range(npol):# loop on config
    #     for col in range(trainModelVar.shape[1]):# loop on vars
    #         c[:, r1] = c[:, r1]*trainModelVar[:, col]**L1[r1, col] #x**j * y**i
    #         # c_test[:, r1] =c_test[:, r1]*X_test[:,col]**L1[r1,col]
    #     a[:, r1] = c[:, r1]*sw
    #
    # coeff = np.linalg.lstsq(a, trainResults*sw, rcond=None)[0]
    c2[:,1:] = trainModelVar
            # c_test[:, r1] =c_test[:, r1]*X_test[:,col]**L1[r1,col]
   # for r1 in range(npol):# loop on config
    bb=np.repeat([[sw]],c2.shape[1],axis=1)
    a2 = c2*bb[0, :, :].T

    coeff2 = np.linalg.lstsq(a2, trainResults*sw, rcond=None)[0]
    return c2.dot(coeff2)
def f1(x,args):
    a,y = args['aa2'],args['yy1']
    return sum((a.dot(x)-y)**2)

def polyfit_nd_coeff(pref,trainModelVar, trainResults, testModelVar,scale_params,var,varT, degree, sigz=None, weights=None):
    """
    Fit a bivariate polynomial of given DEGREE to a set of points
    (X, Y, Z), assuming errors SIGZ in the Z variable only.

    For example with DEGREE=1 this function fits a plane

       z = a + b*x + c*y

    while with DEGREE=2 the function fits a quadratic surface

       z = a + b*x + c*x^2 + d*y + e*x*y + f*y^2

    """
    from scipy.optimize import minimize
    from scipy.optimize import LinearConstraint
    if weights is None:
        if sigz is None:
            sw = 1.
        else:
            sw = 1./sigz
    else:
        sw = np.sqrt(weights)

    npol = trainModelVar.shape[1] + 1
    # a = np.empty((trainModelVar.shape[0], npol))
    # c = np.ones_like(a)
    # c_test= np.ones_like(a)
    a2 = np.empty((trainModelVar.shape[0], npol))
    c2 = np.ones_like(a2)
    c_test2= np.ones_like(a2)
    k = 0
    #X=np.column_stack((x, y));
    L1 = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]])
    #re=2**X.shape[1]
    # for r1 in range(npol):  # loop on config
    #     for col in range(trainModelVar.shape[1]):# loop on vars
    #         c[:, r1] = c[:, r1]*trainModelVar[:, col]**L1[r1, col]  # x**j * y**i
    #
    #         c_test[0, r1] = c_test[0, r1]*testModelVar[col]**L1[r1, col]
    #     a[:, r1] = c[:, r1]*sw

    #coeff = np.linalg.lstsq(a, trainResults*sw, rcond=None)[0]
    try:
        c2[:, 1:] = trainModelVar
        c_test2[:, 1:] = testModelVar
    except:
        q=2
            # c_test[:, r1] =c_test[:, r1]*X_test[:,col]**L1[r1,col]
   # for r1 in range(npol):# loop on config
    bb=np.repeat([[sw]], c2.shape[1], axis=1)
    a2 = c2*bb[0, :, :].T
    # with contrains moti
    lb=(((0.03-scale_params[0])/scale_params[1])-varT)/1 # or /60 ??? moti
    if pref['preProcessing type']=='scaling (0-1)':
       lb=-varT+(0.03-scale_params[0])/(scale_params[1]-scale_params[0]) # or /60 ??? moti
    coeff2 = np.linalg.lstsq(a2, trainResults*sw, rcond=None)[0]
    if coeff2.dot(c_test2[0, :])<lb:
        y1=trainResults*sw
        additional1={'aa2':a2,'yy1':y1}
        linear_constraint = LinearConstraint([c_test2[0, :]], [lb], [np.inf])
        res = minimize(fun=f1,x0= np.ones([len(c_test2[0, :]),]), args=additional1, method='trust-constr',
                    constraints=linear_constraint)
        coeff2=res.x
    return coeff2, c_test2[0, :]

def sort_param_comb(pref, resultsVec):
    """
    Desciption:
    A function which sorts the results vector and returns the sorted results vector and his
    indexes.
    Inputs:
      1. pref- Preferences dictionary
      2. resultsVec - vector containing results for all combinations
    Outputs:
      1. sortedResultsVecIdx- Indexes of best results
      2. sortedResultsVec- resultsVec after sort. first value represents the best result (based on best Hyper parameters
                           combination).
      3. bestParams- Dictionary containing best variables for linear eqation, variables for distance and fraction value
                     for current run, selected according to their results.
    """
    sortedResultsVecIdx = resultsVec.argsort()
    sortedResultsVec = resultsVec[sortedResultsVecIdx]
    bestParams = pref['Combinations'][sortedResultsVecIdx[0]]
    # combine 1st & 2sc configs , moti
    bestParams_f1 = pref['Combinations'][sortedResultsVecIdx[0]]['features']
    bestParams_f2 = pref['Combinations'][sortedResultsVecIdx[1]]['features']
    bestParams_ff=bestParams_f1# +bestParams_f2
    unique_con = [x for i, x in enumerate(bestParams_ff) if i == bestParams_ff.index(x)]
    bestParams['features']=unique_con
    return sortedResultsVecIdx, sortedResultsVec, bestParams

def run_and_test_full_model(pref, results, modelingDataCombined, validationData,scale_params):
    """
    Desciption:
      A function which simulates full rolling model, using the best linear model for each of the selected variables.
    Inputs:
      1. pref- Preferences dictionary
      2. results- Dictionary, containing a results dictionary for each variable
      3. modelingDataCombined- Data frame with all relevant measurements from all experiments used for modeling(train and
                          test), according to the set of best combinations.
      4. testData- dictionary containing interpolated data frame for every validation experiment.
    Outputs:
      1. modeledVars- Dictionary containing data frames for each test experiment, with all modeled variable values.
    """
    # Initial conditions and constants


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


    Settings, Const, modeledVars = simulation_initialization(validationData, pref)

    currModelStateInit = pd.concat([modeledVars.iloc[0], validationData[pref['Data variables']].iloc[0],
                               validationData[pref['featuresDist']].iloc[0]], axis=0)  #  vector containing current relevant modeled values and controlled parameters
    # Remove duplications from "allRelTrainDataInit" dataframe
    currModelNames = list(currModelStateInit.index)
    nameIdx = []
    nameVec = []
    for idx in range(len(currModelNames)):
        if currModelNames[idx] not in nameVec:
            nameIdx.append(idx)
            nameVec.append(currModelNames[idx])
    currModelState = currModelStateInit.iloc[nameIdx]
    frac1=bestParams['frac'];dt1 = 20
    t_end=modeledVars.index[-1]
    # run model every time step for all modeled variables
    for t in range(0, t_end-(dt1+1), Settings['DT']):  #  start from t=1[minutes]
        currModelState[pref['Data variables']] =\
            validationData[pref['Data variables']].iloc[t]
        #  If featuresDist is a modeled parameter, we should not update it from data!!
        currModelState[pref['featuresDist']] = validationData[pref['featuresDist']].iloc[t]

        for var in pref['Variables']:
            delVariableName = var + '_del'
            if t % dt1:
                modeledVars[var].iloc[t+1] =modeledVars[var].iloc[t]

            else:
                varT=modeledVars[var].to_numpy()[t]
                bestParams = results[var]['bestParams']
                relTestData = currModelState[bestParams['features']]#.to_numpy()
                allRelTestDataInit = pd.concat([currModelState[bestParams['features']],currModelState[bestParams['featuresDist']]]
                                , axis=1).dropna()

                allRelTestData = allRelTestDataInit.T.drop_duplicates().T  # Remove duplications from "allRelTrainDataInit" dataframe

                testDistVar = currModelState[bestParams['featuresDist']].to_numpy()
                test_dist={}; train_dist={}; distSumSqr = 0
                for varForDist in bestParams['featuresDist']:
                    test_dist[varForDist] = np.repeat(testDistVar.T,
                                                      dataDict[var]['trainDistVar'].size, axis=0)
                    train_dist[varForDist] = np.repeat(dataDict[var]['trainDistVar'],
                                                       len(testDistVar),
                                                       axis=1)

                    distVarSqr = (test_dist[varForDist] - train_dist[varForDist]) ** 2
                    distSumSqr+=distVarSqr
                npoints=int(np.ceil(bestParams['frac']*dataDict[var]['trainDistVar'][:,0].size))
                dist=np.sqrt(distSumSqr)
                w=np.argsort(dist,axis=0)[:npoints]
                deltaVar, x ,bias_re,bias_vel= loess_nd_test_point_mat\
                            (pref,pd.DataFrame(relTestData).T,bestParams['features'],dataDict[var]['relTrainData'], dataDict[var]['trainDistVar'],
                             dataDict[var]['trainResults'],dist,w,scale_params[var],var,varT, frac=frac1)
                for jj in range(1,1*len(pref['Combinations']),1): # in case of bias go to the next config
                    ii=results[var]['sortedCombinations'][jj]
                    if bias_vel<0.7 or bias_vel>1.3:
                        break
                    if len(pref['Combinations'][ii]['features']) <2:
                        continue
                    allRelTrainDataInitB = pd.concat([modelingDataCombined[pref['Combinations'][ii]['features']],
                                         modelingDataCombined[pref['Combinations'][ii]['featuresDist']],
                                         modelingDataCombined[delVariableName]], axis=1).dropna()

                    allModelingData = allRelTrainDataInitB.T.drop_duplicates().T  # Remove duplications from "allRelTrainDataInit" dataframe

                    #dataDict[var]['relTrainData'] = allModelingData[pref['Combinations'][ii]['features']].to_numpy()  # relevant features for linear equation.
                    relTestData = currModelState[pref['Combinations'][ii]['features']]
                    deltaVar, x ,bias_re,bias_vel= loess_nd_test_point_mat\
                            (pref,pd.DataFrame(relTestData).T,pref['Combinations'][ii]['features'],
                             allModelingData[pref['Combinations'][ii]['features']].to_numpy(), dataDict[var]['trainDistVar'],
                             dataDict[var]['trainResults'],dist,w,scale_params[var],var,varT, frac=frac1)
                    demo=2
                modeledVars[var].iloc[t+1] = \
                        modeledVars[var].iloc[t] + dt1*deltaVar/60
                modeledVars[var+'_biasVel'].iloc[t+1:t+1+dt1]=list(bias_re*np.ones([1,dt1]).T)
            currModelState[var] = modeledVars[var].iloc[t+1]

    return modeledVars


def gold_hyper(pref, testData, modeledVars):
    # function gold_hyper calculate the goodness of predicted model (modeledVars) to test data (testData)
   # input:
    gold_var = {}
    gold_mean_comul = 0
    for exp in modeledVars.keys():
        #expLength = int(round(testData[exp]['TimeMeas'].iloc[-1])) + 1 #  Experiments length in minutes
        gold_var[exp] = pd.DataFrame()
        for var in pref['Variables']:
           # calculte goodness of prediction  for spesific exp and specific featchare
            gold_var[exp][var] = [sum(abs((testData[exp][var]-modeledVars[exp][var])/
                                      (abs(testData[exp][var]) +
                                       abs(modeledVars[exp][var]))))/len(testData[exp][var])]
        gold_mean_comul = gold_mean_comul+gold_var[exp].mean(axis=1)  # assuming uniform weights for the featchers
    gold_mean = gold_mean_comul/len(testData.keys())
    return gold_mean



def simulation_initialization(expData, pref):

    """
    Desciption:
      A function which initializes vectores for each of the modeled variables, sets costant values and
      determines setting for the rolling model protocol.
    Inputs:
      1. expData- Data frame of measured values for specific experiment.
      2. pref- Preferences dictionary
    Outputs:
      1. Settings- Dictionary containing settings for rolling model protocol
      2. Const- Dictionary containing Chemical, Physical and technical constants of the model
      3. modeledVarsForExp- Data frame containing vectors at experiment's length of each modeled variables, filled with
                            the initial values of the experiment.
    """
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
        modeledVarsForExp[var+'_biasVel'] = expData[var].iloc[0] * np.zeros([expLength, ])
    # InitCond['X_0'] = 0.15*np.ones([])
    # InitCond['Vl_0'] = 80
    # InitCond['Vg_0'] = Const['TOTAL_VOLUME'] - InitCond['Vl_0']
    # InitCond['DO_0'] = Const['DO_MAX']
    # InitCond['P1_0'] = 0

    # return values
    return Settings, Const, modeledVarsForExp


def run_var_model_for_all_CV(paramComb, pref, dataMeasurements, variable,scale_params):
    """
    Desciption:
      A function which runs model for a specific variable for all relevant cross validation
      options
    Inputs:
      1. paramComb- Data frame of measured values for specific experiment.
      2. pref- Preferences dictionary
      3. dataMeasurements- Dictionary containing data frames of data for selected experiments.
      4. variable- Variable to model
    Outputs:
      1. scoreForParamComb- Sum of the score for all CV options
    """
    CVTrain = pref['CVTrain']
    CVTest = pref['CVTest']
    scoreForParamComb = 0
    if (variable in pref['Combinations'][paramComb]['features']) or\
            len(pref['Combinations'][paramComb]['features']) < 2:  # moti
        scoreForParamComb = 1e8
    else:
        CVend = 1 # pref['numCVOpt'] moti
        for CVOpt in range(0, CVend):  # 'numCVOpt' is the number of train/test division options
            trainData, testData, CVTrain, CVTest = \
                devide_data_comb(dataMeasurements, CVTrain, CVTest)

            # Take dictionary of dataframe for every experiment, and output a dataframe with all experiments together
            # features = pref['Combinations'][paramComb]['features']
            trainDataCombined, testDataCombined = \
                combineData(trainData, testData)

            # Calculate mean goal function score for all test experiments and sum for all possibilities of Cross Validation
            # scoreForParamComb += run_var_model(variable, pref['Combinations'][paramComb],
            #                                     trainDataCombined, testDataCombined)
            # Optional upgrade for run_var_model:
            scoreForParamComb += run_var_model_mat(pref,variable, pref['Combinations'][paramComb],
                                                trainDataCombined, testDataCombined,scale_params)
    return scoreForParamComb
