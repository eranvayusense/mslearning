def load_data(isLoadData, selectedExp):
    import pickle
    if isLoadData:
        data, offlineData = read_data(selectedExp)
    else:
        file_name = "allData.p"
        with open(file_name, 'rb') as f:
            allData = pickle.load(f)
        data = {key: allData[key] for key in allData.keys() & selectedExp}
        file_name = "allOfflineData.p"
        with open(file_name, 'rb') as f:
            allOfflineData = pickle.load(f)
        string = '_dex'
        offlineData = {key: allOfflineData[key] for key in allOfflineData.keys() & [x + string for x in selectedExp]}
    return data, offlineData

def load_data_df(isLoadData, selectedExp):
    import pickle
    if isLoadData:
        fileName = 'RnD_data_new.p'
        with open(fileName, 'rb') as f:
            data = pickle.load(f)
    else:
        fileName = 'RnD_data_interpolated.p'
        data = dict()
        with open(fileName, 'rb') as f:
            allData = pickle.load(f)
        for exp in selectedExp:
            data[exp] = allData[exp]



    return data

def data_interp(data, offlineData, selectedExp, Const):
    import numpy as np
    import datetime
    dataDictInterp = dict()
    from scipy import interpolate
    for exp in selectedExp:
        dataOnlineRel = data[exp]
        dataOfflineRel = offlineData[exp + '_dex']
        timesOfflineMin = dataOfflineRel['Age'].values * 60
        if np.isnan(timesOfflineMin).any():
            findLastIdx = np.argwhere(np.isnan(timesOfflineMin))[0][0]
        else:
            findLastIdx = len(timesOfflineMin)
        timesOfflineMin = timesOfflineMin[0:findLastIdx]
        timesOfflineMin[0] = 0
        timesOnlineMin = dataOnlineRel['Age'].values * 60
        finalTime = max(timesOnlineMin[-1], timesOfflineMin[-1])
        relDO = dataOnlineRel['DO'].values
        relDO[relDO < 1] = 1
        relDO = relDO * Const['DO_MAX'] / 100
        DOInterpFunc = interpolate.interp1d(timesOnlineMin, relDO, fill_value='extrapolate')
        DOInterp = DOInterpFunc(range(0, int(round(finalTime))))

        # DOInterp = np.interp(range(0, finalTime), timesOnlineMin, relDO)
        kanaData = dataOfflineRel['Kanamycin'][0:findLastIdx].values
        isNanKana = np.isnan(kanaData)
        kanaInterpFunc = interpolate.interp1d(timesOfflineMin[~isNanKana], kanaData[~isNanKana],
                                              fill_value='extrapolate')
        kanaInterp = kanaInterpFunc(range(0, int(round(finalTime))))

        tobraData = dataOfflineRel['Tobramycin'][0:findLastIdx].values
        isNanTobra = np.isnan(tobraData)
        tobraInterpFunc = interpolate.interp1d(timesOfflineMin[~isNanTobra], tobraData[~isNanTobra],
                                               fill_value='extrapolate')
        tobraInterp = tobraInterpFunc(range(0, int(round(finalTime))))

        ammoniaData = dataOfflineRel['Ammonia[percent]'][0:findLastIdx].values
        isNanAmmonia = np.isnan(ammoniaData)
        ammoniaInterpFunc = interpolate.interp1d(timesOfflineMin[~isNanAmmonia], ammoniaData[~isNanAmmonia],
                                                 fill_value='extrapolate')
        ammoniaInterp = ammoniaInterpFunc(range(0, int(round(finalTime))))

        dexData = dataOfflineRel['Dextrose[percent]'][
                  0:findLastIdx].values * 10  # multiply by 10 to change from [%] to [g/l]
        isNanDex = np.isnan(dexData)
        dexInterpFunc = interpolate.interp1d(timesOfflineMin[~isNanDex], dexData[~isNanDex],
                                             fill_value='extrapolate')
        dexInterp = dexInterpFunc(range(0, int(round(finalTime))))

        agitationData = dataOnlineRel['Agitation'].values
        agitationInterpFunc = interpolate.interp1d(timesOnlineMin, agitationData, fill_value='extrapolate')
        agitationInterp = agitationInterpFunc(range(0, int(round(finalTime))))

        airflowData = dataOnlineRel['Airflow'].values
        airflowInterpFunc = interpolate.interp1d(timesOnlineMin, airflowData, fill_value='extrapolate')
        airflowInterp = airflowInterpFunc(range(0, int(round(finalTime))))

        pHData = dataOnlineRel['pH'].values
        pHInterpFunc = interpolate.interp1d(timesOnlineMin, pHData, fill_value='extrapolate')
        pHInterp = pHInterpFunc(range(0, int(round(finalTime))))

        ammoniaFeedData = dataOnlineRel['Fa'].values
        ammoniaFeedInterpFunc = interpolate.interp1d(timesOnlineMin, ammoniaFeedData, fill_value='extrapolate')
        ammoniaFeedInterp = ammoniaFeedInterpFunc(range(0, int(round(finalTime))))

        dexFeedData = dataOnlineRel['Fs'].values
        dexFeedInterpFunc = interpolate.interp1d(timesOnlineMin, dexFeedData, fill_value='extrapolate')
        dexFeedInterp = dexFeedInterpFunc(range(0, int(round(finalTime))))

        if 'Incyte' in dataOfflineRel:
            IncyteData = dataOfflineRel['Incyte'].values[1:-1]
            IncyteData[IncyteData < 0] = 0
            IncyteTimes = dataOfflineRel['Incyte time'].values[1:-1]
            if type(IncyteTimes[0]) is str:
                dataTimeZero = datetime.datetime.strptime(IncyteTimes[0], '%d/%m/%Y %H:%M:%S')  # data time format
            else:
                strTime = IncyteTimes[0].strftime('%d/%m/%Y %H:%M:%S')
                dataTimeZero = datetime.datetime.strptime(strTime, '%m/%d/%Y %H:%M:%S')
            IncyteTime = np.zeros([1, len(IncyteTimes)])
            for time in range(0, len(IncyteTimes)):
                if type(IncyteTimes[time]) is str:
                    dataTime = datetime.datetime.strptime(IncyteTimes[time], '%d/%m/%Y %H:%M:%S')  # data time format
                else:
                    strTime = IncyteTimes[time].strftime('%d/%m/%Y %H:%M:%S')
                    dataTime = datetime.datetime.strptime(strTime, '%m/%d/%Y %H:%M:%S')
                IncyteTime[0, time] = (dataTime - dataTimeZero).seconds / 60 + (dataTime - dataTimeZero).days * 24 * 60
            IncyteInterpFunc = interpolate.interp1d(IncyteTime[0], IncyteData, fill_value='extrapolate')
            incyteInterp = IncyteInterpFunc(range(0, int(round(finalTime))))
            incyteInterp[incyteInterp < 0] = 0
            dataDictInterp[exp] = {'timesOfflineMin': timesOfflineMin, 'timesOnlineMin': timesOnlineMin,
                                   'finalTime': finalTime,
                                   'DO': DOInterp, 'kanaInterp': kanaInterp, 'tobraInterp': tobraInterp,
                                   'A': ammoniaInterp, 'agitationInterp': agitationInterp,
                                   'airflowInterp': airflowInterp, 'pHInterp': pHInterp,
                                   'timeInterp': range(0, int(round(finalTime))), 'S': dexInterp,
                                   'incyte': incyteInterp, 'FaInterp': ammoniaFeedInterp, 'FsInterp': dexFeedInterp}
        else:
            dataDictInterp[exp] = {'timesOfflineMin': timesOfflineMin, 'timesOnlineMin': timesOnlineMin,
                                   'finalTime': finalTime,
                                   'DO': DOInterp, 'kanaInterp': kanaInterp, 'tobraInterp': tobraInterp,
                                   'A': ammoniaInterp, 'agitationInterp': agitationInterp,
                                   'airflowInterp': airflowInterp, 'pHInterp': pHInterp,
                                   'timeInterp': range(0, int(round(finalTime))), 'S': dexInterp,
                                   'FaInterp': ammoniaFeedInterp, 'FsInterp': dexFeedInterp}

    return dataDictInterp

def data_interp_df(data, selectedExp):
    import pandas as pd
    import numpy as np
    import scipy
    interpData = dict()#empty data frame
    interpDataOrginized = dict()
    for exp in selectedExp:
        interpData[exp] = data[exp]
        interpData[exp].index = interpData[exp].index * 60
        #npInterpData = interpData[exp].to_numpy()

        # interpData[exp].index = interpData[exp].index.astype(int)
        for time in range(0,int(interpData[exp].index[-1])):
            abs_sub = abs(time - interpData[exp].index.to_numpy())
            min_idx = np.array([np.where(abs_sub == np.amin(abs_sub))[0][0]])
            if time == 0:
                nextRow = interpData[exp].loc[interpData[exp].index[min_idx]]
                nextRow.rename(index = {nextRow.index[0]:time}, inplace=True)
                interpDataOrginized[exp] = nextRow
            else:
                nextRow = interpData[exp].loc[interpData[exp].index[min_idx]]
                nextRow.rename(index = {nextRow.index[0]:time}, inplace=True)
                interpDataOrginized[exp] = interpDataOrginized[exp].append(nextRow)
        interpDataOrginized[exp] =interpDataOrginized[exp].interpolate()

        interpDataOrginized[exp].rename(columns={'Dextrose[percent]': 'S', 'Ammonia[percent]':'A', }, inplace=True)
    return interpDataOrginized


def devide_data(wantedData):
    import math
    import random
    expNames = list(wantedData.keys())
    numOfExp = len(wantedData)
    numExpForTrain = math.floor(numOfExp*0.7)
    # numExpForTest = numOfExp-numExpForTrain
    expIdx = range(0, numOfExp - 1)
    trainIdx = random.sample(expIdx, numExpForTrain)
    testIdx = set(expIdx).difference(trainIdx)
    trainData = {key: wantedData[key] for key in wantedData.keys() & [expNames[i]for i in trainIdx]}
    testData = {key: wantedData[key] for key in wantedData.keys() & [expNames[i]for i in testIdx]}
    trainData['expNames'] = [expNames[i]for i in trainIdx]
    testData['expNames'] = [expNames[i]for i in testIdx]
    return trainData,testData
def devide_data_comb(wantedData,expIdx,trainIdx):
    import math
    import random
    expNames = list(wantedData.keys())


    # numExpForTest = numOfExp-numExpForTrain

    #trainIdx=list([13, 16, 22, 4, 15, 27, 5, 28, 14, 23, 17, 25, 7, 20, 10, 21, 26, 11, 1, 24, 6, 30])
    testIdx = set(expIdx).difference(trainIdx)
    testIdx=list(testIdx)
    trainIdx.extend(testIdx[:2])
    testIdx.extend(trainIdx[:2])
    trainIdx=trainIdx[2:]
    testIdx=testIdx[2:]
    trainData = {key: wantedData[key] for key in wantedData.keys() & [expNames[i]for i in trainIdx]}
    testData = {key: wantedData[key] for key in wantedData.keys() & [expNames[i]for i in testIdx]}
    trainData['expNames'] = [expNames[i]for i in trainIdx]
    testData['expNames'] = [expNames[i]for i in testIdx]
    return trainData,testData,trainIdx

def simulation_initialization():
    #Settings
    Settings = dict()
    Settings['DT'] = 1/60
    Settings['nConfig'] = 10000
    Settings['nConfigPerIter'] = 1000
    Settings['case'] = 1

    #Constants
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

    #initial condictions
    InitCond = dict()
    InitCond['X_0'] = 0.15
    InitCond['Vl_0'] = 80
    InitCond['Vg_0'] = Const['TOTAL_VOLUME'] - InitCond['Vl_0']
    InitCond['DO_0'] = Const['DO_MAX']
    InitCond['P1_0'] = 0

    #return values
    return Settings,Const,InitCond


def set_initial_conditions(expData, Const):
    import pandas as pd
    import numpy as np
    variablesDF = pd.DataFrame()
    variablesDF['X'] = 0.15 * np.ones([expData.shape[0]]) #[g/L]
    variablesDF['P'] = 0 * np.ones([expData.shape[0]]) #[g/L]
    variablesDF['S'] = expData['S'][0]*10 * np.ones([expData.shape[0]]) #[g/L]
    variablesDF['DO'] = expData['DO'][0] * (Const['DO_MAX']/100) * np.ones([expData.shape[0]]) #[g/L]
    variablesDF['A'] = expData['A'][0] * np.ones([expData.shape[0]]) #[g/L]
    variablesDF['Vl'] = 80 * np.ones([expData.shape[0]]) #[L]
    variablesDF['Vg'] = 70 * np.ones([expData.shape[0]]) #[L]
    variablesDF['mu'] = np.zeros([expData.shape[0]])
    variablesDF['mu_pp'] = np.zeros([expData.shape[0]])
    return variablesDF

def orginize_const_dict(functionCombos,varTypes,nConfig):
    def rand_vec_generator(type, inputRange, length):
        import numpy as np
        import random
        if type == 1:
            my_number = 1
            if inputRange[0] < 0:
                my_number = random.choice((-1, 1))
            expRange = np.log10(np.abs(inputRange))
            calcConst = expRange[1] - expRange[0]
            vec = my_number * np.power(10, expRange[0] + calcConst * np.random.uniform(size=length))
        elif type == 0:
            calcConst = inputRange[1] - inputRange[0]
            vec = inputRange[0] + calcConst * np.random.uniform(size=length)
        return vec
    import pandas as pd
    data = pd.read_excel('param_options.xlsx')
    constDict = dict()
    for func in functionCombos:
        rel_function_data = data[(data['function'] == func)]
        constDict[func]=dict()
        for var in varTypes:
            rel_var_data = rel_function_data[(rel_function_data['variable'] == var)]
            constDict[func][var] = dict()
            for idxVal in range(0, rel_var_data.shape[0]):
                relData = rel_var_data.iloc[idxVal, :]
                rangeVec = [relData['min range'], relData['max range']]
                constDict[func][var][relData['constant']] = rand_vec_generator(relData['exp'], rangeVec, nConfig)

    return constDict



def runFunctionSearch(functionCombos, varVec, trainData, nConfig, constDict):
    import numpy as np
    ansDict = dict()
    for functionVec in functionCombos:
        # ansMat = np.zeros([nConfig, len(varVec)])
        functionVecName = functionVec[0][0] + functionVec[1][0] + functionVec[2][0]
        ansDict[functionVecName] = dict()
        counter = -1
        scoreMat = np.zeros([len(trainData.keys())-1, nConfig])
        for exp in trainData.keys():
            if exp == 'expNames':
                continue
            counter += 1
            ansDict[functionVecName][exp] = dict()
            for idx in range(len(varVec)):
                ansDict[functionVecName][exp][varVec[idx]] = dict()
                ansDict[functionVecName][exp][varVec[idx]]['valMat'] = run_func(functionVec[idx], varVec[idx],
                                                                                constDict, trainData[exp], nConfig)
            ansDict[functionVecName][exp]['func_val'] = ansDict[functionVecName][exp][varVec[0]]['valMat'] * \
                                                        ansDict[functionVecName][exp][varVec[1]]['valMat'] * \
                                                        ansDict[functionVecName][exp][varVec[2]]['valMat']
            del ansDict[functionVecName][exp][varVec[0]]['valMat'], ansDict[functionVecName][exp][varVec[1]]['valMat'], \
            ansDict[functionVecName][exp][varVec[2]]['valMat']
            expLength = trainData[exp].index[-1]+1
            ansDict[functionVecName][exp]['prod'] = np.zeros([nConfig, expLength])
            for time in range(0, expLength):
                ansDict[functionVecName][exp]['prod'][:, time] = ansDict[functionVecName][exp]['prod'][:, time - 1] + \
                                                                 trainData[exp]['Incyte'][time] * \
                                                                 ansDict[functionVecName][exp]['func_val'][:, time]
            del ansDict[functionVecName][exp]['func_val']
            scoreMat[counter, :] = goal_function(ansDict[functionVecName][exp]['prod'], trainData[exp]['Tobramycin'])
        ansDict[functionVecName]['sumScoreMat'] = np.sum(scoreMat, axis=0)
        ansDict[functionVecName]['sortedSumScoreMat'] = np.sort(ansDict[functionVecName]['sumScoreMat'])
        ansDict[functionVecName]['sortedSumScoreMatIdx'] = np.argsort(ansDict[functionVecName]['sumScoreMat'])
    return ansDict

def run_func(function,var,constDict,trainData,nConfig):
    import math
    import numpy as np
    finalTime = int(trainData.tail(1).index[0])+1
    valMat = np.zeros([nConfig, finalTime])
    for time in range(0, finalTime):
        if function == 'michaelis_menten':
            KName = [*constDict[function][var]]
            K = constDict[function][var][KName[0]]
            valMat[:, time] = trainData[var][time]/(trainData[var][time]+K)
        elif function == 'bell_michaelis_menten':
            KNames = [*constDict[function][var]]
            K = constDict[function][var][KNames[0]]
            K2 = constDict[function][var][KNames[1]]
            div = np.sqrt(K*K2)/(K+np.sqrt(K*K2)+np.power(np.sqrt(K*K2), 2)/K2)
            valMat[:, time] = (trainData[var][time]/(K+trainData[var][time]+np.power(trainData[var][time], 2)/K2))/div
        elif function == 'second_degree':
            KNames = [*constDict[function][var]]
            a = constDict[function][var][KNames[0]]
            b = constDict[function][var][KNames[1]]
            valMat[:, time] = a + b*trainData[var][time]

        elif function == 'third_degree':
            KNames = [*constDict[function][var]]
            a = constDict[function][var][KNames[0]]
            b = constDict[function][var][KNames[1]]
            c = constDict[function][var][KNames[2]]
            valMat[:, time] = a + b * trainData[var][time]

    return valMat


def goal_function(modelData, expData):
    import numpy as np
    import matplotlib.pyplot as plt
    # scoreTest = np.corrcoef(expData, modelData[175, :])
    coefMat = np.corrcoef(expData, modelData)
    score = coefMat[0, 1:coefMat.shape[1]+1]
    sortedScore = np.sort(score)
    sortedIdx = np.argsort(score)
    return score
    # plt.figure()
    # plt.plot(range(0, len(expData)), expData, modelData[sortedIdx[-1], :])

['Mean dO [40-end]', 'Mean Ammonia [40-end]', 'Mean Dextrose [40-end]', 'Mean pH [40-end]',
                     'Mean Agitation [40-end]', 'Sum Ammonia feeding [40-end]', 'Time dextrose low'
                     'Peak dO level']
def feature_extractor(selectedTypesOfFeatures, interpWantedData):
    import numpy as np
    import pandas as pd
    featureNames = []
    featuresDF = pd.DataFrame()
    relExp = list(interpWantedData.keys())
    if 'expNames' in relExp:
        relExp.remove('expNames')
    for feature in selectedTypesOfFeatures:
        if feature == 'Mean dO [40-end]':
            featName = 'meanDO'
            featureNames.append(featName)
            MeandO=[]
            for exp in relExp:
                MeandO.append(np.mean(interpWantedData[exp]['DO'][40*60:]))
            featuresDF[featName] = MeandO

        elif feature == 'Mean Ammonia [40-end]':
            featName = 'meanAmm'
            featureNames.append(featName)
            MeanAmm = []
            for exp in relExp:
                MeanAmm.append(np.mean(interpWantedData[exp]['A'][40 * 60:]))
            featuresDF[featName] = MeanAmm

        elif feature == 'Mean Dextrose [40-end]':
            featName = 'meanS'
            featureNames.append(featName)
            MeanS=[]
            for exp in relExp:
                MeanS.append(np.mean(interpWantedData[exp]['S'][40*60:]))
            featuresDF[featName] = MeanS

        elif feature == 'Mean pH [40-end]':
            featName = 'meanpH'
            featureNames.append(featName)
            MeanpH=[]
            for exp in relExp:
                MeanpH.append(np.mean(interpWantedData[exp]['pH_x'][40*60:]))
            featuresDF[featName] = MeanpH

        elif feature == 'Mean Agitation [40-end]':
            featName = 'meanAgi'
            featureNames.append(featName)
            MeanAgi=[]
            for exp in relExp:
                MeanAgi.append(np.mean(interpWantedData[exp]['Agitation'][40*60:]))
            featuresDF[featName] = MeanAgi

        elif feature == 'Sum Ammonia feeding [40-end]':
            featName = 'meanAmmFeed'
            featureNames.append(featName)
            SumAmmFeed=[]
            for exp in relExp:
                SumAmmFeed.append(np.mean(interpWantedData[exp]['Fa'][40*60:]))
            featuresDF[featName] = SumAmmFeed

        elif feature == 'Time dextrose low':
            featName = 'TimeDexLow'
            featureNames.append(featName)
            timeDexLow=[]
            for exp in relExp:
                
                MeanAgi.append(np.mean(interpWantedData[exp]['agitationInterp'][40*60:]))
            featuresDF[featName] = MeanAgi
    finalTitter = []
    finalImpurity = []
    for exp in relExp:
        lastIdx = interpWantedData[exp].index[-1]
        finalTitter.append(interpWantedData[exp]['Tobramycin'][lastIdx])
        finalImpurity.append(interpWantedData[exp]['Kanamycin'][lastIdx]/
                             (interpWantedData[exp]['Kanamycin'][lastIdx]+interpWantedData[exp]['Tobramycin'][lastIdx]))
    resultsDF = pd.DataFrame()
    resultsDF['finalTitter'] = finalTitter
    resultsDF['finalImpurity'] = finalImpurity
    return featureNames, featuresDF, resultsDF


def feature_extractor_multiple_meas(selectedTypesOfFeatures, interpWantedData):
    import numpy as np
    import pandas as pd
    featureNames = []
    featuresDF = pd.DataFrame()
    relMeasDF = []
    relExp = list(interpWantedData.keys())
    if 'expNames' in relExp:
        relExp.remove('expNames')
    for exp in relExp:
        expDF = pd.DataFrame()
        for feature in selectedTypesOfFeatures:
            if feature == 'meanDO':
                featName = 'meanDO'
                if featName not in featureNames:
                    featureNames.append(featName)
                expDF[featName] = interpWantedData[exp]['DO'][40:]
            elif feature == 'meanAmm':
                featName = 'meanAmm'
                if featName not in featureNames:
                    featureNames.append(featName)
                expDF[featName] = interpWantedData[exp]['Ammonia[percent]'][40:]
            elif feature == 'meanS':
                featName = 'meanS'
                if featName not in featureNames:
                    featureNames.append(featName)
                expDF[featName] = interpWantedData[exp]['Dextrose[percent]'][40:]
            elif feature == 'meanpH':
                featName = 'meanpH'
                if featName not in featureNames:
                    featureNames.append(featName)
                expDF[featName] = interpWantedData[exp]['pH_x'][40:]
            elif feature == 'meanAgi':
                featName = 'meanAgi'
                if featName not in featureNames:
                    featureNames.append(featName)
                expDF[featName] = interpWantedData[exp]['Agitation'][40:]
            elif feature == 'meanAmmFeed':
                featName = 'meanAmmFeed'
                if featName not in featureNames:
                    featureNames.append(featName)
                expDF[featName] = interpWantedData[exp]['Fa'][40:]
            elif feature == 'TimeDexLow':
                featName = 'TimeDexLow'
                if featName not in featureNames:
                    featureNames.append(featName)
                expDF[featName] = interpWantedData[exp]['Fa'][40:]

        Titter = []
        Impurity = []
        expDF['Titter'] = interpWantedData[exp]['Tobramycin']
        titterNotNan = pd.notna(expDF['Titter'])
        titterNotNanIdx = expDF.index[titterNotNan]
        # for meas in range(len(titterNotNanIdx.to_list())):
            
        expDF['Titter'][titterNotNanIdx]
        expDF['Impurity'] = interpWantedData[exp]['Kanamycin'] /\
                            (interpWantedData[exp]['Kanamycin'] + interpWantedData[exp]['Tobramycin'])

        featuresDF = featuresDF.append(expDF)
    featuresDF.dropna(inplace=True)
    resultsDF = pd.DataFrame()
    resultsDF['Titter'] = featuresDF['Titter']
    resultsDF['Impurity'] = featuresDF['Impurity']
    featuresDF.drop('Titter', axis=1, inplace=True)
    featuresDF.drop('Impurity', axis=1, inplace=True)
    return featureNames, featuresDF, resultsDF




def read_rnd_data(selectedExp):
    import numpy as np
    import pandas as pd
    import pickle

    # Extract existing data
    file_name = "RnD_Data.p"
    with open(file_name, 'rb') as f:
        RnD_Data = pickle.load(f)
    data = {key: RnD_Data[key] for key in RnD_Data.keys() & selectedExp}
    return data

def correlation_function(selectedTypesOfTest, selectedTypesOfVar, typesOfVariables,
                         offlineData, data, preprocessingTechnique):
    import numpy as np
    import pandas as pd
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.preprocessing import RobustScaler
    if selectedTypesOfTest == 'Test all':
        relVar = typesOfVariables
    else:
        relVar = selectedTypesOfVar
    allexpDataframe = pd.DataFrame()
    #load all the data
    for exp in data.keys():
        ammoniaData = np.array(offlineData[exp + '_dex']['Ammonia[percent]'])
        ammFeedData = np.array(data[exp]['Fa'])
        pHData = np.array(data[exp]['pH'])
        pHDataOffline = np.array(offlineData[exp + '_dex']['pH'])
        timesOffline = np.array(offlineData[exp + '_dex']['Age'])
        timesOnline = np.array(data[exp]['Age'])
        DOData = np.array(data[exp]['DO'])
        dexData = np.array(offlineData[exp + '_dex']['Dextrose[percent]'])
        dexFeedData = np.array(data[exp]['Fs'])
        agiData = np.array(data[exp]['Agitation'])
        livepHData = np.array(data[exp]['pH'])
        tobraData = np.array(offlineData[exp + '_dex']['Tobramycin'])
        KanaData = np.array(offlineData[exp + '_dex']['Kanamycin'])
        airFlowData = np.array(data[exp]['Airflow'])

        #find measurements times
        isAmmoniaNotNan = ~np.isnan(ammoniaData)
        isDexNotNaN = ~np.isnan(dexData)
        isTobraNotNan = ~np.isnan(tobraData)
        isTimeNotNan = ~np.isnan(timesOffline)
        offlineTimeZeros = timesOffline
        offlineTimeZeros[~isTimeNotNan] = 0
        isProduction = offlineTimeZeros>24
        notNanTimes = isTimeNotNan & isProduction
        offlineMeasIdx = np.where(isAmmoniaNotNan & isDexNotNaN & isTobraNotNan & notNanTimes)
        # ammoniaMeas = ammoniaData[isAllMeas]
        offlineMeasTimes = timesOffline[offlineMeasIdx]
        expDataFrame = pd.DataFrame()
        dataMat = np.zeros([len(relVar) , len(offlineMeasTimes)])
        onlineMeasIndex = np.zeros([1, len(offlineMeasTimes)])
        for i in range(0, len(offlineMeasTimes)):
            onlineMeasIndex[0, i] = np.argmin(abs(timesOnline - offlineMeasTimes[i]))
        onlineMeasIndex = onlineMeasIndex.astype(int)[0]
        for variable in relVar:
            varCounter = 0
            if variable =='DO':
                dataMat[:varCounter] = DOData[onlineMeasIndex]
                expDataFrame['DO'] = DOData[onlineMeasIndex]
            elif variable =='S':
                dataMat[:varCounter] = dexData[offlineMeasIdx]
                expDataFrame['S'] = dexData[offlineMeasIdx]
            elif variable == 'Ammonia':
                dataMat[:varCounter] = ammoniaData[offlineMeasIdx]
                expDataFrame['Ammonia'] = ammoniaData[offlineMeasIdx]
            elif variable == 'CO2':
                dataMat[:varCounter] = CO2Data[onlineMeasIndex]
                expDataFrame['CO2'] = CO2Data[onlineMeasIndex]
            elif variable == 'Production':
                dataMat[:varCounter] = tobraData[offlineMeasIdx]
                expDataFrame['Production'] = tobraData[offlineMeasIdx]
            elif variable == 'Incyte':
                dataMat[:varCounter] = dexData[offlineMeasIdx]
                expDataFrame['Incyte'] = dexData[offlineMeasIdx]
            elif variable == 'pH':
                dataMat[:varCounter] = pHData[onlineMeasIndex]
                expDataFrame['pH'] = pHData[onlineMeasIndex]
            elif variable == 'Ammonia feeding':
                dataMat[:varCounter] = ammFeedData[onlineMeasIndex]
                expDataFrame['Ammonia feeding'] = ammFeedData[onlineMeasIndex]
            elif variable == 'Dextrose feeding':
                dataMat[:varCounter] = dexFeedData[onlineMeasIndex]
                expDataFrame['Dextrose feeding'] = dexFeedData[onlineMeasIndex]
            elif variable == 'Airflow':
                dataMat[:varCounter] = airFlowData[onlineMeasIndex]
                expDataFrame['Airflow'] = airFlowData[onlineMeasIndex]
            elif variable == 'Agitation':
                dataMat[:varCounter] = agiData[onlineMeasIndex]
                expDataFrame['Agitation'] = agiData[onlineMeasIndex]
            varCounter += 1

        allexpDataframe = allexpDataframe.append(expDataFrame, ignore_index=True, sort=False)
    if preprocessingTechnique == 'Standardize (Robust scalar)':
        allexpDataframeProcessed = RobustScaler().fit_transform(allexpDataframe)
        allexpDataframeProcessed = pd.DataFrame(allexpDataframeProcessed, columns=expDataFrame.columns)
        # allexpDataframeProcessed = transformer.transform(allexpDataframe)
    elif preprocessingTechnique == 'No preprocessing':
        allexpDataframeProcessed = allexpDataframe
    elif preprocessingTechnique == 'scaling (0-1)':
        allexpDataframeProcessed == MinMaxScaler().fit(allexpDataframe)
        allexpDataframeProcessed = pd.DataFrame(allexpDataframeProcessed, columns=expDataFrame.columns)
    correlationDataFrame = allexpDataframeProcessed.corr(method='pearson')
    return correlationDataFrame, allexpDataframe



def correlation_function_df(selectedTypesOfTest, selectedTypesOfVar, typesOfVariables,
                         data, preprocessingTechnique):
    import numpy as np
    import pandas as pd
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.preprocessing import RobustScaler
    if selectedTypesOfTest == 'Test all':
        relVar = typesOfVariables
    else:
        relVar = selectedTypesOfVar
    allexpDataframe = pd.DataFrame()
    #load all the data
    for exp in data.keys():
        expFeaturesDF = pd.DataFrame()
        for variable in relVar:
            varCounter = 0
            if variable =='DO':
                expFeaturesDF[variable] = data[exp]['DO']
                # dataMat[:varCounter] = DOData[onlineMeasIndex]
                # expDataFrame['DO'] = DOData[onlineMeasIndex]
            elif variable =='S':
                expFeaturesDF[variable] = data[exp]['Dextrose[percent]']
                # dataMat[:varCounter] = dexData[offlineMeasIdx]
                # expDataFrame['S'] = dexData[offlineMeasIdx]
            elif variable == 'Ammonia':
                expFeaturesDF[variable] = data[exp]['Ammonia[percent]']
                # dataMat[:varCounter] = ammoniaData[offlineMeasIdx]
                # expDataFrame['Ammonia'] = ammoniaData[offlineMeasIdx]
            elif variable == 'CO2':
                expFeaturesDF[variable] = data[exp]['CO2']
                # dataMat[:varCounter] = CO2Data[onlineMeasIndex]
                # expDataFrame['CO2'] = CO2Data[onlineMeasIndex]
            elif variable == 'Production':
                expFeaturesDF[variable] = data[exp]['Tobramycin']
                # dataMat[:varCounter] = tobraData[offlineMeasIdx]
                # expDataFrame['Production'] = tobraData[offlineMeasIdx]
            elif variable == 'Incyte':
                expFeaturesDF[variable] = data[exp]['Incyte']
                # dataMat[:varCounter] = dexData[offlineMeasIdx]
                # expDataFrame['Incyte'] = dexData[offlineMeasIdx]
            elif variable == 'pH':
                expFeaturesDF[variable] = data[exp]['pH_x']
                # dataMat[:varCounter] = pHData[onlineMeasIndex]
                # expDataFrame['pH'] = pHData[onlineMeasIndex]
            elif variable == 'Ammonia feeding':
                expFeaturesDF[variable] = data[exp]['Fa']
                # dataMat[:varCounter] = ammFeedData[onlineMeasIndex]
                # expDataFrame['Ammonia feeding'] = ammFeedData[onlineMeasIndex]
            elif variable == 'Dextrose feeding':
                expFeaturesDF[variable] = data[exp]['Fs']
                # dataMat[:varCounter] = dexFeedData[onlineMeasIndex]
                # expDataFrame['Dextrose feeding'] = dexFeedData[onlineMeasIndex]
            elif variable == 'Airflow':
                expFeaturesDF[variable] = data[exp]['Airflow']
                # dataMat[:varCounter] = airFlowData[onlineMeasIndex]
                # expDataFrame['Airflow'] = airFlowData[onlineMeasIndex]
            elif variable == 'Agitation':
                expFeaturesDF[variable] = data[exp]['Agitation']
                # dataMat[:varCounter] = agiData[onlineMeasIndex]
                # expDataFrame['Agitation'] = agiData[onlineMeasIndex]
            varCounter += 1

        allexpDataframe = allexpDataframe.append(expFeaturesDF, ignore_index=True, sort=False)
    allexpDataframe.dropna(inplace=True)
    if preprocessingTechnique == 'Standardize (Robust scalar)':
        allexpDataframeProcessed = RobustScaler().fit_transform(allexpDataframe)
        allexpDataframeProcessed = pd.DataFrame(allexpDataframeProcessed, columns=allexpDataframe.columns)
        # allexpDataframeProcessed = transformer.transform(allexpDataframe)
    elif preprocessingTechnique == 'No preprocessing':
        allexpDataframeProcessed = allexpDataframe
    elif preprocessingTechnique == 'scaling (0-1)':
        allexpDataframeProcessed = MinMaxScaler().fit_transform(allexpDataframe)
        allexpDataframeProcessed = pd.DataFrame(allexpDataframeProcessed, columns=allexpDataframe.columns)
    correlationDataFrame = allexpDataframeProcessed.corr(method='pearson')
    return correlationDataFrame, allexpDataframe

def polyfit_2d_coeff(x, y, z,x_test,y_test, degree, sigz=None, weights=None):
    """
    Fit a bivariate polynomial of given DEGREE to a set of points
    (X, Y, Z), assuming errors SIGZ in the Z variable only.

    For example with DEGREE=1 this function fits a plane

       z = a + b*x + c*y

    while with DEGREE=2 the function fits a quadratic surface

       z = a + b*x + c*x^2 + d*y + e*x*y + f*y^2

    """
    import numpy as np
    if weights is None:
        if sigz is None:
            sw = 1.
        else:
            sw = 1./sigz
    else:
        sw = np.sqrt(weights)

    npol = int((degree+1)*(degree+2)/2)
    a = np.empty((x.size, npol))
    c = np.empty_like(a)
    c_test=np.empty(npol)
    k = 0
    for i in range(degree+1):
        for j in range(degree-i+1):
            c[:, k] = x**j * y**i
            c_test[ k] = x_test**j * y_test**i
            a[:, k] = c[:, k]*sw
            k += 1

    coeff = np.linalg.lstsq(a, z*sw, rcond=None)[0]

    return coeff,c_test
def polyfit_nd_coeff(X, z,X_test, degree, sigz=None, weights=None):
    """
    Fit a bivariate polynomial of given DEGREE to a set of points
    (X, Y, Z), assuming errors SIGZ in the Z variable only.

    For example with DEGREE=1 this function fits a plane

       z = a + b*x + c*y

    while with DEGREE=2 the function fits a quadratic surface

       z = a + b*x + c*x^2 + d*y + e*x*y + f*y^2

    """
    import numpy as np
    if weights is None:
        if sigz is None:
            sw = 1.
        else:
            sw = 1./sigz
    else:
        sw = np.sqrt(weights)

    npol = X.shape[1] + 1
    a = np.empty((X.shape[0], npol))
    c = np.ones_like(a)
    c_test= np.ones_like(a)
    k = 0
    #X=np.column_stack((x, y));
    L1=np.array([[0,0,0,0,0],[1,0,0,0,0],[0,1,0,0,0],[0,0,1,0,0],[0,0,0,1,0],[0,0,0,0,1]])
    #re=2**X.shape[1]
    for r1 in range(npol):# loop on config
        for col in range(X.shape[1]):# loop on vars
            c[:, r1] =c[:, r1]*X[:,col]**L1[r1,col] #x**j * y**i
            c_test[0, r1] =c_test[0, r1]*X_test[col]**L1[r1,col]
        a[:, r1] = c[:, r1]*sw


    coeff = np.linalg.lstsq(a, z*sw, rcond=None)[0]

    return coeff,c_test[0,:]

def loess_2d_test_point(x1, y1, z,x_test,y_test,z_test, frac=0.5, degree=1, rescale=False,
             npoints=None, sigz=None):

    """
    zout, wout = loess_2d(x, y, z, frac=0.5, degree=1)
    gives a LOESS smoothed estimate of the quantity z at the sets
    of coordinates (x,y).

    """
    from loess.loess_2d import biweight_sigma,biweight_mean,rotate_points,polyfit_2d
    import numpy as np

    x1=np.append(x_test,x1)
    y1=np.append(y_test,y1)
    z=np.append(z_test,z)
    if frac == 0:
        return z, np.ones_like(z)

    assert x1.size == y1.size == z.size, 'Input vectors (X, Y, Z) must have the same size'

    if npoints is None:
        npoints = int(np.ceil(frac*x1.size))

    if rescale:

        # Robust calculation of the axis of maximum variance
        #
        nsteps = 180
        angles = np.arange(nsteps)
        sig = np.zeros(nsteps)
        for j, ang in enumerate(angles):
            x2, y2 = rotate_points(x1, y1, ang)
            sig[j] = biweight_sigma(x2)
        k = np.argmax(sig) # Find index of max value
        x2, y2 = rotate_points(x1, y1, angles[k])
        x = (x2 - biweight_mean(x2)) / biweight_sigma(x2)
        y = (y2 - biweight_mean(y2)) / biweight_sigma(y2)

    else:

        x = x1
        y = y1

    zout = np.empty_like(x)
    wout = np.empty_like(x)

    #for j, (xj, yj) in enumerate(zip(x, y)):
    xj=x[0]
    yj=y[0]
    dist = np.sqrt((x - xj)**2 + (y - yj)**2)
    dist[0]=1e20 # in order to exclude the test point label from the smoothing process
    w = np.argsort(dist)[:npoints]
    distWeights = (1 - (dist[w]/dist[w[-1]])**3)**3  # tricube function distance weights
    zfit = polyfit_2d(x[w], y[w], z[w], degree, weights=distWeights)

    # Robust fit from Sec.2 of Cleveland (1979)
    # Use errors if those are known.
    #
    bad = []
    for p in range(10):  # do at most 10 iterations

        if sigz is None:  # Errors are unknown
            aerr = np.abs(zfit - z[w])  # Note ABS()
            mad = np.median(aerr)  # Characteristic scale
            uu = (aerr/(6*mad))**2  # For a Gaussian: sigma=1.4826*MAD
        else:  # Errors are assumed known
            uu = ((zfit - z[w])/(4*sigz[w]))**2  # 4*sig ~ 6*mad

        uu = uu.clip(0, 1)
        biWeights = (1 - uu)**2
        totWeights = distWeights*biWeights
        zfit = polyfit_2d(x[w], y[w], z[w], degree, weights=totWeights)
        badOld = bad
        bad = np.where(biWeights < 0.34)[0] # 99% confidence outliers
        if np.array_equal(badOld, bad):
            break
    coeff,c_test = polyfit_2d_coeff(x[w], y[w], z[w],x_test,y_test, degree, weights=totWeights)

    zout = c_test.dot(coeff)# zfit[0]
    wout = biWeights[0]

    return zout, wout
#def loess_nd_test_point(X, z, X_test, z_test, frac=0.5, degree=1, rescale=False,
   #          npoints=None, sigz=None):
def loess_nd_test_point(X, XDist, z, X_test, XDist_test, z_test, frac=0.5, degree=1, rescale=False,
                         npoints=None, sigz=None):

    """
    zout, wout = loess_2d(x, y, z, frac=0.5, degree=1)
    gives a LOESS smoothed estimate of the quantity z at the sets
    of coordinates (x,y).

    """
    #XDist = X
   # XDist_test = X_test
    from loess.loess_2d import biweight_sigma,biweight_mean,rotate_points,polyfit_2d
    import numpy as np
   # X=np.vstack([X_test,X])
    #x1=np.append(x_test,x1)
    #y1=np.append(y_test,y1)
    #z=np.append(z_test,z)
    if frac == 0:
        return z, np.ones_like(z)

    #assert x1.size == y1.size == z.size, 'Input vectors (X, Y, Z) must have the same size'

    if npoints is None:
        npoints = int(np.ceil(frac*X[:, 0].size))

    if rescale:

        # Robust calculation of the axis of maximum variance
        #
        nsteps = 180
        angles = np.arange(nsteps)
        sig = np.zeros(nsteps)
        #for j, ang in enumerate(angles):
           # x2, y2 = rotate_points(x1, y1, ang)
           # sig[j] = biweight_sigma(x2)
        k = np.argmax(sig) # Find index of max value
       # x2, y2 = rotate_points(x1, y1, angles[k])
       # x = (x2 - biweight_mean(x2)) / biweight_sigma(x2)
       # y = (y2 - biweight_mean(y2)) / biweight_sigma(y2)

    else:
       demo=1
       # x = x1
       # y = y1

    # zout = np.empty_like(X[:,1].size)
    # wout = np.empty_like(X[:,1].size)

   # for j, (xj, yj) in enumerate(zip(x, y)):
   #  xj=X[0,0]
   #  yj=X[0,1]
    distSumSqr = 0
    for varForDist in range(XDist_test.size):
        distVarSqr = (XDist[:, varForDist]- XDist_test[varForDist]) ** 2
        distSumSqr += distVarSqr
    dist = np.sqrt(distSumSqr)
    # dist = np.sqrt((X[:,0] - xj)**2 + (X[:,1]  - yj)**2)
    # dist[0]=1e20 # in order to exclude the test point label from the smoothing process
    w = np.argsort(dist)[:npoints]
    distWeights = (1 - (dist[w]/dist[w[-1]])**3)**3  # tricube function distance weights
    zfit = polyfit_nd(X[w,:], z[w], degree, weights=distWeights)

    # Robust fit from Sec.2 of Cleveland (1979)
    # Use errors if those are known.
    #
    bad = []
    for p in range(10):  # do at most 10 iterations

        if sigz is None:  # Errors are unknown
            aerr = np.abs(zfit - z[w])  # Note ABS()
            mad = np.median(aerr)  # Characteristic scale
            uu = (aerr/(6*mad))**2  # For a Gaussian: sigma=1.4826*MAD
        else:  # Errors are assumed known
            uu = ((zfit - z[w])/(4*sigz[w]))**2  # 4*sig ~ 6*mad

        uu = uu.clip(0, 1)
        biWeights = (1 - uu)**2
        totWeights = distWeights*biWeights
        zfit =polyfit_nd(X[w,:], z[w], degree, weights=totWeights)
        badOld = bad
        bad = np.where(biWeights < 0.34)[0] # 99% confidence outliers
        if np.array_equal(badOld, bad):
            break
    coeff,c_test = polyfit_nd_coeff(X[w,:], z[w],X_test, degree, weights=totWeights)

    zout = c_test.dot(coeff)# zfit[0]
    wout = biWeights[0]

    return zout, wout
def polyfit_nd(X, z, degree, sigz=None, weights=None):
    """
    Fit a bivariate polynomial of given DEGREE to a set of points
    (X, Y, Z), assuming errors SIGZ in the Z variable only.

    For example with DEGREE=1 this function fits a plane

       z = a + b*x + c*y

    while with DEGREE=2 the function fits a quadratic surface

       z = a + b*x + c*x^2 + d*y + e*x*y + f*y^2

    """
    import numpy as np
    if weights is None:
        if sigz is None:
            sw = 1.
        else:
            sw = 1./sigz
    else:
        sw = np.sqrt(weights)

    # npol = int((degree+1)*(degree+2)/2)#Number of organs in polynom
    npol = X.shape[1] + 1
    a = np.empty((X.shape[0], npol))
    c = np.ones_like(a)
    c_test= np.ones_like(a)
    k = 0
    #X=np.column_stack((x, y));
    L1=np.array([[0,0,0,0,0],[1,0,0,0,0],[0,1,0,0,0],[0,0,1,0,0],[0,0,0,1,0],[0,0,0,0,1]])
    re=2**X.shape[1]
    for r1 in range(npol):# loop on config
        for col in range(X.shape[1]):# loop on vars
            c[:, r1] =c[:, r1]*X[:,col]**L1[r1,col] #x**j * y**i
           # c_test[:, r1] =c_test[:, r1]*X_test[:,col]**L1[r1,col]
        a[:, r1] = c[:, r1]*sw


    coeff = np.linalg.lstsq(a, z*sw, rcond=None)[0]

    return c.dot(coeff)
