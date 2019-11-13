import numpy as np
from read_data import read_data
import pickle
from function_for_GUI import *
import itertools
from tkinter import *
import matplotlib.pyplot as plt
from itertools import combinations
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler
#import skmisc.loess
from loess.loess_2d import loess_2d
from loess.examples import test_loess_1d
from plotbin.plot_velfield import plot_velfield
# from PIL import ImageTk, Image
import os

from tkinter import messagebox
# top = Tk()
# C = Canvas(top, bg="blue", height=250, width=300)
# filename = PhotoImage(file="C:\\Users\\admin\PycharmProjects\\tobra_model\\statistics_pic1.png")
# background_label = Label(top, image=filename)
# background_label.place(x=0, y=0, relwidth=1, relheight=1)
# C.pack()
# top.mainloop

experimentsOptions = ['0119A', '0119B', '0319A', '0319B', '0419A', '0419B', '0519A', '0519B', '0619A', '0619B', '0719A', '0819A',
               '0819B', '040217', '040117', '330615', '340615', '350615', '370615', '380615', '390615', '400615', '410615',
              '430615', '450815', '470815', '0019_REF', '0019_IC', '0119_REF', '0119_IC', '0219_REF', '0219_IC']
incyteExp = ['0419A', '0519A', '0519B', '0619A', '0619B', '0719A', '0819A', '0119_IC', '0219_IC']
exp2019 = ['0119A', '0119B', '0319A', '0319B', '0419A', '0419B', '0519A', '0519B', '0619A', '0619B', '0719A', '0819A',
               '0819B', '0019_REF', '0019_IC', '0119_REF', '0119_IC', '0219_REF', '0219_IC']

preprocessingOptions = ['No preprocessing', 'scaling (0-1)', 'Standardize (Robust scalar)']

master = Tk()

def get_list(event):
    """
    function to read the listbox selection
    and put the result in an entry widget
    """
    # get selected line index
    # index = expListbox.curselection()[0]

    index = expListbox.curselection()
    # get the line's text
    for i in range(0, len(index)):
        selText = expListbox.get(index[i])
    # delete previous text in enter1
    # enter1.delete(0, 50)
    # now display the selected text
    selectedExperiments.append(selText)

def passForword():
    global preprocessingTechnique
    preprocessingTechnique = preProcessingListbox.get(preProcessingListbox.curselection())
    # numIter.append(numIterEntry.get())
    master.destroy()
    return preprocessingTechnique

C = Canvas(master, bg="blue", height=250, width=300)
# filename = PhotoImage(file="C:\\Users\\admin\PycharmProjects\\tobra_model\\statistics_pic1_with_logo.png")
filename = PhotoImage(file="statistics_pic1_with_logo.png")
background_label = Label(master, image=filename)
background_label.place(x=0, y=0, relwidth=1, relheight=1)
master.geometry('900x500')
selectedExperiments = []
titleLabel = Label(master, text='Summary Statistics app', font=('Helvetica', '17', 'bold'), fg='blue')
titleLabel.grid(row=0, column=1, sticky=E, pady=20)
methodLabel = Label(master, text='Method type:', font=('Helvetica', '10', 'bold'))
methodLabel.grid(row=1, column=1, sticky=E, pady=10, padx=10)
methodType = StringVar(master)
methodType.set("LOESS") # default value

methodDropDown = OptionMenu(master, methodType, "K-means", "Function Search", "Correlations", "LOESS")
methodDropDown.grid(row=1, column=2, sticky=E, pady=10)




#load data or use .p file
loadDataText = Label(master, text='load excel data?', font=('Helvetica', '10', 'bold'))
loadDataText.grid(row=2, column=1, sticky=E, pady=20, padx=10)
isLoadData = IntVar()
Checkbutton(master, variable=isLoadData).grid(row=2, column=2, sticky=W, pady=20)

expText = Label(master, text='choose relevant experiments:', font=('Helvetica', '10', 'bold'))
expText.grid(row=3, column=1, sticky=E, padx=10)
scroll = Scrollbar(master)
scroll.grid(row=3, column=3)
expListbox = Listbox(master, selectmode='multiple')
expListbox.grid(row=3, column=2)
expListbox.config(yscrollcommand=scroll.set)
scroll.config(command=expListbox.yview)

for item in experimentsOptions:
    expListbox.insert(END, item)
# idex = expListbox.curselection()
# relExp = expListbox.get(idex)

# expListbox.bind('<ButtonRelease-1>', get_list)

isAllExp = IntVar(value=1)
Checkbutton(master, text="All experiments", variable=isAllExp).grid(row=3, column=4, sticky=W, padx=10)
is2019Exp = IntVar()
# check2019 = StringVar()
Checkbutton(master, text="2019 experiments", variable=is2019Exp).grid(row=3, column=5, sticky=W, padx=10)
isIncyteExp = IntVar()
Checkbutton(master, text="Incyte experiments", variable=isIncyteExp).grid(row=3, column=6, sticky=W, padx=10)

preProcessingText = Label(master, text='choose preprocessing technique:', font=('Helvetica', '10', 'bold'))
preProcessingText.grid(row=4, column=1, sticky=E, padx=10, pady=10)
preProcessingListbox = Listbox(master, height=3, width=25, selectmode=SINGLE)
# preProcessingListbox.select_set(2)

preProcessingListbox.event_generate("<<ListboxSelect>>")
preProcessingListbox.grid(row=4, column=2, pady=10)

for item in preprocessingOptions:
    preProcessingListbox.insert(END, item)
# idex = expListbox.curselection()
# relExp = expListbox.get(idex)
# expListbox.bind('<ButtonRelease-1>', get_list)
preProcessingListbox.selection_set(1)

button = Button(text="continue", command=passForword, width=20, height=1, font=('Helvetica', '17'))
button.place(rely=0.95, relx=0.5, anchor=CENTER)

master.mainloop()

#get GUI data
methodType = methodType.get()
isLoadData = isLoadData.get()

if isAllExp.get():
    selectedExp = experimentsOptions
elif is2019Exp.get():
    selectedExp = exp2019
elif isIncyteExp.get():
    selectedExp = incyteExp
else:
    selectedExp = selectedExperiments




if methodType == "Function Search":
    typesOfFunctions=['second_degree', 'third_degree', 'michaelis_menten', 'bell_michaelis_menten']
    def get_list_func(event):
        """
        function to read the listbox selection
        and put the result in an entry widget
        """
        # get selected line index
        # index = expListbox.curselection()[0]

        index = funcListbox.curselection()
        # get the line's text
        for i in range(0, len(index)):
            selText = funcListbox.get(index[i])
        # delete previous text in enter1
        # enter1.delete(0, 50)
        # now display the selected text
        selectedTypesOfFunctions.append(selText)
    def passForword():
        global nConfig
        nConfig = int(numIterEntry.get())
        # numIter.append(numIterEntry.get())
        funcSearch.destroy()
        return nConfig



    funcSearch = Tk()
    C = Canvas(funcSearch, bg="blue", height=250, width=300)
    # filename = PhotoImage(file="C:\\Users\\admin\PycharmProjects\\tobra_model\\statistics_pic1_with_logo.png")
    filename = PhotoImage(file="statistics_pic1_with_logo.png")
    background_label = Label(funcSearch, image=filename)
    background_label.place(x=0, y=0, relwidth=1, relheight=1)
    funcSearch.geometry('900x500')
    selectedTypesOfFunctions = []
    titleLabel = Label(funcSearch, text='Function search algorithm', font=('Helvetica', '17', 'bold'), fg='blue')
    titleLabel.grid(row=0, column=1, sticky=E, pady=20)
    VariablesText = Label(funcSearch, text='Variables:', font=('Helvetica', '10', 'bold'))
    VariablesText.grid(row=1, column=1, sticky=E, pady=10, padx=20)
    dexVar = IntVar(value=1)
    Checkbutton(funcSearch, text="S (Dextrose)", variable=dexVar).grid(row=1, column=2, sticky=W, pady=10, padx=10)
    DOVar = IntVar(value=1)
    Checkbutton(funcSearch, text="DO (Oxygen)", variable=DOVar).grid(row=1, column=3, sticky=W, pady=10, padx=10)
    ammVar = IntVar(value=1)
    Checkbutton(funcSearch, text="A (Ammonia)", variable=ammVar).grid(row=1, column=4, sticky=W, pady=10, padx=10)

    numIterText = Label(funcSearch, text='Insert number of iterations:', font=('Helvetica', '10', 'bold'))
    numIterText.grid(row=2, column=1, sticky=E, pady=10, padx=20)
    numIterEntry = Entry(funcSearch, text='1000')
    numIterEntry.grid(row=2, column=2, sticky=E, pady=10)
    numIterEntry.insert(0, "1000")


    funcText = Label(funcSearch, text='Functions to run:', font=('Helvetica', '10', 'bold'))
    funcText.grid(row=3, column=1, sticky=E, pady=10, padx=20)
    funcListbox = Listbox(funcSearch, selectmode='multiple')
    funcListbox.grid(row=3, column=2, pady=10)
    for item in typesOfFunctions:
        funcListbox.insert(END, item)
    # idex = expListbox.curselection()
    # relExp = expListbox.get(idex)
    funcListbox.bind('<ButtonRelease-1>', get_list_func)

    button = Button(text="continue", command=passForword, width=20, height=1, font=('Helvetica', '17'))
    button.place(rely=0.95, relx=0.5, anchor=CENTER)
    funcSearch.mainloop()

    #get GUI data
    isDex = dexVar.get()
    isDO = DOVar.get()
    isAmm = ammVar.get()
    varVec = []
    if isDex:
        varVec.append('S')
    if isDO:
        varVec.append('DO')
    if isAmm:
        varVec.append('A')

    # load the data
    Settings, Const, InitCond = simulation_initialization()
    # data = read_rnd_data(selectedExp)
    # interpData=data['0419A'].interpolate(limit_area='inside')
    #load data from dataframes:

    #load data from dictionaries.
    # data, offlineData = load_data(isLoadData, selectedExp)
    # interpWantedData = data_interp(data, offlineData, selectedExp, Const)
    if isLoadData:
        data = load_data_df(isLoadData, selectedExp)
        interpWantedData = data_interp_df(data, selectedExp)
    else:
        interpWantedData = load_data_df(isLoadData, selectedExp)

    trainData, testData = devide_data(interpWantedData)
    if isLoadData:
        del data, interpWantedData
    else:
        del interpWantedData
    functionCombos = [p for p in itertools.product(selectedTypesOfFunctions, repeat=3)]
    constDict = orginize_const_dict(selectedTypesOfFunctions, varVec, nConfig)
    ansDict = runFunctionSearch(functionCombos, varVec, trainData, nConfig, constDict)

    bestScore = 0
    for functionVec in functionCombos:
        functionVecName = functionVec[0][0] + functionVec[1][0] + functionVec[2][0]
        if ansDict[functionVecName]['sortedSumScoreMat'][-1] > bestScore:
            bestScore = ansDict[functionVecName]['sortedSumScoreMat'][-1]
            bestScoreFunctions = functionVecName
            bestScoreFunctionVec = functionVec


    exp = list(trainData.keys())[1]
    #bestScoreFunctions = 'bbb'
    # bestScoreFunctions = 'ttt'
    relExp = list(trainData.keys())
    if 'expNames' in relExp:
        relExp.remove('expNames')
    for exp in relExp:
        plt.figure()
        devideConst = np.mean(ansDict[bestScoreFunctions][exp]['prod'][ansDict[bestScoreFunctions]['sortedSumScoreMatIdx'][-1]]) / \
                np.mean(trainData[exp]['Tobramycin'])

        plt.plot(np.array(trainData[exp].index),
                 ansDict[bestScoreFunctions][exp]['prod'][ansDict[bestScoreFunctions]['sortedSumScoreMatIdx'][-1]]/devideConst)
        plt.plot(np.array(trainData[exp].index), trainData[exp]['Tobramycin'])
        plt.legend(['Model', 'Data'])
        plt.xlabel('Time [min]')
        plt.ylabel('product [microgram/gram]')
        plt.title(bestScoreFunctions + ' best model vs data production')

    plt.figure()
    plt.plot(trainData[exp]['Tobramycin'],
             ansDict[bestScoreFunctions][exp]['prod'][ansDict[bestScoreFunctions]['sortedSumScoreMatIdx'][-1]])
    plt.xlabel('data')
    plt.ylabel('model')
    plt.title(bestScoreFunctions + ' best model vs data production')



elif methodType == "K-means":
    def get_list_features(event):
        """
        function to read the listbox selection
        and put the result in an entry widget
        """
        # get selected line index
        # index = expListbox.curselection()[0]
        global selectedTypesOfFeatures
        index = FeaturesListbox.curselection()
        # get the line's text
        selText = []
        for i in range(0, len(index)):
            # selText = FeaturesListbox.get(index[i])
            selText.append(FeaturesListbox.get(index[i]))
        # delete previous text in enter1
        # enter1.delete(0, 50)
        # now display the selected text
        # selectedTypesOfFeatures.append(selText)
        selectedTypesOfFeatures = selText
    def passForword():
        global nConfig, nMeans
        nConfig = int(numIterEntry.get())
        nMeans = int(numMeansEntry.get())
        # numIter.append(numIterEntry.get())
        K_means.destroy()
        return nConfig



    from sklearn.cluster import KMeans
    selectedTypesOfFeatures = []
    # typesOfFeatures=['Mean dO [40-end]', 'Mean Ammonia [40-end]', 'Mean Dextrose [40-end]', 'Mean pH [40-end]',
    #                  'Mean Agitation [40-end]', 'Sum Ammonia feeding [40-end]', 'TimeDexLow'
    #                  'Peak dO level']
    typesOfFeatures = ['meanDO', 'meanAmm', 'meanS', 'meanpH', 'meanAgi', 'meanAmmFeed', 'TimeDexLow']
    K_means = Tk()
    C = Canvas(K_means, bg="blue", height=250, width=300)
    # filename = PhotoImage(file="C:\\Users\\admin\PycharmProjects\\tobra_model\\statistics_pic1_with_logo.png")
    filename = PhotoImage(file="statistics_pic1_with_logo.png")
    background_label = Label(K_means, image=filename)
    background_label.place(x=0, y=0, relwidth=1, relheight=1)
    K_means.geometry('900x500')
    titleLabel = Label(K_means, text='K-means algorithm', font=('Helvetica', '17', 'bold'), fg='blue')
    titleLabel.grid(row=0, column=1, sticky=E, pady=20)
    FeaturesText = Label(K_means, text='Features:', font=('Helvetica', '10', 'bold'))
    FeaturesText.grid(row=1, column=1, sticky=E, pady=10, padx=20)

    FeaturesListbox = Listbox(K_means, selectmode='multiple')
    FeaturesListbox.grid(row=1, column=2, pady=10)
    for item in typesOfFeatures:
        FeaturesListbox.insert(END, item)
    # idex = expListbox.curselection()
    # relExp = expListbox.get(idex)
    FeaturesListbox.bind('<ButtonRelease-1>', get_list_features)

    VariablesText = Label(K_means, text='Reference Variables:', font=('Helvetica', '10', 'bold'))
    VariablesText.grid(row=2, column=1, sticky=E, pady=10, padx=20)
    ProdRef = IntVar(value=1)
    Checkbutton(K_means, text="Product", variable=ProdRef).grid(row=2, column=2, sticky=W, pady=10, padx=10)
    ImpurityRef = IntVar(value=1)
    Checkbutton(K_means, text="Impurity", variable=ImpurityRef).grid(row=2, column=3, sticky=W, pady=10, padx=10)

    numIterText = Label(K_means, text='Insert number of iterations:', font=('Helvetica', '10', 'bold'))
    numIterText.grid(row=3, column=1, sticky=E, pady=10, padx=20)
    numIterEntry = Entry(K_means, text='1000')
    numIterEntry.grid(row=3, column=2, sticky=E, pady=10)
    numIterEntry.insert(0, "1000")

    numMeansText = Label(K_means, text='Insert number of means:', font=('Helvetica', '10', 'bold'))
    numMeansText.grid(row=4, column=1, sticky=E, pady=10, padx=20)
    numMeansEntry = Entry(K_means, text='5')
    numMeansEntry.grid(row=4, column=2, sticky=E, pady=10)
    numMeansEntry.insert(0, "5")

    button = Button(text="continue", command=passForword, width=20, height=1, font=('Helvetica', '17'))
    button.place(rely=0.95, relx=0.5, anchor=CENTER)
    K_means.mainloop()


    #get GUI data
    isProdRef = ProdRef.get()
    isImpurityRef = ImpurityRef.get()

    #execute Kmeans algorithm
    Settings, Const, InitCond = simulation_initialization()
    #if isLoadData:
    #   data = load_data_df(isLoadData, selectedExp)
    #    interpWantedData = data_interp_df(data, selectedExp)
    #else:
    #    interpWantedData = load_data_df(isLoadData, selectedExp)

    data = load_data_df(1, selectedExp)
    trainData, testData = devide_data(data)
    #if isLoadData:
    #    del data, interpWantedData
    #else:
    #    del interpWantedData
    del data
    # data, offlineData = load_data(isLoadData, selectedExp)
    # interpWantedData = data_interp(data, offlineData, selectedExp, Const)
    # trainData, testData = devide_data(interpWantedData)
    # del data, offlineData, interpWantedData
    featureNames, featuresTrainDF, resultsTrainDF = feature_extractor_multiple_meas(selectedTypesOfFeatures, trainData)
    featureNames, featuresTestDF, resultsTestDF = feature_extractor_multiple_meas(selectedTypesOfFeatures, testData)

    numMeasTrain = featuresTrainDF.shape[0]
    allFeatAvgDF = featuresTrainDF.append(featuresTestDF)
    if preprocessingTechnique == 'Standardize (Robust scalar)':
        FeaturesProcessed = RobustScaler().fit_transform(allFeatAvgDF)
        # FeatureTestProcessed = RobustScaler().fit_transform(featuresTestDF)
        FeaturesProcessed = pd.DataFrame(FeaturesProcessed, columns=featureNames)
        # FeatureTestProcessed = pd.DataFrame(FeatureTestProcessed, columns=featureNames)
    elif preprocessingTechnique == 'No preprocessing':
        FeaturesProcessed = featuresTrainDF
        # FeatureTestProcessed = featuresTestDF
    elif preprocessingTechnique == 'scaling (0-1)':
        FeaturesProcessed = MinMaxScaler().fit_transform(allFeatAvgDF)
        # FeatureTestProcessed = MinMaxScaler().fit_transform(featuresTestDF)
        FeaturesProcessed = pd.DataFrame(FeaturesProcessed, columns=featureNames)
        # FeatureTestProcessed = pd.DataFrame(FeatureTestProcessed, columns=featureNames)

    FeatureTrainProcessed = FeaturesProcessed.iloc[0:numMeasTrain]
    FeatureTestProcessed = FeaturesProcessed.iloc[numMeasTrain:]
    k = KMeans(n_clusters=nMeans)
    k.fit(FeatureTrainProcessed)
    predicted_classification = k.predict(FeatureTestProcessed)
    plt.figure()
    plt.scatter(resultsTestDF['Titter'], resultsTestDF['Impurity'], c=predicted_classification)
    plt.figure()
    plt.scatter(resultsTrainDF['Titter'], resultsTrainDF['Impurity'], c=k.labels_)
    # plt.show()
    q=2

elif methodType == "Correlations":
    # def get_list_tests(event):
    #     """
    #     function to read the listbox selection
    #     and put the result in an entry widget
    #     """
    #     index = testListbox.curselection()
    #     # get the line's text
    #     for i in range(0, len(index)):
    #         selText = testListbox.get(index[i])
    #     # delete previous text in enter1
    #     # enter1.delete(0, 50)
    #     # now display the selected text
    #     selectedTypesOfTest.append(selText)

    def get_list_var(event):
        global selectedTypesOfVar, selectedTypesOfTest
        index = varListbox.curselection()
        indextest = testListbox.curselection()
        # get the line's text
        selText = []
        selTextTest = []
        for i in range(0, len(index)):
            # selText = FeaturesListbox.get(index[i])
            selText.append(varListbox.get(index[i]))

        for i in range(0, len(indextest)):
            # selText = FeaturesListbox.get(index[i])
            selTextTest.append(testListbox.get(indextest[i]))
        # delete previous text in enter1
        # enter1.delete(0, 50)
        # now display the selected text
        # selectedTypesOfFeatures.append(selText)
        selectedTypesOfVar = selText
        selectedTypesOfTest = selTextTest

        """
        function to read the listbox selection
        and put the result in an entry widget
        """
        # index = varListbox.curselection()
        # indextest = testListbox.curselection()
        # # get the line's text
        # for i in range(0, len(index)):
        #     selText = varListbox.get(index[i])
        # for i in range(0, len(indextest)):
        #     selTextTest = testListbox.get(indextest[i])
        # # delete previous text in enter1
        # # enter1.delete(0, 50)
        # # now display the selected text
        # selectedTypesOfVar.append(selText)
        # selectedTypesOfTest.append(selTextTest)


    def passForword():
        global numCorr
        numCorr = numCorrDisplayEntry.get()
        # numIter.append(numIterEntry.get())
        correlations.destroy()
        return numCorr

    selectedTypesOfTest = []
    selectedTypesOfVar = []
    typesOfCorrelations = ['Test all', 'Test specific variables']
    typesOfVariables = ['DO', 'S', 'Ammonia', 'CO2', 'Production', 'Incyte', 'pH', 'Ammonia feeding',
                        'Dextrose feeding', 'Airflow', 'Agitation']


    correlations = Tk()
    C = Canvas(correlations, bg="blue", height=250, width=300)
    # filename = PhotoImage(file="C:\\Users\\admin\PycharmProjects\\tobra_model\\statistics_pic1_with_logo.png")
    filename = PhotoImage(file="statistics_pic1_with_logo.png")
    background_label = Label(correlations, image=filename)
    background_label.place(x=0, y=0, relwidth=1, relheight=1)
    correlations.geometry('900x500')
    titleLabel = Label(correlations, text='Correlations algorithm', font=('Helvetica', '17', 'bold'), fg='blue')
    titleLabel.grid(row=0, column=1, sticky=E, pady=10)
    testText = Label(correlations, text='Test type:', font=('Helvetica', '10', 'bold'))
    testText.grid(row=1, column=1, sticky=E, pady=10, padx=20)
    testListbox = Listbox(correlations, selectmode=SINGLE, exportselection=0, height=2)
    testListbox.grid(row=1, column=2, pady=10)
    for item in typesOfCorrelations:
        testListbox.insert(END, item)
    # idex = expListbox.curselection()
    # relExp = expListbox.get(idex)
    counter = 0
    varText = Label(correlations, text='Variables:', font=('Helvetica', '10', 'bold'))
    varText.grid(row=2, column=1, sticky=E, pady=10, padx=20)
    varListbox = Listbox(correlations, selectmode='multiple', exportselection=0)
    varListbox.grid(row=2, column=2, pady=10)

    for item in typesOfVariables:
        varListbox.insert(END, item)
    # idex = expListbox.curselection()
    # relExp = expListbox.get(idex)
    varListbox.bind('<ButtonRelease-1>', get_list_var)

    VariablesText = Label(correlations, text='Algorithm display:', font=('Helvetica', '10', 'bold'))
    VariablesText.grid(row=3, column=1, sticky=E, pady=0, padx=20)
    isDisplayMatrix = IntVar(value=1)
    Checkbutton(correlations, text="Matrix display", variable=isDisplayMatrix).grid(row=3, column=2, sticky=W, padx=20)
    numCorrDisplayText = Label(correlations, text='Number of best correlation to display:', font=('Helvetica', '10', 'bold'))
    numCorrDisplayText.grid(row=3, column=3, sticky=E, padx=10)
    numCorrDisplayEntry = Entry(correlations, text='all')
    numCorrDisplayEntry.grid(row=3, column=4, sticky=E, pady=10, padx=10)
    numCorrDisplayEntry.insert(0, "all")

    button = Button(text="continue", command=passForword, width=20, height=1, font=('Helvetica', '17'))
    button.place(rely=0.95, relx=0.5, anchor=CENTER)
    correlations.mainloop()

    # get GUI data
    isDisplayMatrix = isDisplayMatrix.get()

    # load the data
    Settings, Const, InitCond = simulation_initialization()

    data = load_data_df(1, selectedExp)
    # trainData, testData = devide_data(data)
    # data, offlineData = load_data(isLoadData, selectedExp)
    # interpWantedData = data_interp(data, offlineData, selectedExp, Const)
    correlationDataFrame, allexpDataframe = correlation_function_df(selectedTypesOfTest, selectedTypesOfVar,
                                                                 typesOfVariables, data, preprocessingTechnique)

    if isDisplayMatrix:
        fig, ax = plt.subplots()
        # hide axes
        fig.patch.set_visible(False)
        ax.axis('off')
        ax.axis('tight')
        ax.table(cellText=correlationDataFrame.values, colLabels=correlationDataFrame.columns, loc='center')
        fig.tight_layout()

    npCorrelationMat = correlationDataFrame.to_numpy()
    possibleCombos = list(combinations(selectedTypesOfVar, 2))
    if numCorr == 'all':
        numCorr = len(possibleCombos)
    else:
        numCorr = int(numCorr)
    scoreVec = np.zeros(len(possibleCombos))
    counter = 0
    for combo in possibleCombos:
        scoreVec[counter] = correlationDataFrame[combo[0]][combo[1]]
        counter += 1

    absScoreVec = abs(scoreVec)
    sortedIndexes = absScoreVec.argsort()
    for comboIdx in range(1, numCorr):
        relCombo = possibleCombos[sortedIndexes[-comboIdx]]
        relScore = absScoreVec[sortedIndexes[-comboIdx]]
        plt.figure()
        plt.scatter(allexpDataframe[relCombo[0]], allexpDataframe[relCombo[1]])
        plt.title("score #{}: {} as a function of {}".format(round(relScore,2), relCombo[1], relCombo[0]))
        plt.xlabel(relCombo[0])
        plt.ylabel(relCombo[1])
    # plt.show()




elif methodType == "LOESS":

    def get_list_features(event):
        """
        function to read the listbox selection
        and put the result in an entry widget
        """
        global selectedTypesOfFeatures
        index = FeaturesListbox.curselection()
        # get the line's text
        selText = []
        for i in range(0, len(index)):
            selText.append(FeaturesListbox.get(index[i]))
        selectedTypesOfFeatures = selText



    def get_list_degree(event):
        """
        function to read the listbox selection
        and put the result in an entry widget
        """
        global degreeVal
        index = degreeListbox.curselection()
        # get the line's text
        selText = []
        for i in range(0, len(index)):
            selText.append(degreeListbox.get(index[i]))
        degreeVal = selText

    def passForword():
        global selectedTypesOfFeatures, selectedTypesOfR_filter,degreeVal, fractionMinVal, fractionMaxVal
        indexFeatures = FeaturesListbox.curselection()

        # get the line's text
        selText = []
        for i in range(0, len(indexFeatures)):
            selText.append(FeaturesListbox.get(indexFeatures[i]))
        selectedTypesOfFeatures = selText

        indexR_filter = R_filterListbox.curselection()
        # get the line's text
        selText = []
        for i in range(0, len(indexR_filter)):
            selText.append(R_filterListbox.get(indexR_filter[i]))
        selectedTypesOfR_filter  = selText

        indexDegree = degreeListbox.curselection()
        # get the line's text
        selText = []
        for i in range(0, len(indexDegree)):
            selText.append(degreeListbox.get(indexDegree[i]))
        degreeVal = selText
        fractionMinVal = float(FracMinValEntry.get())
        fractionMaxVal = float(FracMaxValEntry.get())
        loess.destroy()



    selectedTypesOfFeatures = []
    # typesOfFeatures = ['Mean dO [40-end]', 'Mean Ammonia [40-end]', 'Mean Dextrose [40-end]', 'Mean pH [40-end]',
    #                    'Mean Agitation [40-end]', 'Sum Ammonia feeding [40-end]', 'TimeDexLow'
    #                                                                               'Peak dO level']
    typesOfFeatures = ['meanDO', 'meanAmm', 'meanS', 'meanpH', 'meanAgi', 'meanAmmFeed', 'Time', 'TimeDexLow']
    typesOfR_filter = ['meanDO', 'meanAmm', 'meanS', 'meanpH', 'meanAgi', 'meanAmmFeed', 'Time', 'TimeDexLow']
    typesOfDegree = ['First degree', 'Second degree']
    loess = Tk()
    # C = Canvas(loess, bg="blue", height=250, width=300)
    # # filename = PhotoImage(file="C:\\Users\\admin\PycharmProjects\\tobra_model\\statistics_pic1_with_logo.png")
    # filename = PhotoImage(file="statistics_pic1_with_logo.png")
    # background_label = Label(loess, image=filename)
    # background_label.place(x=0, y=0, relwidth=1, relheight=1)
    loess.geometry('900x500')
    titleLabel = Label(loess, text='LOESS algorithm', font=('Helvetica', '17', 'bold'), fg='blue')
    titleLabel.grid(row=0, column=1, sticky=E, pady=20)
    #Text for features
    FeaturesText = Label(loess, text='Features:', font=('Helvetica', '10', 'bold'))
    FeaturesText.grid(row=1, column=1, sticky=E, pady=10, padx=20)
    #Listbox with selection options for wanted features
    FeaturesListbox = Listbox(loess, selectmode='multiple')
    FeaturesListbox.grid(row=1, column=2, pady=10)
    for item in typesOfFeatures:
        FeaturesListbox.insert(END, item)
    # idex = expListbox.curselection()
    # relExp = expListbox.get(idex)
    FeaturesListbox.selection_set(2)
    FeaturesListbox.selection_set(6)
    # FeaturesListbox.bind('<ButtonRelease-1>', get_list_features)

    # text of R_filter
    R_filterText = Label(loess, text='R_filter:', font=('Helvetica', '10', 'bold'))
    R_filterText.grid(row=1, column=3, sticky=E, pady=10, padx=20)
    #Listbox with selection options for wanted features
    R_filterListbox = Listbox(loess, selectmode='multiple', exportselection=0)
    R_filterListbox.grid(row=1, column=4, pady=10)
    for item in typesOfR_filter:
        R_filterListbox.insert(END, item)
    # idex = expListbox.curselection()
    # relExp = expListbox.get(idex)
    R_filterListbox.selection_set(1)


    degreeText = Label(loess, text='Polynom degree:', font=('Helvetica', '10', 'bold'))
    degreeText.grid(row=2, column=1, sticky=E, pady=10, padx=20)
    # Listbox with selection options for wanted features
    degreeListbox = Listbox(loess, selectmode=SINGLE, exportselection=0, height=2)
    degreeListbox.grid(row=2, column=2, pady=10)
    for item in typesOfDegree:
        degreeListbox.insert(END, item)
    # idex = expListbox.curselection()
    # relExp = expListbox.get(idex)
    degreeListbox.selection_set(first=0)
    # degreeListbox.bind('<ButtonRelease-1>', get_list_degree)


    fracMinText = Label(loess, text='Minimal fraction for group:', font=('Helvetica', '10', 'bold'))
    fracMinText.grid(row=3, column=1, sticky=E, pady=10, padx=20)
    FracMinValEntry = Entry(loess)
    FracMinValEntry.grid(row=3, column=2, pady=10)
    FracMinValEntry.delete(0, END)
    FracMinValEntry.insert(0, "0.2")

    fracMaxText = Label(loess, text='Maximal fraction for group:', font=('Helvetica', '10', 'bold'))
    fracMaxText.grid(row=3, column=3, sticky=E, pady=10, padx=20)
    FracMaxValEntry = Entry(loess)
    FracMaxValEntry.grid(row=3, column=4, pady=10)
    FracMaxValEntry.delete(0, END)
    FracMaxValEntry.insert(0, "0.4")



    button = Button(text="continue", command=passForword, width=20, height=1, font=('Helvetica', '17'))
    button.place(rely=0.95, relx=0.5, anchor=CENTER)
    loess.mainloop()

if degreeVal == ['First degree']:
    deg = 1
elif degreeVal == ['Second degree']:
    deg = 2

# fractionMinVal = float(FracMinValEntry.get())
# fractionMaxVal = float(FracMaxValEntry.get())

# os.system("pause")

# selectedTypesOfFeatures = ['Mean dO [40-end]', 'Mean Ammonia [40-end]', 'Mean Dextrose [40-end]', 'Mean pH [40-end]',
#                  'Mean Agitation [40-end]', 'Sum Ammonia feeding [40-end]', 'Time dextrose low'
#                  'Peak dO level']
# selectedTypesOfFeatures = ['Mean dO [40-end]', 'Mean Ammonia [40-end]']
Settings, Const, InitCond = simulation_initialization()
data = load_data_df(1, selectedExp)
trainData, testData = devide_data(data)
#featureNames, featuresTrainDF, resultsTrainDF = feature_extractor_multiple_meas(selectedTypesOfFeatures, trainData)
#featureNames, featuresTestDF, resultsTestDF = feature_extractor_multiple_meas(selectedTypesOfFeatures, testData)
featureNames, featuresTrainDF, resultsTrainDF = feature_extractor_multiple_meas(typesOfFeatures, trainData)
featureNames, featuresTrainDF, resultsTrainDF = feature_extractor_multiple_meas(typesOfR_filter, trainData)
featureNames, featuresTestDF, resultsTestDF = feature_extractor_multiple_meas(typesOfFeatures, testData)
featureNames, featuresTestDF, resultsTestDF = feature_extractor_multiple_meas(typesOfR_filter, testData)

trainTimes = featuresTrainDF.index.values #times of all measurements in numpy format
testTimes = featuresTestDF.index.values #times of all measurements in numpy format
featuresTrainAvgDF = pd.DataFrame()
resultsTrainDeltaDF = pd.DataFrame()
for meas in range(len(trainTimes)-1):
    if trainTimes[meas] < trainTimes[meas+1]:
        avgTime = (trainTimes[meas] + trainTimes[meas + 1]) / 2
        avgFeatures = (featuresTrainDF.iloc[meas] + featuresTrainDF.iloc[meas+1])/2
        avgFeatures.name = avgTime
        deltaResults = (resultsTrainDF.iloc[meas+1] - resultsTrainDF.iloc[meas])/(trainTimes[meas + 1] - trainTimes[meas])
        deltaResults.name = avgTime
        featuresTrainAvgDF = featuresTrainAvgDF.append(avgFeatures)
        resultsTrainDeltaDF = resultsTrainDeltaDF.append(deltaResults)
featuresTrainAvgDF = featuresTrainAvgDF.reindex(featuresTrainDF.columns, axis=1)
resultsTrainDeltaDF = resultsTrainDeltaDF.reindex(resultsTrainDF.columns, axis=1)
featuresTrainAvgDF['Time'] = featuresTrainAvgDF.index

featuresTestAvgDF = pd.DataFrame()
resultsTestDeltaDF = pd.DataFrame()
for meas in range(len(testTimes)-1):
    if testTimes[meas] < testTimes[meas+1]:
        avgTime = (testTimes[meas] + testTimes[meas + 1]) / 2
        avgFeatures = (featuresTestDF.iloc[meas] + featuresTestDF.iloc[meas+1])/2
        avgFeatures.name = avgTime
        deltaResults = (resultsTestDF.iloc[meas+1] - resultsTestDF.iloc[meas])/(testTimes[meas + 1] - testTimes[meas])
        deltaResults.name = avgTime
        featuresTestAvgDF = featuresTestAvgDF.append(avgFeatures)
        resultsTestDeltaDF = resultsTestDeltaDF.append(deltaResults)
featuresTestAvgDF = featuresTestAvgDF.reindex(featuresTestDF.columns, axis=1)
resultsTestDeltaDF = resultsTestDeltaDF.reindex(resultsTestDF.columns, axis=1)
featuresTestAvgDF['Time'] = featuresTestAvgDF.index
t1=featuresTestAvgDF['Time'].to_numpy()
ind_new_exp=np.diff(t1)

featureNames.append('Time')#append time to feature names for further analysis
numMeasTrain = featuresTrainAvgDF.shape[0]
allFeatAvgDF = featuresTrainAvgDF.append(featuresTestAvgDF)
if preprocessingTechnique == 'Standardize (Robust scalar)':
    FeaturesProcessed = RobustScaler().fit_transform(allFeatAvgDF)
    # FeatureTestProcessed = RobustScaler().fit_transform(featuresTestAvgDF)
    resultsTrainProcessed = (resultsTrainDeltaDF)
    resultsTestProcessed = (resultsTestDeltaDF)
    FeaturesProcessed = pd.DataFrame(FeaturesProcessed, columns=featureNames)
    # FeatureTestProcessed = pd.DataFrame(FeatureTestProcessed, columns=featureNames)
    resultsTrainProcessed = pd.DataFrame(resultsTrainProcessed, columns=['Titter', 'Impurity'])
    resultsTestProcessed = pd.DataFrame(resultsTestProcessed, columns=['Titter', 'Impurity'])
elif preprocessingTechnique == 'No preprocessing':
    FeaturesProcessed = allFeatAvgDF
    # FeatureTestProcessed = featuresTestAvgDF
    resultsTrainProcessed = resultsTrainDeltaDF
    resultsTestProcessed = resultsTestDeltaDF
elif preprocessingTechnique == 'scaling (0-1)':
    FeaturesProcessed = MinMaxScaler().fit_transform(allFeatAvgDF)
    # FeatureTestProcessed = MinMaxScaler().fit_transform(featuresTestAvgDF)
    resultsTrainProcessed = (resultsTrainDeltaDF)
    resultsTestProcessed = (resultsTestDeltaDF)
    FeaturesProcessed = pd.DataFrame(FeaturesProcessed, columns=featureNames)
    # FeatureTestProcessed = pd.DataFrame(FeatureTestProcessed, columns=featureNames)
    resultsTrainProcessed = pd.DataFrame(resultsTrainProcessed, columns=['Titter', 'Impurity'])
    resultsTestProcessed = pd.DataFrame(resultsTestProcessed, columns=['Titter', 'Impurity'])

FeatureTrainProcessed = FeaturesProcessed.iloc[0:numMeasTrain]
FeatureTestProcessed = FeaturesProcessed.iloc[numMeasTrain:]
featuresTrainNP = FeatureTrainProcessed.to_numpy()
titterTrainNP = resultsTrainProcessed['Titter'].to_numpy()
featuresTestNP = FeatureTestProcessed.to_numpy()
trainTimes = resultsTrainProcessed.index.values #times of all measurements in numpy format
# sectionDecisionType = "radius"
# radius = 2 #hours
# loessModel = skmisc.loess(featuresTrainNP, titterTrainNP)
# selectedVar = ['meanS', 'meanAmm']# The options are: ['meanDO', 'meanAmm','meanS','meanpH','meanAgi','meanAmmFeed', 'Time dextrose low' 'Time']
VarNP=np.empty((len(FeatureTrainProcessed[selectedTypesOfFeatures[0]].to_numpy()),len(selectedTypesOfFeatures) ))
VartestNP=np.empty((len(FeatureTestProcessed[selectedTypesOfFeatures[0]].to_numpy()),len(selectedTypesOfFeatures) ))
for i2 in range(len(selectedTypesOfFeatures)):
    VarNP[:,i2] = FeatureTrainProcessed[selectedTypesOfFeatures[i2]].to_numpy()
    VartestNP[:,i2] = FeatureTestProcessed[selectedTypesOfFeatures[i2]].to_numpy()
VarfilterNP=np.empty((len(FeatureTrainProcessed[selectedTypesOfR_filter[0]].to_numpy()),len(selectedTypesOfR_filter) ))
VarfiltertestNP=np.empty((len(FeatureTestProcessed[selectedTypesOfR_filter[0]].to_numpy()),len(selectedTypesOfR_filter) ))
for i2 in range(len(selectedTypesOfR_filter)):
    VarfilterNP[:,i2] = FeatureTrainProcessed[selectedTypesOfR_filter[i2]].to_numpy()
    VarfiltertestNP[:,i2] = FeatureTestProcessed[selectedTypesOfR_filter[i2]].to_numpy()

#firstVarNP = FeatureTrainProcessed[selectedTypesOfFeatures[0]].to_numpy()
#secondVarNP = FeatureTrainProcessed[selectedTypesOfFeatures[1]].to_numpy()
# zout, wout = loess_2d(firstVarNP, secondVarNP, titterTrainNP, frac=0.2, degree=1, rescale=True)
# test
#firstVarTestNP = FeatureTestProcessed[selectedTypesOfFeatures[0]].to_numpy()
#secondVarTestNP = FeatureTestProcessed[selectedTypesOfFeatures[1]].to_numpy()
titterTestNP = resultsTestProcessed['Titter'].to_numpy()
z_smoot_test=np.empty(VartestNP.shape[0])
# nd
X=VarNP


for i11 in range(len(z_smoot_test)):
    X_test=VartestNP[i11,:]
    X_filter_test=VarfiltertestNP[i11,:]
    zout1, wout = loess_nd_test_point(VarNP,VarfilterNP, titterTrainNP,X_test ,X_filter_test,titterTestNP[i11], frac=fractionMinVal, degree=deg, rescale=False)
    z_smoot_test[i11]=zout1
#for i11 in range(len(firstVarTestNP)):
  #  zout1, wout = loess_2d_test_point(firstVarNP, secondVarNP, titterTrainNP, firstVarTestNP[i11],
 #                                     secondVarTestNP[i11],titterTestNP[i11], frac=fractionMinVal, degree=deg, rescale=False)
 #   z_smoot_test[i11]=zout1
err_v=(z_smoot_test-titterTestNP)/titterTestNP
k_s=0
for z1 in range(len(ind_new_exp)-1):
    if ind_new_exp[z1]<=0:
        #f1,ax=plt.figure();
        fig, ax = plt.subplots()
        h1,=ax.plot(t1[k_s:z1+1], titterTestNP[k_s:z1+1], 'ro', label='Data'),
        h2,=ax.plot(t1[k_s:z1+1], z_smoot_test[k_s:z1+1], 'bo', label='predict')
        plt.title(list(['std=', featureNames]))
       #ax.axis('equal')
        leg = ax.legend();
        #plt.xlabel(selectedTypesOfFeatures[0])
        plt.ylabel('dp/dt')
        k_s= z1+1


plt.show()
plt.figure();plt.plot(np.arange(z_smoot_test.size),err_v,'ro')
plt.title(np.std(err_v))
plt.figure()
plt.clf()
if fractionMaxVal-fractionMinVal < 0.4:
    fracOptions = np.arange(fractionMinVal, fractionMaxVal, 0.05)
else:
    fracOptions = np.arange(fractionMinVal, fractionMaxVal, 0.1)

plt.subplot(3, 3, 1)
plot_velfield(firstVarNP, secondVarNP, titterTrainNP)
plt.xlabel(selectedTypesOfFeatures[0])
plt.ylabel(selectedTypesOfFeatures[1])
plt.title("True Function")

for idx in range(len(fracOptions)):
    zout, wout = loess_2d(firstVarNP, secondVarNP, titterTrainNP, frac=fracOptions[idx], degree=1, rescale=True)
    plt.subplot(3, 3, idx+2)
    plot_velfield(firstVarNP, secondVarNP, zout)
    plt.xlabel(selectedTypesOfFeatures[0])
    plt.ylabel(selectedTypesOfFeatures[1])
    plt.title("Fraction="+str(fracOptions[idx]))

plt.show()
# plt.subplot(132)
# plot_velfield(x, y, zran)
# plt.title("With Noise Added")
# plt.tick_params(labelleft=False)
plt.subplot(121)
plot_velfield(firstVarNP, secondVarNP, titterTrainNP)
plt.xlabel(selectedTypesOfFeatures[0])
plt.ylabel(selectedTypesOfFeatures[1])
plt.title("True Function")

plt.subplot(122)
plot_velfield(firstVarNP, secondVarNP, zout)
plt.title("LOESS Recovery")
plt.xlabel(selectedTypesOfFeatures[0])
plt.ylabel(selectedTypesOfFeatures[1])
plt.tick_params(labelleft=False)
plt.show()

plt.figure()
plot_velfield(featuresTrainNP[:, 1], featuresTrainNP[:, 2], titterTrainNP, 1)
for sample in range(featuresTestDF.size):
    currentTime = featuresTestDF.iloc[sample].name
    timeDelta = abs(trainTimes-currentTime)
    if sectionDecisionType == "radius":
        relMeasIdx = np.where(timeDelta <= radius)
        currentTime = featuresTrainDF.iloc[relMeasIdx]


# experiments to be tested
# variables = ['S', 'DO', 'A']# which variables to analyse
# isLoadFromExcel = 0# 0 if data loaded from .p file, 1 if from excel
q=2
# initialize settings, constants and initial conditions
plt.show()

# load the data
# if isLoadFromExcel:
#     data, offlineData = read_data(experiments)
# else:
#     file_name = "allData.p"
#     with open(file_name, 'rb') as f:
#         allData = pickle.load(f)
#     data = {key: allData[key] for key in allData.keys() & experiments}
#     file_name = "allOfflineData.p"
#     with open(file_name, 'rb') as f:
#         allOfflineData = pickle.load(f)
#     string = '_dex'
#     offlineData = {key: allOfflineData[key] for key in allOfflineData.keys() & [x+string for x in experiments]}
#
# #Interpulate data
#
#
# #Initialize
# functionCombos = [p for p in itertools.product(functionOptions, repeat=3)]


