import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
def show_results(modeledVars, interpDataPP, pref,gold_mean):
    testData = {exp: interpDataPP[exp] for exp in pref['CVTest']}
    numOfVar = len(pref['Variables'])
    numOfRows = int(math.ceil(numOfVar/2))
    for exp in testData.keys():
        plt.figure()
        for varIdx, varName in enumerate(pref['Variables'], start=1):
            plt.subplot(numOfRows, 2, varIdx)
            plt.plot(testData[exp]['TimeMeas'], testData[exp][varName], modeledVars[exp][varName])
            plt.title([varName+', gold='+str(gold_mean)])
            plt.legend(['Modeled data', 'Measured data'])

