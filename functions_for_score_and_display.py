import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
def show_results(modeledVars, validationData, pref):
# Disciption:
#   A function which displays the modeled values againts actual measurements for all test experiments.
# Inputs:
#   1. modeledVars- Dictionary containing data frames for each test experiment, with all modeled variable values.
#   2. validationData- dictionary containing interpolated data frame for every validation experiment.
#   3. pref- Preferences dictionary
# Outputs:
#   1. graphs- number of graphs (according to number of test experiments), displaying model against actual measurements
#              for the selected variables.

    # testData = {exp: interpDataPP[exp] for exp in pref['CVTest']}
    numOfVar = len(pref['Variables'])
    numOfRows = int(math.ceil(numOfVar/2))
    for exp in validationData.keys():
        plt.figure()
        for varIdx, varName in enumerate(pref['Variables'], start=1):
            plt.subplot(numOfRows, 2, varIdx)
            plt.plot(validationData[exp]['TimeMeas'], validationData[exp][varName], modeledVars[exp][varName])
            plt.title(varName)
            plt.legend(['Modeled data', 'Measured data'])

