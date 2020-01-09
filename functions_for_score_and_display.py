import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

def show_results(scale_params, modeledVars, validationData, pref, gold_mean, results):
# Discription:
#   A function which displays the modeled values againts actual measurements for all test experiments.
# Inputs:
#   1. modeledVars- Dictionary containing data frames for each test experiment, with all modeled variable values.
#   2. validationData- dictionary containing interpolated data frame for every validation experiment.
#   3. pref- Preferences dictionary
# Outputs:
#   1. graphs- number of graphs (according to number of test experiments), displaying model against actual measurements
#              for the selected variables.

    numOfVar = len(pref['Variables'])
    numOfRows = int(math.ceil(numOfVar/2))
    for exp in modeledVars.keys():
        fig = plt.figure()
        fig.suptitle('Experiment: ' + exp + ', score: ' + str(round(gold_mean[0], 2)), fontsize=18)
        for varIdx, varName in enumerate(pref['Variables'], start=1):
            plt.subplot(numOfRows, 2, varIdx)
            c1=modeledVars[exp][varName+'_biasVel']
            if  pref['preProcessing type']=='scaling (0-1)':
                scale11=1/(scale_params[varName][1]-scale_params[varName][0])
                plt.plot(validationData[exp]['TimeMeas'],
                         (validationData[exp][varName]+scale_params[varName][0]*scale11)/scale11,
                          s=10)
                plt.plot(validationData[exp]['TimeMeas'],
                         (modeledVars[exp][varName]+scale_params[varName][0]*scale11)/scale11,
                          c=cc ,s=10)
                plt.colorbar()
            else:


                cc=1*(modeledVars[exp][varName+'_biasVel']+1e-8-min(c1))/(max(c1)+1e-8-min(c1))
#                cc[0:500]=1.80*cc[0:500]
                plt.scatter(validationData[exp]['TimeMeas'],
                     validationData[exp][varName]*scale_params[varName][1]+scale_params[varName][0]
                                                                   ,s=10 )
                plt.scatter(validationData[exp]['TimeMeas'],
                        modeledVars[exp][varName]*scale_params[varName][1]+scale_params[varName][0]
                           ,c=cc ,s=10)
                plt.colorbar()
              #        modeledVars[exp][varName]*scale_params[varName][1]+scale_params[varName][0],
              #              ,cmap='viridis')
              #           ,c=(np.append(c1.T,c1.T,axis=1)-c1.min())/(c1.max()-c1.min()))
              #   colorValues = np.array([1, 1, 1, 0, 0]).astype(float)
              #
              #   for ind,y,c in zip(np.arange(1,len(validationData[exp]['TimeMeas'])),validationData[exp][varName]*scale_params[varName][1]+scale_params[varName][0], colorValues):
              #       plt.scatter(validationData[exp]['TimeMeas'][ind], y)#,  color=([0.1843, 0.3098, 0.3098]))
              # #  plt.show()
            plt.title(varName + ', Features=' + str(results[varName]['bestParams']['features']) + '\n, distance=' +
                      str(results[varName]['bestParams']['featuresDist']) + ', Fraction=' +
                      str(results[varName]['bestParams']['frac']),
                      fontsize=9)
            plt.legend(['Measured data', 'Modeled data'])

