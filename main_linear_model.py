# import all relevant packages and functions
import numpy as np
import pandas as pd

from functions_for_orginization import *
from function_for_smoothing import *
from functions_for_modeling_stage import *
from functions_for_score_and_display import *

# Create preferences for run
# output: pref- a dictionary containing all relevant preferences for next run (each preference is either a number or a string)
pref = setPreferences()

#Initialize results dictionary with a dict for each variable which will contain the results for him
results = {}#dict.fromkeys(pref['Variables'], {})

# Load interpulated data and preprocess data
interpData, interpDataPP = load_data(pref['Process Type'], pref['preProcessing type'],
                                     pref['Is filter data'], relExp=pref['Relevant experiments'],
                                     isLoadInterpolated=pref['isLoadInterpolated'])#load data of relevant experiments

dataMeasurements = meas_creator(interpDataPP, pref)


# Run calibration for rolling model
for variable in pref['Variables']: #Run over every variable in model

    # Initialize results vector for variable, each slot will represent the quality of specific Hyper parameters combination
    results[variable] = {}
    results[variable]['resultsVec'] = np.zeros((len(pref['Combinations'])))

    for paramComb in pref['Combinations'].keys(): #Run over all combinations of hyper parameters (features, frac, etc.) including index value
        # no point in modeling a variable using data of the same variable.
        if variable in pref['Combinations'][paramComb]['features']:
            results[variable]['resultsVec'][paramComb] = 1e8
            continue
        for CVOpt in pref['numCVOpt']: #'numCVOpt' is the number of train/test division options
            trainData, testData, pref['CVTrain'], pref['CVTest'] = \
                devide_data_comb(dataMeasurements, pref['CVTrain'], pref['CVTest']) # I think there is no need for "pref['trainIdx']", it could be found

            # Take dictionary of dataframe for every experiment, and output a dataframe with all experiments together
            # features = pref['Combinations'][paramComb]['features']
            trainDataCombined, testDataCombined =\
                combineData(trainData, testData)

            # Calculate mean goal function score for all test experiments and sum for all possibilities of Cross Validation
            results[variable]['resultsVec'][paramComb] += run_var_model(variable, pref['Combinations'][paramComb],
                                                                        trainDataCombined, testDataCombined, pref)

    # Sort all possible Hyper parameters combinations according to 'resultsVec' values and find best configuration
    results[variable]['sortedCombinations'], results[variable]['sortedResultsVec'], results[variable]['bestParams'] = \
        sort_param_comb(pref, results[variable]['resultsVec'])

# Run full linear model using best Hyper parameters configuration for each variable according to last CV division and
# display comparison
modeledVars = run_and_test_full_model(pref, results, trainDataCombined, interpDataPP)



show_results(modeledVars, interpDataPP, pref)
q=2
# Next step is to find the variables which reduces model's accuracy. could be by taking out one variable from the model
# each time and finding the improvement in "fullModelScore".












