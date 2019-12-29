# import all relevant packages and functions
from functions_for_orginization import *
from functions_for_modeling_stage import *
from functions_for_score_and_display import *

# Create preferences for run
pref = setPreferences()

#Initialize results dictionary
results = {}  # Dictionary, containing a results dictionary for each variable

# Load interpulated data and preprocess data for relevant experiments
interpData, interpDataPP,scale_params = load_data(pref['Process Type'], pref['preProcessing type'],
                                     pref['Is filter data'], relExp=pref['Relevant experiments'],
                                     isLoadInterpolated=pref['isLoadInterpolated'])

# Define modeling data and validation data
interpDataPPModeling = {exp: interpDataPP[exp] for exp in pref['Modeling experiments']}
interpDataPPValid = {exp: interpDataPP[exp] for exp in pref['Validation experiments']}

# Create measurements dictionary for stand alone variable LLR
dataMeasurements = meas_creator(interpDataPPModeling, pref)


# Run calibration for rolling model
for variable in pref['Variables']: #Run over every modeled variable

    # Initialize results vector for variable, each slot will represent the quality of specific Hyper parameters combination
    results[variable] = {}
    results[variable]['resultsVec'] = np.zeros((len(pref['Combinations'])))

    for paramComb in pref['Combinations'].keys(): #Run over all combinations of hyper parameters (features, frac, etc.) including index value
        # If variable is using himself as an input of the linear equation, give bad score
        if (variable in pref['Combinations'][paramComb]['features']) or len(pref['Combinations'][paramComb]['features'])<2:# moti
            results[variable]['resultsVec'][paramComb] = 1e8
            continue
        CVend= 1# pref['numCVOpt']  moti
        for CVOpt in range(0, CVend): #'numCVOpt' is the number of train/test division options
            trainData, testData, pref['CVTrain'], pref['CVTest'] = \
                devide_data_comb(dataMeasurements, pref['CVTrain'], pref['CVTest'])

            # Take dictionary of dataframe for every experiment, and output a dataframe with all experiments together
            # features = pref['Combinations'][paramComb]['features']
            trainDataCombined, testDataCombined =\
                combineData(trainData, testData)

            # Calculate mean goal function score for all test experiments and sum for all possibilities of Cross Validation
            #results[variable]['resultsVec'][paramComb] += run_var_model(variable, pref['Combinations'][paramComb],
            #                                                            trainDataCombined, testDataCombined)   #  moti
            results[variable]['resultsVec'][paramComb] += run_var_model_mat(variable, pref['Combinations'][paramComb],
                                                                        trainDataCombined, testDataCombined)
    # Sort all possible Hyper parameters combinations according to 'resultsVec' values and find best configuration
    results[variable]['sortedCombinations'], results[variable]['sortedResultsVec'], results[variable]['bestParams'] = \
        sort_param_comb(pref, results[variable]['resultsVec'])

# Run full linear model using best Hyper parameters configuration for each variable according to last CV division and
# display comparison
allModelingData = trainDataCombined.append(testDataCombined)
modeledVars, gold_mean = run_and_test_full_model(pref, results, allModelingData, interpDataPPValid)

# Display results of validation experiments for best configuration (data VS model)
show_results(scale_params,modeledVars, interpDataPPValid, pref, gold_mean)
q=2
# Next step is to find the variables which reduces model's accuracy. could be by taking out one variable from the model
# each time and finding the improvement in "fullModelScore".












