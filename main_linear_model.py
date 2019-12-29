# import all relevant packages and functions
from functions_for_orginization import *
from functions_for_modeling_stage import *
from functions_for_score_and_display import *
import multiprocessing as mp
import time

if __name__ == '__main__':
    # Create preferences for run
    pref = setPreferences()

    # Initialize results dictionary
    results = {}  # Dictionary, containing a results dictionary for each variable

    # Load interpolated data and pre-process data for relevant experiments
    interpData, interpDataPP = load_data(pref['Process Type'], pref['preProcessing type'],
                                         pref['Is filter data'], relExp=pref['Relevant experiments'],
                                         isLoadInterpolated=pref['isLoadInterpolated'])

    # Define modeling data and validation data
    interpDataPPModeling = {exp: interpDataPP[exp] for exp in pref['Modeling experiments']}
    interpDataPPValid = {exp: interpDataPP[exp] for exp in pref['Validation experiments']}

    # Create measurements dictionary for stand alone variable LLR
    dataMeasurements = meas_creator(interpDataPPModeling, pref)

    # Run calibration for rolling model
    for variable in pref['Variables']:  # Run over every modeled variable

        # Initialize results vector for variable, each slot will represent the quality of specific Hyper parameters combination
        results[variable] = {}
        modelingT0 = time.time()
        if pref['Is run parallel']:  # Run parallel computing
            pool = mp.Pool(mp.cpu_count())  # Raise all available processors
            resultsVecForVar = \
                pool.starmap(run_var_model_for_all_CV,
                             [(paramComb, pref, dataMeasurements, variable)
                              for paramComb in pref['Combinations'].keys()])
            pool.close()
            results[variable]['resultsVec'] = np.array(resultsVecForVar)

        else:# Run one processor
            results[variable]['resultsVec2'] = np.zeros((len(pref['Combinations'])))
            for paramComb in pref['Combinations'].keys():
                results[variable]['resultsVec2'][paramComb] = \
                    run_var_model_for_all_CV(paramComb, pref, dataMeasurements, variable)

        modelingTime = time.time() - modelingT0  # Time stamp (toc)
        # Sort all possible Hyper parameters combinations according to 'resultsVec' values and find best configuration
        results[variable]['sortedCombinations'], results[variable]['sortedResultsVec'],\
        results[variable]['bestParams'] = \
            sort_param_comb(pref, results[variable]['resultsVec'])

    # Run full linear model using best Hyper parameters configuration for each variable
    # according to last CV division and display comparison

    # create pool of measurements for validation
    dataMeasurementsCombined, empty = combineData(dataMeasurements, [], isTestCombine=False)

    # Run full rolling model
    modeledVars = {}
    if pref['Is run parallel']:  # Run parallel computing

        pool = mp.Pool(mp.cpu_count())  # Raise all available processors
        modeledVarsList = \
            pool.starmap(run_and_test_full_model,
                         [(pref, results, dataMeasurementsCombined, interpDataPPValid[exp])
                          for exp in interpDataPPValid.keys()])
        pool.close()
        modeledVars = list_to_dict(modeledVarsList, list(interpDataPPValid.keys()))
    else:
        for exp in list(interpDataPPValid.keys())[0:1]:# validate only the first experiment. eran
            modeledVars1 =\
                run_and_test_full_model(pref, results, dataMeasurementsCombined, interpDataPPValid[exp])


    gold_mean = gold_hyper(pref, interpDataPPValid, modeledVars)



    # Display results of validation experiments for best configuration (data VS model)
    show_results(modeledVars, interpDataPPValid, pref, gold_mean)
    q=2
    # Next step is to find the variables which reduces model's accuracy. could be by taking out one variable from the model
    # each time and finding the improvement in "fullModelScore".



# # Run over all combinations of hyper parameters (features, frac, etc.) including index value
# # If variable is using himself as an input of the linear equation, give bad score
# if (variable in pref['Combinations'][paramComb]['features']) or\
#         len(pref['Combinations'][paramComb]['features'])<2:# moti
#     results[variable]['resultsVec2'][paramComb] = 1e8
#     continue
# CVend = 1  # pref['numCVOpt'] moti
# for CVOpt in range(0, CVend):  # 'numCVOpt' is the number of train/test division options
#     trainData, testData, pref['CVTrain'], pref['CVTest'] = \
#         devide_data_comb(dataMeasurements, pref['CVTrain'], pref['CVTest'])
#
#     # Take dictionary of dataframe for every experiment, and output a dataframe with all experiments together
#     # features = pref['Combinations'][paramComb]['features']
#     trainDataCombined, testDataCombined =\
#         combineData(trainData, testData)
#
#     # Calculate mean goal function score for all test experiments and sum for all possibilities of Cross Validation
#     results[variable]['resultsVec2'][paramComb] +=\
#         run_var_model(variable, pref['Combinations'][paramComb],
#                       trainDataCombined, testDataCombined)
