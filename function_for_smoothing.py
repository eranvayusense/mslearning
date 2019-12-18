import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
def filter_data(data, processType='Tobramycin'):
    # from function_for_GUI import *
    # Filter data
    # Output: A dictionary with the same format as data, containing filtered data according to predetermined filter method.
    # Either load from file or run new filtration
    smoothedData = {}
    if processType == 'Tobramycin':
        for exp in data.keys():
            smoothedData[exp] = pd.DataFrame(index=data[exp].index)
            for var in data[exp].columns:
                varData = data[exp][var]
                if var == 'Tobramycin':
                    smoothedData[exp][var] = moovin_avg_var(varData, 2, isNumWindows=True)
                elif var == 'Temp':
                    smoothedData[exp][var] = polyfit_var(varData, 5)
                elif var == 'Airflow':
                    smoothedData[exp][var] = moovin_avg_var(varData, 40)
                elif var == 'Agitation':
                    # smoothedData[exp][var] = moovin_avg_var(varData, 60)
                    smoothedData[exp][var] = data[exp][var]
                elif var == 'pH_x':
                    smoothedData[exp][var] = moovin_avg_var(varData, 60)
                elif var == 'DO':
                    smoothedData[exp][var] = moovin_avg_var(varData, 60)
                elif var == 'PCV':
                    smoothedData[exp][var] = polyfit_var(varData, 4)
                elif var == 'Kanamycin':
                    smoothedData[exp][var] = moovin_avg_var(varData, 2, isNumWindows=True)
                else:
                    smoothedData[exp][var] = data[exp][var]
            smoothedData[exp].index
    elif processType == 'BiondVax':# We need to build the BiondVax filtering
        for exp in data.keys():
            smoothedData[exp] = pd.DataFrame()
            for var in data[exp].columns:
                varData = data[exp][var]
                if var == 'DO':
                    smoothedData[exp][var] = moovin_avg_var(varData, 60)
                else:
                    smoothedData[exp][var] = data[exp][var]
    return smoothedData


def data_polyfit(data, var, deg_s, deg_e):
    filtData = {}
    for exp in list(data.keys())[0:6]:
        plt.figure()
        filtData[exp] = pd.DataFrame()
        #for var in data[exp].columns:
        #    if var == 'Dextrose[percent]':
        notNanVal = np.invert(np.isnan(data[exp][var].to_numpy()))
        plt.plot(data[exp].index.values[notNanVal], data[exp][var].to_numpy()[notNanVal])
        # plt.hold(True)
        PlotLegend = ['Data']
        for deg in range(deg_s, deg_e):
            varCoef = np.polyfit(data[exp].index.values[notNanVal], data[exp][var].to_numpy()[notNanVal], deg)
            # filtData[exp][var] = np.polyval(varCoef, data[exp].index.values)
            polyfit = np.polyval(varCoef, data[exp].index.values[notNanVal])
            plt.plot(data[exp].index.values[notNanVal], polyfit)
            plt.title('Experiment: ' + exp+', '+var)
            PlotLegend.append('Polynom degree: ' + str(deg))
        plt.legend(PlotLegend)
    plt.show()

def data_FFT(data):
    FFTData = {}
    for exp in data.keys():
        plt.figure()
        FFTData[exp] = pd.DataFrame()
        for var in data[exp].columns:
            if var == 'Fa':
                notNanVal = np.invert(np.isnan(data[exp][var].to_numpy()))
                FFTVarData = np.fft.fft(data[exp][var].to_numpy()[notNanVal])
                freq = np.fft.fftfreq(data[exp][var].to_numpy()[notNanVal].shape[0])
                plt.plot(freq, FFTVarData.real, freq, FFTVarData.imag)
    plt.show()
def moving_average(data,var,len1):
    movingAvgData = {}
    for exp in list(data.keys())[0:6]:
        plt.figure()
        movingAvgData[exp] = pd.DataFrame()
        #for var in data[exp].columns:
           # if var == 'Dextrose[percent]':
        notNanVal = np.invert(np.isnan(data[exp][var].to_numpy()))
        notNanVal = pd.Series(notNanVal)
        relevantDataNP = data[exp][var].to_numpy()[notNanVal]
        plt.plot(data[exp].index.values[notNanVal], relevantDataNP)
        PlotLegend = ['Data']
        s1=int(len1/20)
        e1=int(len1/5)
        for winOpt in range(s1, e1, 10):
            winLength = round(sum(notNanVal) / winOpt)
            movingavgVarData = pd.DataFrame(relevantDataNP).rolling(winLength).mean()
            movingavgVarData[0][0:winLength-1] = relevantDataNP[0:winLength-1]
            plt.plot(data[exp].index.values[notNanVal], movingavgVarData)
            PlotLegend.append('win size: ' + str(winLength))
            plt.title('Experiment: ' + exp+', '+var)
        plt.legend(PlotLegend)
    plt.show()

def moovin_avg_var(data, numOfParts, isNumWindows=False):
    smoothVarData = np.nan * np.ones(shape=len(data))
    notNanVal = np.invert(np.isnan(data.to_numpy()))
    # notNanVal = pd.Series(notNanVal)
    relevantDataNP = data.to_numpy()[notNanVal]
    if isNumWindows:
        winLength = numOfParts
    else:
        winLength = int(round(sum(notNanVal) / numOfParts))
    movingavgVarData = pd.DataFrame(relevantDataNP).rolling(winLength).mean()
    movingavgVarData[0][0:winLength - 1] = relevantDataNP[0:winLength - 1]
    smoothVarData[notNanVal] = movingavgVarData.to_numpy().transpose()[0]
    return smoothVarData

def polyfit_var(data, deg):
    smoothVarData = np.nan * np.ones(shape=len(data))
    notNanVal = np.invert(np.isnan(data.to_numpy()))
    varCoef = np.polyfit(data.index.values[notNanVal], data.to_numpy()[notNanVal], deg)
    polyfit = np.polyval(varCoef, data.index.values[notNanVal])
    smoothVarData[notNanVal] = polyfit
    return smoothVarData