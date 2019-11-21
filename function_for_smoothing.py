def data_polyfit(data,var,deg_s, deg_e):
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
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

# def data_polyfit(data):
#     import pandas as pd
#     import numpy as np
#     import matplotlib.pyplot as plt
#     filtData = {}
#     for exp in data.keys():
#         plt.figure()
#         filtData[exp] = pd.DataFrame()
#         for var in data[exp].columns:
#             if var == 'Dextrose[percent]':
#                 notNanVal = np.invert(np.isnan(data[exp][var].to_numpy()))
#                 plt.plot(data[exp].index.values[notNanVal], data[exp][var].to_numpy()[notNanVal])
#                 # plt.hold(True)
#                 PlotLegend = ['Data']
#                 for deg in range(3, 7):
#                     varCoef = np.polyfit(data[exp].index.values[notNanVal], data[exp][var].to_numpy()[notNanVal], deg)
#                     # filtData[exp][var] = np.polyval(varCoef, data[exp].index.values)
#                     polyfit = np.polyval(varCoef, data[exp].index.values[notNanVal])
#                     plt.plot(data[exp].index.values[notNanVal], polyfit)
#                     plt.title('Experiment: ' + exp)
#                     PlotLegend.append('Polynom degree: ' + str(deg))
#                 plt.legend(PlotLegend)
#     plt.show()

def data_FFT(data):
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
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
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
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

# def moving_average(data):
#     import pandas as pd
#     import numpy as np
#     import matplotlib.pyplot as plt
#     movingAvgData = {}
#     for exp in data.keys():
#         plt.figure()
#         movingAvgData[exp] = pd.DataFrame()
#         for var in data[exp].columns:
#             if var == 'Dextrose[percent]':
#                 notNanVal = np.invert(np.isnan(data[exp][var].to_numpy()))
#                 notNanVal = pd.Series(notNanVal)
#                 relevantDataNP = data[exp][var].to_numpy()[notNanVal]
#                 plt.plot(data[exp].index.values[notNanVal], relevantDataNP)
#                 PlotLegend = ['Data']
#                 for winOpt in range(20, 70, 10):
#                     winLength = round(sum(notNanVal) / winOpt)
#                     movingavgVarData = pd.DataFrame(relevantDataNP).rolling(winLength).mean()
#                     movingavgVarData[0][0:winLength-1] = relevantDataNP[0:winLength-1]
#                     plt.plot(data[exp].index.values[notNanVal], movingavgVarData)
#                     PlotLegend.append('win size: ' + str(winLength))
#                     plt.title('Experiment: ' + exp)
#         plt.legend(PlotLegend)
#     plt.show()

q=2
            # elif var == 'S':
            #     varCoef = np.polyfit(data[exp].index.values, data[exp][var].to_numpy, 4)
            #     filtData[exp][var] = np.polyval(varCoef, data[exp].index.values)
