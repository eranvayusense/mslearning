
"""
 Script organizing R&D data from 'allData' and 'allOnlineData' into a dictionary containing dataframes of merged data,
 then saving it to a .py file
"""
import numpy as np
import pandas as pd
import pickle
import datetime
from operator import itemgetter
# from functions_for_data import *
RnD_Data = {}

file_name = "allData.p"
with open(file_name, 'rb') as f:
    online_data = pickle.load(f)
file_name = "allOfflineData.p"
with open(file_name, 'rb') as f:
    offline_data = pickle.load(f)

for exp in online_data.keys():

    # Organize online data
    curr_online = online_data[exp]
    curr_online['Age'] = curr_online['Age'].round(decimals=2)
    curr_online.set_index('Age', inplace=True)

    # Organize offline data
    curr_offline = offline_data[exp + '_dex']
    flag = 0
    if len(curr_offline.columns) > 7:

        # Organize the Incyte data

        IncyteData = curr_offline['Incyte'].values[1:-1]
        IncyteData[IncyteData < 0] = 0
        IncyteTimes = curr_offline['Incyte time'].values[1:-1]
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
        IncyteTime = IncyteTime/60
        IncyteTime = IncyteTime.round(decimals=2)

        curr_offline = curr_offline.drop(columns=['Incyte time', 'Incyte'])
        curr_offline = curr_offline.dropna(how='all')
        flag = 1

    curr_offline['Age'] = curr_offline['Age'].round(decimals=2)
    curr_offline.set_index('Age', inplace=True)
    check_insertion = curr_offline.copy()

    # Adjust the size of the offline data to match the size of the online data
    matching_samples = set(curr_online.index).intersection(curr_offline.index)
    sample = np.empty((1, curr_offline.shape[1]))
    sample[:] = np.nan

    for time in curr_online.index:
        if time not in matching_samples:
            curr_offline.loc[time, :] = sample[0, :]

    curr_data = pd.merge(curr_online, curr_offline, how='outer', left_index=True, right_index=True)

    # Repeat to insert Incyte data
    if flag > 0:
        Incyte_times = set(IncyteTime[0, :])
        Regular_times = set(curr_data.index)
        all_time_samples = list(Incyte_times|Regular_times)
        sample = np.empty((1, curr_data.shape[1]))
        sample[:] = np.nan

        IncyteDF = pd.DataFrame(data=IncyteData, index=IncyteTime[0, :])
        IncyteDF.columns = ['Incyte']

        for time in all_time_samples:
            if time not in curr_data.index:
                curr_data.loc[time, :] = sample[0, :]
            if time not in IncyteTime[0,:]:
                IncyteDF.loc[time] = np.nan

        IncyteDF = IncyteDF.sort_index()
        IncyteDF = pd.to_numeric(IncyteDF['Incyte'], errors='coerce')
        IncyteDF = IncyteDF.interpolate(method='polynomial', order=2)
        curr_data = pd.merge(curr_data, IncyteDF, how='outer', left_index=True, right_index=True)

    RnD_Data[exp] = curr_data
    print(exp)


print('DONE DONE DONE')
output = open('RnD_Data_5_1.p', 'wb')
pickle.dump(RnD_Data, output)
output.close()
print('DONE DONE DONE')


# exp_name = '040117'
# online_data = pd.read_excel('RND_orginized_data.xlsx', exp_name)
# offline_data = pd.read_excel('RND_orginized_data.xlsx', exp_name + '_dex')
#
# insert_batch_rnd('PLEASE_WORK',online_data,offline_data)