def insert_batch_rnd(exp_name, online_data, offline_data):
    import numpy as np
    import pandas as pd
    import pickle

    # Extract existing data
    file_name = "RnD_Data.p"
    with open(file_name, 'rb') as f:
        RnD_Data = pickle.load(f)

    # Organize new data
    if online_data.shape[1] > 10:
        online_data.columns = ['Age', 'Temp', 'Airflow', 'Pressure', 'Agitation', 'Weight',
                               'pH', 'DO', 'Fa', 'Fs', 'CO2']
    else:
        online_data.columns = ['Age', 'Temp', 'Airflow', 'Pressure', 'Agitation', 'Weight', 'pH', 'DO', 'Fa', 'Fs']

    online_data['Age'] = online_data['Age'].round(decimals=2)
    online_data.set_index('Age', inplace=True)

    if offline_data.shape[1] > 7:
        offline_data.columns = ['Age', 'pH', 'PCV', 'Dextrose[percent]', 'Ammonia[percent]',
                                'Kanamycin', 'Tobramycin', 'Incyte time', 'Incyte']
    else:
        offline_data.columns = ['Age', 'pH', 'PCV', 'Dextrose[percent]', 'Ammonia[percent]',
                                'Kanamycin', 'Tobramycin']

    offline_data['Age'] = offline_data['Age'].round(decimals=2)
    offline_data.set_index('Age', inplace=True)

    # Adjust the size of the offline data to match the size of the online data
    matching_samples = set(online_data.index).intersection(offline_data.index)
    sample = np.empty((1, offline_data.shape[1]))
    sample[:] = np.nan

    for time in online_data.index:
        if time not in matching_samples:
            offline_data.ix[time, :] = sample[0, :]

    # Merge and insert into a dictionary
    curr_data = pd.merge(online_data, offline_data, how='outer', left_index=True, right_index=True)
    RnD_Data[exp_name] = curr_data

    # Save the updated data
    output = open('RnD_Data.p', 'wb')
    pickle.dump(RnD_Data, output)
    output.close()


def insert_batch_prod(exp_name, data):
    import numpy as np
    import pandas as pd
    import pickle

    # Extract existing data
    file_name = "'Production_Data.p'"
    with open(file_name, 'rb') as f:
        RnD_Data = pickle.load(f)

    # Organize new data
    data.columns = ['Time', 'Temp', 'Airflow', 'Pressure', 'Agitation', 'Weight', 'pH', 'DO', 'Fs',
                    'Fa', 'Age',
                    'sum_Fs', 'sum_Fa', 'Tobramycin', 'Kanamycin', 'Impurity', 'A']

    # Save the updated data
    output = open('Production_Data.p', 'wb')
    pickle.dump(RnD_Data, output)
    output.close()
