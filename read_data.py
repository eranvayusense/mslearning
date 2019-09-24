def read_data(experiments):
    import numpy as np
    import pandas as pd
    import matplotlib as plt
    # experiments = ['0119A','0319A','0419A','0519A']
    offlineSheets=[]
    for exp in experiments:
        offlineSheets.append(exp+'_dex')
    # experiments = ['0119A','0319A']
    allData=[]
    counter=0
    data = pd.read_excel('C:\\Users\Admin\Documents\simulation_upgrade\Data\RND_orginized_data.xlsx', sheet_name = experiments)
    offlineData=pd.read_excel('C:\\Users\Admin\Documents\simulation_upgrade\Data\RND_orginized_data.xlsx', sheet_name = offlineSheets)
    for exp in experiments:
        if data[exp].shape[1] > 10:
            data[exp].columns = data[exp].columns = ['Age', 'Temp', 'Airflow', 'Pressure', 'Agitation', 'Weight',
                                                     'pH', 'DO', 'Fa', 'Fs', 'CO2']
        else:
            data[exp].columns = ['Age', 'Temp', 'Airflow', 'Pressure', 'Agitation', 'Weight', 'pH', 'DO', 'Fa', 'Fs']



        if offlineData[exp+'_dex'].shape[1] > 7:
            offlineData[exp+'_dex'].columns = ['Age', 'pH', 'PCV', 'Dextrose[percent]', 'Ammonia[percent]',
                                               'Kanamycin', 'Tobramycin', 'Incyte time', 'Incyte']
        else:
            offlineData[exp+'_dex'].columns = ['Age', 'pH', 'PCV', 'Dextrose[percent]', 'Ammonia[percent]',
                                               'Kanamycin', 'Tobramycin']







    # data[data.eq('%').any(1)] == '?'
    # is_in=data[data.isin(['%']).any(1)]
    return data, offlineData
