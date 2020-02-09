def loadControlled():
    import numpy as np
    import pandas as pd
    import os.path
    import sys
    import matplotlib as plt
    FileName3="C:\simulation_upgrade\intelligent controller solution/controlledParams.txt"
    FileID = open(FileName3, 'r')
    conStruct = pd.read_csv(FileID, delimiter=',', skiprows=0)
    #MeasStruct.columns = ["Age", "Temp", "Pressure", "Airflow", "Agitation", "pH", "DO", "CO2", "Mass", "Power_uptake", "Incyte", "Fs", "Fa", "Aconcen", "AMeasTime"]
    formatSpec=1

    FileID.close()
    fa=conStruct.to_numpy()[:,6:15]
    loadedControlledParams={'Time':conStruct.to_numpy()[:,0],
                            'AMeas':conStruct.to_numpy()[:,1],
                            'AWanted':conStruct.to_numpy()[:,2],
                            'F_s':conStruct.to_numpy()[:,3],
                            'Agitation':conStruct.to_numpy()[:,4],
                            'isAmmoniaUpdated':conStruct.to_numpy()[:,5],'F_a':fa}
    return loadedControlledParams
loadControlled()
# def load_data():
#     FileName='C:\simulation_upgrade\dataFile.txt'
#     FileID = open(FileName, 'r')
#     MeasStruct = pd.read_csv(FileID, delimiter=',', skiprows=1)
#     MeasStruct.columns = ["Age", "Temp", "Pressure", "Airflow", "Agitation", "pH", "DO", "CO2", "Mass", "Power_uptake", "Incyte", "Fs", "Fa", "Aconcen", "AMeasTime"]
#     formatSpec=1
#     FileID.close()
#     MeasStruct['Fa'][MeasStruct['Fa'].mask(MeasStruct['Fa']<0)]=0
#     return MeasStruct
# def fprintf(stream, format_spec, *args):
#     stream.write(format_spec % args)
# # experiments = ['0119A','0319A','0419A','0519A']
# def orginize_meas(measString):
#     DataFileName='dataFile.txt';
#     ControlledFileName='controlledParams.txt';
#     measVec=[]
#     for  item in measString:
#        measVec.append( float(item))
#     #measVec=['50','37.1985','0.22766','26','285','6.3671','100','0','99.5716','1','0.255','0','0','1.58']
#     #measVec=[float(x) for x in measVec]
#     #offlineSheets=[]
#
#     if ~os.path.isfile(DataFileName):
#         sys.stdout  = open(DataFileName, 'a+');
#         fprintf(sys.stdout,'%18s %18s %18s %18s %18s %18s %18s %18s %18s %18s %18s %18s %18s %18s\r\n',
#             'Age [Min] ,','Temp [°C] ,','Pressure [Bar] ,','Airflow [L/Min] ,','Agitation [RPM] ,','pH [-] ,','DO [%%] ,','CO2 [%%] ,','Mass [Kg] ,','Power uptake [W] ,','Incyte [pF/Cm] ,','Fs [Ml/Hrs] ,','Fa [Ml/Hrs]','Aconcen [g/L]')
#         if measVec[0]>1:
#             for time in np.arange(1,measVec[0]-1):
#                     fprintf(sys.stdout,'\n ')
#                     fprintf(sys.stdout,'%14.4f',time)
#                     for item in measVec[1:]:
#                          fprintf(sys.stdout,'%14.4f ',float(item))
#         fprintf(sys.stdout,'\n ')
#         for item in measVec:
#             fprintf(sys.stdout,'%14.4f ',float(item))
#     else:
#         sys.stdout  = open(DataFileName, 'a+')
#         MeasStruct=load_data()
#         if measVec[0]>MeasStruct['Time'][-1]+1:
#             for time in np.arange(MeasStruct.Time[-1]+1,measVec[0]-1,1):
#                  fprintf(sys.stdout,'\n ')
#                  fprintf(sys.stdout,'%14.4f',time)
#                  for item in measVec[1:]:
#                       fprintf(sys.stdout,'%14.4f ',float(item))
#             fprintf(sys.stdout,'\n ')
#         fprintf(sys.stdout,'\n ')
#         for item in measVec:
#             fprintf(sys.stdout,'%14.4f ',float(item))
#         fprintf(sys.stdout,'\n ')
#     sys.stdout.close()
#     if ~os.path.isfile(ControlledFileName):
#         sys.stdout  = open(ControlledFileName, 'a+');
#         fprintf(sys.stdout,'%24s %24s %24s %24s %24s %24s %24s %24s %24s %24s %24s %24s %24s %24s %24s %24s %24s\r\n',
#             'Time [Minutes] ,','Measured Ammonia [g/l] ,','Wanted Ammonia [g/l] ,','F_s [g/h] ,',
#             'Agitation [RPM] ,','Ammonia updated? ,','F_a 1 [g/h] ,','F_a 2 [g/h] ,','F_a 3 [g/h] ,',
#             'F_a 4 [g/h] ,','F_a 5 [g/h] ,','F_a 6 [g/h] ,','F_a 7 [g/h] ,','F_a 8 [g/h],',
#             'F_a 9 [g/h],','F_a 10 [g/h],','F_a 11 [g/h],')
#         fprintf(sys.stdout, '%20.4f ,%20.4f ,%20.4f ,%20.4f ,%20.4f ,%20.4f ,%20.4f ,%20.4f ,'
#                             '%20.4f ,%20.4f ,%20.4f ,%20.4f ,%20.4f ,%20.4f ,%20.4f ,%20.4f ,%20.4f\r\n',
#             0,1,1,0,285,1,0,0,0,0,0,0,0,0,0,0,0)
#         for T in np.arange(60,23*60,60):
#             fprintf(sys.stdout, '%20.4f ,%20.4f ,%20.4f ,%20.4f ,%20.4f ,%20.4f ,%20.4f ,%20.4f ,'
#                                 '%20.4f ,%20.4f ,%20.4f ,%20.4f ,%20.4f ,%20.4f ,%20.4f ,%20.4f ,%20.4f\r\n',
#              T,1,1,0,285,0,0,0,0,0,0,0,0,0,0,0,0)
#         sys.stdout.close()
#     Time=measVec[0]
#     AconcenMeas=measVec[13]
#     return Time,AconcenMeas
# Time,AconcenMeas=orginize_meas(['50','37.1985','0.22766','26','285','6.3671','100','0','99.5716','1','0.255','0','0','1.58'])

# data[data.eq('%').any(1)] == '?'
# is_in=data[data.isin(['%']).any(1)]
#return data,offlineData
