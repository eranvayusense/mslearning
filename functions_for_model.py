def set_initial_constants_and_settings():#(MeasStruct):

    #%% Set settings
    Settings={'nConfig':10,'nConfigPerIter':1,'DT':1/60,# length of step [hours]
    'AmmoniaStableTime':120,# time for ammonia concentration to stabilize on a new value
    'SelectedCase':'28'}

    
    #%% Set Constants values
    fat=295; gly=92.09;NUM_C_IN_FATTY=19;NUM_C_IN_GLY=3
    TRIGLY=3*fat+gly;NUM_C_IN_TRIGLY=3*NUM_C_IN_FATTY+NUM_C_IN_GLY
    Constants={'CO2_MOLAR_MASS':44.01,# [g/mol] molar mass value of CO2
                   'DEXTROSE_MOLAR_MASS': 180.16,# [g/mol] molar mass value if dextrose
        'GLYCEROL_MOLAR_MASS':gly,# [g/mol] molar mass value if glycerol
        'FATTY_ACID_MOLAR_MASS':fat,# %[g/mol] molar mass value if fatty acids
        'TRIGLYCERIDES_MOLAR_MASS':TRIGLY, #;%[g/mol] molar mass value if triglycerides
        'R_CONST':0.082057,#;%gas constant[l*atm/Kelvin*mol]
        'NUM_C_IN_DEXSTROSE':6,#;%number of carbons in dextrose molecules
        'NUM_C_IN_GLYCEROL':NUM_C_IN_GLY,#;%number of carbons in glycerol molecules
        'NUM_C_IN_FATTY_ACID':NUM_C_IN_FATTY,#;%number of carbons in fatty acid molecules
        'NUM_C_IN_TRIGLYCERIDES':NUM_C_IN_TRIGLY,#;%number of carbons in triglyceride molecule
        'OIL_DENSITY':0.9,#;%density of soybean oil [g/ml]
        'OIL_UPTAKE_DELAY':10,# ;%time difference between low dextrose and oil uptake [hours]
        'ADD_TIME_TO_UTC':1,#; %time difference in hours between UTC and Hungary
        'DO_MAX':7e-3,#;%maximum concentration of dissolved oxygen in 35 celcius water[g/l]
        'TOTAL_VOLUME':150,#;%total fermentor volume [l] for R&D and [m^3] for production
        'TIME_DIFFERENCE':1,#;%difference between UTC and Hungary time [Hours]
        'AGI_REF_VAL':275,# %refference value for oxygen insertion by agitation
        'AF_REF_VAL':0.35,#; %refference value for oxygen insertion by airflow
        'AMMONIA_FEEDING_CONCENTRATION':0.25,#;%concentration of the solution for the ammonia
        'DEXTROSE_FEEDING_CONCENTRATION':0.5,#;%concentration of the dextrose feeding solution
        'Cab':14,#;%concentration [Molar] of the Ammonia solution (needs to be validated)
        'LOW_DEX_VAL':5,#;%dextrose concentration [g/l] defined as low
        'DEPRESSION_MAX_VAL':1/3}

    #%% Set Initial conditions values
    Vl_0=80
    InitialCond={'X_0':0.15,#; %Initial biomass concentration [pF/cm] currently from incyte measurements and units
     'Vl_0':Vl_0,#;%liquid volume [l] for R&D and [m^3] for production -TO BE TAKEN FROM MEASUREMENTS!!
     'gas_V':Constants['TOTAL_VOLUME']-Vl_0,#;%gas volume in fermentor [l] for R&D and [m^3] for production
    # #%params.S_0=64.5;%initial dextrose concentration[g/l]
     'DO_0':Constants['DO_MAX'],#;% initial dissolved oxygen concentration-TO BE TAKEN FROM MEASUREMENTS!!
    # # % if isempty(MeasStruct.AConcen(end))
    # # %     InitialCond.A_0=savedData.A;% initial amonia concentration-TO BE TAKEN FROM MEASUREMENTS!!
    # # % else
    # # %     InitialCond.A_0=MeasStruct.AConcen(1);
    # # % end
     'A_0':0,
     'P1_0':0,#;%initial tobramycin weight
     'P2_0':0,'SoybeanOil':30}
    # 'CO2_0':0,#;%initial CO2 concentration [%]
    # ,#;%concentration [%] of soybean oil
    # 'nSteps':MeasStruct['Time'][-1]}
    return Settings, InitialCond,Constants
def devide_data(wantedData):
    import math
    import random
    expNames = list(wantedData.keys())
    numOfExp = len(wantedData)
    numExpForTrain = math.floor(numOfExp*0.7)
    # numExpForTest = numOfExp-numExpForTrain
    expIdx = range(0, numOfExp - 1)
    trainIdx = random.sample(expIdx, numExpForTrain)
    testIdx = set(expIdx).difference(trainIdx)
    trainData = {key: wantedData[key] for key in wantedData.keys() & [expNames[i]for i in trainIdx]}
    testData = {key: wantedData[key] for key in wantedData.keys() & [expNames[i]for i in testIdx]}
    trainData['expNames'] = [expNames[i]for i in trainIdx]
    testData['expNames'] = [expNames[i]for i in testIdx]
    return trainData,testData
def load_data_df(isLoadData, selectedExp):
    import pickle
    if isLoadData:
        fileName = 'RnD_data_new.p'
        with open(fileName, 'rb') as f:
            data = pickle.load(f)
    else:
        fileName = 'RnD_data_interpolated.p'
        data = dict()
        with open(fileName, 'rb') as f:
            allData = pickle.load(f)
        for exp in selectedExp:
            data[exp] = allData[exp]
    return data
def rand_vec_generator(type,range,NUM_OF_CONFIGURATIONS):
    import numpy as np
    if  type=='exp':
            expRange=np.log10(range)
            calcConst=(expRange[1]-expRange[0])
            randomVec=10**(expRange[0]+calcConst*np.random.uniform(0,1,NUM_OF_CONFIGURATIONS))
            isExp=1
            return randomVec,isExp
    elif  type== 'linear':
            calcConst=(range[1]-range[0])
            randomVec=range[0]+calcConst*np.random.uniform(0,1,NUM_OF_CONFIGURATIONS)
            isExp=0
            return randomVec,isExp
def mat_creater_random_co2_only(NUM_OF_CONFIGURATIONS,varargin):

# %% a function which creates a random matrix where:
# % number of columns is the number of parameters
# % number of rows is the number of confiugration selectedExp
    if (varargin)==0:
        DT=1/60;
    else:
        DT=varargin;


    mu_x,isExp_mu_x = rand_vec_generator('exp',[1e-1,1e2],NUM_OF_CONFIGURATIONS);

    [K_x,isExp_K_x] = rand_vec_generator('exp',[1e-2,1e2],NUM_OF_CONFIGURATIONS);
    [Y_xs,isExp_Y_xs] = rand_vec_generator('linear',[0,1],NUM_OF_CONFIGURATIONS);
    #%Y_xs=bestConfig(3)+0*Y_xs;
    [m_x,isExp_m_x] = rand_vec_generator('exp',[0.5,50],NUM_OF_CONFIGURATIONS);
    #%m_x=bestConfig(4)+0*m_x;
    [K_ox,isExp_K_ox] = rand_vec_generator('exp',[1e-4,1e-2],NUM_OF_CONFIGURATIONS);
    [Y_xo,isExp_Y_xo] = rand_vec_generator('exp',[1e1,1e6],NUM_OF_CONFIGURATIONS);
    [m_o,isExp_m_o] = rand_vec_generator('exp',[1e-6,1],NUM_OF_CONFIGURATIONS);
    [K_la,isExp_K_la] = rand_vec_generator('exp',[1e-1,1e2],NUM_OF_CONFIGURATIONS);
    [mu_p,isExp_mu_p] = rand_vec_generator('exp',[1e-1,1e2],NUM_OF_CONFIGURATIONS);
    [K_I,isExp_K_I] = rand_vec_generator('linear',[0,70],NUM_OF_CONFIGURATIONS);
    [K_p,isExp_K_p] = rand_vec_generator('exp',[1e-3,10],NUM_OF_CONFIGURATIONS);
    [K_op,isExp_K_op] = rand_vec_generator('exp',[1e-4,1e0],NUM_OF_CONFIGURATIONS);
    [Y_ps,isExp_Y_ps] = rand_vec_generator('linear',[0,1],NUM_OF_CONFIGURATIONS);
    [Y_pa,isExp_Y_pa] = rand_vec_generator('exp',[1e1,1e5],NUM_OF_CONFIGURATIONS);
    [Y_po,isExp_Y_po] = rand_vec_generator('exp',[1e0,1e4],NUM_OF_CONFIGURATIONS);
    [K,isExp_K] = rand_vec_generator('exp',[1e-6,1e-1],NUM_OF_CONFIGURATIONS);
    [K_d,isExp_K_d] = rand_vec_generator('exp',[1e-6,1e-2],NUM_OF_CONFIGURATIONS);
    [K_xa,isExp_K_xa] = rand_vec_generator('exp',[1e0,1e2],NUM_OF_CONFIGURATIONS);
    [Y_xa,isExp_Y_xa] = rand_vec_generator('exp',[1e-3,1e1],NUM_OF_CONFIGURATIONS);
    [K_ps2,isExp_K_ps2] = rand_vec_generator('exp',[1e-4,1e0],NUM_OF_CONFIGURATIONS);
    [K_ph,isExp_K_ph] = rand_vec_generator('exp',[10**-7.8,1e-7],NUM_OF_CONFIGURATIONS);
    [a,isExp_a] = rand_vec_generator('linear',[0.2,1],NUM_OF_CONFIGURATIONS);
    [b,isExp_b] = rand_vec_generator('linear',[1,3],NUM_OF_CONFIGURATIONS);
    [C,isExp_C] = rand_vec_generator('linear',[1,31],NUM_OF_CONFIGURATIONS);
    [Y_c,isExp_Y_c] = rand_vec_generator('exp',[10^-3,10^-1],NUM_OF_CONFIGURATIONS);
    [K_inhib,isExp_K_inhib]=rand_vec_generator('linear',[0.1,0.9],NUM_OF_CONFIGURATIONS);
    [K_aph,isExp_K_aph]=rand_vec_generator('exp',[1e5,1e11],NUM_OF_CONFIGURATIONS);
    [gamma,isExp_gamma]=rand_vec_generator('exp',[1e-8,1e-1],NUM_OF_CONFIGURATIONS);
    [oil_utility,isExp_oil_utility]=rand_vec_generator('linear',[1e-8,0.4],NUM_OF_CONFIGURATIONS);
    [oil_feed_time,isExp_oil_feed_time]=rand_vec_generator('linear',[30/DT,90/DT],NUM_OF_CONFIGURATIONS);
    param_mat = [ mu_x, K_x, Y_xs, m_x, K_ox, Y_xo, m_o, K_la, mu_p,K_I,K_p,K_op,Y_ps,Y_pa,Y_po,K,K_d,K_xa,Y_xa,K_ps2,
            K_ph, a,b,C,Y_c,K_inhib,K_aph,gamma,oil_utility,oil_feed_time ];

    isExpVec=[ isExp_mu_x,
            isExp_K_x,
            isExp_Y_xs,
            isExp_m_x,
            isExp_K_ox,
            isExp_Y_xo,
            isExp_m_o,
            isExp_K_la,
            isExp_mu_p,
            isExp_K_I,
            isExp_K_p,
            isExp_K_op,
            isExp_Y_ps,
            isExp_Y_pa,
            isExp_Y_po,
            isExp_K,
            isExp_K_d,
            isExp_K_xa,
            isExp_Y_xa,
            isExp_K_ps2,
            isExp_K_ph,
            isExp_a,
            isExp_b,
            isExp_C,
            isExp_Y_c,
            isExp_K_inhib,
            isExp_K_aph,
            isExp_gamma,
            isExp_oil_utility,
            isExp_oil_feed_time            ]

    paramsNames=[ 'mu_x'
            'K_x',
            'Y_xs',
            'm_x',
            'K_ox',
            'Y_xo',
            'm_o',
            'K_la',
            'mu_p',
            'K_I',
            'K_p',
            'K_op',
            'Y_ps',
            'Y_pa',
            'Y_po',
            'K',
            'K_d',
            'K_xa',
            'Y_xa',
            'K_ps2',
            'K_ph',
            'a',
            'b',
            'C',
            'Y_c',
            'K_inhib',
            'K_aph',
            'gamma',
            'oil_utility',
            'oil_feed_time']

    return param_mat,isExpVec,paramsNames
def loadParams(firstConfig,lastConfig,paramMat):
    import numpy as np
    import pandas as pd
    import os.path
    import sys
    import scipy.io

    matF=r'C:\Users\Admin\VAYU Sense AG\VAYU ltd - Documents\R&D\algo 7.2\tobramycin modeling\simulation upgrade results\14-Jul-2019 14_55_35 params xmin fromMoti model number 28.mat'
    mat_params=r'C:\Users\admin\VAYU Sense AG\VAYU ltd - Documents\R&D\algo 7.2\tobramycin modeling\simulation upgrade results\params819.mat'

    param_config = scipy.io.loadmat(matF)
    parms1=paramMat#param_config['xMin']
    params={'mu_x':parms1[0][int(firstConfig):int(lastConfig)+1],
                                'K_x':parms1[1][int(firstConfig):int(lastConfig)+1],
                            'Y_xs':parms1[2][int(firstConfig):int(lastConfig)+1],
                            'm_x':parms1[3][int(firstConfig):int(lastConfig)+1],
                            'K_ox':parms1[4][int(firstConfig):int(lastConfig)+1],
                            'Y_xo':parms1[5][int(firstConfig):int(lastConfig)+1],'m_o':parms1[6][int(firstConfig):int(lastConfig)+1],
            'K_la':parms1[7][int(firstConfig):int(lastConfig)+1],
                            'mu_p':parms1[8][int(firstConfig):int(lastConfig)+1],
                            'K_I':parms1[9][int(firstConfig):int(lastConfig)+1],
                            'K_p':parms1[10][int(firstConfig):int(lastConfig)+1],'K_op':parms1[11][int(firstConfig):int(lastConfig)+1],
             'Y_ps':parms1[12][int(firstConfig):int(lastConfig)+1],
                            'Y_pa':parms1[13][int(firstConfig):int(lastConfig)+1],'Y_po':parms1[14][int(firstConfig):int(lastConfig)+1],
            'K':parms1[15][int(firstConfig):int(lastConfig)+1],
                            'K_d':parms1[16][int(firstConfig):int(lastConfig)+1],
                            'K_xa':parms1[17][int(firstConfig):int(lastConfig)+1],
                            'Y_xa':parms1[18][int(firstConfig):int(lastConfig)+1],'K_ps2':parms1[19][int(firstConfig):int(lastConfig)+1],
              'K_ph':parms1[20][int(firstConfig):int(lastConfig)+1],
             'a':parms1[21][int(firstConfig):int(lastConfig)+1],
                            'b':parms1[22][int(firstConfig):int(lastConfig)+1],'C':parms1[23][int(firstConfig):int(lastConfig)+1],
            'Y_c':parms1[24][int(firstConfig):int(lastConfig)+1],
                            'K_inhib':parms1[25][int(firstConfig):int(lastConfig)+1],
                            'K_aph':parms1[26][int(firstConfig):int(lastConfig)+1],
                            'gamma':parms1[27][int(firstConfig):int(lastConfig)+1],
            'oil_utility':parms1[28][int(firstConfig):int(lastConfig)+1],'oil_feed_time':parms1[29][int(firstConfig):int(lastConfig)+1]}
    return params
def find_golden_score(lenT,params,ImportedData,Settings,InitialCond,Constants):
   # % a function which finds the golden score of a specific experiment for a set of parameters
    import numpy as np
    params['nSteps']=lenT
    #%% Set vectors for differential equations, and run growth stage (if saparated)
    GrowthData = growth_stage(Settings,InitialCond,ImportedData,params,Constants);
    ProdData=prod_stage_bilder(GrowthData,InitialCond,ImportedData,params,Constants,Settings);
    #%% calculate golden score
   # isAllRelevant=all(~isnan([ProdData.X,ProdData.S,ProdData.DO,ProdData.A,ProdData.P1,ProdData.CO2])); %1 if model is good, 0 if not
    #if 1:#% isAllRelevant
   #     [goldenScoreVec,Xmedian]=gold_function(Constants,ProdData,ImportedData,Settings);
   # else:
    goldenScoreVec=np.inf
    Xmedian=np.inf

    return goldenScoreVec,Xmedian
def  growth_stage(Settings,InitialCond,ImportedData,params,Constants):
    import numpy as np
    # %a function which first initializes the vectors for all measured variables,
    # % then calculates the relevant variables values during the growth stage according to data
    # %feeding/conditions data from experiment and a vector of paraeters.
    # %the output is a struct where each variable has one set of modeled values
    # %for each parameter vector inserted.
    #    %%determine if we are in appendix builder or parameters calibration
    if 'nCombination2Display' in Settings.keys():
        nConfig=Settings.nCombinationsPerIter;
    else:
        nConfig=len(params['mu_x'])
    #%% define vectores for variables with initial conditions
  #  h_0=10**(-ImportedData['PH'][0]);
    #H=h_0*np.ones(nConfig,ImportedData['nSteps']);
    pH_x=np.zeros([nConfig,params['nSteps']]);
   # PH[0]=ImportedData['PH'][0]
    X=InitialCond['X_0']*np.ones([nConfig,params['nSteps']])#;%biomass concentration vector
    V=InitialCond['Vl_0']*np.ones([nConfig,params['nSteps']])#;%vloume vector
    S=10*ImportedData['S'][0]*np.ones([nConfig,params['nSteps']])#;%dextrose concentration vector
    DO=Constants['DO_MAX']*ImportedData['DO'][0]/100**np.ones([nConfig,params['nSteps']])#;%oxygen concentration vector
    A=ImportedData['A'][0]*np.ones([nConfig,params['nSteps']]);
    P1=InitialCond['P1_0']*np.ones([nConfig,params['nSteps']])#;%tobramycin concentration vector
    P2=InitialCond['P2_0']*np.zeros([nConfig,params['nSteps']])#;%kanamycin concentration vector (to be added)
    mu=np.zeros([nConfig,params['nSteps']]);
    mu_pp=np.zeros([nConfig,params['nSteps']]);
    mu_pp_oil=np.zeros([nConfig,params['nSteps']]);
    SOil=np.zeros([nConfig,params['nSteps']]);
    #%% create struct with all relevant data for next stages
    GrowthData=dict({'X': X, 'V': V, 'S': S, 'DO': DO, 'A': A, 'P1': P1, 'P2': P2,
      'mu': mu,'mu_pp':mu_pp,'pH_x':pH_x,'mu_pp_oil':mu_pp_oil,'SOil':SOil})# ,'H',H, 'PH',PH

    return GrowthData
def prod_stage_bilder(GrowthData,InitialCond,ImportedData,params,Constants,Settings):
    import numpy.ma as ma
    import numpy as np
    import scipy.io
    matF=r'C:\Users\Admin\VAYU Sense AG\VAYU ltd - Documents\R&D\algo 7.2\tobramycin modeling\simulation upgrade results\14-Jul-2019 14_55_35 params xmin fromMoti model number 28.mat'

    param_config = scipy.io.loadmat(matF)
    params['mu_x']=param_config['xMin'][0,0];params['K_x']=param_config['xMin'][0,1];params['Y_xs']=param_config['xMin'][0,2];
    params['m_x']=param_config['xMin'][0,3];params['K_ox']=param_config['xMin'][0,4];params['Y_xo']=param_config['xMin'][0,5];
    params['m_o']=param_config['xMin'][0,6];params['K_la']=param_config['xMin'][0,7];params['mu_p']=param_config['xMin'][0,8];
    params['K_I']=param_config['xMin'][0,9];params['K_p']=param_config['xMin'][0,10];params['K_op']=param_config['xMin'][0,11];
    params['Y_ps']=param_config['xMin'][0,12];params['ypa']=param_config['xMin'][0,13];params['Y_po']=param_config['xMin'][0,14];
    params['K']=param_config['xMin'][0,15];params['K_d']=param_config['xMin'][0,16];params['kxa']=param_config['xMin'][0,17];
    params['yxa']=param_config['xMin'][0,18];params['K_ps2']=param_config['xMin'][0,19];params['kph']=param_config['xMin'][0,20];
    params['a']=param_config['xMin'][0,21];params['b']=param_config['xMin'][0,22];params['C']=param_config['xMin'][0,23];
    params['yc']=param_config['xMin'][0,24];params['kinhib']=param_config['xMin'][0,25];params['kaph']=param_config['xMin'][0,26];
    params['gamm']=param_config['xMin'][0,27];params['oil']=param_config['xMin'][0,28];params['oilfe']=param_config['xMin'][0,29];
#%a function which calculates the relevant variables values during the
#%production stage according to data feeding/conditions data from experiment and a vector of paraeters.
#%the output is a struct where each variable has one set of modeled values
#%for each parameter vector inserted.

#%% load variable vectores from growth stage
    X=GrowthData['X'];#%biomass concentration vector
    #%V=GrowthData['V']#;%vloume vector
    V=InitialCond['Vl_0'];
    S=GrowthData['S'];#%dextrose concentration vector
    DO=GrowthData['DO'];#%oxygen concentration vector
    A=GrowthData['A'];#%amonia concentration vector
    P1=GrowthData['P1'];#%tobramycin concentration vector
    #P2=GrowthData.P2;%kanamycin concentration vector
    #H=GrowthData['H'];
    pH_x=GrowthData['pH_x'];
    mu=GrowthData['mu'];
    mu_pp=GrowthData['mu_pp'];
    mu_ag=0*mu_pp;
    mu_pp_oil=GrowthData['mu_pp_oil']
    SOil=GrowthData['SOil']
    relCombos=[];
    soyBeanFeedMat=np.zeros([len(params['oil_feed_time']),params['nSteps']])#;%Soybean feed vector initialization   soyBeanFeedMat=zeros(length(params.oil_feed_time),ImportedData.nSteps);
    oilUptakeDelay=Constants['OIL_UPTAKE_DELAY']/Settings['DT'];
    isDexLow=np.zeros([len(params['oil_feed_time']),1]);
    C=np.ones([len(params['oil_feed_time']),params['nSteps']]);
    ImportedData['airFlowVVM']=ImportedData['Airflow']
    for t in np.arange(1,params['nSteps']):
        mask1= isDexLow==0 * S[:,t-1]<5
        relRows=np.nonzero(1*mask1);
        for idx in np.arange(0,len(mask1)):
            if mask1[idx]:
                 isDexLow[relRows]=1;
                 relFeedingIdx=np.arange(t+oilUptakeDelay,t+oilUptakeDelay+params['oil_feed_time'][idx])#;%period of soybean oil feeding
    #%                 norm_feeding_bell=new_bell(relFeedingIdx,length(relFeedingIdx)/0.5,...
    #%                 0.1,t+oilUptakeDelay+(params.oil_feed_time(idx)/2));% normalized sigmoid value for feeding
                 c1=t+oilUptakeDelay+(params['oil_feed_time'][idx]/2);
                 sig=params['oil_feed_time'][idx]/4;
                 norm_feeding_bell=np.exp(-((relFeedingIdx-c1)/sig)**2);
                 sum_norm_feeding_bell=np.sum([norm_feeding_bell], axis=1);
                 feeding_bell=norm_feeding_bell*InitialCond['SoybeanOil']/sum_norm_feeding_bell;
                 ind2=np.nonzero(relFeedingIdx<=params['nSteps'])
                 soyBeanFeedMat[idx,np.int64(np.take(relFeedingIdx,ind2))]=feeding_bell[relFeedingIdx<=params['nSteps']]*params['oil_utility'][idx]
                 soyBeanFeedMat[idx,np.int64(np.take(relFeedingIdx,ind2))]=0;
                 relDepressionIdx=np.arange(t,t+oilUptakeDelay);
                 recoveringIdx=np.arange(t+oilUptakeDelay,params['nSteps']);
        #%                 tau=t+oilUptakeDelay;
                 tau=c1;
                 C[idx,np.int64(relDepressionIdx)]=params['oil_utility'][idx];
                 C[idx,np.int64(recoveringIdx)]=1-(1-params['oil_utility'][idx])*np.exp(-((recoveringIdx-(t+oilUptakeDelay))/(tau-(t+oilUptakeDelay))));
        rel_mo=params['m_o']*C[:,t]
        rel_mx=params['m_x']*C[:,t];
        div=np.sqrt(params['K_I']*params['K_ps2'])/(params['K_I']+(np.sqrt(params['K_I']*params['K_ps2']))+
                                        ((np.sqrt(params['K_I']*params['K_ps2']))**2)/params['K_ps2']);
        norm_mich_ment=(S[:,t-1]/(params['K_I']+S[:,t-1]+(S[:,t-1]**2/params['K_ps2'])))/div;
        norm_mich_ment_oil=(SOil[:,t-1]/(params['K_I']+SOil[:,t-1]+
                (SOil[:,t-1]**2/params['K_ps2'])))/div;
        mu[:,t]=C[:,t]*params['mu_x']*(S[:,t-1]/(params['K_x']+S[:,t-1]))*(DO[:,t-1]/(params['K_ox']+DO[:,t-1]))*(A[:,t-1]/(params['K_xa']+A[:,t-1])) #...%Dissolved oxygen part of Michaelis Menten
                #;%Ammonia part of Michaelis Menten[1/h]
        alpha1=4e-4
        ag05=290;
        mu_ag[:,t]=1/(1+np.exp(alpha1*ag05*(ImportedData['Agitation'][t]-ag05)));
        mu_pp[:,t]=C[:,t]*params['mu_p']*(A[:,t-1]/(params['K_p']+A[:,t-1]))* (DO[:,t-1]/(DO[:,t-1]+params['K_op']))* norm_mich_ment#;%Dextrose part of Michaelis Menten
        mu_pp[:,t]=mu_pp[:,t]*mu_ag[:,t];
        mu_pp_oil[:,t]=C[:,t]*params['mu_p']*(A[:,t-1]/(params['K_p']+A[:,t-1]))* (DO[:,t-1]/(DO[:,t-1]+params['K_op']))*norm_mich_ment_oil;#%Dextrose part of Michaelis Menten
        mu_pp_oil[:,t]=mu_pp_oil[:,t]*mu_ag[:,t];
        m_xOil=0#rel_mx*SOil[:,t-1]/(S[:,t-1]+SOil[:,t-1]);
        m_xDex=rel_mx*S[:,t-1]/(S[:,t-1]+SOil[:,t-1]);
        minusSOil=Settings['DT']*X[:,t-1]*(mu_pp_oil[:,t]/params['Y_ps']+m_xOil);
        plusSOil=0#SOil[:,t-1]+soyBeanFeedMat[:,t];
        isOverDraft=np.nonzero(minusSOil>plusSOil);
        coefOil=0#np.ones([len(params['K_x']),1]);
        #coefOil[isOverDraft]=plusSOil[isOverDraft]/minusSOil[isOverDraft];
        SOil[:,t]=0#SOil[:,t-1]+soyBeanFeedMat[:,t]-Settings['DT']*X[:,t-1]*coefOil*(mu_pp_oil[:,t]/params['Y_ps']+m_xOil);
        plusDO=DO[:,t-1]+Settings['DT']*(params['K_la']*((ImportedData['Agitation'][t]/(Constants['AGI_REF_VAL']**2)* (ImportedData['airFlowVVM'][t]/Constants['AF_REF_VAL']))*(Constants['DO_MAX']-DO[:,t-1])));
        minusDO=Settings['DT']*(X[:,t-1]*((mu[:,t]/params['Y_xo'])+
                rel_mo+(mu_pp[:,t]+mu_pp_oil[:,t])/params['Y_po']))
        DOCoef=(plusDO-(Constants['DO_MAX']/25))/minusDO;
        plusS=S[:,t-1]+Settings['DT']*(ImportedData['Fs'][t]/V);
        minusS=Settings['DT']*(X[:,t-1]*((mu[:,t]/params['Y_xs'])+m_xDex+mu_pp[:,t]/params['Y_ps']));
        SCoef=(plusS-0.1)/minusS;
        reductionCoef=np.ones([len(DOCoef),1])*(np.min([DOCoef,SCoef]))
        naturalS=S[:,t-1]+Settings['DT']*(-X[:,t-1]* ((mu[:,t]/params['Y_xs'])+m_xDex+
                mu_pp[:,t]/params['Y_ps'])+ImportedData['Fs'][t]/V);
        coefS=S[:,t-1]+Settings['DT']*(-X[:,t-1]*reductionCoef* ((mu[:,t]/params['Y_xs'])+m_xDex+
                mu_pp[:,t]/params['Y_ps'])+ImportedData['Fs'][t]/V);
        naturalDO=DO[:,t-1]+Settings['DT']*(X[:,t-1]*
                (-(mu[:,t]/params['Y_xo'])-rel_mo-(mu_pp[:,t]+mu_pp_oil[:,t])/params['Y_po'])+params['K_la']*
                ((ImportedData['Agitation'][t]/Constants['AGI_REF_VAL'])**2*
                (ImportedData['airFlowVVM'][t]/Constants['AF_REF_VAL']))*(Constants['DO_MAX']-DO[:,t-1]));
        coefDO=DO[:,t-1]+Settings['DT']*(X[:,t-1]*reductionCoef*
                (-(mu[:,t]/params['Y_xo'])-rel_mo-(mu_pp[:,t]+mu_pp_oil[:,t])/params['Y_po'])+params['K_la']*
                ((ImportedData['Agitation'][t]/Constants['AGI_REF_VAL'])**2*
                (ImportedData['airFlowVVM'][t]/Constants['AF_REF_VAL']))*(Constants['DO_MAX']-DO[:,t-1]));
        isCoef=(coefS>naturalS) + (coefDO>naturalDO);
        S[:,t]=naturalS;
        S[np.nonzero(isCoef),t]=coefS[np.nonzero(isCoef)];
        DO[:,t]=naturalDO;
        DO[np.nonzero(isCoef),t]=coefDO[np.nonzero(isCoef)];
        DO[:,t]=np.min([Constants['DO_MAX'],DO[:,t]]);
        Coef=np.ones([len(naturalDO),1]);
        #if 1*isCoef:
        Coef[np.nonzero(1*isCoef)]=reductionCoef[np.nonzero(1*isCoef)];
        X[:,t]=X[:,t-1]+Settings['DT']*X[:,t-1]*(Coef*mu[:,t]-params['K_d']*A[:,t-1]);
        P1[:,t]=np.max([0,P1[:,t-1]+Settings['DT']*
                ((Coef*mu_pp[:,t]+coefOil*mu_pp_oil[:,t])*X[:,t-1]-params['K']*P1[:,t-1])])#;%in [g/l]
        A[:,t]=ImportedData['A'][t]

    DOPercent=DO*100/Constants['DO_MAX']
        #%% creation of a struct containing all of the data.
    ProdData=dict({'X', X, 'V', V, 'S', S, 'DO', DO, 'A', A, 'P1', P1,'P2',
           'mu', mu,
           'DOPercent',DOPercent,'mu_pp',mu_pp,'mu_pp_oil',mu_pp_oil})#;'PH',PH




    return ProdData
#%%%%%%% old #############
