def set_initial_constants_and_settings():#(MeasStruct):

    #%% Set settings
    Settings={'nConfig':1,'nConfigPerIter':1,'DT':1/60,# length of step [hours]
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
     'P2_0':0}
    # 'CO2_0':0,#;%initial CO2 concentration [%]
    # 'SoybeanOil':30,#;%concentration [%] of soybean oil
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
            randomVec=10**(expRange[0]+calcConst*np.random.uniform(NUM_OF_CONFIGURATIONS,1))
            isExp=1
            return randomVec,isExp
    elif  type== 'linear':
            calcConst=(range[1]-range[0])
            randomVec=range[0]+calcConst*np.random.uniform(NUM_OF_CONFIGURATIONS,1)
            isExp=0
            return randomVec,isExp
def mat_creater_random_co2_only(bestConfig,NUM_OF_CONFIGURATIONS,varargin):

# %% a function which creates a random matrix where:
# % number of columns is the number of parameters
# % number of rows is the number of confiugration selectedExp
    if len(varargin)==0:
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
    [K_ph,isExp_K_ph] = rand_vec_generator('exp',[10^-7.8,10^-7],NUM_OF_CONFIGURATIONS);
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
    parms1=param_config['xMin']
    params={'mu_x':parms1[firstConfig:lastConfig+1,0],
                                'K_x':parms1[firstConfig:lastConfig+1,1],
                            'Y_xs':parms1[firstConfig:lastConfig+1,2],
                            'm_x':parms1[firstConfig:lastConfig+1,3],
                            'K_ox':parms1[firstConfig:lastConfig+1,4],
                            'Y_xo':parms1[firstConfig:lastConfig+1,5],'m_o':parms1[firstConfig:lastConfig+1,6],
            'K_la':parms1[firstConfig:lastConfig+1,7],
                            'mu_p':parms1[firstConfig:lastConfig+1,8],
                            'K_I':parms1[firstConfig:lastConfig+1,9],
                            'K_p':parms1[firstConfig:lastConfig+1,10],'K_op':parms1[firstConfig:lastConfig+1,11],
             'Y_ps':parms1[firstConfig:lastConfig+1,12],
                            'Y_pa':parms1[firstConfig:lastConfig+1,13],'Y_po':parms1[firstConfig:lastConfig+1,14],
            'K':parms1[firstConfig:lastConfig+1,15],
                            'K_d':parms1[firstConfig:lastConfig+1,16],
                            'K_xa':parms1[firstConfig:lastConfig+1,17],
                            'Y_xa':parms1[firstConfig:lastConfig+1,18],'K_ps2':parms1[firstConfig:lastConfig+1,19],
              'K_ph':parms1[firstConfig:lastConfig+1,20],
             'a':parms1[firstConfig:lastConfig+1,21],
                            'b':parms1[firstConfig:lastConfig+1,22],'C':parms1[firstConfig:lastConfig+1,23],
            'Y_c':parms1[firstConfig:lastConfig+1,24],
                            'K_inhib':parms1[firstConfig:lastConfig+1,25],
                            'K_aph':parms1[firstConfig:lastConfig+1,26],
                            'gamma':parms1[firstConfig:lastConfig+1,27],
            'oil_utility':parms1[firstConfig:lastConfig+1,28],'oil_feed_time':parms1[firstConfig:lastConfig+1,29]}
    return params
def find_golden_score(params,ImportedData,Settings,InitialCond,Constants)
   # % a function which finds the golden score of a specific experiment for a set of parameters

    #%% Set vectors for differential equations, and run growth stage (if saparated)
    GrowthData = growth_stage(Settings,InitialCond,ImportedData,params,Constants);
    ProdData=prod_stage_bilder(GrowthData,InitialCond,ImportedData,params,Constants,Settings);
    %% calculate golden score
    isAllRelevant=all(~isnan([ProdData.X,ProdData.S,ProdData.DO,ProdData.A,ProdData.P1,ProdData.CO2])); %1 if model is good, 0 if not
    if 1% isAllRelevant
        [goldenScoreVec,Xmedian]=gold_function(Constants,ProdData,ImportedData,Settings);
    else
        goldenScoreVec=inf;Xmedian=inf;
    end
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
    h_0=10**(-ImportedData['PH'][0]);
    H=h_0*np.ones(nConfig,ImportedData['nSteps']);
    PH=np.zeros(nConfig,ImportedData['nSteps']);
    PH[0]=ImportedData['PH'][0]
    X=InitialCond['X_0']*np.ones(nConfig,ImportedData['nSteps'])#;%biomass concentration vector
    V=InitialCond['Vl_0']*np.ones(nConfig,ImportedData['nSteps'])#;%vloume vector
    S=ImportedData['sConcen'][:,0]*np.ones(nConfig,ImportedData['nSteps'])#;%dextrose concentration vector
    DO=Constants.DO_MAX*ImportedData.DO(1)/100**np.ones(nConfig,ImportedData['nSteps'])#;%oxygen concentration vector
    A=ImportedData.A(1)*np.ones(nConfig,ImportedData['nSteps']);
    P1=InitialCond.P1_0*np.ones(nConfig,ImportedData['nSteps'])#;%tobramycin concentration vector
    P2=InitialCond.P2_0*np.zeros(nConfig,ImportedData['nSteps'])#;%kanamycin concentration vector (to be added)
    mu=np.zeros(nConfig,ImportedData['nSteps']);
    mu_pp=np.zeros(nConfig,ImportedData['nSteps']);
    #mu_pp_oil=zeros(nConfig,ImportedData.nSteps);
    #SOil=zeros(nConfig,ImportedData.nSteps);
    #%% create struct with all relevant data for next stages
    GrowthData=dict({'X', X, 'V', V, 'S', S, 'DO', DO, 'A', A, 'P1', P1, 'P2', P2,
      'mu', mu,'mu_pp',mu_pp,'H',H, 'PH',PH})

    return GrowthData
def prod_stage_bilder(GrowthData,InitialCond,ImportedData,params,Constants,Settings):
    import numpy.ma as ma
    import numpy as np
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
    H=GrowthData['H'];
    PH=GrowthData['PH'];
    mu=GrowthData['mu'];
    mu_pp=GrowthData['mu_pp'];
    relCombos=[];
    soyBeanFeedMat=np.zeros(len(params['oil_feed_time']),InitialCond['nSteps'])#;%Soybean feed vector initialization   soyBeanFeedMat=zeros(length(params.oil_feed_time),ImportedData.nSteps);
    oilUptakeDelay=Constants.OIL_UPTAKE_DELAY/Settings.DT;
    isDexLow=np.zeros(len(params.oil_feed_time),1);
    C=np.ones(len(params['oil_feed_time']),ImportedData.nSteps);
    for t in np.arange(1,InitialCond['nSteps']):
        relRows=np.where(isDexLow==0 & S[:,t-1]<5);
        for idx in range (0,relRows):
             isDexLow[relRows]=1;
             relFeedingIdx=t+np.arange(oilUptakeDelay,t)+oilUptakeDelay+params['oil_feed_time'][idx]#;%period of soybean oil feeding
#%                 norm_feeding_bell=new_bell(relFeedingIdx,length(relFeedingIdx)/0.5,...
#%                 0.1,t+oilUptakeDelay+(params.oil_feed_time(idx)/2));% normalized sigmoid value for feeding
             c1=t+oilUptakeDelay+(params['oil_feed_time'][idx]/2);
             sig=params['oil_feed_time'][idx]/4;
             norm_feeding_bell=np.exp(-((relFeedingIdx-c1)/sig)**2);
             sum_norm_feeding_bell=sum(norm_feeding_bell,2);
             feeding_bell=norm_feeding_bell*InitialCond['SoybeanOil']/sum_norm_feeding_bell;
             soyBeanFeedMat[idx,relFeedingIdx[relFeedingIdx<=ImportedData['nSteps']]]=feeding_bell[relFeedingIdx<=ImportedData['nSteps']]*params['oil_utility'][idx]
             soyBeanFeedMat[idx,relFeedingIdx[relFeedingIdx<=ImportedData['nSteps']]]=0;
             relDepressionIdx=np.arange(t,t+oilUptakeDelay);
             recoveringIdx=np.arange(t+oilUptakeDelay,ImportedData['nSteps']);
    #%                 tau=t+oilUptakeDelay;
             tau=c1;
             C[idx,relDepressionIdx]=params['oil_utility'][idx];
             C[idx,recoveringIdx]=1-(1-params['oil_utility'][idx])*np.exp(-((recoveringIdx-(t+oilUptakeDelay))/(tau-(t+oilUptakeDelay))));
        rel_mo=params['m_o']*C[:,t]
        rel_mx=params['m_x']*C[:,t];
        div=np.sqrt(params['K_I']*params['K_ps2'])/
                (params['K_I']+(np.sqrt(params['K_I']*params['K_ps2']))+
                (np.sqrt(params['K_I']*params['K_ps2']))**2/params['K_ps2']);
        norm_mich_ment=(S[:,t-1]/(params['K_I']+S[:,t-1])+
                (S[:,t-1]**2/params['K_ps2'])))/div;
        norm_mich_ment_oil=(SOil[:,t-1]/(params['K_I']+SOil[:,t-1]+
                (SOil[:,t-1]**2/params['K_ps2'])))/div;
        mu[:,t]=C[:,t]*params['mu_x']*(S[:,t-1]/(params['K_x']+S[:,t-1]))*#...%Dextrose part of Michaelis Menten
                (DO[:,t-1]/(params['K_ox']+DO[:,t-1]))*#...%Dissolved oxygen part of Michaelis Menten
                (A[:,t-1]/(params['K_xa']+A[:,t-1]))#;%Ammonia part of Michaelis Menten[1/h]
        alpha1=4e-4
        ag05=290;
        mu_ag[:,t]=1./(1+np.exp(alpha1*ag05*(ImportedData['agitation'][:,t]-ag05)));
        mu_pp[:,t]=C[:,t]*params['mu_p']*(A[:,t-1]/(params['K_p']+A[:,t-1]))*
                (DO[:,t-1]/([DO[:,t-1]+params['K_op']))*#...%Dissolved oxygen part of Michaelis Menten
                norm_mich_ment;%Dextrose part of Michaelis Menten
        mu_pp[:,t]=mu_pp[:,t]*mu_ag[:,t];
        mu_pp_oil[:,t]=C[:,t]*params['mu_p']*(A[:,t-1]/(params['K_p']+A[:,t-1]))*
                (DO[:,t-1]/(DO[:,t-1]+params['K_op']))*#...%Dissolved oxygen part of Michaelis Menten
                norm_mich_ment_oil;%Dextrose part of Michaelis Menten
        mu_pp_oil[:,t]=mu_pp_oil[:,t]*mu_ag[:,t];
        m_xOil=rel_mx*SOil[:,t-1]/(S[:,t-1]+SOil[:,t-1]);
        m_xDex=rel_mx*S[:,t-1]/(S[:,t-1]+SOil[:,t-1]);
        minusSOil=Settings['DT']*X[:,t-1]*(mu_pp_oil[:,t]/params['Y_ps']+m_xOil);
        plusSOil=SOil[:,t-1]+soyBeanFeedMat[:,t];
        isOverDraft=minusSOil>plusSOil;
        coefOil=np.ones(len(params['K_x']),1);
        coefOil[isOverDraft]=plusSOil[isOverDraft]/minusSOil[isOverDraft];
        SOil[:,t]=SOil[:,t-1]+soyBeanFeedMat[:,t]-Settings['DT']*X[:,t-1]*coefOil*(mu_pp_oil[:,t]/params['Y_ps']+m_xOil);
        plusDO=DO[:,t-1]+Settings['DT']*(params['K_la']*((ImportedData['agitation'][:,t]/ImportedData['AGI_REF_VAL']**2*
                (ImportedDataairFlowVVM[:,t]/Constants['AF_REF_VAL']))*(Constants['DO_MAX']-DO[:,t-1]));
        minusDO=Settings['DT']*(X[:,t-1]*((mu[:,t]/params['Y_xo'])
                +rel_mo+(mu_pp[:,t]+mu_pp_oil[:,t])/params['Y_po']));
        DOCoef=(plusDO-(Constants['DO_MAX']/25))/minusDO;
        plusS=S[:,t-1]+Settings['DT']*(ImportedData['F_s'][:,t]/V);
        minusS=Settings['DT']*(X[:,t-1]*((mu[:,t]/params['Y_xs'])+m_xDex+mu_pp[:,t]/params['Y_ps']));
        SCoef=(plusS-0.1)./minusS;
        reductionCoef=np.min(DOCoef,SCoef);
        naturalS=S[:,t-1]+Settings['DT']*(-X[:,t-1].* ((mu[:,t]./params['Y_xs'])+m_xDex+
                mu_pp[:,t]./params['Y_ps'])+ImportedData['F_s'][:,t]./V);

        #%%%%%%% old #############
             idx=1;
            relRows= T>2 and np.transpose(LoggedSignals.isDexLow)==0 and np.transpose(LoggedSignals.S[T])<5;
            if relRows:



      #% dymamics

                    LoggedSignals['isDexLow']=1;
                    relFeedingIdx=T+np.arange(oilUptakeDelay,T)+oilUptakeDelay+params1['params']['oil_feed_time'].item(0);#%period of soybean oil feeding
     #               norm_feeding_bell=self.new_bell(relFeedingIdx,length(relFeedingIdx)/0.5,...
    #%                 0.1,t+oilUptakeDelay+(params1.oil_feed_time(idx)/2));% normalized sigmoid value for feeding
                    c1=T+oilUptakeDelay+params1['params']['oil_feed_time'].item(0)/2
                    sig=params1['params']['oil_feed_time'].item(0)/4
                    norm_feeding_bell=np.exp(-((relFeedingIdx-c1)/sig)^2);
                    sum_norm_feeding_bell=np.sum(norm_feeding_bell,2);
                    feeding_bell=norm_feeding_bell*params1['params']['InitialCond'].item(0)[0,0][7]/sum_norm_feeding_bell
                    LoggedSignals.soyBeanFeedMat[idx,relFeedingIdx(relFeedingIdx<=nSteps)]=feeding_bell(relFeedingIdx<=nSteps)*params1['params']['oil_utility'].item(0);
                    LoggedSignals.soyBeanFeedMat[idx,relFeedingIdx(relFeedingIdx<=nSteps)]=0;
                    relDepressionIdx=np.arange(T,T+oilUptakeDelay)
                    recoveringIdx=T+np.arange(oilUptakeDelay,nSteps)
    #%                 tau=T+oilUptakeDelay;
                    tau=c1;
                    LoggedSignals.C[idx,relDepressionIdx]=params1['params']['oil_utility'].item(0);
                    LoggedSignals.C[idx,recoveringIdx]=1-(1-params1['params']['oil_utility'].item(0))*np.exp(-((recoveringIdx-(T+oilUptakeDelay))
                                                                                           /(tau-(T+oilUptakeDelay))));

#%                 DO[idx,t-1]=min(DO(idx,t-1)+0.15*Constants.DO_MAX,Constants.DO_MAX);

 #                if 0#%strcmp( params1.test1,'DO10')
#                                                    DO=0.2*Constants.DO_MAX;

            rel_mo=params1['params']['m_o']*LoggedSignals.C[T];
            rel_mx=params1['params']['m_x']*LoggedSignals.C[T];
            div=np.sqrt(params1['params']['K_I']*params1['params']['K_ps2'])/(params1['params']['K_I']+(np.sqrt(params1['params']['K_I']*params1['params']['K_ps2']))+
                                                        (np.sqrt(params1['params']['K_I']*params1['params']['K_ps2']))^2/params1['params']['K_ps2']);
            norm_mich_ment=(LoggedSignals.S[T]/(['params']['K_I']+LoggedSignals.S[T]+
                (LoggedSignals.S[T]^2./params1['params']['K_ps2'])))/div;
            norm_mich_ment_oil=(SOil/(params1['params']['K_I']+SOil+...
                        (SOil^2./params1['params']['K_ps2'])))/div;
            mu=LoggedSignals.C[T]*params1['params']['mu_x']*(LoggedSignals.S[T]/(params1['params']['K_x']+LoggedSignals.S[T]))*(DO/(params1['params']['K_ox']+DO))*(A/(params1['params']['K_xa']+A))
            alpha1=4e-4; ag05=290;
            mu_ag=1/(1+np.exp(alpha1*ag05*(LoggedSignals.Ag[T]-ag05)))
            mu_pp=LoggedSignals.C[T]*params1['params']['mu_p']*(A/(params1['params']['K_p'].ravel()+A))*(DO/(DO+params1['params']['K_op']))*norm_mich_ment;
            mu_pp=mu_pp*mu_ag;
            mu_pp_oil=LoggedSignals.C[T]*params1['params']['mu_p'](1)*(A/(params1['params']['K_p']+A))*(DO/(DO+params1['params']['K_op']))*norm_mich_ment_oil #;%Dextrose part of Michaelis Menten
            mu_pp_oil=mu_pp_oil*mu_ag;
            m_xOil=rel_mx*SOil/(LoggedSignals.S[T]+SOil);
            m_xDex=rel_mx*LoggedSignals.S[T]/(LoggedSignals.S[T]+SOil);
            minusSOil=Settings.DT*X*(mu_pp_oil/params1['params']['Y_ps']+m_xOil);
            plusSOil=SOil+ LoggedSignals.soyBeanFeedMat[T];
            isOverDraft=minusSOil>plusSOil;
            coefOil=np.ones(len(params1.K_x(1)),1);
            coefOil[isOverDraft]=plusSOil(isOverDraft)/minusSOil(isOverDraft);
            SOil=SOil+ LoggedSignals.soyBeanFeedMat[T]-Settings.DT*X*coefOil*(mu_pp_oil/params1['params']['Y_ps']+m_xOil);

            plusDO=DO+Settings.DT*(params1['params']['K_la']*
                ((LoggedSignals.Ag[T]/Constants.AGI_REF_VAL)^2*(Constants.airFlowVVM/Constants.AF_REF_VAL))*(Constants.DO_MAX-DO));
            minusDO=Settings.DT*(X*
                ((mu/params1['params']['Y_xo'])+rel_mo+(mu_pp+mu_pp_oil)/params1['params']['Y_po']));
            DOCoef=(plusDO-(Constants.DO_MAX/25))/minusDO;
            plusS=LoggedSignals.S[T]+Settings.DT*(LoggedSignals.F_s[T]/V);
            minusS=Settings.DT*(X*
                ((mu/params1['params']['Y_xs'])+m_xDex+
                mu_pp/params1['params']['Y_ps']));
            SCoef=(plusS-0.1)/minusS;
            reductionCoef=min(DOCoef,SCoef);
            naturalS=LoggedSignals.S[T]+Settings.DT*(-X*((mu/params1['params']['Y_xs'])+m_xDex+mu_pp/params1['params']['Y_ps'])+LoggedSignals.F_s[T]/V);
            coefS=LoggedSignals.S[T]+Settings.DT*(-X*reductionCoef*
                ((mu/params1['params']['Y_xs'])+m_xDex+
                mu_pp/params1['params']['Y_ps'])+LoggedSignals.F_s[T]/V);

            naturalDO=DO+Settings.DT*(X*...
                (-(mu/params1['params']['Y_xo'])-rel_mo-(mu_pp+mu_pp_oil)/params1['params']['Y_po'])+params1['params']['K_la']*
                ((LoggedSignals.Ag[T]/Constants.AGI_REF_VAL)^2*
                (Constants.airFlowVVM/Constants.AF_REF_VAL))*(Constants.DO_MAX-DO));
            coefDO=DO+Settings.DT*(X*reductionCoef*
                (-(mu/params1['params']['Y_xo'])-rel_mo-(mu_pp+mu_pp_oil)/params1['params']['Y_po'])+params1['params']['K_la']*
                ((LoggedSignals.Ag[T]/Constants.AGI_REF_VAL)^2*
                (Constants.airFlowVVM/Constants.AF_REF_VAL))*(Constants.DO_MAX-DO));
            isCoef=coefS>naturalS or coefDO>naturalDO;
            LoggedSignals.S[T+1]=naturalS;
            #%LoggedSignals.S[T](find(isCoef))=coefS(find(isCoef));
            if isCoef:
                LoggedSignals.S[T+1]=coefS;

            DO=naturalDO;
            DO[isCoef.nonzero()]=coefDO(isCoef.nonzero());
            DO=min(Constants.DO_MAX,DO);
            Coef=np.ones(len(naturalDO),1);
            Coef[isCoef.nonzero()]=reductionCoef(isCoef.nonzero());

            #%A=1;%;params1.impData1{1,rnd1}.A;
            X=X+Settings.DT*X*(Coef*mu-params1['params']['K_d']*A);
            LoggedSignals.P1[T+1]=np.max(0,LoggedSignals.P1[T]+Settings.DT*
                ((Coef*mu_pp+coefOil*mu_pp_oil)*X-params1['params']['K']*LoggedSignals.P1[T]));#%in [g/l]
            LoggedSignals.A1[T]=A; LoggedSignals.DO1[T]=DO; LoggedSignals.X1[T]=X;  LoggedSignals.A[T]=A;

    return ProdData
