def model_for_controller(InitialCond,ImportedData,params,Constants,Settings):
    import numpy as np
    # % h_0=10.^(-ImportedData.PH(1));
    # % H=h_0*ones(nConfig,ImportedData.nSteps);
    # % PH=np.zeros(nConfig,ImportedData.nSteps);
    # % PH(1)=ImportedData.PH(1);
        # %% Set vectors for relevan time interval
    flag1=1;
    X=InitialCond['X_0']*np.ones(1,InitialCond['nSteps']);#%biomass concentration vector
    V=InitialCond['Vl_0']#;%vloume vector
    S=np.multiply(InitialCond['S_0'],np.ones(1,InitialCond['nSteps'])),#;%dextrose concentration vector
    DO=Constants['DO_MAX']*InitialCond['DO_0']/100*np.ones(1,InitialCond['nSteps'])#;%oxygen concentration vector
    A=np.multiply(InitialCond['A_0'],np.ones(1,InitialCond['nSteps']))#;

    P1=np.multiply(InitialCond['P1_0'],np.ones(1,InitialCond['nSteps']))#;%tobramycin concentration vector
    mu=np.zeros(1,InitialCond['nSteps'])#;%growth rate vector initialization
    mu_pp=np.zeros(1,InitialCond['nSteps'])#;%production rate vector initialization
    mu_ag=np.zeros(1,InitialCond['nSteps'])#;%agitation negative influence initialization
    mu_pp_oil=np.zeros(1,InitialCond['nSteps'])#;%oil based production rate vector initialization
    SOil=np.zeros(1,InitialCond['nSteps'])#;%oil concentration vector initialization
    CO2=np.multiply(InitialCond['CO2_0'],np.ones(1,InitialCond['nSteps']))#;%CO2 concentration vector initialization
    soyBeanFeedMat=np.zeros(1,InitialCond['nSteps'])#;%Soybean feed vector initialization
    oilUptakeDelay=Constants['OIL_UPTAKE_DELAY']/Settings['DT']
    isDexLow=0#;%switch representing Dextrose concentration state
    C=np.ones(1,InitialCond['nSteps']);
    CO2Conversion=Constants['NUM_C_IN_DEXSTROSE']\
                  *InitialCond['Vl_0']*Constants['CO2_MOLAR_MASS']/(InitialCond['gas_V']*Constants['DEXTROSE_MOLAR_MASS'])#;%conversion between co2 concen in liquid [g/l] to concen in air [g/l]
    isLowDO=0;


      #% dymamics
    for t in np.arange(1,InitialCond['nSteps']):
            idx=1;
            relRows= T>2 and np.transpose(LoggedSignals.isDexLow)==0 and np.transpose(LoggedSignals.S[T])<5;
            if relRows:
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
