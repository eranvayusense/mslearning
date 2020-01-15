
"""classic Acrobot task"""
import numpy as np
from numpy import sin, cos, pi
import pandas as pd
from gym import core, spaces
from gym.utils import seeding

__copyright__ = "Copyright 2013, RLPy http://acl.mit.edu/RLPy"
__credits__ = ["Alborz Geramifard", "Robert H. Klein", "Christoph Dann",
               "William Dabney", "Jonathan P. How"]
__license__ = "BSD 3-Clause"
__author__ = "Christoph Dann <cdann@cdann.de>"

# SOURCE:
# https://github.com/rlpy/rlpy/blob/master/rlpy/Domains/Acrobot.py

class AcrobotEnv(core.Env):

    """
    Acrobot is a 2-link pendulum with only the second joint actuated.
    Initially, both links point downwards. The goal is to swing the
    end-effector at a height at least the length of one link above the base.
    Both links can swing freely and can pass by each other, i.e., they don't
    collide when they have the same angle.
    **STATE:**
    The state consists of the sin() and cos() of the two rotational joint
    angles and the joint angular velocities :
    [cos(theta1) sin(theta1) cos(theta2) sin(theta2) thetaDot1 thetaDot2].
    For the first link, an angle of 0 corresponds to the link pointing downwards.
    The angle of the second link is relative to the angle of the first link.
    An angle of 0 corresponds to having the same angle between the two links.
    A state of [1, 0, 1, 0, ..., ...] means that both links point downwards.
    **ACTIONS:**
    The action is either applying +1, 0 or -1 torque on the joint between
    the two pendulum links.
    .. note::
        The dynamics equations were missing some terms in the NIPS paper which
        are present in the book. R. Sutton confirmed in personal correspondence
        that the experimental results shown in the paper and the book were
        generated with the equations shown in the book.
        However, there is the option to run the domain with the paper equations
        by setting book_or_nips = 'nips'
    **REFERENCE:**
    .. seealso::
        R. Sutton: Generalization in Reinforcement Learning:
        Successful Examples Using Sparse Coarse Coding (NIPS 1996)
    .. seealso::
        R. Sutton and A. G. Barto:
        Reinforcement learning: An introduction.
        Cambridge: MIT press, 1998.
    .. warning::
        This version of the domain uses the Runge-Kutta method for integrating
        the system dynamics and is more realistic, but also considerably harder
        than the original version which employs Euler integration,
        see the AcrobotLegacy class.
    """

    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : 15
    }

    dt = .2

    LINK_LENGTH_1 = 1.  # [m]
    LINK_LENGTH_2 = 1.  # [m]
    LINK_MASS_1 = 1.  #: [kg] mass of link 1
    LINK_MASS_2 = 1.  #: [kg] mass of link 2
    LINK_COM_POS_1 = 0.5  #: [m] position of the center of mass of link 1
    LINK_COM_POS_2 = 0.5  #: [m] position of the center of mass of link 2
    LINK_MOI = 1.  #: moments of inertia for both links

    MAX_VEL_1 = 4 * np.pi
    MAX_VEL_2 = 9 * np.pi

    AVAIL_TORQUE = [-1., 0., +1]

    torque_noise_max = 0.

    #: use dynamics equations from the nips paper or the book
    book_or_nips = "book"
    action_arrow = None
    domain_fig = None
    actions_num = 3

    def __init__(self,exp1):
        self.viewer = None
        high = np.array([1.0, 1.0, 1.0, 1.0, self.MAX_VEL_1, self.MAX_VEL_2])
        low = -high
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        self.action_space = spaces.Discrete(3)
        self.state = None
        self.seed()
        self.params=self.read_data(exp1)
    def read_data(self,experiments):
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
        data = pd.read_excel('C:\simulation_upgrade\Data\RND_orginized_data.xlsx',sheet_name = experiments)
        offlineData=pd.read_excel('C:\simulation_upgrade\Data\RND_orginized_data.xlsx',sheet_name = offlineSheets)
        for exp in experiments:
            if data[exp].shape[1] > 10:
                data[exp].columns = data[exp].columns = ['Age','Temp','Airflow','Pressure','Agitation','Weight','pH','DO','Fa','Fs','CO2']
            else:
                data[exp].columns = ['Age', 'Temp', 'Airflow', 'Pressure', 'Agitation', 'Weight', 'pH', 'DO', 'Fa', 'Fs']

            if offlineData[exp+'_dex'].shape[1]>7:
                offlineData[exp+'_dex'].columns = ['Age','pH','PCV','Dextrose[percent]','Ammonia[percent]','Kanamycin','Tobramycin','Incyte time','Incyte']
            else:
                offlineData[exp+'_dex'].columns = ['Age','pH','PCV','Dextrose[percent]','Ammonia[percent]','Kanamycin','Tobramycin']

        # data[data.eq('%').any(1)] == '?'
        # is_in=data[data.isin(['%']).any(1)]
        return data,offlineData

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        ii=0.5
        bellMax=0.65
        lowDexVal=0.1
        highDexVal=4
        cum_fs=0
        state1=27.0620+ii*5*np.random(1)
        Agi=271
        T= 24*60-1;
        A=0.564+ii*0.2*np.random(1);
        X =5.38+ ii*0.8*np.random(1)
        DO= (6.47e-04)+ii*(1.4e-4)*np.random(1);
        if S>bellMax:
          DexState=(S-bellMax)/(highDexVal-bellMax)
        else:
           DexState=-(S-bellMax)/(lowDexVal-bellMax)
        self.state = [(DO-0.0005)/0.0005 ,0.02*100-2.2,  (A-1.05)/0.3, (T-3500)/4000,cum_fs,
         DexState, ((Agi-270)/(280-260))]
        return self.state

    def step(self, a,LoggedSignals,params):
        #s = self.state
        nSteps=8200;
        Constants= params.Constants, Settings=params.Settings, T70=0

        oilUptakeDelay=Constants.OIL_UPTAKE_DELAY/Settings.DT;
        V=80; Amax=1.21; Amin=0.79;

        state1= LoggedSignals.State,  T=LoggedSignals.T
        mask8=(T%480)>475#% for RT : exeute action of Ag/ammonia only each 8h
        if params.RT==0:
            mask8=1

        S=LoggedSignals.S
        if ( params.control=='valid') &  LoggedSignals.Start>0:
              LoggedSignals.Start=0;T=0;params.RT=0# temp
        if T>LoggedSignals.Start:
            A=LoggedSignals.A1[T]
            SOil=LoggedSignals.SOil
            DO=LoggedSignals.DO1[T], X=LoggedSignals.X1[T]
        else:
            # start
            A=LoggedSignals.A
            LoggedSignals.isDexLow=0
            LoggedSignals.soyBeanFeedMat=np.zeros(np.len(params.oil_feed_time),nSteps)
            SOil=0
            DO=LoggedSignals.DO
            X=LoggedSignals.X/Constants.prop_biomss
            #LoggedSignals.S=State(6);
        t_se=480;delA=params.delAmat(Action)
        if ~mask8:
            delA=0
        if params.RT==1:
            t_se=60
        if T==0:
            t_se=24*60-1
        LoggedSignals.F_s[1440] =0
        LoggedSignals.Ag[1440] =285
        for i in range(1,t_se):
            T=T+1 #;% if T>1; LoggedSignals.F_s[T] =LoggedSignals.F_s(T-1) ;end
            if T<24*60:
                LoggedSignals.fs_min[T]=0; LoggedSignals.fs_max[T]=0
                LoggedSignals.ag_min[T]=Settings.agL(1); LoggedSignals.ag_max[T]=Settings.agH(1);
                if T==1:
                    LoggedSignals.Ag[T] =285
                LoggedSignals.F_s[T] =0
                A=params.meanA24[T];
            elif T>24*60-1 & T<70*60:
                A=A+delA
                temA=A
                A=temA-delA if temA>Amax else A #   isApple = True if fruit == 'Apple' else False
                A=temA-delA if temA<Amin else A
                delA=0
                if  not( np.where(LoggedSignals.S<2 & LoggedSignals.S>0,1,'first')):#LoggedSignals.S[T]>5
                    if T==24*60:
                        LoggedSignals.F_s[T]=100; LoggedSignals.Ag[T] =271;
                        A=1.19
                    LoggedSignals.fs_min[T]=Settings.fsL(2); LoggedSignals.fs_max[T]=Settings.fsH(2)
                    LoggedSignals.ag_min[T]=Settings.agL(2); LoggedSignals.ag_max[T]=Settings.agH(2)
                else:
                    if T== np.where(LoggedSignals.S<2 & LoggedSignals.S>0,1,'first'):
                        LoggedSignals.T2=T
                        LoggedSignals.F_s[T] =140
                    LoggedSignals.fs_min[T]=params.Settings.fsL(3); LoggedSignals.fs_max[T]=params.Settings.fsH(3);
                    LoggedSignals.ag_min[T]=Settings.agL(2); LoggedSignals.ag_max[T]=Settings.agH(2);

            elif T>70*60-1:
                A=A+delA
                temA=A
                A=temA-delA if temA>Amax else A #   isApple = True if fruit == 'Apple' else False
                A=temA-delA if temA<Amin else A
                delA=0
                LoggedSignals.fs_min[T]=params.Settings.fsL(4); LoggedSignals.fs_max[T]=params.Settings.fsH(4);
                LoggedSignals.ag_min[T]=params.Settings.agL(3); LoggedSignals.ag_max[T]=params.Settings.agH(3);
                if T==70*60:
                    LoggedSignals.F_s[T] =110;            LoggedSignals.Ag[T] = LoggedSignals.Ag(T-1);


            delFs=params.fsM(Action);
            if i>1 or (LoggedSignals.F_s[T] +params.fsM(Action))>LoggedSignals.fs_max[T] or (LoggedSignals.F_s[T] +params.fsM(Action))<LoggedSignals.fs_min[T]:
                delFs=0;
            LoggedSignals.F_s[T+1] =LoggedSignals.F_s[T] +delFs#;% FS solution
            LoggedSignals.F_s[T] =Constants.DEXTROSE_FEEDING_CONCENTRATION*LoggedSignals.F_s[T];
            if T<30*60:# % late feeding
                  LoggedSignals.F_s[T]=0;
            delAg=params.agM(Action);
            if i==1 and isfield(LoggedSignals,'T2')and T>=LoggedSignals.T2 and T<=(LoggedSignals.T2+24*60):
               delAg =np.min(params.agM)

            if ~mask8 or i>1 or (LoggedSignals.Ag[T] +delAg)>LoggedSignals.ag_max[T] or (LoggedSignals.Ag[T] +delAg)<LoggedSignals.ag_min[T]:
                delAg=0;

            # for T>70 , Ag rate is 16h (for t<70 it equal to 8h)
                 #    T70=[T70; T];if T70(end)-T70(end-1)<1.2*t_se; delAg=0; end

            LoggedSignals.Ag[T+1] =LoggedSignals.Ag[T] +delAg;
            # LoggedSignals.Ag[T] =max(LoggedSignals.Ag[T] ,LoggedSignals.ag_max[T]);
            if T==LoggedSignals.Start+1: # %T==1
                #%LoggedSignals.Ag=zeros(1,9000);
                LoggedSignals.rnd1=1+floor(rand(1)*length(params.impData1));
                LoggedSignals.AgOld=280#;%params.impData1{1,LoggedSignals.rnd1}.agitation[T];
                LoggedSignals.S[T]=S; LoggedSignals.P1[T]=0;#%params.impData1{1,LoggedSignals.rnd1}.sConcen(1);



            if i==1:
                   # %  LoggedSignals.Ag[T]=LoggedSignals.AgOld+params.agM( Action);
                LoggedSignals.AgOld=LoggedSignals.Ag[T];

            rnd1=LoggedSignals.rnd1;
        #% type
            """
              if strcmp( params.control,'FsAg'):
                i=i
             elif  params.control=='valid':
                LoggedSignals.F_s[T] =params.impData1{1,rnd1}.F_s[T];% LoggedSignals.Ag[T]= LoggedSignals.AgOld; %LoggedSignals.Ag[T]=params.agM( Action);
             LoggedSignals.Ag[T]=params.impData1{1,rnd1}.agitation[T];
             A=params.impData1{1,rnd1}.A[T];
             Constants.airFlowVVM=params.impData1{1,rnd1}.airFlowVVM[T];
             if T==1% daynamics verfication
                LoggedSignals.S[T]=  params.impData1{1,1}.sConcen(1);
                X=0.15;
              % State(2)=params.impData1{1,1}.biomass(1);
               DO=0.007;%=params.impData1{1,1}.DOg2l(1);
                A=params.impData1{1,1}.A(1);
    
            """

        #% dymamics
            idx=1;
            relRows= T>2 and np.transpose(LoggedSignals.isDexLow)==0 and np.transpose(LoggedSignals.S[T])<5;
            if relRows:
                    LoggedSignals.isDexLow=1;
                    relFeedingIdx=T+np.arange(oilUptakeDelay,T)+oilUptakeDelay+params.oil_feed_time(idx);#%period of soybean oil feeding
     #               norm_feeding_bell=self.new_bell(relFeedingIdx,length(relFeedingIdx)/0.5,...
    #%                 0.1,t+oilUptakeDelay+(params.oil_feed_time(idx)/2));% normalized sigmoid value for feeding
                    c1=T+oilUptakeDelay+params.oil_feed_time(idx)/2
                    sig=params.oil_feed_time(idx)/4
                    norm_feeding_bell=exp(-((relFeedingIdx-c1)/sig)^2);
                    sum_norm_feeding_bell=np.sum(norm_feeding_bell,2);
                    feeding_bell=norm_feeding_bell*params.InitialCond.SoybeanOil/sum_norm_feeding_bell
                    LoggedSignals.soyBeanFeedMat[idx,relFeedingIdx(relFeedingIdx<=nSteps)]=feeding_bell(relFeedingIdx<=nSteps)*params.oil_utility(idx);
                    LoggedSignals.soyBeanFeedMat[idx,relFeedingIdx(relFeedingIdx<=nSteps)]=0;
                    relDepressionIdx=np.arange(T,T+oilUptakeDelay)
                    recoveringIdx=T+np.arange(oilUptakeDelay,nSteps)
    #%                 tau=T+oilUptakeDelay;
                    tau=c1;
                    LoggedSignals.C[idx,relDepressionIdx]=params.oil_utility(idx);
                    LoggedSignals.C[idx,recoveringIdx]=1-(1-params.oil_utility(idx))*exp(-((recoveringIdx-(T+oilUptakeDelay))
                                                                                           /(tau-(T+oilUptakeDelay))));

#%                 DO[idx,t-1]=min(DO(idx,t-1)+0.15*Constants.DO_MAX,Constants.DO_MAX);

 #                if 0#%strcmp( params.test1,'DO10')
#                                                    DO=0.2*Constants.DO_MAX;

            rel_mo=params.m_o[1]*LoggedSignals.C[T];
            rel_mx=params.m_x(1)*LoggedSignals.C[T];
            div=np.sqrt(params.K_I(1)*params.K_ps2(1))/(params.K_I(1)+(sqrt(params.K_I(1)*params.K_ps2(1)))+
                                                        (sqrt(params.K_I(1)*params.K_ps2(1)))^2/params.K_ps2(1));
            norm_mich_ment=(LoggedSignals.S[T]/(params.K_I(1)+LoggedSignals.S[T]+
                (LoggedSignals.S[T]^2./params.K_ps2(1))))/div;
            norm_mich_ment_oil=(SOil/(params.K_I(1)+SOil+...
                        (SOil^2./params.K_ps2(1))))/div;
            mu=LoggedSignals.C[T]*params.mu_x(1)*(LoggedSignals.S[T]/(params.K_x(1)+LoggedSignals.S[T]))*(DO/(params.K_ox(1)+DO))*(A/(params.K_xa(1)+A))
            alpha1=4e-4; ag05=290;
            mu_ag=1/(1+exp(alpha1*ag05*(LoggedSignals.Ag[T]-ag05)))
            mu_pp=LoggedSignals.C[T]*params.mu_p(1)*(A/(params.K_p.ravel()+A))*(DO/(DO+params.K_op(1)))*norm_mich_ment;
            mu_pp=mu_pp*mu_ag;
            mu_pp_oil=LoggedSignals.C[T]*params.mu_p(1)*(A/(params.K_p(1)+A))*(DO/(DO+params.K_op(1)))*norm_mich_ment_oil #;%Dextrose part of Michaelis Menten
            mu_pp_oil=mu_pp_oil*mu_ag;
            m_xOil=rel_mx*SOil/(LoggedSignals.S[T]+SOil);
            m_xDex=rel_mx*LoggedSignals.S[T]/(LoggedSignals.S[T]+SOil);
            minusSOil=Settings.DT*X*(mu_pp_oil/params.Y_ps(1)+m_xOil);
            plusSOil=SOil+ LoggedSignals.soyBeanFeedMat[T];
            isOverDraft=minusSOil>plusSOil;
            coefOil=np.ones(length(params.K_x(1)),1);
            coefOil[isOverDraft]=plusSOil(isOverDraft)/minusSOil(isOverDraft);
            SOil=SOil+ LoggedSignals.soyBeanFeedMat[T]-Settings.DT*X*coefOil*(mu_pp_oil/params.Y_ps(1)+m_xOil);



            plusDO=DO+Settings.DT*(params.K_la(1)*
                ((LoggedSignals.Ag[T]/Constants.AGI_REF_VAL)^2*(Constants.airFlowVVM/Constants.AF_REF_VAL))*(Constants.DO_MAX-DO));
            minusDO=Settings.DT*(X*
                ((mu/params.Y_xo(1))+rel_mo+(mu_pp+mu_pp_oil)/params.Y_po(1)));
            DOCoef=(plusDO-(Constants.DO_MAX/25))/minusDO;
            plusS=LoggedSignals.S[T]+Settings.DT*(LoggedSignals.F_s[T]/V);
            minusS=Settings.DT*(X*
                ((mu/params.Y_xs(1))+m_xDex+
                mu_pp/params.Y_ps(1)));
            SCoef=(plusS-0.1)/minusS;
            reductionCoef=min(DOCoef,SCoef);
            naturalS=LoggedSignals.S[T]+Settings.DT*(-X*((mu/params.Y_xs(1))+m_xDex+mu_pp/params.Y_ps(1))+LoggedSignals.F_s[T]/V);
            coefS=LoggedSignals.S[T]+Settings.DT*(-X*reductionCoef*
                ((mu/params.Y_xs(1))+m_xDex+
                mu_pp/params.Y_ps(1))+LoggedSignals.F_s[T]/V);

            naturalDO=DO+Settings.DT*(X*...
                (-(mu/params.Y_xo(1))-rel_mo-(mu_pp+mu_pp_oil)/params.Y_po(1))+params.K_la(1)*
                ((LoggedSignals.Ag[T]/Constants.AGI_REF_VAL)^2*
                (Constants.airFlowVVM/Constants.AF_REF_VAL))*(Constants.DO_MAX-DO));
            coefDO=DO+Settings.DT*(X*reductionCoef*
                (-(mu/params.Y_xo(1))-rel_mo-(mu_pp+mu_pp_oil)/params.Y_po(1))+params.K_la(1)*
                ((LoggedSignals.Ag[T]/Constants.AGI_REF_VAL)^2*
                (Constants.airFlowVVM/Constants.AF_REF_VAL))*(Constants.DO_MAX-DO));
            isCoef=coefS>naturalS or coefDO>naturalDO;
            LoggedSignals.S[T+1]=naturalS;
            #%LoggedSignals.S[T](find(isCoef))=coefS(find(isCoef));
            if isCoef:
                LoggedSignals.S[T+1]=coefS;

            DO=naturalDO;
            DO[find(isCoef)]=coefDO(find(isCoef));
            DO=min(Constants.DO_MAX,DO);
            Coef=ones(length(naturalDO),1);
            Coef[find(isCoef)]=reductionCoef(find(isCoef));

            #%A=1;%;params.impData1{1,rnd1}.A;
            X=X+Settings.DT*X*(Coef*mu-params.K_d(1)*A);
            LoggedSignals.P1[T+1]=np.max(0,LoggedSignals.P1[T]+Settings.DT*
                ((Coef*mu_pp+coefOil*mu_pp_oil)*X-params.K(1)*LoggedSignals.P1[T]));#%in [g/l]
            LoggedSignals.A1[T]=A; LoggedSignals.DO1[T]=DO; LoggedSignals.X1[T]=X;  LoggedSignals.A[T]=A;
        for t in range(0, t_end-(dt1+1), Settings['DT']):  #  start from t=1[minutes]
            currModelState[pref['Data variables']] =\
                validationData[pref['Data variables']].iloc[t]
            #  If featuresDist is a modeled parameter, we should not update it from data!!
            currModelState[pref['featuresDist']] = validationData[pref['featuresDist']].iloc[t]

            for var in pref['Variables']:
                delVariableName = var + '_del'
            if t % dt1:
                modeledVars[var].iloc[t+1] =modeledVars[var].iloc[t]

            else:
                varT=modeledVars[var].to_numpy()[t]
                bestParams = results[var]['bestParams']
                relTestData = currModelState[bestParams['features']]#.to_numpy()
                allRelTestDataInit = pd.concat([currModelState[bestParams['features']],currModelState[bestParams['featuresDist']]]
                                , axis=1).dropna()

                allRelTestData = allRelTestDataInit.T.drop_duplicates().T  # Remove duplications from "allRelTrainDataInit" dataframe

                testDistVar = currModelState[bestParams['featuresDist']].to_numpy()
                test_dist={}; train_dist={}; distSumSqr = 0
                for varForDist in bestParams['featuresDist']:
                    test_dist[varForDist] = np.repeat(testDistVar.T,
                                                      dataDict[var]['trainDistVar'].size, axis=0)
                    train_dist[varForDist] = np.repeat(dataDict[var]['trainDistVar'],
                                                       len(testDistVar),
                                                       axis=1)

                    distVarSqr = (test_dist[varForDist] - train_dist[varForDist]) ** 2
                    distSumSqr+=distVarSqr
                npoints=int(np.ceil(bestParams['frac']*dataDict[var]['trainDistVar'][:,0].size))
                dist=np.sqrt(distSumSqr)
                w=np.argsort(dist,axis=0)[:npoints]
                deltaVar, x ,bias_re,bias_vel= loess_nd_test_point_mat\
                            (pref,pd.DataFrame(relTestData).T,bestParams['features'],dataDict[var]['relTrainData'], dataDict[var]['trainDistVar'],
                             dataDict[var]['trainResults'],dist,w,scale_params[var],var,varT, frac=frac1)
                for jj in range(1,1*len(pref['Combinations']),1): # in case of bias go to the next config
                    ii=results[var]['sortedCombinations'][jj]
                    if bias_vel<0.7 or bias_vel>1.3:
                        break
                    if len(pref['Combinations'][ii]['features']) <2:
                        continue
                    allRelTrainDataInitB = pd.concat([modelingDataCombined[pref['Combinations'][ii]['features']],
                                         modelingDataCombined[pref['Combinations'][ii]['featuresDist']],
                                         modelingDataCombined[delVariableName]], axis=1).dropna()

                    allModelingData = allRelTrainDataInitB.T.drop_duplicates().T  # Remove duplications from "allRelTrainDataInit" dataframe

                    #dataDict[var]['relTrainData'] = allModelingData[pref['Combinations'][ii]['features']].to_numpy()  # relevant features for linear equation.
                    relTestData = currModelState[pref['Combinations'][ii]['features']]
                    deltaVar, x ,bias_re,bias_vel= loess_nd_test_point_mat\
                            (pref,pd.DataFrame(relTestData).T,pref['Combinations'][ii]['features'],
                             allModelingData[pref['Combinations'][ii]['features']].to_numpy(), dataDict[var]['trainDistVar'],
                             dataDict[var]['trainResults'],dist,w,scale_params[var],var,varT, frac=frac1)
                    demo=2
                modeledVars[var].iloc[t+1] = \
                        modeledVars[var].iloc[t] + dt1*deltaVar/60
                modeledVars[var+'_biasVel'].iloc[t+1:t+1+dt1]=list(bias_re*np.ones([1,dt1]).T)
            currModelState[var] = modeledVars[var].iloc[t+1]

            if T>(nSteps-480):
                break

        # ODEINT IS TOO SLOW!
        # ns_continuous = integrate.odeint(self._dsdt, self.s_continuous, [0, self.dt])
        # self.s_continuous = ns_continuous[-1] # We only care about the state
        # at the ''final timestep'', self.dt

        sh=0;
        bellMax=0.65;
        lowDexVal=0.1;
        highDexVal=4;
        if LoggedSignals.S(T)>bellMax:
            DexState=(LoggedSignals.S(T)-bellMax)/(highDexVal-bellMax);
        else:
            DexState=-(LoggedSignals.S(T)-bellMax)/(lowDexVal-bellMax);

        LoggedSignals.T=T;
        LoggedSignals.State[1]=(mean( LoggedSignals.DO1(np.arange(T+1-120-sh,T-sh)))-0.0005)/0.0005;
        #% LoggedSignals.State(2)=Constants.prop_biomss*(mean( LoggedSignals.X1(T+1-120:T))-4)/3;%(LoggedSignals.DO1(T-sh)-LoggedSignals.DO1(T-240-sh))/1e-7;
        LoggedSignals.State[2]=0.02*LoggedSignals.F_s(T+1)-2.2;
        LoggedSignals.State[3]=(LoggedSignals.A(T-sh)-1.05)/0.3;
        LoggedSignals.State[4]=((T-sh)-3500)/4000;
        LoggedSignals.State[5]=(LoggedSignals.S(T-sh)-LoggedSignals.S(T-60-sh))/0.3;#%(T/60)*mean( LoggedSignals.F_s(1:T))/V;

        LoggedSignals.State[6]=DexState;
        LoggedSignals.State[7]=((LoggedSignals.Ag(T-sh)-270)/(280-260));
        LoggedSignals.SOil=SOil;
        #% state shrvul
        NextObs = LoggedSignals.State;
#% Reward upon time
        if T>nSteps-480:
            IsDone =1;#%NextObs(3)<-100.05; %abs(X) > EnvConstants.XThreshold || abs(Theta) > EnvConstants.ThetaThresholdRadians;
            Reward=0;
        else:
            IsDone =1;#%NextObs(3)<-100.05; %abs(X) > EnvConstants.XThreshold || abs(Theta) > EnvConstants.ThetaThresholdRadians;
            Reward=0;
        self.state = ns
        terminal = self._terminal()
        reward = -1. if not terminal else 0.
        return (NextObs,reward,IsDone,LoggedSignals, {})
        # return [NextObs,Reward,IsDone,LoggedSignals]
    def _get_ob(self):
        state1= self.state
        return np.array([cos(s[0]), np.sin(s[0]), cos(s[1]), sin(s[1]), s[2], s[3]])

    def _terminal(self):
        state1= self.state
        return bool(-np.cos(s[0]) - np.cos(s[1] + s[0]) > 1.)

    def _dsdt(self, s_augmented, t):
        m1 = self.LINK_MASS_1
        m2 = self.LINK_MASS_2
        l1 = self.LINK_LENGTH_1
        lc1 = self.LINK_COM_POS_1
        lc2 = self.LINK_COM_POS_2
        I1 = self.LINK_MOI
        I2 = self.LINK_MOI
        g = 9.8
        a = s_augmented[-1]
        state1= s_augmented[:-1]
        theta1 = s[0]
        theta2 = s[1]
        dtheta1 = s[2]
        dtheta2 = s[3]
        d1 = m1 * lc1 ** 2 + m2 * \
            (l1 ** 2 + lc2 ** 2 + 2 * l1 * lc2 * np.cos(theta2)) + I1 + I2
        d2 = m2 * (lc2 ** 2 + l1 * lc2 * np.cos(theta2)) + I2
        phi2 = m2 * lc2 * g * np.cos(theta1 + theta2 - np.pi / 2.)
        phi1 = - m2 * l1 * lc2 * dtheta2 ** 2 * np.sin(theta2) \
               - 2 * m2 * l1 * lc2 * dtheta2 * dtheta1 * np.sin(theta2)  \
            + (m1 * lc1 + m2 * l1) * g * np.cos(theta1 - np.pi / 2) + phi2
        if self.book_or_nips == "nips":
            # the following line is consistent with the description in the
            # paper
            ddtheta2 = (a + d2 / d1 * phi1 - phi2) / \
                (m2 * lc2 ** 2 + I2 - d2 ** 2 / d1)
        else:
            # the following line is consistent with the java implementation and the
            # book
            ddtheta2 = (a + d2 / d1 * phi1 - m2 * l1 * lc2 * dtheta1 ** 2 * np.sin(theta2) - phi2) \
                / (m2 * lc2 ** 2 + I2 - d2 ** 2 / d1)
        ddtheta1 = -(d2 * ddtheta2 + phi1) / d1
        return (dtheta1, dtheta2, ddtheta1, ddtheta2, 0.)

    def render(self, mode='human'):
        from gym.envs.classic_control import rendering

        state1= self.state

        if self.viewer is None:
            self.viewer = rendering.Viewer(500,500)
            bound = self.LINK_LENGTH_1 + self.LINK_LENGTH_2 + 0.2  # 2.2 for default
            self.viewer.set_bounds(-bound,bound,-bound,bound)

        if state1 is None: return None

        p1 = [-self.LINK_LENGTH_1 *
              np.cos(s[0]), self.LINK_LENGTH_1 * np.sin(s[0])]

        p2 = [p1[0] - self.LINK_LENGTH_2 * np.cos(s[0] + s[1]),
              p1[1] + self.LINK_LENGTH_2 * np.sin(s[0] + s[1])]

        xys = np.array([[0,0], p1, p2])[:,::-1]
        thetas = [s[0]-np.pi/2, s[0]+s[1]-np.pi/2]
        link_lengths = [self.LINK_LENGTH_1, self.LINK_LENGTH_2]

        self.viewer.draw_line((-2.2, 1), (2.2, 1))
        for ((x,y),th,llen) in zip(xys, thetas, link_lengths):
            l,r,t,b = 0, llen, .1, -.1
            jtransform = rendering.Transform(rotation=th, translation=(x,y))
            link = self.viewer.draw_polygon([(l,b), (l,t), (r,t), (r,b)])
            link.add_attr(jtransform)
            link.set_color(0,.8, .8)
            circ = self.viewer.draw_circle(.1)
            circ.set_color(.8, .8, 0)
            circ.add_attr(jtransform)

        return self.viewer.render(return_rgb_array = mode=='rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

def wrap(x, m, M):
    """
    :param x: a scalar
    :param m: minimum possible value in range
    :param M: maximum possible value in range
    Wraps ``x`` so m <= x <= M; but unlike ``bound()`` which
    truncates, ``wrap()`` wraps x around the coordinate system defined by m,M.\n
    For example, m = -180, M = 180 (degrees), x = 360 --> returns 0.
    """
    diff = M - m
    while x > M:
        x = x - diff
    while x < m:
        x = x + diff
    return x

def bound(x, m, M=None):
    """
    :param x: scalar
    Either have m as scalar, so bound(x,m,M) which returns m <= x <= M *OR*
    have m as length 2 vector, bound(x,m, <IGNORED>) returns m[0] <= x <= m[1].
    """
    if M is None:
        M = m[1]
        m = m[0]
    # bound x between min (m) and Max (M)
    return min(max(x, m), M)


def rk4(derivs, y0, t, *args, **kwargs):
    """
    Integrate 1D or ND system of ODEs using 4-th order Runge-Kutta.
    This is a toy implementation which may be useful if you find
    yourself stranded on a system w/o scipy.  Otherwise use
    :func:`scipy.integrate`.
    *y0*
        initial state vector
    *t*
        sample times
    *derivs*
        returns the derivative of the system and has the
        signature ``dy = derivs(yi, ti)``
    *args*
        additional arguments passed to the derivative function
    *kwargs*
        additional keyword arguments passed to the derivative function
    Example 1 ::
        ## 2D system
        def derivs6(x,t):
            d1 =  x[0] + 2*x[1]
            d2 =  -3*x[0] + 4*x[1]
            return (d1, d2)
        dt = 0.0005
        t = arange(0.0, 2.0, dt)
        y0 = (1,2)
        yout = rk4(derivs6, y0, t)
    Example 2::
        ## 1D system
        alpha = 2
        def derivs(x,t):
            return -alpha*x + exp(-t)
        y0 = 1
        yout = rk4(derivs, y0, t)
    If you have access to scipy, you should probably be using the
    scipy.integrate tools rather than this function.
    """

    try:
        Ny = len(y0)
    except TypeError:
        yout = np.zeros((len[T],), np.float_)
    else:
        yout = np.zeros((len[T], Ny), np.float_)

    yout[0] = y0


    for i in np.arange(len[T] - 1):

        thist = t[i]
        dt = t[i + 1] - thist
        dt2 = dt / 2.0
        y0 = yout[i]

        k1 = np.asarray(derivs(y0, thist, *args, **kwargs))
        k2 = np.asarray(derivs(y0 + dt2 * k1, thist + dt2, *args, **kwargs))
        k3 = np.asarray(derivs(y0 + dt2 * k2, thist + dt2, *args, **kwargs))
        k4 = np.asarray(derivs(y0 + dt * k3, thist + dt, *args, **kwargs))
        yout[i + 1] = y0 + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
    return yout
""""
© 2019 GitHub, Inc.
Terms
Privacy
Security
Status
Help
Contact GitHub
Pricing
API
Training
Blog
About
"""
