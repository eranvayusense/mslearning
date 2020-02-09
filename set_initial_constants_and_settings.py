def set_initial_constants_and_settings(MeasStruct):

    #%% Set settings
    Settings={'nConfig':1,'nConfigPerIter':1,'DT':1/60,# length of step [hours]
    'AmmoniaStableTime':120,# time for ammonia concentration to stabilize on a new value
    'SelectedCase':'26'}

    
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
    #%params.S_0=64.5;%initial dextrose concentration[g/l]
    'DO_0':MeasStruct['DO'][1],#;% initial dissolved oxygen concentration-TO BE TAKEN FROM MEASUREMENTS!!
    # % if isempty(MeasStruct.AConcen(end))
    # %     InitialCond.A_0=savedData.A;% initial amonia concentration-TO BE TAKEN FROM MEASUREMENTS!!
    # % else
    # %     InitialCond.A_0=MeasStruct.AConcen(1);
    # % end
    'A_0':MeasStruct['AConcen'][1],
    'P1_0':0,#;%initial tobramycin weight
    'S_0':60,
    'CO2_0':0,#;%initial CO2 concentration [%]
    'SoybeanOil':30,#;%concentration [%] of soybean oil
    'nSteps':MeasStruct['Time'][-1]}
    return InitialCond,Settings,Constants
