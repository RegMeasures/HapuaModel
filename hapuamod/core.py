# -*- coding: utf-8 -*-
""" Core hapua model

"""

# import standard packages
import logging
import pandas as pd

# import local modules
from hapuamod import loadmod
from hapuamod import riv
from hapuamod import coast

def run(ModelConfigFile):
    """ Main hapuamod run script
    
    Parameters:
        ModelConfigFile (string): File name (including path if required) of 
            the main model config file.
    """
    
    #%% Set up logging
    RootLogger = logging.getLogger()
    RootLogger.setLevel(logging.DEBUG)
    
    ConsoleHandler = logging.StreamHandler()
    ConsoleHandler.setLevel(logging.DEBUG)
    RootLogger.addHandler(ConsoleHandler)
    
    #%% Load the model
    Config = loadmod.readConfig(ModelConfigFile)
    (FlowTs, WaveTs, SeaLevelTs, Origin, BaseShoreNormDir, ShoreX, ShoreY, 
     LagoonY, LagoonElev, RiverElev, OutletX, OutletY, OutletElev, OutletWidth, 
     Dx, TimePars, PhysicalPars, OutputOpts) = loadmod.loadModel(Config)
    
    #%% Generate initial conditions for river model
    RivFlow = interpolate_at(FlowTs, TimePars['StartTime'])[0]
    SeaLevel = interpolate_at(SeaLevelTs, TimePars['StartTime'])[0]
    
    (ChanDx, ChanElev, ChanWidth, ChanArea) = \
    riv.assembleChannel(RiverElev, ShoreX, LagoonY, LagoonElev, 
                        OutletX, OutletY, OutletElev, OutletWidth, 
                        PhysicalPars['RiverWidth'], Dx)
    
    (ChanDep, ChanVel) = riv.solveSteady(ChanDx, ChanElev, ChanWidth, 
                                         PhysicalPars['Roughness'], 
                                         RivFlow, SeaLevel)
    
    #%% Main timestepping loop
    HydTime = TimePars['StartTime']
    MorTime = TimePars['StartTime']
    LogTime = TimePars['StartTime']
    while HydTime <= TimePars['EndTime']:
        
        
        
        
        
        #%% Run the shoreline model

        
        # Run the river model
        RivFlow = interpolate_at(FlowTs, HydTime)[0]
        SeaLevel = interpolate_at(SeaLevelTs, HydTime)[0]
        (ChanDep, ChanVel) = riv.solveFullPreissmann(ChanElev, ChanWidth, 
                                                     ChanDep, ChanVel, ChanDx, 
                                                     TimePars['HydDt'], 
                                                     PhysicalPars['Roughness'], 
                                                     RivFlow, SeaLevel, 
                                                     0.6, 0.001, 20, 9.81)
        
        # Morphology
        if HydTime >= MorTime:
            
            # Run shoreline model
            WavesAtT = interpolate_at(WaveTs, MorTime)
            EDir_h = WavesAtT.EDir_h[0]
            WavePower = WavesAtT.WavePower[0]
            WavePeriod = WavesAtT.WavePeriod[0]
            Wlen_h = WavesAtT.Wlen_h[0]
            
            LST = coast.longShoreTransport(ShoreY, Dx, WavePower, WavePeriod, 
                                           Wlen_h, EDir_h, PhysicalPars)
            ShoreY += coast.shoreChange(LST, Dx, TimePars['MorDt'], PhysicalPars)
            MorTime += TimePars['MorDt']
            
        # updates to user
        if HydTime >= LogTime:
            logging.info('Time = %s', HydTime)
            LogTime += OutputOpts['LogInt']
        
        # increment time
        HydTime += TimePars['HydDt']

def interpolate_at(Df, Time):
    """ Linearly interpolate dataframe for specific time
    Interpolate wave, river flow and sea level input data for specific 
    model time.
    
    New_df = interpolate_at(Df, Time)
    
    Parameters:
        Df(DataFrame): dataframe or series with float or datetime index
        Time(datetime): new index(s) to output data at
    
    Returns:
        New_df: new data frame containing linearly interpolated data at the 
                specified time.
    """
    New_idxs = pd.DatetimeIndex([Time])
    #df = df.drop_duplicates().dropna()
    Df = Df.reindex(Df.index.append(New_idxs).unique())
    Df = Df.sort_index()
    Df = Df.interpolate(method='time')
    return Df.loc[New_idxs]

