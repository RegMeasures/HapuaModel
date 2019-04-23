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
from hapuamod import visualise

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
     TimePars, PhysicalPars, NumericalPars, OutputOpts) = loadmod.loadModel(Config)
    
    #%% Generate initial conditions for river model
    RivFlow = interpolate_at(FlowTs, pd.DatetimeIndex([TimePars['StartTime']])).values
    SeaLevel = interpolate_at(SeaLevelTs, pd.DatetimeIndex([TimePars['StartTime']])).values
    
    (ChanDx, ChanElev, ChanWidth, ChanArea) = \
    riv.assembleChannel(RiverElev, ShoreX, LagoonY, LagoonElev, 
                        OutletX, OutletY, OutletElev, OutletWidth, 
                        PhysicalPars['RiverWidth'], NumericalPars['Dx'])
    
    (ChanDep, ChanVel) = riv.solveSteady(ChanDx, ChanElev, ChanWidth, 
                                         PhysicalPars['Roughness'], 
                                         RivFlow, SeaLevel)
    
    #%% Prepare plotting
    LivePlot = OutputOpts['PlotInt'] > pd.Timedelta(0)
    if LivePlot:
        LsLines = visualise.longSection(ChanDx, ChanElev, ChanDep, ChanVel)
    
    #%% Main timestepping loop
    MorTime = TimePars['StartTime']
    LogTime = TimePars['StartTime']
    PlotTime = TimePars['StartTime']
    while MorTime <= TimePars['EndTime']:
                
        # Run the river model for all the timesteps upto the next morphology step
        
        # HydTimes = np.arange(MorTime, MorTime+TimePars['MorDt'], TimePars['HydDt'])
        HydTimes = pd.date_range(MorTime, MorTime+TimePars['MorDt'], 
                                 freq=TimePars['HydDt'], closed='right')
        
        RivFlow = interpolate_at(FlowTs, HydTimes).values
        SeaLevel = interpolate_at(SeaLevelTs, HydTimes).values
        (ChanDep, ChanVel) = riv.solveFullPreissmann(ChanElev, ChanWidth, 
                                                     ChanDep, ChanVel, ChanDx, 
                                                     TimePars['HydDt'], 
                                                     PhysicalPars['Roughness'], 
                                                     RivFlow, SeaLevel, 
                                                     NumericalPars)
        
        # Run shoreline model
        WavesAtT = interpolate_at(WaveTs, pd.DatetimeIndex([MorTime]))
        EDir_h = WavesAtT.EDir_h[0]
        WavePower = WavesAtT.WavePower[0]
        WavePeriod = WavesAtT.WavePeriod[0]
        Wlen_h = WavesAtT.Wlen_h[0]
        
        LST = coast.longShoreTransport(ShoreY, NumericalPars['Dx'], WavePower, 
                                       WavePeriod, Wlen_h, EDir_h, 
                                       PhysicalPars)
        ShoreY += coast.shoreChange(LST, NumericalPars['Dx'], 
                                    TimePars['MorDt'], PhysicalPars)
            
        # updates to user
        if MorTime >= LogTime:
            logging.info('Time = %s', MorTime)
            LogTime += OutputOpts['LogInt']
            
        # plotting
        if LivePlot:
            if MorTime >= PlotTime:
                visualise.updateLongSection(LsLines, ChanDx, ChanElev, 
                                            ChanDep, ChanVel)
                PlotTime += OutputOpts['PlotInt']
        
        # increment time
        MorTime += TimePars['MorDt']

def interpolate_at(Df, New_idxs):
    """ Linearly interpolate dataframe for specified index values
    interpolate_at is used to Linearly interpolate wave, river flow and sea 
    level input data for specific model times.
    
    New_df = interpolate_at(Df, Time)
    
    Parameters:
        Df(DataFrame): dataframe or series with float or datetime index
        Time(datetime): new index(s) to output data at
    
    Returns:
        New_df: new data frame containing linearly interpolated data at the 
                specified time.
    """
    Df = Df.reindex(Df.index.append(New_idxs).unique())
    Df = Df.sort_index()
    Df = Df.interpolate(method='time')
    return Df.loc[New_idxs]
