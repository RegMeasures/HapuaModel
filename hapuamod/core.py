# -*- coding: utf-8 -*-
""" Core hapua model

"""

# import standard packages
import logging
import pandas as pd
import numpy as np

# import local modules
from hapuamod import loadmod
from hapuamod import riv
from hapuamod import coast
from hapuamod import mor
from hapuamod import visualise

def run(ModelConfigFile):
    """ Main hapuamod run script
    
    Parameters:
        ModelConfigFile (string): File name (including path if required) of 
            the main model config file.
    """
    
    #%% Set up logging
    RootLogger = logging.getLogger()
    RootLogger.setLevel(logging.INFO)
    
    ConsoleHandler = logging.StreamHandler()
    ConsoleHandler.setLevel(logging.INFO)
    RootLogger.addHandler(ConsoleHandler)
    
    #%% Load the model
    Config = loadmod.readConfig(ModelConfigFile)
    (FlowTs, WaveTs, SeaLevelTs, Origin, ShoreNormDir, 
     ShoreX, ShoreY, LagoonElev, BarrierElev, OutletElev, 
     RiverElev, OutletEndX, OutletEndWidth, OutletEndElev,
     TimePars, PhysicalPars, NumericalPars, OutputOpts) = loadmod.loadModel(Config)
    
    #%% Generate initial conditions for river model
    RivFlow = interpolate_at(FlowTs, pd.DatetimeIndex([TimePars['StartTime']])).values
    SeaLevel = interpolate_at(SeaLevelTs, pd.DatetimeIndex([TimePars['StartTime']])).values
    
    (ChanDx, ChanElev, ChanWidth, LagArea, OnlineLagoon, OutletChanIx) = \
        mor.assembleChannel(ShoreX, ShoreY, LagoonElev, OutletElev,
                            OutletEndX, OutletEndWidth, OutletEndElev, 
                            RiverElev, PhysicalPars['RiverWidth'], 
                            NumericalPars['Dx'])
    
    (ChanDep, ChanVel) = riv.solveSteady(ChanDx, ChanElev, ChanWidth, 
                                         PhysicalPars['Roughness'], 
                                         NumericalPars['Beta'], 
                                         RivFlow, SeaLevel)
    
    #%% Set up variables to hold output timeseries
    OutputTs = pd.DataFrame(list(zip([RivFlow],[RivFlow],[SeaLevel])),
                                     columns=['Qin','Qout','SeaLevel'],
                                     index=pd.DatetimeIndex([TimePars['StartTime']]))
    
    #%% Prepare plotting
    LivePlot = OutputOpts['PlotInt'] > pd.Timedelta(0)
    if LivePlot:
        LsLines = visualise.longSection(ChanDx, ChanElev, ChanWidth, ChanDep, ChanVel, 
                                        np.zeros(ChanElev.size))
        BdyFig = visualise.BdyCndFig(OutputTs)
        ModelFig = visualise.modelView(ShoreX, ShoreY, OutletEndX, OutletChanIx)
    
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
        
        riv.solveFullPreissmann(ChanElev, ChanWidth, LagArea, 
                                ChanDep, ChanVel, ChanDx, 
                                TimePars['HydDt'], PhysicalPars['Roughness'], 
                                RivFlow, SeaLevel, NumericalPars)
        
        # Calculate bedload transport
        Bedload = riv.calcBedload(ChanElev, ChanWidth, ChanDep, ChanVel, 
                                  ChanDx, PhysicalPars)
        assert Bedload.size == ChanElev.size
        
        # Run shoreline model
        WavesAtT = interpolate_at(WaveTs, pd.DatetimeIndex([MorTime]))
        EDir_h = WavesAtT.EDir_h[0]
        WavePower = WavesAtT.WavePower[0]
        WavePeriod = WavesAtT.WavePeriod[0]
        Wlen_h = WavesAtT.Wlen_h[0]
        Hs_offshore = WavesAtT.Hsig_Offshore[0]
        
        LST = coast.longShoreTransport(ShoreY, NumericalPars['Dx'], WavePower, 
                                       WavePeriod, Wlen_h, EDir_h, 
                                       PhysicalPars)
        
        # Calculate runup & overtopping potential
        Runup = coast.runup(WavePeriod, Hs_offshore, PhysicalPars['BeachSlope'])
        
        # Update morphology
        mor.updateMorphology(ShoreX, ShoreY, LagoonElev, OutletElev, BarrierElev,
                             OutletEndX, OutletEndWidth, OutletEndElev, 
                             RiverElev, PhysicalPars['RiverWidth'], OnlineLagoon, 
                             OutletChanIx, ChanWidth, ChanDep, ChanDx,
                             LST, Bedload, NumericalPars['Dx'], TimePars['MorDt'], 
                             PhysicalPars)
        
        (ChanDx, ChanElev, ChanWidth, LagArea, OnlineLagoon, OutletChanIx) = \
            mor.assembleChannel(ShoreX, ShoreY, LagoonElev, OutletElev,
                                OutletEndX, OutletEndWidth, OutletEndElev, 
                                RiverElev, PhysicalPars['RiverWidth'], 
                                NumericalPars['Dx'])
        
        # Store outputs
        OutputTs = OutputTs.append(pd.DataFrame(list(zip([RivFlow[-1]],
                                                         [ChanDep[-1]*ChanVel[-1]*ChanWidth[-1]],
                                                         [SeaLevel[-1]])),
                                                columns=['Qin','Qout','SeaLevel'],
                                                index=pd.DatetimeIndex([MorTime])))
        
        # updates to user
        if MorTime >= LogTime:
            logging.info('Time = %s', MorTime)
            LogTime += OutputOpts['LogInt']
            
        # plotting
        if LivePlot:
            if MorTime >= PlotTime:
                visualise.updateLongSection(LsLines, ChanDx, ChanElev, 
                                            ChanWidth, ChanDep, ChanVel, 
                                            Bedload)
                visualise.updateBdyCndFig(BdyFig, OutputTs)
                visualise.updateModelView(ModelFig, ShoreX, ShoreY, OutletEndX, OutletChanIx)
                PlotTime += OutputOpts['PlotInt']
        
        # increment time
        MorTime += TimePars['MorDt']
        
    return(OutputTs)
    
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
