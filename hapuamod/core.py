# -*- coding: utf-8 -*-
""" Core hapua model

"""

# import standard packages
import logging
import pandas as pd
import numpy as np

# import local modules
from . import loadmod
from . import riv
from . import coast
from . import mor
from . import visualise
from . import out

def main(ModelConfigFile, Overwrite=False):
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
    (ModelName, FlowTs, WaveTs, SeaLevelTs, Origin, ShoreNormDir, ShoreX, 
     ShoreY, ShoreZ, RiverElev, OutletEndX, OutletEndWidth, OutletEndElev,
     TimePars, PhysicalPars, NumericalPars, OutputOpts) = loadmod.loadModel(ModelConfigFile)
    
    #%% Generate initial conditions for river model
    RivFlow = interpolate_at(FlowTs, pd.DatetimeIndex([TimePars['StartTime']])).values
    SeaLevel = interpolate_at(SeaLevelTs, pd.DatetimeIndex([TimePars['StartTime']])).values
    
    (ChanDx, ChanElev, ChanWidth, LagArea, ChanDep, ChanVel, 
     OnlineLagoon, OutletChanIx, ChanFlag, Closed) = \
        riv.assembleChannel(ShoreX, ShoreY, ShoreZ,
                            OutletEndX, OutletEndWidth, OutletEndElev, 
                            RiverElev, PhysicalPars['RiverWidth'], 
                            np.zeros(RiverElev.size), np.zeros(RiverElev.size),
                            np.zeros(ShoreX.size), np.zeros(ShoreX.size), 
                            np.zeros(ShoreX.size), np.zeros(ShoreX.size),
                            np.zeros(2), np.zeros(2), NumericalPars['Dx'],
                            PhysicalPars)
    
    (ChanDep, ChanVel) = riv.solveSteady(ChanDx, ChanElev, ChanWidth, 
                                         PhysicalPars['Roughness'], 
                                         RivFlow, SeaLevel, NumericalPars)
    
    (LagoonWL, LagoonVel, OutletDep, OutletVel, OutletEndDep, OutletEndVel) = \
        riv.storeHydraulics(ChanDep, ChanVel, OnlineLagoon, OutletChanIx, 
                            ChanFlag, ShoreZ[:,3], Closed)
    
    #%% Create output file and write initial conditions
    out.newOutFile(OutputOpts['OutFile'], ModelName, TimePars['StartTime'], 
                   ShoreX, NumericalPars['Dx'],  RiverElev, 
                   Origin, ShoreNormDir, PhysicalPars['RiverWidth'],
                   Overwrite)
    out.writeCurrent(OutputOpts['OutFile'], TimePars['StartTime'],
                     ShoreY, ShoreZ, LagoonWL, LagoonVel, np.zeros(ShoreX.size), 
                     OutletDep, OutletVel, np.zeros(ShoreX.size), 
                     np.zeros(ShoreX.size-1), np.zeros(ShoreX.size), np.zeros(ShoreX.size), 
                     RiverElev, ChanDep[ChanFlag==0], ChanVel[ChanFlag==0], np.zeros(RiverElev.size),
                     OutletEndX, OutletEndElev, OutletEndWidth, 
                     OutletEndDep, OutletEndVel, np.zeros(2), Closed)
    
    #%% Set up variables to hold output timeseries
    OutputTs = pd.DataFrame(list(zip([RivFlow],[RivFlow],[SeaLevel])),
                                     columns=['Qin','Qout','SeaLevel'],
                                     index=pd.DatetimeIndex([TimePars['StartTime']]))
    
    #%% Prepare plotting
    LivePlot = OutputOpts['PlotInt'] > pd.Timedelta(0)
    if LivePlot:
        LsLines = visualise.longSection(ChanDx, ChanElev, ChanWidth, ChanDep, ChanVel, 
                                        np.zeros(ChanElev.size-1))
        BdyFig = visualise.BdyCndFig(OutputTs)
        ModelFig = visualise.modelView(ShoreX, ShoreY, OutletEndX, OutletEndWidth,
                                       OutletChanIx, PhysicalPars['RiverWidth'],
                                       ShoreZ=ShoreZ, 
                                       WavePower=0, EDir_h=0, 
                                       LST=np.zeros(ShoreX.size-1),
                                       CST=np.zeros(ShoreX.size))
    
    #%% Main timestepping loop
    MorTime = TimePars['StartTime']
    OutTime = TimePars['StartTime'] + OutputOpts['OutInt']
    LogTime = TimePars['StartTime'] + OutputOpts['LogInt']
    PlotTime = TimePars['StartTime'] + OutputOpts['PlotInt']
    while MorTime <= TimePars['EndTime']:
        
        # Re-assemble the combined river channel incase it has evolved
        (ChanDx, ChanElev, ChanWidth, LagArea, ChanDep, ChanVel, 
         OnlineLagoon, OutletChanIx, ChanFlag, Closed) = \
            riv.assembleChannel(ShoreX, ShoreY, ShoreZ, 
                                OutletEndX, OutletEndWidth, OutletEndElev, 
                                RiverElev, PhysicalPars['RiverWidth'], 
                                ChanDep[ChanFlag==0], ChanVel[ChanFlag==0], 
                                LagoonWL, LagoonVel, OutletDep, OutletVel,
                                OutletEndDep, OutletEndVel, 
                                NumericalPars['Dx'], PhysicalPars)
                
        # Run the river model for all the timesteps upto the next morphology step
        HydTimes = pd.date_range(MorTime, MorTime+TimePars['MorDt'], 
                                 freq=TimePars['HydDt'], closed='right')
        
        RivFlow = interpolate_at(FlowTs, HydTimes).values
        SeaLevel = interpolate_at(SeaLevelTs, HydTimes).values
        
        # try:
        riv.solveFullPreissmann(ChanElev, ChanWidth, LagArea, Closed,
                                ChanDep, ChanVel, ChanDx, 
                                TimePars['HydDt'], PhysicalPars['Roughness'], 
                                RivFlow, SeaLevel, NumericalPars)
#        except Exception as ErrMsg:
#            logging.warning(ErrMsg)
#            logging.warning('Unsteady hydraulics failed at %s. Falling back to quasi-steady for this timestep.' % 
#                            HydTimes[-1].strftime("%m/%d/%Y, %H:%M:%S"))
#            (ChanDep, ChanVel) = riv.solveSteady(ChanDx, ChanElev, ChanWidth, 
#                                                 PhysicalPars['Roughness'], 
#                                                 RivFlow[-1], SeaLevel[-1], 
#                                                 NumericalPars)
        
        # Store the hydraulics ready to use for initial conditions in the next loop
        (LagoonWL, LagoonVel, OutletDep, OutletVel, OutletEndDep, OutletEndVel) = \
            riv.storeHydraulics(ChanDep, ChanVel, OnlineLagoon, OutletChanIx, 
                                ChanFlag, ShoreZ[:,3], Closed)
        
        # Calculate bedload transport
        Bedload = riv.calcBedload(ChanElev, ChanWidth, ChanDep, ChanVel, 
                                  ChanDx, PhysicalPars, NumericalPars['Psi'])
        assert Bedload.size == ChanElev.size-1
        
        # Calculate longshore transport
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
        (CST_tot, OverwashProp) = coast.overtopping(Runup, SeaLevel[-1], ShoreY, 
                                                    ShoreZ, PhysicalPars)
        
        # increment time
        MorTime += TimePars['MorDt']
        
        # If it is an output or plotting timestep then put bedload into model co-ordinates
        if MorTime >= OutTime or (LivePlot and MorTime >= PlotTime):
            (LagoonBedload, OutletBedload, OutletEndBedload) = \
                riv.storeBedload(Bedload, ShoreX.size, OnlineLagoon, OutletChanIx, 
                                 ChanFlag, Closed)
                
        # Update morphology
        mor.updateMorphology(ShoreX, ShoreY, ShoreZ,
                             OutletEndX, OutletEndWidth, OutletEndElev, 
                             RiverElev, PhysicalPars['RiverWidth'], OnlineLagoon, 
                             OutletChanIx, LagoonWL, OutletDep,
                             ChanWidth, ChanDep, ChanDx, ChanFlag, 
                             Closed, LST, Bedload, CST_tot, OverwashProp,
                             NumericalPars['Dx'], TimePars['MorDt'], 
                             PhysicalPars)
        
        
        
        # Save outputs
        if MorTime >= OutTime:
            out.writeCurrent(OutputOpts['OutFile'], MorTime, 
                             ShoreY, ShoreZ, LagoonWL, LagoonVel, LagoonBedload,
                             OutletDep, OutletVel, OutletBedload,
                             LST, CST_tot, OverwashProp,
                             RiverElev, ChanDep[ChanFlag==0], ChanVel[ChanFlag==0], Bedload[ChanFlag[:-1]==0],
                             OutletEndX, OutletEndElev, OutletEndWidth, 
                             OutletEndDep, OutletEndVel, OutletEndBedload, Closed)
            OutputTs = OutputTs.append(pd.DataFrame(list(zip([RivFlow[-1]],
                                                             [ChanDep[-1]*ChanVel[-1]*ChanWidth[-1]],
                                                             [SeaLevel[-1]])),
                                                    columns=['Qin','Qout','SeaLevel'],
                                                    index=pd.DatetimeIndex([MorTime])))
            OutTime += OutputOpts['OutInt']
        
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
                visualise.updateModelView(ModelFig, ShoreX, ShoreY, OutletEndX, 
                                          OutletEndWidth, OutletChanIx, 
                                          Closed=Closed, ShoreZ=ShoreZ, 
                                          WavePower=WavePower, EDir_h=EDir_h, 
                                          LST=LST, CST=CST_tot)
                PlotTime += OutputOpts['PlotInt']
    
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
