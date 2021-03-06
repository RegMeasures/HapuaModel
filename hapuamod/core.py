# -*- coding: utf-8 -*-
""" Core hapua model

"""

# import standard packages
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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
    
    #%% Load the model
    (ModelName, FlowTs, WaveTs, SeaLevelTs, Origin, ShoreNormDir, ShoreX, 
     ShoreY, ShoreZ, RiverElev, OutletEndX, OutletEndWidth, OutletEndElev, 
     RiverWL, RiverVel, LagoonWL, LagoonVel, OutletDep, OutletVel, 
     OutletEndDep, OutletEndVel, Closed, TimePars, PhysicalPars, 
     NumericalPars, OutputOpts) = loadmod.loadModel(ModelConfigFile)
    
    #%% Generate initial conditions for river model - only necessary if not hotstarting
    RivFlow = interpolate_at(FlowTs, pd.DatetimeIndex([TimePars['StartTime']])).values
    SeaLevel = interpolate_at(SeaLevelTs, pd.DatetimeIndex([TimePars['StartTime']])).values
    WavesAtT = interpolate_at(WaveTs, pd.DatetimeIndex([TimePars['StartTime']]))
    EDir_h = WavesAtT.EDir_h[0]
    Hs_offshore = WavesAtT.Hsig_Offshore[0]
    
    (ChanDx, ChanElev, ChanWidth, LagArea, LagLen, ChanDep, ChanVel, 
     OnlineLagoon, OutletChanIx, ChanFlag, Closed) = \
        riv.assembleChannel(ShoreX, ShoreY, ShoreZ,
                            OutletEndX, OutletEndWidth, OutletEndElev, 
                            Closed, RiverElev, RiverWL - RiverElev, RiverVel,
                            LagoonWL, LagoonVel, OutletDep, OutletVel,
                            OutletEndDep, OutletEndVel, NumericalPars['Dx'],
                            PhysicalPars)

    if np.all(LagoonVel==0):
        # If not hotstarting
        (ChanDep, ChanVel) = riv.solveSteady(ChanDx, ChanElev, ChanWidth, 
                                             PhysicalPars['RoughnessManning'], 
                                             RivFlow, SeaLevel, NumericalPars)
        
        (LagoonWL, LagoonVel, OutletDep, OutletVel, OutletEndDep, OutletEndVel) = \
            riv.storeHydraulics(ChanDep, ChanVel, OnlineLagoon, OutletChanIx, 
                                ChanFlag, ShoreZ[:,3], Closed)
    
    #%% Create output files and write initial conditions
    out.newOutFile(OutputOpts['OutFile'], ModelName, TimePars['StartTime'], 
                   ShoreX, NumericalPars['Dx'],  RiverElev, 
                   Origin, ShoreNormDir, PhysicalPars,
                   Overwrite)
    out.writeCurrent(OutputOpts['OutFile'], TimePars['StartTime'], 
                     SeaLevel[-1], RivFlow[-1], Hs_offshore, EDir_h,
                     ShoreY, ShoreZ, LagoonWL, LagoonVel, np.zeros(ShoreX.size), 
                     OutletDep, OutletVel, np.zeros(ShoreX.size), 
                     np.zeros(ShoreX.size-1), np.zeros(ShoreX.size), np.zeros(ShoreX.size), 
                     RiverElev, ChanDep[ChanFlag==0], ChanVel[ChanFlag==0], np.zeros(RiverElev.size),
                     OutletEndX, OutletEndElev, OutletEndWidth, 
                     OutletEndDep, OutletEndVel, np.zeros(2), Closed)
    
    out.newTsOutFile(OutputOpts['TsOutFile'], ModelName, 
                     TimePars['StartTime'], Overwrite)
    out.writeTsOut(OutputOpts['TsOutFile'], TimePars['StartTime'], 
                   SeaLevel[-1], RivFlow[-1], Hs_offshore, EDir_h, 
                   np.nanmean(LagoonWL), ChanDep[-1]*ChanVel[-1]*ChanWidth[-1], 
                   0, OutletEndX, Closed, TimePars['MorDtMin'])    
    
    #%% Prepare plotting
    LivePlot = OutputOpts['PlotInt'] > pd.Timedelta(0)
    if LivePlot:
        plt.ion()
        LongSecFig = visualise.longSection(ChanDx, ChanElev, ChanWidth, ChanDep, ChanVel, 
                                           np.zeros(ChanElev.size-1))
        BdyPlotTs = pd.DataFrame(list(zip([RivFlow],[RivFlow],[SeaLevel])),
                                 columns=['Qin','Qout','SeaLevel'],
                                 index=pd.DatetimeIndex([TimePars['StartTime']]))
        BdyFig = visualise.bdyCndFig(BdyPlotTs)
        ModelFig = visualise.modelView(ShoreX, ShoreY, OutletEndX, OutletEndWidth,
                                       OutletChanIx, PhysicalPars['RiverWidth'],
                                       PhysicalPars['SpitWidth'], ShoreZ=ShoreZ, 
                                       WavePower=0, EDir_h=0, 
                                       LST=np.zeros(ShoreX.size-1),
                                       CST=np.zeros(ShoreX.size))
        
    #%% Main timestepping loop
    MorTime = TimePars['StartTime']
    OutTime = TimePars['StartTime'] + OutputOpts['OutInt']
    TsOutTime = TimePars['StartTime'] + OutputOpts['TsOutInt']
    LogTime = TimePars['StartTime'] + OutputOpts['LogInt']
    PlotTime = TimePars['StartTime'] + OutputOpts['PlotInt']
    
    MorDt = TimePars['MorDtMin'] # Initial morphological timestep
    while MorTime <= TimePars['EndTime']:
        
        # Re-assemble the combined river channel incase it has evolved
        (ChanDx, ChanElev, ChanWidth, LagArea, LagLen, ChanDep, ChanVel, 
         OnlineLagoon, OutletChanIx, ChanFlag, Closed) = \
            riv.assembleChannel(ShoreX, ShoreY, ShoreZ, 
                                OutletEndX, OutletEndWidth, OutletEndElev, 
                                Closed, RiverElev, 
                                ChanDep[ChanFlag==0], ChanVel[ChanFlag==0], 
                                LagoonWL, LagoonVel, OutletDep, OutletVel,
                                OutletEndDep, OutletEndVel, 
                                NumericalPars['Dx'], PhysicalPars)
                
        # Run the river model for all the timesteps upto the next morphology step
        HydTimes = pd.date_range(MorTime, MorTime + MorDt, 
                                 freq=TimePars['HydDt'], closed='right')
        
        RivFlow = interpolate_at(FlowTs, HydTimes).values
        SeaLevel = interpolate_at(SeaLevelTs, HydTimes).values
        
        try:
            riv.solveFullPreissmann(ChanElev, ChanWidth, LagArea, LagLen, 
                                    Closed, ChanDep, ChanVel, ChanDx, 
                                    TimePars['HydDt'], RivFlow, SeaLevel, 
                                    NumericalPars, PhysicalPars)
        except Exception as ErrMsg:
            logging.warning(ErrMsg)
            logging.warning('Unsteady hydraulics failed at %s. Falling back to quasi-steady for this timestep.' % 
                            HydTimes[-1].strftime("%m/%d/%Y, %H:%M:%S"))
            (ChanDep, ChanVel) = riv.solveSteady(ChanDx, ChanElev, ChanWidth, 
                                                PhysicalPars['RoughnessManning'], 
                                                RivFlow[-1], SeaLevel[-1], 
                                                NumericalPars)
        
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
        
        # Update morphology
        (MorDt, Breach) = mor.updateMorphology(ShoreX, ShoreY, ShoreZ,
                                               OutletEndX, OutletEndWidth, OutletEndElev, 
                                               RiverElev, OnlineLagoon, 
                                               OutletChanIx, LagoonWL, OutletDep,
                                               ChanWidth, ChanDep, ChanDx, ChanFlag, 
                                               Closed, LST, Bedload, CST_tot, OverwashProp,
                                               MorDt, 
                                               PhysicalPars, TimePars, NumericalPars)
        
        # Prevent model from running beyond specified end time
        if MorDt > TimePars['EndTime'] - MorTime:
            MorDt = TimePars['EndTime'] - MorTime
        
        # increment time
        MorTime += MorDt
        
        # If it is an output or plotting timestep then put bedload into model co-ordinates
        if MorTime >= OutTime or (LivePlot and MorTime >= PlotTime):
            (LagoonBedload, OutletBedload, OutletEndBedload) = \
                riv.storeBedload(Bedload, ShoreX.size, OnlineLagoon, OutletChanIx, 
                                 ChanFlag, Closed)
        
        # Save outputs
        if MorTime >= OutTime:
            out.writeCurrent(OutputOpts['OutFile'], MorTime, 
                             SeaLevel[-1], RivFlow[-1], Hs_offshore, EDir_h, 
                             ShoreY, ShoreZ, LagoonWL, LagoonVel, LagoonBedload,
                             OutletDep, OutletVel, OutletBedload,
                             LST, CST_tot, OverwashProp,
                             RiverElev, ChanDep[ChanFlag==0], ChanVel[ChanFlag==0], Bedload[ChanFlag[:-1]==0],
                             OutletEndX, OutletEndElev, OutletEndWidth, 
                             OutletEndDep, OutletEndVel, OutletEndBedload, Closed)
            OutTime += OutputOpts['OutInt']
        
        if MorTime >= TsOutTime:
            out.writeTsOut(OutputOpts['TsOutFile'], MorTime, 
                           SeaLevel[-1], RivFlow[-1], Hs_offshore, EDir_h, 
                           np.nanmean(LagoonWL), ChanDep[-1]*ChanVel[-1]*ChanWidth[-1], 
                           Bedload[0], OutletEndX, Closed, MorDt)
            TsOutTime += OutputOpts['TsOutInt']
        
        # updates to user
        if MorTime >= LogTime:
            logging.info('Time = %s, MorDt = %.1fs', MorTime, MorDt.seconds)
            LogTime += OutputOpts['LogInt']
            
        # plotting
        if LivePlot:
            if MorTime >= PlotTime:
                visualise.updateLongSection(LongSecFig, ChanDx, ChanElev, 
                                            ChanWidth, ChanDep, ChanVel, Bedload)
                BdyPlotTs = BdyPlotTs.append(pd.DataFrame(list(zip([RivFlow[-1]],
                                                                   [ChanDep[-1]*ChanVel[-1]*ChanWidth[-1]],
                                                                   [SeaLevel[-1]])),
                                                          columns=['Qin','Qout','SeaLevel'],
                                                          index=pd.DatetimeIndex([MorTime])))
                visualise.updateBdyCndFig(BdyFig, BdyPlotTs)
                visualise.updateModelView(ModelFig, ShoreX, ShoreY, OutletEndX, 
                                          OutletEndWidth, OutletChanIx, 
                                          Closed=Closed, ShoreZ=ShoreZ, 
                                          WavePower=WavePower, EDir_h=EDir_h, 
                                          LST=LST, CST=CST_tot)
                PlotTime += OutputOpts['PlotInt']
        
        # Adjust closed flag if breach has occured
        if Breach:
            Closed = False
    
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
    FromIx = np.where(Df.index <= New_idxs[0])[0][-1]
    ToIx = np.where(Df.index >= New_idxs[-1])[0][0]
    Selected = Df[FromIx:ToIx+1]
    Selected = Selected.reindex(Selected.index.append(New_idxs).unique())
    Selected = Selected.sort_index()
    Selected = Selected.interpolate(method='time')
    return Selected.loc[New_idxs]
