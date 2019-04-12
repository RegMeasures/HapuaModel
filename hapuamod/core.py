# -*- coding: utf-8 -*-
""" Core functions for timestepping through model

"""

# import standard packages
import numpy as np
import logging
import pandas as pd

# import local modules
import hapuamod.riv as riv
import hapuamod.coast as coast

def timestep(Time, Dx, Dt, WaveTs, SeaLevelTs, FlowTs, PhysicalPars, 
             ShoreX, ShoreY, LagoonY, LagoonElev, RiverElev, 
             OutletX, OutletY, OutletElev, OutletWidth):
    """ Run a single timestep of the entire model
    """
    
    logging.info('Processing Time = %s',Time)
    
    #%% Interpolate input variables for specific time
    WavesAtT = interpolate_at(WaveTs, Time)
    EDir_h = WavesAtT.EDir_h[0]
    WavePower = WavesAtT.WavePower[0]
    WavePeriod = WavesAtT.WavePeriod[0]
    Wlen_h = WavesAtT.Wlen_h[0]
    
#    RivFlow = interpolate_at(FlowTs, Time)
#    SeaLevel = interpolate_at(SeaLevelTs, Time)
    
    #%% Run the shoreline model
    LST = coast.longShoreTransport(ShoreY, Dx, WavePower, WavePeriod, Wlen_h, 
                                EDir_h, PhysicalPars)

    ShoreY += coast.shoreChange(LST, Dx, Dt, PhysicalPars)
    
    #%% Run the river model
    

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

