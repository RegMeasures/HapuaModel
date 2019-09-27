# -*- coding: utf-8 -*-

# Import standard packages
import netCDF4
import numpy as np
import pandas as pd
import os
import logging
import argparse

ResultsFile = 'HurunuiModelOutputs.nc'

StartTimestep = 0
EndTimestep = []

# open netcdf file for reading
logging.info('Reading data from %s' % ResultsFile)
NcFile = netCDF4.Dataset(ResultsFile, mode='r', format='NETCDF4_CLASSIC') 

# set end-timestep
if not EndTimestep:
    EndTimestep = NcFile.dimensions['time'].size
elif EndTimestep > NcFile.dimensions['time'].size:
    EndTimestep = NcFile.dimensions['time'].size
    logging.warning('EndTimestep exceeds number of timesteps in output file. EndTimestep has been reset to %i.' % EndTimestep)

# read in static data
ShoreX = NcFile.variables['transect_x'][:]

# read in first timestep
(ShoreY, OutletEndX, OutletEndWidth, OutletChanIx, 
 ShoreZ, WavePower, EDir_h, LST, CST) = readTimestep(NcFile, StartTimestep)


# Set up initial figure
ModelFig = modelView(ShoreX, ShoreY, OutletEndX, OutletEndWidth, OutletChanIx, 
                     ShoreZ=None, WavePower=None, EDir_h=0, LST=None, CST=None, 
                     WaveScaling=0.01, CstScaling=0.00005, LstScaling=0.0001,
                     QuiverWidth=0.002)

updateModelView(ModelFig, ShoreX, ShoreY, OutletEndX, OutletEndWidth, 
                OutletChanIx, Closed=False, 
                ShoreZ=None, WavePower=None, EDir_h=0, LST=None, CST=None)

def readTimestep(NcFile, TimestepNo)
    ShoreY = NcFile.variables['shoreline_y'][TimestepNo,:]
    OutletEndX = NcFile.variables['outlet_end_x'][TimestepNo,:]
    OutletEndWidth = NcFile.variables['outlet_end_width'][TimestepNo,:]
    OutletChanIx = NcFile.variables['outlet_end_x'][TimestepNo,:]
    ShoreZ=None
    WavePower=None
    EDir_h=0
    LST=None
    CST=None