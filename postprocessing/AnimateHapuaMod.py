# -*- coding: utf-8 -*-

# Import standard packages
import netCDF4
import numpy as np
import os
import logging
import argparse
from matplotlib import animation
import matplotlib.pyplot as plt

# import local modules
from hapuamod import visualise

ResultsFile = 'HurunuiModelOutputs.nc'

StartTimestep = 0
EndTimestep = []

def main(ResultsFile, StartTimestep, EndTimestep):
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
    ModelFig = visualise.modelView(ShoreX, ShoreY, OutletEndX, OutletEndWidth, OutletChanIx, 
                                   ShoreZ=None, WavePower=None, EDir_h=0, LST=None, CST=None, 
                                   WaveScaling=0.01, CstScaling=0.00005, LstScaling=0.0001,
                                   QuiverWidth=0.002)
    FigToAnimate = ModelFig['PlanFig']
    
    ani = animation.FuncAnimation(FigToAnimate, animate, frames=range(StartTimestep,EndTimestep), interval=500)
    
    ani.save('test.mp4', fps=30)
    
    ani.event_source.stop()
    del ani
    plt.close(FigToAnimate)


def readTimestep(NcFile, TimeIx):
    NTransects = NcFile.dimensions['transect_x'].size
    
    # ShoreY
    ShoreY = np.empty((NTransects, 5))
    ShoreY[:,0] = NcFile.variables['shoreline_y'][TimeIx,:]
    ShoreY[:,1] = NcFile.variables['outlet_bank1_y'][TimeIx,:]
    ShoreY[:,2] = NcFile.variables['outlet_bank2_y'][TimeIx,:]
    ShoreY[:,3] = NcFile.variables['lagoon_y'][TimeIx,:]
    ShoreY[:,4] = NcFile.variables['cliff_y'][TimeIx,:]
    
    OutletEndX = NcFile.variables['outlet_end_x'][TimeIx,:]
    OutletEndWidth = NcFile.variables['outlet_end_width'][TimeIx,:]
    Closed = bool(NcFile.variables['outlet_closed'][TimeIx])
    
    # ShoreZ
    ShoreZ=np.empty((NTransects, 4))
    ShoreZ[:,0] = NcFile.variables['barrier_crest_z'][TimeIx,:]
    ShoreZ[:,1] = NcFile.variables['outlet_bed_z'][TimeIx,:]
    ShoreZ[:,2] = NcFile.variables['inner_barrier_crest_z'][TimeIx,:]
    ShoreZ[:,3] = NcFile.variables['lagoon_bed_z'][TimeIx,:]
    
    WavePower=None
    EDir_h=0
    LST = NcFile.variables['lst'][TimeIx,:] / 3600
    CST = NcFile.variables['cst'][TimeIx,:] / 3600
    
    # Calculate OutletChanIx
    if Closed:
        # Outlet closed
        OutletChanIx = np.empty(0)
    elif OutletEndX[0] < OutletEndX[1]:
        # Outlet angles from L to R
        OutletChanIx = np.where(np.logical_and(OutletEndX[0] <= ShoreX, 
                                               ShoreX <= OutletEndX[1]))[0]
    else:
        # Outlet from R to L
        OutletChanIx = np.flipud(np.where(np.logical_and(OutletEndX[1] <= ShoreX,
                                                         ShoreX <= OutletEndX[0]))[0])
    
    return(ShoreY, OutletEndX, OutletEndWidth, OutletChanIx,
           ShoreZ, WavePower, EDir_h, LST, CST)
    
def animate(TimeIx):
    (ShoreY, OutletEndX, OutletEndWidth, OutletChanIx, 
     ShoreZ, WavePower, EDir_h, LST, CST) = readTimestep(NcFile, TimeIx)
    visualise.updateModelView(ModelFig, ShoreX, ShoreY, OutletEndX, OutletEndWidth, 
                              OutletChanIx, Closed=False, 
                              ShoreZ=None, WavePower=None, EDir_h=0, LST=None, CST=None)
    #return [ModelFig['ChannelLine'], ModelFig['ShoreLine']]