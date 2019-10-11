# -*- coding: utf-8 -*-
""" 
"""

# Import standard packages
import netCDF4
import numpy as np
import os
import logging
import argparse
import matplotlib
matplotlib.use('Agg')
from matplotlib import animation
import matplotlib.pyplot as plt

# import local modules
from hapuamod import visualise

def readTimestep(NcFile, TimeIx):
    NTransects = NcFile.dimensions['transect_x'].size
    
    ShoreX = NcFile.variables['transect_x'][:]
    
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
    
    return(ShoreX, ShoreY, OutletEndX, OutletEndWidth, OutletChanIx,
           ShoreZ, WavePower, EDir_h, LST, CST, Closed)
    
def animate(TimeIx, NcFile, ModelFig):
    logging.info('Animating timestep no %i' % TimeIx)
    (ShoreX, ShoreY, OutletEndX, OutletEndWidth, OutletChanIx, 
     ShoreZ, WavePower, EDir_h, LST, CST, Closed) = readTimestep(NcFile, TimeIx)
    visualise.updateModelView(ModelFig, ShoreX, ShoreY, OutletEndX, OutletEndWidth, 
                              OutletChanIx, Closed=Closed, 
                              ShoreZ=None, WavePower=None, EDir_h=0, LST=None, CST=None)
    #return [ModelFig['ChannelLine'], ModelFig['ShoreLine']]
    
def main(ResultsFile, AnimationFile, StartTimestep=0, EndTimestep=[]):
    """ Main function for creating an animation of hapuamod results
        
        main(ResultsFile, AnimationFile, StartTimestep, EndTimestep)
        
        Parameters:
            ResultsFile (str): Filename of the netCDF file containing model 
                               results
            AnimationFile (str): Filename of the movie file to create (*.mp4 
                                 seems to work but other formats may be 
                                 possible?)
            StartTimestep (int): First output timestep to animate 
            EndTimestep (int): Last output timestep to animate 
    """
    
    # open netcdf file for reading
    logging.info('Reading data from %s' % ResultsFile)
    NcFile = netCDF4.Dataset(ResultsFile, mode='r', format='NETCDF4_CLASSIC') 
    
    # set default end-timestep
    if not EndTimestep:
        EndTimestep = NcFile.dimensions['time'].size
    elif EndTimestep > NcFile.dimensions['time'].size:
        EndTimestep = NcFile.dimensions['time'].size
        logging.warning('EndTimestep exceeds number of timesteps in output file. EndTimestep has been reset to %i.' % EndTimestep)
    logging.info('Generating animation for timesteps %i to %i' % (StartTimestep, EndTimestep))
    
    # read in first timestep
    (ShoreX, ShoreY, OutletEndX, OutletEndWidth, OutletChanIx, 
     ShoreZ, WavePower, EDir_h, LST, CST, Closed) = readTimestep(NcFile, StartTimestep)
    
    # Set up initial figure
    ModelFig = visualise.modelView(ShoreX, ShoreY, OutletEndX, OutletEndWidth, OutletChanIx, 
                                   ShoreZ=None, WavePower=None, EDir_h=0, LST=None, CST=None, 
                                   WaveScaling=0.01, CstScaling=0.00005, LstScaling=0.0001,
                                   QuiverWidth=0.002)
    FigToAnimate = ModelFig['PlanFig']
    
    # Set up animation function
    ani = animation.FuncAnimation(FigToAnimate, animate, 
                                  frames=range(StartTimestep,EndTimestep), 
                                  fargs=(NcFile, ModelFig), interval=500)
    
    # Run the animation and save to a file
    ani.save(AnimationFile, fps=30)
    
    ani.event_source.stop()
    del ani
    plt.close(FigToAnimate)

if __name__ == "__main__":
    # Set up logging
    RootLogger = logging.getLogger()
    RootLogger.setLevel(logging.INFO)
    
    ConsoleHandler = logging.StreamHandler()
    ConsoleHandler.setLevel(logging.INFO)
    RootLogger.addHandler(ConsoleHandler)
    
    # Parse arguments from commandline
    Parser = argparse.ArgumentParser(description='Generate animations from HapuaMod netCDF output file')
    Parser.add_argument('ResultsFile', help='HapuaMod netCDF formatted results file', type=str)
    Parser.add_argument('AnimationFile', help='mp4 video output file', type=str)
    Parser.add_argument('-s', '--StartTimestep', type=int, default=0,
                        help='Optional timestep number to start animation from (integer)')
    Parser.add_argument('-e', '--EndTimestep', type=int, 
                        help='Optional timestep number to end animation at (integer)')
    ArgsIn = Parser.parse_args()
    main(ArgsIn.ResultsFile, ArgsIn.AnimationFile, ArgsIn.StartTimestep, ArgsIn.EndTimestep)