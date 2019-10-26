# -*- coding: utf-8 -*-
""" 
"""

# Import standard packages
import netCDF4
import logging
import argparse
import matplotlib
matplotlib.use('Agg')
from matplotlib import animation
import matplotlib.pyplot as plt

# import local modules
from . import visualise
from . import out
    
def animate(TimeIx, NcFile, ModelFig):
    logging.info('Animating timestep no %i' % TimeIx)
    (ShoreX, ShoreY, ShoreZ, 
     OutletEndX, OutletEndWidth, OutletEndElev, OutletChanIx,
     WavePower, EDir_h, LST, CST, Closed, RiverElev) = out.readTimestep(NcFile, TimeIx)
    visualise.updateModelView(ModelFig, ShoreX, ShoreY, OutletEndX, OutletEndWidth, 
                              OutletChanIx, Closed=Closed, 
                              ShoreZ=None, WavePower=None, EDir_h=0, LST=None, CST=None)
    #return [ModelFig['ChannelLine'], ModelFig['ShoreLine']]
    
def main(ResultsFile, AnimationFile, StartTimestep=0, EndTimestep=None, 
         ResampleInt=1, FrameRate=5, AreaOfInterest=None):
    """ Main function for creating an animation of hapuamod results
        
        main(ResultsFile, AnimationFile, StartTimestep, EndTimestep)
        
        Parameters:
            ResultsFile (str): Filename of the netCDF file containing model 
                               results
            AnimationFile (str): Filename of the movie file to create (*.mp4 
                                 seems to work but other formats may be 
                                 possible?)
            StartTimestep (int): First output timestep to animate (optional)
            EndTimestep (int): Last output timestep to animate (optional)
            ResampleInt (int): Frequency of output timesteps to use in 
                               animation (optional, default=1)
            FrameRate (float): Frames rate (frames/s) for output animation
                               (optional, default=5)
            AreaOfInterest (tuple): Area to zoom plot to. Specified in model 
                                    co-ordinates in the format 
                                    "(Xmin, Xmax, Ymin, Ymax)" (Optional)
    """
    
    # open netcdf file for reading
    logging.info('Reading data from %s' % ResultsFile)
    NcFile = netCDF4.Dataset(ResultsFile, mode='r', format='NETCDF4_CLASSIC') 
    
    # set default end-timestep
    if EndTimestep is None:
        EndTimestep = NcFile.dimensions['time'].size
    elif EndTimestep > NcFile.dimensions['time'].size:
        EndTimestep = NcFile.dimensions['time'].size
        logging.warning('EndTimestep exceeds number of timesteps in output file. EndTimestep has been reset to %i.' % EndTimestep)
    logging.info('Generating animation for timesteps %i to %i' % (StartTimestep, EndTimestep))
    
    # read in first timestep
    (ShoreX, ShoreY, ShoreZ, 
     OutletEndX, OutletEndWidth, OutletEndElev, OutletChanIx,
     WavePower, EDir_h, LST, CST, Closed, RiverElev) = out.readTimestep(NcFile, StartTimestep)
    
    # Set up initial figure
    ModelFig = visualise.modelView(ShoreX, ShoreY, OutletEndX, OutletEndWidth, OutletChanIx, 100,
                                   ShoreZ=None, WavePower=None, EDir_h=0, LST=None, CST=None, 
                                   WaveScaling=0.01, CstScaling=0.00005, LstScaling=0.0001,
                                   QuiverWidth=0.002, AreaOfInterest=AreaOfInterest)
    FigToAnimate = ModelFig['PlanFig']
    
    # Set up animation function
    ani = animation.FuncAnimation(FigToAnimate, animate, 
                                  frames=range(StartTimestep, EndTimestep, 
                                               ResampleInt), 
                                  fargs=(NcFile, ModelFig), 
                                  interval=1000/FrameRate)
    
    # Run the animation and save to a file
    ani.save(AnimationFile, fps=FrameRate)
    
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
                        help='Optional timestep number to start animation from ' +
                             '(integer, default=0)')
    Parser.add_argument('-e', '--EndTimestep', type=int, default=None,
                        help='Optional timestep number to end animation at ' +
                             '(integer, default=last available timestep)')
    Parser.add_argument('-r', '--ResampleInt', type=int, default=1,
                        help='Optional output timestep resampling interval, ' +
                             'for example a value of 3 would animate every ' +
                             'third output timestep. (integer, default=1)')
    Parser.add_argument('-f', '--FrameRate', type=float, default=5,
                        help='Optional frame rate in frames per second ' +
                             '(float, default=5)')
    Parser.add_argument('-a', '--AreaOfInterest', type=float, default=None, 
                        nargs='+',
                        help='Optional area of interest to zoom the output ' +
                             'animation to. The area should be specified as 4 ' +
                             'parameters in model co-ordinates in the format ' +
                             '"Xmin Xmax Ymin Ymax"')
    ArgsIn = Parser.parse_args()
    
    # Convert AreaOfInterest (if specified) into a tuple 
    if not ArgsIn.AreaOfInterest is None:
        ArgsIn.AreaOfInterest = tuple(ArgsIn.AreaOfInterest)
    
    # Run main animation function with specified arguments
    main(ArgsIn.ResultsFile, ArgsIn.AnimationFile, 
         StartTimestep = ArgsIn.StartTimestep, 
         EndTimestep = ArgsIn.EndTimestep, 
         ResampleInt = ArgsIn.ResampleInt, 
         FrameRate = ArgsIn.FrameRate,
         AreaOfInterest = ArgsIn.AreaOfInterest)