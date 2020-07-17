# -*- coding: utf-8 -*-
""" Functions to generate animations from HapuaMod netCDF output file
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
from . import riv
    
def animateMap(TimeIx, NcFile, ModelFig):
    """ Animation timestepping function for generating plan view animation
    """
    logging.info('Animating timestep no %i' % TimeIx)
    
    (SeaLevel, ShoreX, ShoreY, ShoreZ, LagoonWL, LagoonVel, OutletWL, 
     OutletVel, OutletEndX, OutletEndWidth, OutletEndElev, OutletEndVel, 
     OutletEndWL, OutletChanIx, WavePower, EDir_h, LST, CST, Closed, 
     RiverElev, RiverWL, RiverVel, ModelTime) = out.readTimestep(NcFile, TimeIx)
    
    visualise.updateModelView(ModelFig, ShoreX, ShoreY, OutletEndX, OutletEndWidth, 
                              OutletChanIx, Closed=Closed, 
                              ShoreZ=None, WavePower=None, EDir_h=0, LST=None, 
                              CST=None, PlotTime=ModelTime)
    #return [ModelFig['ChannelLine'], ModelFig['ShoreLine']]
    
def animateTransect(TimeIx, NcFile, TransectFig):
    """ Animation timestepping function for transect view animation
    """
    logging.info('Animating timestep no %i' % TimeIx)
    
    (SeaLevel, ShoreX, ShoreY, ShoreZ, LagoonWL, LagoonVel, OutletWL, 
     OutletVel, OutletEndX, OutletEndWidth, OutletEndElev, OutletEndVel, 
     OutletEndWL, OutletChanIx, WavePower, EDir_h, LST, CST, Closed, 
     RiverElev, RiverWL, RiverVel, ModelTime) = out.readTimestep(NcFile, TimeIx)
    
    visualise.updateTransectFig(TransectFig, ShoreY, ShoreZ, 
                                LagoonWL, OutletWL, SeaLevel,
                                PlotTime=ModelTime)

def animateLongSec(TimeIx, NcFile, LongSecFig, PhysicalPars):
    """ Animation timestepping function for generating longsection animation
    """
    logging.info('Animating timestep no %i' % TimeIx)
    
    (SeaLevel, ShoreX, ShoreY, ShoreZ, LagoonWL, LagoonVel, OutletWL, 
     OutletVel, OutletEndX, OutletEndWidth, OutletEndElev, OutletEndVel, 
     OutletEndWL, OutletChanIx, WavePower, EDir_h, LST, CST, Closed, 
     RiverElev, RiverWL, RiverVel, ModelTime) = out.readTimestep(NcFile, TimeIx)
    
    Dx = ShoreX[1] - ShoreX[0]
    OutletDep = OutletWL - ShoreZ[:,1]
    OutletEndDep = OutletEndWL - OutletEndElev
    RiverDep = RiverWL - RiverElev
    
    (ChanDx, ChanElev, ChanWidth, LagArea, LagLen, ChanDep, ChanVel, 
     OnlineLagoon, OutletChanIx, ChanFlag, Closed) = \
        riv.assembleChannel(ShoreX, ShoreY, ShoreZ, 
                            OutletEndX, OutletEndWidth, OutletEndElev, 
                            Closed, RiverElev, RiverDep, RiverVel, 
                            LagoonWL, LagoonVel, OutletDep, OutletVel, 
                            OutletEndDep, OutletEndVel, Dx, PhysicalPars)
            
    visualise.updateLongSection(LongSecFig, ChanDx, ChanElev, ChanWidth, 
                                ChanDep, ChanVel, PlotTime=ModelTime)
    
def main(ResultsFile, AnimationFile, StartTimestep=0, EndTimestep=None, 
         ResampleInt=1, FrameRate=5, AreaOfInterest=None, TransectX=None,
         Longsection=False):
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
            TransectX (float): If specified then plot transect closest to 
                               specified X value instead of map view (optional)
            Longsection (bool): Flag to trigger plotting of long section rather
                                than plan view (Optional, default=False)
    """
    
    # Some input validation
    if Longsection and not(TransectX is None):
        logging.warning('Both Longsection and TransectX have been specified. ' +
                        'Only Transect animation will be produced.')
    
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
    (SeaLevel, ShoreX, ShoreY, ShoreZ, LagoonWL, LagoonVel, OutletWL, 
     OutletVel, OutletEndX, OutletEndWidth, OutletEndElev, OutletEndVel, 
     OutletEndWL, OutletChanIx, WavePower, EDir_h, LST, CST, Closed, 
     RiverElev, RiverWL, RiverVel, ModelTime) = out.readTimestep(NcFile, StartTimestep)
    
    if not(TransectX is None):
        # Transect animation
        logging.info('Animating transect closest to X = %.1f' % TransectX)
        
        # Set up initial figure
        BeachSlope = NcFile.BeachSlope
        BackshoreElev = NcFile.BackshoreElev
        ClosureDepth = NcFile.ClosureDepth
        BeachTopElev = NcFile.BeachTopElev
        TransectFig = visualise.newTransectFig(ShoreX, ShoreY, ShoreZ, LagoonWL, OutletWL, 
                                               SeaLevel, BeachSlope, BackshoreElev, 
                                               ClosureDepth, BeachTopElev, TransectX, 
                                               AreaOfInterest=AreaOfInterest)
        FigToAnimate = TransectFig['TransFig']
        
        # Set up the animation function
        ani = animation.FuncAnimation(FigToAnimate, animateTransect, 
                                      frames=range(StartTimestep, EndTimestep, 
                                                   ResampleInt), 
                                      fargs=(NcFile, TransectFig), 
                                      interval=1000/FrameRate)
    elif Longsection:
        # Longsection animation
        logging.info('Animating longsection view')
        
        # Set up initial figure
        OutletDep = OutletWL - ShoreZ[:,1]
        OutletEndDep = OutletEndWL - OutletEndElev
        RiverDep = RiverWL - RiverElev
        Dx = ShoreX[1] - ShoreX[0]
        PhysicalPars = {'RiverWidth': NcFile.RiverWidth,
                        'MinOutletWidth': NcFile.MinOutletWidth,
                        'MaxOutletElev': NcFile.MaxOutletElev} 
        (ChanDx, ChanElev, ChanWidth, LagArea, LagLen, ChanDep, ChanVel, 
         OnlineLagoon, OutletChanIx, ChanFlag, Closed) = \
            riv.assembleChannel(ShoreX, ShoreY, ShoreZ, 
                                OutletEndX, OutletEndWidth, OutletEndElev, 
                                Closed, RiverElev, RiverDep, RiverVel, 
                                LagoonWL, LagoonVel, OutletDep, OutletVel, 
                                OutletEndDep, OutletEndVel, Dx, PhysicalPars)
        LongSecFig = visualise.longSection(ChanDx, ChanElev, ChanWidth, 
                                           ChanDep, ChanVel, PlotTime=ModelTime,
                                           AreaOfInterest=AreaOfInterest)
        FigToAnimate = LongSecFig['RivFig']
        
        # Set up animation function
        ani = animation.FuncAnimation(FigToAnimate, animateLongSec, 
                                      frames=range(StartTimestep, EndTimestep, 
                                                   ResampleInt), 
                                      fargs=(NcFile, LongSecFig, PhysicalPars), 
                                      interval=1000/FrameRate)  
    else:
        # Map view animation
        logging.info('Animating map view')
        
        # Set up initial figure
        RiverWidth = NcFile.RiverWidth
        SpitWidth = NcFile.SpitWidth
        ModelFig = visualise.modelView(ShoreX, ShoreY, OutletEndX, OutletEndWidth, 
                                       OutletChanIx, RiverWidth, SpitWidth,
                                       ShoreZ=None, WavePower=None, EDir_h=0, LST=None, CST=None, 
                                       WaveScaling=0.01, CstScaling=0.00005, LstScaling=0.0001,
                                       QuiverWidth=0.002, AreaOfInterest=AreaOfInterest)
        FigToAnimate = ModelFig['PlanFig']
        
        # Set up animation function
        ani = animation.FuncAnimation(FigToAnimate, animateMap, 
                                      frames=range(StartTimestep, EndTimestep, 
                                                   ResampleInt), 
                                      fargs=(NcFile, ModelFig), 
                                      interval=1000/FrameRate)
        
    # Run the animation and save to a file
    ani.save(AnimationFile, fps=FrameRate, dpi=200)
    
    ani.event_source.stop()
    del ani
    plt.close(FigToAnimate)
    NcFile.close()

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
    Parser.add_argument('-t', '--TransectX', type=float, default=None,
                        help='If specified then plot transect closest to ' +
                              'specified X coordinate, optional - default = ' +
                              'plot map view (float)')
    Parser.add_argument('-l', '--Longsection', action='store_true',
                        help='If specified this flag toggles animation of a ' +
                             'long section rather than map view')
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
         AreaOfInterest = ArgsIn.AreaOfInterest,
         TransectX = ArgsIn.TransectX,
         Longsection = ArgsIn.Longsection)