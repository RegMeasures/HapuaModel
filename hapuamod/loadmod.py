# -*- coding: utf-8 -*-

# import standard packages
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math
from configobj import ConfigObj
import os
import shapefile
import logging

# import local packages
from hapuamod import geom

def readConfig(ModelConfigFile):
    """ Reads model config file
    
    readConfig parses the model config file (*.cnf) into a dict style variable
    using the configobj package. In addition readConfig resolves relative file 
    paths contained in the config file.
    """
    
    # Read the main config file
    logging.info('Loading "%s"' % ModelConfigFile)
    Config = ConfigObj(ModelConfigFile)
    
    # Extract file path and add to other relative file paths as required
    ConfigFilePath = os.path.split(ModelConfigFile)[0]
    Config['BoundaryConditions']['RiverFlow'] = \
        os.path.join(ConfigFilePath, Config['BoundaryConditions']['RiverFlow'])
    Config['BoundaryConditions']['WaveConditions'] = \
        os.path.join(ConfigFilePath, Config['BoundaryConditions']['WaveConditions'])
    Config['BoundaryConditions']['SeaLevel'] = \
        os.path.join(ConfigFilePath, Config['BoundaryConditions']['SeaLevel'])
    Config['SpatialInputs']['Shoreline'] = \
        os.path.join(ConfigFilePath, Config['SpatialInputs']['Shoreline'])
    Config['SpatialInputs']['RiverLocation'] = \
        os.path.join(ConfigFilePath, Config['SpatialInputs']['RiverLocation'])
    Config['SpatialInputs']['LagoonOutline'] = \
        os.path.join(ConfigFilePath, Config['SpatialInputs']['LagoonOutline'])
    Config['SpatialInputs']['OutletLocation'] = \
        os.path.join(ConfigFilePath, Config['SpatialInputs']['OutletLocation'])
    
    # TODO: Add more some validation of inputs here or in loadModel
    
    return Config

def loadModel(Config):
    """ Pre-processes model inputs
    
    loadModel processes the HapuaMod configuration and associated linked files 
    (e.g. shapefile of coastline position etc.) to generate model input 
    variables. Prior to running loadModel the onfig file should first be parsed
    using the readConfig function.
    
    (FlowTs, WaveTs, SeaLevelTs, Origin, BaseShoreNormDir, ShoreX, 
     ShoreY, LagoonY, LagoonElev, RiverElev, OutletX, OutletY, 
     OutletElev, OutletWidth, Dx, Dt, SimTime, 
     PhysicalPars) = loadModel(Config)
    
    Parameters:
        Config (Dict): Model config file read into a dict variable by the 
            readConfig function
    
    Returns:
        FlowTs (DataFrame): Flow timeseries as single column DataFrame with 
            datetime index. River flow given in column "Flow" (m^3/s)
        WaveTs (DataFrame): Wave timeseries DataFrame with datetime index. 
            Columns are:
                - HsOffshore (float64): Offshiore significant wave height (m)
                - WavePeriod (float64): Wave period (s)
                - WavePower (float64): Wave power (W/m wave crest length)
                - Wlen_h (float64): Wavelength (m)
                - EAngle_h (float64): Net direction of wave energy at depth h 
                      (where h is given as PhysicalParameter WaveDataDepth). 
                      EAngle_h is read as a bearing in degrees, but is 
                      converted to a direction relative to BaseShoreNormDir in 
                      radians as part of the pre-processing (radians)
        SeaLevelTs (DataFrame): Sea level timeseries as single column DataFrame
            with datetime index. Column name is SeaLevel (m)
        Origin (np.ndarray(float64)): Real world X and Y coordinates of the
            origin of the model coordinate system (m)
        BaseShoreNormDir (float): Direction of offshore pointing line at 90 
            degrees to overall average coast direction. Computed based on a 
            straightline fitted thruogh the initial condition shoreline 
            position (radians).
        ShoreX, ShoreY (np.ndarray(float64)): positions of discretised 
            shoreline in model coordinate system (m)
        LagoonY (np.ndarray(float64)): position of seaward and landward sides 
            of the lagoon in model coordinate system at transects with 
            x-coordinates given by ShoreX (m)
        LagoonElev (np.ndarray(float64)): 
        RiverElev (np.ndarray(float64)): 
        OutletX, OutletY (np.ndarray(float64)): coordinates of discretised 
            outlet channel nodes in model coordinate system (m)
        OutletElev (np.ndarray(float64)): 
        OutletWidth (np.ndarray(float64)): 
        Dx (float): shoreline discretisation interval (m)
        Dt (pd.Timedelta): timestep
        SimTime (list): Two element list containing the simulation start
            and end times (datetime)
        PhysicalPars (dict): Physical parameters including:
            RhoSed (float): Sediment density (kg/m3)
            RhoSea (float): Seawater density (kg/m3)
            RhoRiv (float): Riverwater density (kg/m3)
            Gravity (float): Gravity (m/s2)
            Kcoef (float): K coefficient for longshore transport 
                (non-dimensional)
            VoidRatio (float): Sediment void ratio (0-1)
            GammaRatio (float): Ratio of water depth at breakpoint to breaking 
                wave height (non-dimensional)
            WaveDataDepth (float): Depth contour of input wave data (m)
            ClosureDepth (float): Closure depth for 1-line shoreline model (m)
            RiverSlope (float): Initial condition slope for river upstream of 
                hapua (m/m)
            GrainSize (float): Sediment size for fluvial sediment transport 
                calculation (assumed uniform) (m) 
            UpstreamLength (float): Length of river upstream of hapua to 
                include in model (m)
            RiverWidth (float): width of river upstream of hapua (assumed 
                       uniform) (m)
            Roughness (float): Manning's 'n' for river hydraulics (m^(1/3)/s) 
            K2coef (float): Calculated from other inputs for use in calculation
                of longshore transport rate. K2 = K / (RhoSed - RhoSea) * g * (1 - VoidRatio))
            BreakerCoef (float): Calculated from other inputs for use in 
                calculation of depth of breaking waves. 
                BreakerCoef = 8 / (RhoSea * Gravity^1.5 * GammaRatio^2)                        
    """
    
    #%% Spatial inputs
    
    # Read the initial shoreline position
    logging.info('Reading initial shoreline position from "%s"' %
                 Config['SpatialInputs']['Shoreline'])
    ShoreShp = shapefile.Reader(Config['SpatialInputs']['Shoreline'])
    # check it is a polyline and there is only one line
    assert ShoreShp.shapeType==3, 'Shoreline shapefile must be a polyline.'
    assert len(ShoreShp.shapes())==1, 'multiple polylines in Shoreline shapefile. There should only be 1.'
    # extract coordinates
    IniShoreCoords = np.asarray(ShoreShp.shape(0).points[:])
    
    # Read the river inflow location
    logging.info('Reading river inflow location from "%s"' %
                 Config['SpatialInputs']['RiverLocation'])
    RiverShp = shapefile.Reader(Config['SpatialInputs']['RiverLocation'])
    # check it is a single point
    assert RiverShp.shapeType==1, 'Shoreline shapefile must be a point.'
    assert len(RiverShp.shapes())==1, 'multiple points in RiverLocation shapefile. There should only be 1.'
    InflowCoord = np.asarray(RiverShp.shape(0).points[:]).squeeze()
    
    # Read the lagoon outline polygon
    logging.info('Reading lagoon outline from from "%s"' %
                 Config['SpatialInputs']['LagoonOutline'])
    LagoonShp = shapefile.Reader(Config['SpatialInputs']['LagoonOutline'])
    # check it is a polygon and there is only one
    assert LagoonShp.shapeType==5, 'Lagoon outline must be a polygon shapefile'
    assert len(LagoonShp.shapes())==1, 'multiple polygons in LagoonOutline shapefile. There should only be 1.'
    LagoonCoords = np.asarray(LagoonShp.shape(0).points[:])
    
    # Read the initial outlet position polyline
    logging.info('Reading lagoon outline from from "%s"' %
                 Config['SpatialInputs']['OutletLocation'])
    OutletShp = shapefile.Reader(Config['SpatialInputs']['OutletLocation'])
    # check it is a polyline and there is only 1
    assert OutletShp.shapeType==3, 'OutletLocation must be a polyline shapefile'
    assert len(OutletShp.shapes())==1, 'multiple polylines in OutletLocation. You can only specify a single initial outlet.'
    OutletCoords = np.asarray(OutletShp.shape(0).points[:])
    
    #%% Develop model coordinate system
    logging.info('Creating shore-parallel model coordinate system')
    
    # Fit a straight reference baseline through the specified shoreline points
    Baseline = np.polyfit(IniShoreCoords[:,0], IniShoreCoords[:,1], 1)
    
    # Defin origin point on baseline (in line with river inflow)
    Origin = np.empty(2)
    Origin[0] = ((InflowCoord[1] + 
                  (InflowCoord[0]/Baseline[0]) - Baseline[1]) / 
                 (Baseline[0] + 1/Baseline[0]))
    Origin[1] = Origin[0] * Baseline[0] + Baseline[1]
    if ((InflowCoord[0] * Baseline[0] + Baseline[1])>0):
        # land is to north of baseline
        BaseShoreNormDir = math.atan2(Baseline[0],-1)
    else:
        # land is to south of baseline
        BaseShoreNormDir = math.atan2(-Baseline[0],1)
    
    #%% Read the boundary condition timeseries
    to_datetime = lambda d: pd.datetime.strptime(d, '%d/%m/%Y %H:%M')
    
    logging.info('Reading flow timeseries from "%s"' % 
                 Config['BoundaryConditions']['RiverFlow'])
    FlowTs = pd.read_csv(Config['BoundaryConditions']['RiverFlow'], 
                         index_col=0, parse_dates=[0],
                         date_parser=to_datetime)
    FlowTs = FlowTs.Flow
    
    logging.info('Reading wave timeseries from "%s"' % 
                 Config['BoundaryConditions']['WaveConditions'])
    WaveTs = pd.read_csv(Config['BoundaryConditions']['WaveConditions'], 
                         index_col=0, parse_dates=[0],
                         date_parser=to_datetime)
    # Convert wave directions into radians in model coordinate system
    WaveTs.EDir_h = np.deg2rad(WaveTs.EDir_h) - (BaseShoreNormDir)
    # Make sure all wave angles are in the range -pi to +pi
    WaveTs.EDir_h = np.mod(WaveTs.EDir_h + np.pi, 2.0 * np.pi) - np.pi 
    
    logging.info('Reading sea level timeseries from "%s"' % 
                 Config['BoundaryConditions']['SeaLevel'])
    SeaLevelTs = pd.read_csv(Config['BoundaryConditions']['SeaLevel'], 
                             index_col=0, parse_dates=[0],
                             date_parser=to_datetime)
    SeaLevelTs = SeaLevelTs.SeaLevel
    
    #%% Initial conditions
    logging.info('Processing initial conditions')
    IniCond = {'OutletWidth': float(Config['InitialConditions']['OutletWidth']),
               'OutletBed': float(Config['InitialConditions']['OutletBed']),
               'LagoonBed': float(Config['InitialConditions']['LagoonBed'])}
    
    #%% Time inputs
    TimePars = {'StartTime': pd.to_datetime(Config['Time']['StartTime']),
                'EndTime': pd.to_datetime(Config['Time']['EndTime']),
                'HydDt': pd.Timedelta(seconds=float(Config['Time']['HydDt'])),
                'MorDt': pd.Timedelta(seconds=float(Config['Time']['MorDt']))}
    # TODO: check mortime is a multiple of hydtime (or replace with morscaling?)
    
    #%% Trim time-series inputs to model time
    assert (FlowTs.index[0] <= TimePars['StartTime'] 
            and FlowTs.index[-1] >= TimePars['EndTime']), \
        'Flow timeseries %s does not extend over full model duration' \
        % Config['BoundaryConditions']['RiverFlow']
    KeepTimes = np.zeros(FlowTs.shape[0], dtype=bool)
    KeepTimes[:-1] = FlowTs.index[1:] > TimePars['StartTime']
    KeepTimes[1:] = FlowTs.index[:-1]<TimePars['EndTime']
    FlowTs = FlowTs[KeepTimes]

    assert (WaveTs.index[0] <= TimePars['StartTime'] 
            and WaveTs.index[-1] >= TimePars['EndTime']), \
        'Wave timeseries %s does not extend over full model duration' \
        % Config['BoundaryConditions']['WaveConditions']
    KeepTimes = np.zeros(WaveTs.shape[0], dtype=bool)
    KeepTimes[:-1] = WaveTs.index[1:] > TimePars['StartTime']
    KeepTimes[1:] = WaveTs.index[:-1]<TimePars['EndTime']
    WaveTs = WaveTs[KeepTimes]
    
    assert (SeaLevelTs.index[0] <= TimePars['StartTime'] 
            and SeaLevelTs.index[-1] >= TimePars['EndTime']), \
        'Sea level timeseries %s does not extend over full model duration' \
        % Config['BoundaryConditions']['SeaLevel']        
    KeepTimes = np.zeros(SeaLevelTs.shape[0], dtype=bool)
    KeepTimes[:-1] = SeaLevelTs.index[1:] > TimePars['StartTime']
    KeepTimes[1:] = SeaLevelTs.index[:-1]<TimePars['EndTime']
    SeaLevelTs = SeaLevelTs[KeepTimes]
    
    #%% Read physical parameters
    logging.info('Processing physical parameters')
    PhysicalPars = {'RhoSed': float(Config['PhysicalParameters']['RhoSed']),
                    'RhoSea': float(Config['PhysicalParameters']['RhoSea']),
                    'RhoRiv': float(Config['PhysicalParameters']['RhoSea']),
                    'Kcoef': float(Config['PhysicalParameters']['Kcoef']),
                    'Gravity': float(Config['PhysicalParameters']['Gravity']),
                    'VoidRatio': float(Config['PhysicalParameters']['VoidRatio']),
                    'GammaRatio': float(Config['PhysicalParameters']['GammaRatio']),
                    'WaveDataDepth': float(Config['PhysicalParameters']['WaveDataDepth']),
                    'ClosureDepth': float(Config['PhysicalParameters']['ClosureDepth']),
                    'RiverSlope': float(Config['PhysicalParameters']['RiverSlope']),
                    'GrainSize': float(Config['PhysicalParameters']['GrainSize']),
                    'UpstreamLength': float(Config['PhysicalParameters']['UpstreamLength']),
                    'RiverWidth': float(Config['PhysicalParameters']['RiverWidth']),
                    'Roughness': float(Config['PhysicalParameters']['RoughnessManning'])}

    GammaLST = ((PhysicalPars['RhoSed'] - PhysicalPars['RhoSea']) * 
                PhysicalPars['Gravity'] * (1 - PhysicalPars['VoidRatio']))
    
    PhysicalPars['K2coef'] = PhysicalPars['Kcoef'] / GammaLST
    PhysicalPars['BreakerCoef'] = 8.0 / (PhysicalPars['RhoSea'] *
                                         PhysicalPars['Gravity']**1.5 *
                                         PhysicalPars['GammaRatio']**2.0)
    
    #%% Read numerical parameters
    Dx = float(Config['NumericalParameters']['AlongShoreDx'])
    NumericalPars = {'Dx': Dx,
                     'Theta': float(Config['NumericalParameters']['Theta']),
                     'ErrTol': float(Config['NumericalParameters']['ErrTol']),
                     'MaxIt': float(Config['NumericalParameters']['MaxIt']),
                     'WarnTol': float(Config['NumericalParameters']['WarnTol'])}

    #%% Read output options
    OutputOpts = {'LogInt': pd.Timedelta(seconds=float(Config['OutputOptions']['LogInt'])),
                  'PlotInt': pd.Timedelta(seconds=float(Config['OutputOptions']['PlotInt']))}
    
    #%% Initialise shoreline variables
    
    # Convert shoreline into model coordinate system
    IniShoreCoords2 = np.empty(IniShoreCoords.shape)
    (IniShoreCoords2[:,0], IniShoreCoords2[:,1]) = geom.real2mod(IniShoreCoords[:,0], IniShoreCoords[:,1], Origin, BaseShoreNormDir)
    if IniShoreCoords2[0,0] > IniShoreCoords2[-1,0]:
        IniShoreCoords2 = IniShoreCoords2 = np.flipud(IniShoreCoords2)
    assert np.all(np.diff(IniShoreCoords2[:,0]) > 0), 'Shoreline includes recurvature???'
    
    # Discretise shoreline at fixed intervals in model coordinate system
    ShoreX = np.arange(math.ceil(IniShoreCoords2[0,0]/Dx)*Dx, 
                       IniShoreCoords2[-1,0], Dx)
    ShoreY = np.interp(ShoreX, IniShoreCoords2[:,0], IniShoreCoords2[:,1])
    
    #%% Initialise lagoon variables
    
    # Convert lagoon outline into model coordinate system
    LagoonCoords2 = np.empty(LagoonCoords.shape)
    (LagoonCoords2[:,0], LagoonCoords2[:,1]) = geom.real2mod(LagoonCoords[:,0], LagoonCoords[:,1], Origin, BaseShoreNormDir)
    
    # Check lagoon extent is within limits of shoreline
    LagoonExtent = [np.amin(LagoonCoords2[:,0]), np.amax(LagoonCoords2[:,0])]
    assert LagoonExtent[1] < ShoreX[-1] and LagoonExtent[0] > ShoreX[0], 'Lagoon exdends beyond shoreline extent. Extend shoreline shapefile to cover full extent of lagoon.'
    
    # Discretise lagoon
    LagoonY = np.full([ShoreX.size,2], np.nan)
    for ii in range(ShoreX.size):
        if LagoonExtent[0] < ShoreX[ii] < LagoonExtent[1]:
            # identify inersections of lagoon polygon and relevant shore normal transect
            YIntersects = geom.intersectPolygon(LagoonCoords2, ShoreX[ii])
            LagoonY[ii,0] = np.amin(YIntersects)
            LagoonY[ii,1] = np.amax(YIntersects)
    
    # Initialise lagoon bed elevation
    LagoonElev = np.full(ShoreX.size, IniCond['LagoonBed'])
    LagoonElev[np.isnan(LagoonY[:,1])] = np.nan
    
    #%% Initialising river variables
    RiverElev = np.flipud(np.arange(IniCond['LagoonBed'],
                                    IniCond['LagoonBed']
                                    + PhysicalPars['RiverSlope']
                                    * PhysicalPars['UpstreamLength'],
                                    PhysicalPars['RiverSlope'] * Dx))
    
    # Convert outlet channel into model coordinate system
    OutletCoords2 = np.empty(OutletCoords.shape)
    (OutletCoords2[:,0], OutletCoords2[:,1]) = geom.real2mod(OutletCoords[:,0], OutletCoords[:,1], Origin, BaseShoreNormDir)
    
    # Check outlet coords ordered from lagoon to sea
    if OutletCoords2[0,1] > OutletCoords2[-1,1]:
        OutletCoords2 = np.flipud(OutletCoords2)
    
    # Trim outlet channel to shoreline and sea
    (OutletX, OutletY) = geom.trimSegment(OutletCoords2[:,0], OutletCoords2[:,1], 
                                          ShoreX, ShoreY)
    (OutletX, OutletY) = geom.trimSegment(np.flipud(OutletX), np.flipud(OutletY), 
                                          ShoreX, LagoonY[:,1])
    OutletX = np.flipud(OutletX)
    OutletY = np.flipud(OutletY)
    
    # Divide outlet channel into appropriate segments
    (OutletX, OutletY) = geom.adjustLineDx(OutletX, OutletY, Dx)
    
    # Create outlet channel variables
    OutletElev = np.linspace(IniCond['LagoonBed'], IniCond['OutletBed'],
                             OutletX.size)
    OutletWidth = np.tile(IniCond['OutletWidth'], OutletX.size)
      
    # Produce a map showing the spatial inputs
#    (ShoreXreal, ShoreYreal) = geom.mod2real(ShoreX, ShoreY, Origin, BaseShoreNormDir)
#    plt.plot(IniShoreCoords[:,0], IniShoreCoords[:,1], 'bx')
#    plt.plot(IniShoreCoords[:,0], IniShoreCoords[:,0] * Baseline[0] + Baseline[1], 'k:')
#    plt.plot(ShoreXreal, ShoreYreal, 'g.')
#    plt.plot(InflowCoord[0], InflowCoord[1],'ro')
#    plt.plot(Origin[0], Origin[1], 'go')
#    plt.axis('equal')
    
    return (FlowTs, WaveTs, SeaLevelTs, Origin, BaseShoreNormDir, ShoreX, 
            ShoreY, LagoonY, LagoonElev, RiverElev, OutletX, OutletY, 
            OutletElev, OutletWidth, TimePars, PhysicalPars, NumericalPars, OutputOpts)