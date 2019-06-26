# -*- coding: utf-8 -*-

# import standard packages
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
    
    if not os.path.isfile(ModelConfigFile):
        logging.error('%s not found' % ModelConfigFile)
    
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
    Config['SpatialInputs']['BarrierBackshore'] = \
        os.path.join(ConfigFilePath, Config['SpatialInputs']['BarrierBackshore'])
    Config['SpatialInputs']['CliffToe'] = \
        os.path.join(ConfigFilePath, Config['SpatialInputs']['CliffToe'])
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
    
    (FlowTs, WaveTs, SeaLevelTs, Origin, ShoreNormDir, 
     ShoreX, ShoreY, LagoonElev, BarrierElev, OutletElev, 
     RiverElev, OutletEndX, OutletEndWidth, OutletEndElev,
     TimePars, PhysicalPars, NumericalPars, OutputOpts) = loadModel(Config)
    
    Parameters:
        Config (Dict): Model config file read into a dict variable by the 
            readConfig function
    
    Returns:
        FlowTs (DataFrame): Flow timeseries as single column DataFrame with 
            datetime index. River flow given in column "Flow" (m^3/s)
        WaveTs (DataFrame): Wave timeseries DataFrame with datetime index. 
            Columns are:
                WavePeriod (float64): Wave period (s)
                Wlen_h (float64): Wavelength (m)
                WavePower (float64): Wave power (W/m wave crest length)
                EDir_h (float64): Net direction of wave energy at depth h 
                    (where h is given as PhysicalParameter WaveDataDepth). 
                    EAngle_h is read as a bearing in degrees, but is 
                    converted to a direction relative to BaseShoreNormDir in 
                    radians as part of the pre-processing (radians)
                Hsig_Offshore (float64): Offshiore significant wave height (m)
        SeaLevelTs (DataFrame): Sea level timeseries as single column DataFrame
            with datetime index. Column name is "SeaLevel" (m)
        Origin (np.ndarray(float64)): Real world X and Y coordinates of the
            origin of the model coordinate system (m)
        ShoreNormDir (float): Direction of offshore pointing line at 90 
            degrees to overall average coast direction. Computed based on a 
            straightline fitted thruogh the initial condition shoreline 
            position (radians).
        ShoreX (np.ndarray(float64)): positions of discretised 
            shoreline in model coordinate system (m)
        ShoreY (np.ndarray(float64)): position of aspects of cross-shore 
            profile in model coordinate system at transects with x-coordinates 
            given by ShoreX (m). The columns of ShoreY represent: 
                0: Shoreline
                1: Seaward side of outlet channel (nan if no outlet at profile)
                2: Lagoonward edge of outlet channel (or nan)
                3: Seaward edge of lagoon (nan if beyond lagoon extent)
                4: Cliff toe position
        LagoonElev (np.ndarray(float64)): elevation of lagoon bed at positions 
            given by ShoreX (m)
        BarrierElev (np.ndarray(float64)): elevation of barrier crest at 
            positions given by ShoreX (m)
        OutletElev (np.ndarray(float64)): elevation of outlet channel bed at 
            positions given by ShoreX (nan where no outlet channel, m)
        RiverElev (np.ndarray(float64)): elevation of river bed cross-sections 
            upstream of lagoon (m)
        OutletEndX (np.ndarray(float64)): X-coordinate of upstream and 
            downstream ends of the outlet channel (m) 
        OutletEndWidth (np.ndarray(float64)): Width of the upstream and 
            downstream ends of the outlet channel (m)
        OutletEndElev (np.ndarray(float64)): Bed level of the upstream and 
            downstream ends of the outlet channel (m)
        TimePars (dict): Time parameters including:
            StartTime (pd.datetime): Start date/time of simulation
            EndTime (pd.datetime): End date/time of simulation period
            HydDt (pd.Timedelta): Hydrodynamic timestep
            MorDt (pd.Timedelta): Morphological timestep
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
            BeachSlope (float): Beachface slope for runup calculation (m/m)
            RiverSlope (float): Initial condition slope for river upstream of 
                hapua (m/m)
            GrainSize (float): Sediment size for fluvial sediment transport 
                calculation (assumed uniform) (m) 
            UpstreamLength (float): Length of river upstream of hapua to 
                include in model (m)
            RiverWidth (float): width of river upstream of hapua (assumed 
                uniform) (m)
            Roughness (float): Manning's 'n' for river hydraulics (m^(1/3)/s) 
            WidthRatio (float): Ratio of channel width to depth for eroding 
                river channel
            BackshoreElev (float): Elevation of lagoon backshore (m)
            K2coef (float): Calculated from other inputs for use in calculation
                of longshore transport rate. K2 = K / (RhoSed - RhoSea) * g * (1 - VoidRatio))
            BreakerCoef (float): Calculated from other inputs for use in 
                calculation of depth of breaking waves. 
                BreakerCoef = 8 / (RhoSea * Gravity^1.5 * GammaRatio^2)
        NumericalPars (dict):
            Dx (float): Discretisation interval for shoreline and river (m)
            Beta (float): Momentum (Boussinesq) coefficient
            Theta (float): Temporal weighting factor for implicit Preissmann 
                scheme 
            ErrTol (float): Error tolerance for depth and velocity in implicit 
                solution to unsteady river hydraulics (m and m/s)
            MaxIt (int): Maximum iterations for implicit solution to unsteady 
                river hydraulics
            WarnTol (float): Warning tolerance for change in water level/
                velocity within a single iteration of the hydrodynamic 
                solution (m and m/s)
        OutputOpts (dict):
            OutFile (string): filename for writing model outputs to 
                (netCDF file)
            LogInt (pd.Timedelta): 
            PlotInt (pd.Timedelta): 
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
    
    # Read the barrier backshore
    logging.info('Reading barrier backshore position from from "%s"' %
                 Config['SpatialInputs']['BarrierBackshore'])
    LagoonShp = shapefile.Reader(Config['SpatialInputs']['BarrierBackshore'])
    # check it is a polyline and there is only one
    assert LagoonShp.shapeType==3, 'BarrierBackshore must be a polyline shapefile'
    assert len(LagoonShp.shapes())==1, 'multiple polygons in BarrierBackshore shapefile. There should only be 1.'
    LagoonCoords = np.asarray(LagoonShp.shape(0).points[:])
    
    # Read the cliff toe position
    logging.info('Reading cliff toe position from from "%s"' %
                 Config['SpatialInputs']['CliffToe'])
    CliffShp = shapefile.Reader(Config['SpatialInputs']['CliffToe'])
    # check it is a polyline and there is only one
    assert CliffShp.shapeType==3, 'CliffToe must be a polyline shapefile'
    assert len(CliffShp.shapes())==1, 'multiple polygons in CliffToe shapefile. There should only be 1.'
    CliffCoords = np.asarray(CliffShp.shape(0).points[:])
    
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
        ShoreNormDir = math.atan2(Baseline[0],-1)
    else:
        # land is to south of baseline
        ShoreNormDir = math.atan2(-Baseline[0],1)
    
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
    WaveTs.EDir_h = np.deg2rad(WaveTs.EDir_h) - (ShoreNormDir)
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
               'LagoonBed': float(Config['InitialConditions']['LagoonBed']),
               'BarrierElev': float(Config['InitialConditions']['BarrierElev'])}
    
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
                    'BeachSlope': float(Config['PhysicalParameters']['ClosureDepth']),
                    'RiverSlope': float(Config['PhysicalParameters']['RiverSlope']),
                    'GrainSize': float(Config['PhysicalParameters']['GrainSize']),
                    'UpstreamLength': float(Config['PhysicalParameters']['UpstreamLength']),
                    'RiverWidth': float(Config['PhysicalParameters']['RiverWidth']),
                    'Roughness': float(Config['PhysicalParameters']['RoughnessManning']),
                    'WidthRatio': float(Config['PhysicalParameters']['WidthDepthRatio']),
                    'BackshoreElev': float(Config['PhysicalParameters']['BackshoreElev'])}

    GammaLST = ((PhysicalPars['RhoSed'] - PhysicalPars['RhoSea']) * 
                PhysicalPars['Gravity'] * (1 - PhysicalPars['VoidRatio']))
    
    PhysicalPars['K2coef'] = PhysicalPars['Kcoef'] / GammaLST
    PhysicalPars['BreakerCoef'] = 8.0 / (PhysicalPars['RhoSea'] *
                                         PhysicalPars['Gravity']**1.5 *
                                         PhysicalPars['GammaRatio']**2.0)
    
    #%% Read numerical parameters
    Dx = float(Config['NumericalParameters']['AlongShoreDx'])
    NumericalPars = {'Dx': Dx,
                     'Beta': float(Config['NumericalParameters']['Beta']),
                     'Theta': float(Config['NumericalParameters']['Theta']),
                     'FrRelax1': float(Config['NumericalParameters']['FrRelax1']),
                     'FrRelax2': float(Config['NumericalParameters']['FrRelax2']),
                     'ErrTol': float(Config['NumericalParameters']['ErrTol']),
                     'MaxIt': int(Config['NumericalParameters']['MaxIt']),
                     'WarnTol': float(Config['NumericalParameters']['WarnTol'])}

    #%% Read output options
    OutputOpts = {'OutFile': Config['OutputOptions']['OutFile'],
                  'LogInt': pd.Timedelta(seconds=float(Config['OutputOptions']['LogInt'])),
                  'PlotInt': pd.Timedelta(seconds=float(Config['OutputOptions']['PlotInt']))}
    
    #%% Initialise shoreline variables
    
    # Convert shoreline into model coordinate system
    IniShoreCoords2 = np.empty(IniShoreCoords.shape)
    (IniShoreCoords2[:,0], IniShoreCoords2[:,1]) = geom.real2mod(IniShoreCoords[:,0], IniShoreCoords[:,1], Origin, ShoreNormDir)
    if IniShoreCoords2[0,0] > IniShoreCoords2[-1,0]:
        IniShoreCoords2 = IniShoreCoords2 = np.flipud(IniShoreCoords2)
    assert np.all(np.diff(IniShoreCoords2[:,0]) > 0), 'Shoreline includes recurvature???'
    
    # Discretise shoreline at fixed intervals in model coordinate system
    ShoreX = np.arange(math.ceil(IniShoreCoords2[0,0]/Dx)*Dx, 
                       IniShoreCoords2[-1,0], Dx)
    ShoreY = np.full([ShoreX.size, 5], np.nan)
    ShoreY[:,0] = np.interp(ShoreX, IniShoreCoords2[:,0], IniShoreCoords2[:,1])
    
    #%% Initialise lagoon and outlet channel variables
    
    # Convert input spatial data into model coordinate system
    LagoonCoords2 = np.empty(LagoonCoords.shape)
    (LagoonCoords2[:,0], LagoonCoords2[:,1]) = geom.real2mod(LagoonCoords[:,0], LagoonCoords[:,1], Origin, ShoreNormDir)
    CliffCoords2 = np.empty(CliffCoords.shape)
    (CliffCoords2[:,0], CliffCoords2[:,1]) = geom.real2mod(CliffCoords[:,0], CliffCoords[:,1], Origin, ShoreNormDir)
    OutletCoords2 = np.empty(OutletCoords.shape)
    (OutletCoords2[:,0], OutletCoords2[:,1]) = geom.real2mod(OutletCoords[:,0], OutletCoords[:,1], Origin, ShoreNormDir)
    
    # Check lagoon extent is within limits of shoreline
    LagoonExtent = [np.amin(LagoonCoords2[:,0]), np.amax(LagoonCoords2[:,0])]
    assert LagoonExtent[1] < ShoreX[-1] and LagoonExtent[0] > ShoreX[0], 'Lagoon exdends beyond shoreline extent. Extend shoreline shapefile to cover full extent of lagoon.'
    
    OutletExtent = [np.amin(OutletCoords2[:,0]), np.amax(OutletCoords2[:,0])]
    
    # Check outlet coords ordered from lagoon to sea
    if OutletCoords2[0,1] > OutletCoords2[-1,1]:
        OutletCoords2 = np.flipud(OutletCoords2)
    
    # Discretise lagoon and outlet channel
    for ii in range(ShoreX.size):
        # find cliff position on transect
        YIntersects = geom.intersectPolyline(CliffCoords2, ShoreX[ii])
        ShoreY[ii,4] = np.amax(YIntersects)
        
        # find seaward edge of lagoon
        if LagoonExtent[0] < ShoreX[ii] < LagoonExtent[1]:
            #  There is some lagoon at current transect
            YIntersects = geom.intersectPolyline(LagoonCoords2, ShoreX[ii])
            ShoreY[ii,3] = np.amax(YIntersects)
        else:
            # No lagoon so seaward edge of lagoon = cliff-line
            ShoreY[ii,3] = ShoreY[ii,4]
            
        # find outlet channel (if there is outlet channel at current transect)    
        if OutletExtent[0] < ShoreX[ii] < OutletExtent[1]:
            YIntersects = geom.intersectPolyline(OutletCoords2, ShoreX[ii])
            # check there is no crazy recurved outlet channel!
            assert YIntersects.size == 1, 'Check/simplify outlet channel - possible weird recurvature?'
            # only insert outlet channel if it fits within barrier
            if ((np.isnan(ShoreY[ii,3]) or 
                ((YIntersects[0] - IniCond['OutletWidth']/2) > ShoreY[ii,3])) and
               ((YIntersects[0] + IniCond['OutletWidth']/2) < ShoreY[ii,0])):
                ShoreY[ii,2] = YIntersects[0] - IniCond['OutletWidth']/2
                ShoreY[ii,1] = YIntersects[0] + IniCond['OutletWidth']/2
    
    # Get outlet angle direction
    OutletToR = OutletCoords2[0,0] < OutletCoords2[-1,0]
    OutletMask = np.logical_not(np.isnan(ShoreY[:,1]))
    
    # Set outlet end coordinates (neatly in-between transects to start with!)
    OutletEndX = np.empty([2])
    if np.all(np.logical_not(OutletMask)):
        # Handle special case that outlet is straight out (or super wide) and didn't intersect any transects
        OutletEndX[0] = np.mean(OutletCoords2[:,1])
        OutletEndX[1] = OutletEndX[0]
        OutletIx = np.where(np.logical_and(OutletEndX[1]-Dx < ShoreX,
                                           ShoreX < OutletEndX[0]))[0][0]
        ShoreY[OutletIx,1] = (ShoreY[OutletIx,0] + ShoreY[OutletIx,3])/2 + IniCond['OutletWidth']/2
        ShoreY[OutletIx,2] = ShoreY[OutletIx,1] - IniCond['OutletWidth']
    else:
        OutletEndX[0] = np.max(ShoreX[OutletMask]) + Dx/2
        OutletEndX[1] = np.min(ShoreX[OutletMask]) - Dx/2
        if OutletToR:
            OutletEndX = np.flipud(OutletEndX)
            
    # Set outlet end width
    OutletEndWidth = np.full(2, IniCond['OutletWidth'])
    
    # Initialise lagoon bed elevation
    LagoonElev = np.full(ShoreX.size, IniCond['LagoonBed'])
    LagoonElev[np.isnan(ShoreY[:,3])] = np.nan
    
    # Initialise outlet channel bed elevation
    OutletElev = np.full(ShoreX.size, np.nan)
    BedLevel = np.linspace(IniCond['LagoonBed'], IniCond['OutletBed'], np.sum(OutletMask)+2)
    if OutletToR:
        OutletElev[OutletMask] = BedLevel[1:-1]
    else:
        OutletElev[OutletMask] = np.flipud(BedLevel[1:-1])
    OutletEndElev = BedLevel[[0,-1]]
    
    # Initialise barrier crest elevation
    BarrierElev = np.full(ShoreX.size, IniCond['BarrierElev'])
    
    #%% Initialising river variables
    RiverElev = np.flipud(np.arange(IniCond['LagoonBed'],
                                    IniCond['LagoonBed']
                                    + PhysicalPars['RiverSlope']
                                    * PhysicalPars['UpstreamLength'],
                                    PhysicalPars['RiverSlope'] * Dx))
      
    # Produce a map showing the spatial inputs
#    (ShoreXreal, ShoreYreal) = geom.mod2real(ShoreX, ShoreY, Origin, BaseShoreNormDir)
#    plt.plot(IniShoreCoords[:,0], IniShoreCoords[:,1], 'bx')
#    plt.plot(IniShoreCoords[:,0], IniShoreCoords[:,0] * Baseline[0] + Baseline[1], 'k:')
#    plt.plot(ShoreXreal, ShoreYreal, 'g.')
#    plt.plot(InflowCoord[0], InflowCoord[1],'ro')
#    plt.plot(Origin[0], Origin[1], 'go')
#    plt.axis('equal')
    
    return (FlowTs, WaveTs, SeaLevelTs, Origin, ShoreNormDir, 
            ShoreX, ShoreY, LagoonElev, BarrierElev, OutletElev, 
            RiverElev, OutletEndX, OutletEndWidth, OutletEndElev,
            TimePars, PhysicalPars, NumericalPars, OutputOpts)