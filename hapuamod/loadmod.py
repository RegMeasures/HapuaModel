# -*- coding: utf-8 -*-

# import standard packages
import pandas as pd
import numpy as np
import math
from configobj import ConfigObj, flatten_errors
from validate import Validator
import os
import shapefile
import logging
import netCDF4
import pkg_resources
import errno

# import local packages
from hapuamod import geom
from hapuamod import out
from hapuamod import synth

class ConfigError(Exception):
    """Exception raised during model config file validation.
    
    Attributes:
        message -- explanation of the error
    """
    
    def __init__(self, message):
        self.message = message

def loadModel(ModelConfigFile):
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
        ModelConfigFile (string): Filename (and path if required) of a valid 
            hapuamod model config file
    
    Returns:
        ModelName (str): Model name specified in config file
        FlowTs (DataFrame): Flow timeseries as single column DataFrame with 
            datetime index. River flow given in column "Flow" (m^3/s)
        WaveTs (DataFrame): Wave timeseries DataFrame with datetime index. 
            Columns are:
                WavePeriod (float64): Wave period (s)
                Wlen_h (float64): Wavelength (m)
                WavePower (float64): Wave power (W/m wave crest length)
                EDir_h (float64): Net direction wave energy arrives FROM at 
                    depth h (where h is given as PhysicalParameter 
                    WaveDataDepth), relative to shore-normal (+ve = arriving 
                    from right of shore normal, -ve = arriving from left of 
                    shore nornmal). EAngle_h is read as a bearing in degrees, 
                    but is converted to a direction relative to 
                    BaseShoreNormDir in radians as part of the pre-processing 
                    (radians)
                Hsig_Offshore (float64): Offshiore significant wave height (m)
        SeaLevelTs (DataFrame): Sea level timeseries as single column DataFrame
            with datetime index. Column name is "SeaLevel" (m)
        Origin (np.ndarray(float64)): Real world X and Y coordinates of the
            origin of the model coordinate system (m)
        ShoreNormDir (float): Direction of offshore pointing line at 90 
            degrees to overall average coast direction. Computed based on a 
            straightline fitted thruogh the initial condition shoreline 
            position (radians, clockwise from North).
        ShoreX (np.ndarray(float64)): positions of discretised 
            shoreline in model coordinate system (m)
        ShoreY (np.ndarray(float64)): position of aspects of cross-shore 
            profile in model coordinate system at transects with x-coordinates 
            given by ShoreX (m). The columns of ShoreY represent: 
                0: Shoreline
                1: Seaward side of outlet channel (nan if no outlet at profile)
                2: Lagoonward edge of outlet channel (or nan)
                3: Seaward edge of lagoon (same as cliff if beyond lagoon extent)
                4: Cliff toe position
        ShoreZ (np.ndarray(float64)): elevation of aspects of cross-shore 
            profile at transects with x-coordinates given by ShoreX (m). The 
            columns of ShoreZ represent:
                0: Barrier crest height
                1: Outlet channel bed elevation (nan if no outlet at profile)
                2: Inner barrier crest height (nan if no outlet at profile)
                3: Lagoon bed elevation (nan if no lagoon at profile)
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
            MorDtMax (pd.Timedelta): Maximum morphological timestep
            MorDtMin (pd.Timedelta): Minimum morphological timestep
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
            MaxOutletElev (float): Maximum elevation of the downstream end of 
                the outlet channel [m] This elevation is applied downstream of 
                the last real outlet channel cross-section and should be below 
                the minimum sea level.
            OT_coef (float): Overtopping coefficient
            OT_exp (float): Overtopping exponent
            OwProp_coef (float): Overwash proportion coefficient
            MinOpForOw (float):
            BeachTopElev (float): Crest height of 'new' barrier created by 
                longshore transport across river mouth. also used as the 
                elevation of the transition from sloped beach to vertical 
                barrier front when plotting [m]
            SpitWidth (float): Crest width of 'new' barrier created by 
                longshore transport across river mouth [m]
            MinOutletWidth (float): Threshold width to trigger outlet closure
            K2coef (float): Calculated from other inputs for use in calculation
                of longshore transport rate. 
                K2 = K / (RhoSed - RhoSea) * g * (1 - VoidRatio))
            BreakerCoef (float): Calculated from other inputs for use in 
                calculation of depth of breaking waves. 
                BreakerCoef = 8 / (RhoSea * Gravity^1.5 * GammaRatio^2)
        NumericalPars (dict):
            Dx (float): Discretisation interval for shoreline and river (m)
            Beta (float): Momentum (Boussinesq) coefficient
            Theta (float): Temporal weighting factor for implicit Preissmann 
                scheme 
            Psi (float): Spatial weighting factor for bedload transport
            ErrTol (float): Error tolerance for depth and velocity in implicit 
                solution to unsteady river hydraulics (m and m/s)
            MaxIt (int): Maximum iterations for implicit solution to unsteady 
                river hydraulics
            WarnTol (float): Warning tolerance for change in water level/
                velocity within a single iteration of the hydrodynamic 
                solution (m and m/s)
            MaxMorChange (float): Maximum morphological change per timestep 
                (used for adaptive timestepping) (m)
        OutputOpts (dict):
            OutFile (string): filename for writing model outputs to 
                (netCDF file)
            OutInt (pd.Timedelta): output interval for writing to netCDF file
            LogInt (pd.Timedelta): interval for writing progress to console/log
            PlotInt (pd.Timedelta): interval for plotting (0 = no plotting)
    """
    
    #%% Read and validate the main config file
    
    # Check the specified model config file exists
    if not os.path.isfile(ModelConfigFile):
        logging.error('%s not found' % ModelConfigFile)
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), 
                                ModelConfigFile)
    
    # Get the absolute file path to the config spec file 'configspec.cnf'
    ConfigSpecFile = pkg_resources.resource_filename(__name__, 'configspec.cnf')
    
    # Read the model config file
    logging.info('Loading "%s"' % ModelConfigFile)
    Config = ConfigObj(ModelConfigFile, configspec=ConfigSpecFile)
    
    # Do basic validation and type conversion
    validator = Validator()
    ValidationResult = Config.validate(validator)
    if ValidationResult != True:
        for (section_list, key, _) in flatten_errors(Config, ValidationResult):
            if key is not None:
                raise ConfigError('The "%s" key in the section "%s" failed validation' %
                                  (key, ', '.join(section_list)))
            else:
                raise ConfigError('The following section was missing:%s ' %
                                  ', '.join(section_list))
    
    ModelName = Config['ModelName']
    
    # Extract file path ready to pre-pend to relative file paths as required
    (ConfigFilePath, ConfigFileName) = os.path.split(ModelConfigFile)  
    
    #%% Time inputs
    logging.info('Reading time inputs')
    TimePars = {'StartTime': pd.to_datetime(Config['Time']['StartTime']),
                'EndTime': pd.to_datetime(Config['Time']['EndTime']),
                'HydDt': pd.Timedelta(seconds=Config['Time']['HydDt']),
                'MorDtMin': pd.Timedelta(seconds=Config['Time']['MorDtMin']),
                'MorDtMax': pd.Timedelta(seconds=Config['Time']['MorDtMax'])}
    # TODO: check mortime is a multiple of hydtime (or replace with morscaling?)
    
    #%% Read physical parameters
    logging.info('Processing physical parameters')
    PhysicalPars = {'RhoSed': Config['PhysicalParameters']['RhoSed'],
                    'RhoSea': Config['PhysicalParameters']['RhoSea'],
                    'RhoRiv': Config['PhysicalParameters']['RhoSea'],
                    'Kcoef': Config['PhysicalParameters']['Kcoef'],
                    'Gravity': Config['PhysicalParameters']['Gravity'],
                    'VoidRatio': Config['PhysicalParameters']['VoidRatio'],
                    'GammaRatio': Config['PhysicalParameters']['GammaRatio'],
                    'WaveDataDepth': Config['PhysicalParameters']['WaveDataDepth'],
                    'ClosureDepth': Config['PhysicalParameters']['ClosureDepth'],
                    'BeachSlope': Config['PhysicalParameters']['BeachSlope'],
                    'RiverSlope': Config['PhysicalParameters']['RiverSlope'],
                    'GrainSize': Config['PhysicalParameters']['GrainSize'],
                    'CritShieldsStress': Config['PhysicalParameters']['CritShieldsStress'],
                    'UpstreamLength': Config['PhysicalParameters']['UpstreamLength'],
                    'RiverWidth': Config['PhysicalParameters']['RiverWidth'],
                    'Roughness': Config['PhysicalParameters']['RoughnessManning'],
                    'WidthRatio': Config['PhysicalParameters']['WidthDepthRatio'],
                    'BackshoreElev': Config['PhysicalParameters']['BackshoreElev'],
                    'MaxOutletElev': Config['PhysicalParameters']['MaxOutletElev'],
                    'OT_coef': Config['PhysicalParameters']['OT_coef'],
                    'OT_exp': Config['PhysicalParameters']['OT_exp'],
                    'OwProp_coef': Config['PhysicalParameters']['OwProp_coef'],
                    'MinOpForOw': Config['PhysicalParameters']['MinOpForOw'],
                    'BeachTopElev': Config['PhysicalParameters']['BeachTopElev'],
                    'SpitWidth': Config['PhysicalParameters']['SpitWidth'],
                    'MinOutletWidth': Config['PhysicalParameters']['MinOutletWidth'],
                    'OutletSedSpreadDist': Config['PhysicalParameters']['OutletSedSpreadDist']}

    GammaLST = ((PhysicalPars['RhoSed'] - PhysicalPars['RhoSea']) * 
                PhysicalPars['Gravity'] * (1 - PhysicalPars['VoidRatio']))
    
    PhysicalPars['K2coef'] = PhysicalPars['Kcoef'] / GammaLST
    PhysicalPars['BreakerCoef'] = 8.0 / (PhysicalPars['RhoSea'] *
                                         PhysicalPars['Gravity']**1.5 *
                                         PhysicalPars['GammaRatio']**2.0)
    
    #%% Read numerical parameters
    Dx = Config['NumericalParameters']['AlongShoreDx']
    NumericalPars = {'Dx': Dx,
                     'Beta': Config['NumericalParameters']['Beta'],
                     'Theta': Config['NumericalParameters']['Theta'],
                     'Psi': Config['NumericalParameters']['Psi'],
                     'ErrTol': Config['NumericalParameters']['ErrTol'],
                     'MaxIt': Config['NumericalParameters']['MaxIt'],
                     'WarnTol': Config['NumericalParameters']['WarnTol'],
                     'MaxMorChange': Config['NumericalParameters']['MaxMorChange']}
    
    #%% Read initial conditions
    if Config['HotStart']['InitialConditionsNetCDF'] is None:
        logging.info('Reading initial conditions')
        IniCond = {'OutletWidth': Config['InitialConditions']['OutletWidth'],
                   'OutletBed': Config['InitialConditions']['OutletBed'],
                   'LagoonBed': Config['InitialConditions']['LagoonBed'],
                   'BarrierElev': Config['InitialConditions']['BarrierElev']}
        
        #%% Initialise river variables
        RiverElev = np.flipud(np.arange(IniCond['LagoonBed'],
                                        IniCond['LagoonBed']
                                        + PhysicalPars['RiverSlope']
                                        * PhysicalPars['UpstreamLength'],
                                        PhysicalPars['RiverSlope'] * Dx))
    
    #%% Read/initialise spatial inputs
    if Config['HotStart']['InitialConditionsNetCDF'] is not None:
        #%% Read spatial inputs from hotstart file
        Config['HotStart']['InitialConditionsNetCDF'] = \
                os.path.join(ConfigFilePath, Config['HotStart']['InitialConditionsNetCDF'])
        logging.info('Hotstarted simulation: Spatial inputs and initial conditions being read from %s' %
                     Config['HotStart']['InitialConditionsNetCDF'])
        NcFile = netCDF4.Dataset(Config['HotStart']['InitialConditionsNetCDF'], 
                                 mode='r', format='NETCDF4_CLASSIC') 
        
        DatetimeOfInterest = pd.to_datetime(Config['HotStart']['HotStartTime'])
        TimeIx = out.closestTimeIx(NcFile, DatetimeOfInterest)
        logging.info('Closest time available in netCDF file to desired hotstart time of %s is %s' % 
                     (DatetimeOfInterest.strftime("%b %d %Y %H:%M:%S"), 
                      netCDF4.num2date(NcFile.variables['time'][TimeIx][0], NcFile.variables['time'].units).strftime("%b %d %Y %H:%M:%S")))
        
        (SeaLevel, ShoreX, ShoreY, ShoreZ, LagoonWL, LagoonVel, OutletWL, 
         OutletVel, OutletEndX, OutletEndWidth, OutletEndElev, OutletEndVel, 
         OutletEndWL, OutletChanIx, WavePower, EDir_h, LST, CST, Closed, 
         RiverElev, RiverWL, RiverVel, ModelTime) = out.readTimestep(NcFile, TimeIx)
        
        Origin = np.array([NcFile.ModelOriginX, NcFile.ModelOriginY])
        ShoreNormDir = np.deg2rad(NcFile.ModelOrientation)
        assert ShoreX[1] - ShoreX[0] == Dx, 'Dx specified in model config file must match hotstart file'
        
    elif Config['SpatialInputs']['Shoreline'] is None:
        #%% Initialise model with default straight shoreline etc
        logging.info('No shoreline location provided - Initialising with simple straight shoreline.')
        assert Config['SpatialInputs']['ShorelineLengthLeft'] is not None, 'ShorelineLengthLeft is required in [SpatialInputs]'
        assert Config['SpatialInputs']['ShorelineLengthRight'] is not None, 'ShorelineLengthRight is required in [SpatialInputs]'
        assert Config['SpatialInputs']['BeachWidth'] is not None, 'BeachWidth is required in [SpatialInputs]'
        
        # No transformation from real world to model coordinates
        Origin = np.zeros(2)
        ShoreNormDir = 0.0
        
        # Create key shore parrallel variables
        ShoreX = np.arange(math.floor(-Config['SpatialInputs']['ShorelineLengthLeft']/Dx)*Dx, 
                           Config['SpatialInputs']['ShorelineLengthRight'] + Dx, Dx)
        ShoreY = np.full([ShoreX.size, 5], np.nan)
        ShoreZ = np.full([ShoreX.size, 4], np.nan)
        
        # Setup basic straight shoreline
        ShoreY[:, 0] = 0.0
        ShoreY[:, 3] = -Config['SpatialInputs']['BeachWidth']
        ShoreY[:, 4] = -Config['SpatialInputs']['BeachWidth']
        
        ShoreZ[:, 0] = IniCond['BarrierElev']
        
        # Setup outlet channel
        OutletEndX = np.asarray([0., 0.])
        OutletEndWidth = np.full(2, IniCond['OutletWidth'])
        OutletEndElev = np.asarray([IniCond['LagoonBed'], IniCond['OutletBed']])
        
        LagoonMask = ShoreX == 0
        ShoreY[LagoonMask, 1] = -PhysicalPars['SpitWidth']
        ShoreY[LagoonMask, 2] = ShoreY[LagoonMask,1] - IniCond['OutletWidth']
        
        ShoreZ[LagoonMask, 1] = (IniCond['OutletBed'] + IniCond['LagoonBed'])/2
        ShoreZ[LagoonMask, 2] = ShoreZ[LagoonMask, 0]
        
        # Setup lagoon
        ShoreY[LagoonMask, 3] = ShoreY[LagoonMask, 2] - 0.001
        ShoreY[LagoonMask, 4] = ShoreY[LagoonMask, 2] - PhysicalPars['RiverWidth']
        ShoreZ[:, 3] = IniCond['LagoonBed']
        
    else:
        #%% Initialise model from supplied real world spatial data
        
        # Read the initial shoreline position
        Config['SpatialInputs']['Shoreline'] = \
                os.path.join(ConfigFilePath, Config['SpatialInputs']['Shoreline'])
        logging.info('Reading initial shoreline position from "%s"' %
                     Config['SpatialInputs']['Shoreline'])
        ShoreShp = shapefile.Reader(Config['SpatialInputs']['Shoreline'])
        # check it is a polyline and there is only one line
        assert ShoreShp.shapeType==3, 'Shoreline shapefile must be a polyline.'
        assert len(ShoreShp.shapes())==1, 'multiple polylines in Shoreline shapefile. There should only be 1.'
        # extract coordinates
        IniShoreCoords = np.asarray(ShoreShp.shape(0).points[:])
        
        # Read the river inflow location
        Config['SpatialInputs']['RiverLocation'] = \
                os.path.join(ConfigFilePath, Config['SpatialInputs']['RiverLocation'])
        logging.info('Reading river inflow location from "%s"' %
                     Config['SpatialInputs']['RiverLocation'])
        RiverShp = shapefile.Reader(Config['SpatialInputs']['RiverLocation'])
        # check it is a single point
        assert RiverShp.shapeType==1, 'Shoreline shapefile must be a point.'
        assert len(RiverShp.shapes())==1, 'multiple points in RiverLocation shapefile. There should only be 1.'
        InflowCoord = np.asarray(RiverShp.shape(0).points[:]).squeeze()
        
        # Read the barrier backshore
        Config['SpatialInputs']['BarrierBackshore'] = \
                os.path.join(ConfigFilePath, Config['SpatialInputs']['BarrierBackshore'])
        logging.info('Reading barrier backshore position from from "%s"' %
                     Config['SpatialInputs']['BarrierBackshore'])
        LagoonShp = shapefile.Reader(Config['SpatialInputs']['BarrierBackshore'])
        # check it is a polyline and there is only one
        assert LagoonShp.shapeType==3, 'BarrierBackshore must be a polyline shapefile'
        assert len(LagoonShp.shapes())==1, 'multiple polygons in BarrierBackshore shapefile. There should only be 1.'
        LagoonCoords = np.asarray(LagoonShp.shape(0).points[:])
        
        # Read the cliff toe position
        Config['SpatialInputs']['CliffToe'] = \
                os.path.join(ConfigFilePath, Config['SpatialInputs']['CliffToe'])
        logging.info('Reading cliff toe position from from "%s"' %
                     Config['SpatialInputs']['CliffToe'])
        CliffShp = shapefile.Reader(Config['SpatialInputs']['CliffToe'])
        # check it is a polyline and there is only one
        assert CliffShp.shapeType==3, 'CliffToe must be a polyline shapefile'
        assert len(CliffShp.shapes())==1, 'multiple polygons in CliffToe shapefile. There should only be 1.'
        CliffCoords = np.asarray(CliffShp.shape(0).points[:])
        
        # Read the initial outlet position polyline
        Config['SpatialInputs']['OutletLocation'] = \
                os.path.join(ConfigFilePath, Config['SpatialInputs']['OutletLocation'])
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
            
        #%% Initialise shoreline variables

        # Convert shoreline polyline into model coordinate system
        IniShoreCoords2 = np.empty(IniShoreCoords.shape)
        (IniShoreCoords2[:,0], IniShoreCoords2[:,1]) = geom.real2mod(IniShoreCoords[:,0], IniShoreCoords[:,1], Origin, ShoreNormDir)
        if IniShoreCoords2[0,0] > IniShoreCoords2[-1,0]:
            IniShoreCoords2 = IniShoreCoords2 = np.flipud(IniShoreCoords2)
        assert np.all(np.diff(IniShoreCoords2[:,0]) > 0), 'Shoreline includes recurvature incompatible with model coordinate system'
        
        # Discretise shoreline at fixed intervals in model coordinate system
        ShoreX = np.arange(math.ceil(IniShoreCoords2[0,0]/Dx)*Dx, 
                           IniShoreCoords2[-1,0], Dx)
        ShoreY = np.full([ShoreX.size, 5], np.nan)
        ShoreZ = np.full([ShoreX.size, 4], np.nan)
        
        # Interpolate shoreline position onto model transects    
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
            # Handle special case that outlet is straight out and didn't intersect any transects
            if OutletToR:
                OutletEndX[0] = Dx * (np.mean(OutletCoords2[:,1])//Dx)
                OutletEndX[1] = OutletEndX[0] + Dx/2
            else:
                OutletEndX[0] = Dx * ((np.mean(OutletCoords2[:,1])//Dx) + 1)
                OutletEndX[1] = OutletEndX[0] - Dx/2
            
            OutletMask = ShoreX == OutletEndX[0]
            assert np.sum(OutletMask)==1, 'Number of outlet transects for straight outlet not equal to 1 in loadModel'
            ShoreY[OutletMask,1] = min(ShoreY[OutletMask,0] - PhysicalPars['SpitWidth'], 
                                       (ShoreY[OutletMask,0] + ShoreY[OutletMask,3])/2 + IniCond['OutletWidth']/2)
            ShoreY[OutletMask,2] = ShoreY[OutletMask,1] - IniCond['OutletWidth']
            ShoreY[OutletMask,3] = min(ShoreY[OutletMask,2] + 0.001, ShoreY[OutletMask,3])
        else:
            if OutletToR:
                # Outlet angles from L to R
                OutletEndX[0] = np.min(ShoreX[OutletMask])
                OutletEndX[1] = np.max(ShoreX[OutletMask]) + Dx/2
            else:
                # Outlet angles from R to L
                OutletEndX[0] = np.max(ShoreX[OutletMask])
                OutletEndX[1] = np.min(ShoreX[OutletMask]) - Dx/2
                
        # Set outlet end width
        OutletEndWidth = np.full(2, IniCond['OutletWidth'])
        
        # Initialise lagoon bed elevation
        ShoreZ[: ,3] = np.full(ShoreX.size, IniCond['LagoonBed'])
        
        # Initialise outlet channel bed elevation
        BedLevel = np.linspace(IniCond['LagoonBed'], IniCond['OutletBed'], np.sum(OutletMask)+2)
        if OutletToR:
            ShoreZ[OutletMask, 1] = BedLevel[1:-1]
        else:
            ShoreZ[OutletMask, 1] = np.flipud(BedLevel[1:-1])
        OutletEndElev = BedLevel[[0,-1]]
        
        # Initialise barrier crest elevation
        ShoreZ[:, 0] = np.full(ShoreX.size, IniCond['BarrierElev'])
        ShoreZ[OutletMask, 2] = np.full(np.sum(OutletMask), IniCond['BarrierElev'])
    
    #%% Read the boundary condition timeseries
    to_datetime = lambda d: pd.datetime.strptime(d, '%d/%m/%Y %H:%M')
    
    # Flow timeseries
    if Config['BoundaryConditions']['RiverFlow'].lower() == 'shotnoise':
        logging.info('Synthetic shot-noise flow hydrograph specified')
        SNPars = Config['BoundaryConditions']['ShotnoiseHydrographParameters']
        if SNPars['HydrographStart'] is not None:
            HydrographStart = pd.to_datetime(SNPars['HydrographStart'])
        else:
            HydrographStart = TimePars['StartTime']
        FlowTs = synth.shotNoise(HydrographStart, TimePars['EndTime'], 
                                 pd.Timedelta(minutes = SNPars['HydrographDt']),
                                 pd.Timedelta(days = SNPars['MeanDaysBetweenEvents']), 
                                 SNPars['MeanEventIncrease'], 
                                 SNPars['FastDecayRate'], SNPars['FastFlowProp'], 
                                 SNPars['SlowDecayRate'], 
                                 pd.Timedelta(days = SNPars['RisingLimbTime']), 
                                 RandomSeed = SNPars['RandomSeed'])
    else:        
        Config['BoundaryConditions']['RiverFlow'] = \
                os.path.join(ConfigFilePath, Config['BoundaryConditions']['RiverFlow'])
        logging.info('Reading flow timeseries from "%s"' % 
                     Config['BoundaryConditions']['RiverFlow'])
        FlowTs = pd.read_csv(Config['BoundaryConditions']['RiverFlow'], 
                             index_col=0, parse_dates=[0],
                             date_parser=to_datetime)
        FlowTs = FlowTs.Flow
    
    # Wave timeseries
    Config['BoundaryConditions']['WaveConditions'] = \
            os.path.join(ConfigFilePath, Config['BoundaryConditions']['WaveConditions'])
    logging.info('Reading wave timeseries from "%s"' % 
                 Config['BoundaryConditions']['WaveConditions'])
    WaveTs = pd.read_csv(Config['BoundaryConditions']['WaveConditions'], 
                         index_col=0, parse_dates=[0],
                         date_parser=to_datetime)
    # Convert wave directions into radians in model coordinate system
    WaveTs.EDir_h = np.deg2rad(WaveTs.EDir_h) - (ShoreNormDir)
    # Make sure all wave angles are in the range -pi to +pi
    WaveTs.EDir_h = np.mod(WaveTs.EDir_h + np.pi, 2.0 * np.pi) - np.pi 
    
    # Sea level timeseries
    if Config['BoundaryConditions']['SeaLevel'].lower() == 'harmonic':
        logging.info('Harmonic tidal boundary specified')
        HTPars = Config['BoundaryConditions']['HarmonicTideParameters']
        SeaLevelTs = synth.harmonicTide(TimePars['StartTime'], TimePars['EndTime'], 
                                        pd.Timedelta(minutes = HTPars['SeaLevelDt']),
                                        HTPars['MeanSeaLevel'], HTPars['TidalRange'])
    else:
        Config['BoundaryConditions']['SeaLevel'] = \
                os.path.join(ConfigFilePath, Config['BoundaryConditions']['SeaLevel'])
        logging.info('Reading sea level timeseries from "%s"' % 
                     Config['BoundaryConditions']['SeaLevel'])
        SeaLevelTs = pd.read_csv(Config['BoundaryConditions']['SeaLevel'], 
                                 index_col=0, parse_dates=[0],
                                 date_parser=to_datetime)
        SeaLevelTs = SeaLevelTs.SeaLevel
    
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
    
    #%% Read output options
    if Config['OutputOptions']['OutFile'] is None:
        OutFile = os.path.splitext(ConfigFileName)[0] + '_outputs.nc'
    else:
        OutFile = Config['OutputOptions']['OutFile']
    OutputOpts = {'OutFile': OutFile,
                  'OutInt': pd.Timedelta(seconds=Config['OutputOptions']['OutInt']),
                  'LogInt': pd.Timedelta(seconds=Config['OutputOptions']['LogInt']),
                  'PlotInt': pd.Timedelta(seconds=Config['OutputOptions']['PlotInt'])}
          
    # Produce a map showing the spatial inputs
    #(ShoreXreal, ShoreYreal) = geom.mod2real(ShoreX, ShoreY, Origin, BaseShoreNormDir)
    #plt.plot(IniShoreCoords[:,0], IniShoreCoords[:,1], 'bx')
    #plt.plot(IniShoreCoords[:,0], IniShoreCoords[:,0] * Baseline[0] + Baseline[1], 'k:')
    #plt.plot(ShoreXreal, ShoreYreal, 'g.')
    #plt.plot(InflowCoord[0], InflowCoord[1],'ro')
    #plt.plot(Origin[0], Origin[1], 'go')
    #plt.axis('equal')
    
    return (ModelName, FlowTs, WaveTs, SeaLevelTs, Origin, ShoreNormDir, 
            ShoreX, ShoreY, ShoreZ, RiverElev, 
            OutletEndX, OutletEndWidth, OutletEndElev,
            TimePars, PhysicalPars, NumericalPars, OutputOpts)