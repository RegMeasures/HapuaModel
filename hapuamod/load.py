# -*- coding: utf-8 -*-

# import standard packages
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math
from configobj import ConfigObj
import os
import shapefile

# import local packages
import hapuamod.geom as geom

def readConfig(ModelConfigFile):
    # Read the main config file
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
    
    # Add some validation here???
    
    return Config

def loadModel(Config):
    
    # Read the boundary condition timeseries
    FlowTs = pd.read_csv(Config['BoundaryConditions']['RiverFlow'], 
                         index_col=0)
    WaveTs = pd.read_csv(Config['BoundaryConditions']['WaveConditions'], 
                         index_col=0)
    SeaLevelTs = pd.read_csv(Config['BoundaryConditions']['SeaLevel'], 
                             index_col=0)
    
    # Read the initial shoreline position
    ShoreShp = shapefile.Reader(Config['SpatialInputs']['Shoreline'])
    # check it is a polyline and there is only one line
    assert ShoreShp.shapeType==3, 'Shoreline shapefile must be a polyline.'
    assert len(ShoreShp.shapes())==1, 'multiple polylines in Shoreline shapefile. There should only be 1.'
    # extract coordinates
    IniShoreCoords = np.asarray(ShoreShp.shape(0).points[:])
    
    # Read the river inflow location
    RiverShp = shapefile.Reader(Config['SpatialInputs']['RiverLocation'])
    # check it is a single point
    assert RiverShp.shapeType==1, 'Shoreline shapefile must be a point.'
    assert len(RiverShp.shapes())==1, 'multiple points in RiverLocation shapefile. There should only be 1.'
    InflowCoord = np.asarray(RiverShp.shape(0).points[:]).squeeze()
    
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
        ShoreNormalDir = math.atan2(Baseline[0],-1)
    else:
        # land is to south of baseline
        ShoreNormalDir = math.atan2(-Baseline[0],1)
        
    # Convert shoreline coords into model coordinate system
    IniShoreCoords2 = np.empty([np.size(IniShoreCoords, axis=0), 2])
    (IniShoreCoords2[:,0], IniShoreCoords2[:,1]) = geom.real2mod(IniShoreCoords[:,0], IniShoreCoords[:,1], Origin, ShoreNormalDir)
    if IniShoreCoords2[0,0] > IniShoreCoords2[-1,0]:
        IniShoreCoords2 = IniShoreCoords2 = np.flipud(IniShoreCoords2)
    assert np.all(np.diff(IniShoreCoords2[:,0]) > 0), 'Shoreline includes recurvature???'
    
    # Discretise shoreline at fixed intervals in model coordinate system
    Dx = float(Config['ModelSettings']['AlongShoreDx'])
    ShoreX = np.arange(math.ceil(IniShoreCoords2[0,0]/Dx)*Dx, 
                       IniShoreCoords2[-1,0], Dx)
    ShoreY = np.interp(ShoreX, IniShoreCoords2[:,0], IniShoreCoords2[:,1])
    
    # Produce a map showing the spatial inputs
    (ShoreXreal, ShoreYreal) = geom.mod2real(ShoreX, ShoreY, Origin, ShoreNormalDir)
    plt.plot(IniShoreCoords[:,0], IniShoreCoords[:,1], 'bx')
    plt.plot(IniShoreCoords[:,0], IniShoreCoords[:,0] * Baseline[0] + Baseline[1], 'k:')
    plt.plot(ShoreXreal, ShoreYreal, 'g.')
    plt.plot(InflowCoord[0], InflowCoord[1],'ro')
    plt.plot(Origin[0], Origin[1], 'go')
    plt.axis('equal')
    
    return (FlowTs, WaveTs, SeaLevelTs, Origin, ShoreNormalDir, ShoreX, ShoreY)