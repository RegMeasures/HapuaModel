import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math
#import calendar
from configobj import ConfigObj
import os
import shapefile

def loadModel(ModelConfigFile):
    
    # Read the main config file
    ConfigFilePath = os.path.split(ModelConfigFile)[0]
    Config = ConfigObj(ModelConfigFile)
    
    # Read the boundary condition timeseries
    FlowFile = os.path.join(ConfigFilePath, 
                            Config['BoundaryConditions']['RiverFlow'])
    FlowTs = pd.read_csv(FlowFile, index_col=0)
    WaveFile = os.path.join(ConfigFilePath, 
                            Config['BoundaryConditions']['WaveConditions'])
    WaveTs = pd.read_csv(WaveFile, index_col=0)
    SeaLevelFile = os.path.join(ConfigFilePath, 
                                Config['BoundaryConditions']['SeaLevel'])
    SeaLevelTs = pd.read_csv(SeaLevelFile, index_col=0)
    
    # Read the initial shoreline position
    ShoreShpFile = os.path.join(ConfigFilePath, 
                                Config['SpatialInputs']['Shoreline'])
    ShoreShp = shapefile.Reader(ShoreShpFile)
    # check it is a polyline and there is only one line
    assert ShoreShp.shapeType==3, 'Shoreline shapefile must be a polyline.'
    assert len(ShoreShp.shapes())==1, 'multiple polylines in Shoreline shapefile. There should only be 1.'
    # extract coordinates
    IniShoreCoords = np.asarray(ShoreShp.shape(0).points[:])
    
    # Read the river inflow location
    RiverShpFile = os.path.join(ConfigFilePath, 
                                Config['SpatialInputs']['RiverLocation'])
    RiverShp = shapefile.Reader(RiverShpFile)
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
    (IniShoreCoords2[:,0], IniShoreCoords2[:,1]) = real2mod(IniShoreCoords[:,0], IniShoreCoords[:,1], Origin, ShoreNormalDir)
    if IniShoreCoords2[0,0] > IniShoreCoords2[-1,0]:
        IniShoreCoords2 = IniShoreCoords2 = np.flipud(IniShoreCoords2)
    assert np.all(np.diff(IniShoreCoords2[:,0]) > 0), 'Shoreline includes recurvature???'
    
    # Discretise shoreline at fixed intervals in model coordinate system
    Dx = float(Config['ModelSettings']['AlongShoreDx'])
    ShoreX = np.arange(math.ceil(IniShoreCoords2[0,0]/Dx)*Dx, 
                       IniShoreCoords2[-1,0], Dx)
    ShoreY = np.interp(ShoreX, IniShoreCoords2[:,0], IniShoreCoords2[:,1])
    
    # Produce a map showing the spatial inputs
    (ShoreXreal, ShoreYreal) = mod2real(ShoreX, ShoreY, Origin, ShoreNormalDir)
    plt.plot(IniShoreCoords[:,0], IniShoreCoords[:,1], 'bx')
    plt.plot(IniShoreCoords[:,0], IniShoreCoords[:,0] * Baseline[0] + Baseline[1], 'k:')
    plt.plot(ShoreXreal, ShoreYreal, 'g.')
    plt.plot(InflowCoord[0], InflowCoord[1],'ro')
    plt.plot(Origin[0], Origin[1], 'go')
    plt.axis('equal')
    
    return (FlowTs, WaveTs, SeaLevelTs, Origin, ShoreNormalDir, ShoreX, ShoreY)

# functions for converting between model and real world coordinate systems
def mod2real(Xmod, Ymod, Origin, ShoreNormalDir):
    Xreal = (Origin[0] + 
             Xmod * np.sin(ShoreNormalDir+np.pi/2) - 
             Ymod * np.cos(ShoreNormalDir+np.pi/2))
    Yreal = (Origin[1] + 
             Xmod * np.cos(ShoreNormalDir+np.pi/2) + 
             Ymod * np.sin(ShoreNormalDir+np.pi/2))
    return (Xreal, Yreal)

def real2mod(Xreal, Yreal, Origin, ShoreNormalDir):
    Xrelative = Xreal - Origin[0]
    Yrelative = Yreal - Origin[1]
    Dist = np.sqrt(Xrelative**2 + Yrelative**2)
    Dir = np.arctan2(Xrelative, Yrelative)
    Xmod = Dist * np.cos(ShoreNormalDir - Dir + np.pi/2)
    Ymod = Dist * np.sin(ShoreNormalDir - Dir + np.pi/2)
    return (Xmod, Ymod)