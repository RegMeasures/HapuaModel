import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
#import calendar
from configobj import ConfigObj
import os
import shapefile

def loadModel(ModelConfigFile):
    
    # Read the main config file
    ConfigFilePath = os.path.split(ModelConfigFile)[0]
    config = ConfigObj(ModelConfigFile)
    
    # Read the boundary condition timeseries
    FlowFile = os.path.join(ConfigFilePath, 
                            config['Boundary conditions']['RiverFlow'])
    FlowTs = pd.read_csv(FlowFile, index_col=0)
    WaveFile = os.path.join(ConfigFilePath, 
                            config['Boundary conditions']['WaveConditions'])
    WaveTs = pd.read_csv(WaveFile, index_col=0)
    SeaLevelFile = os.path.join(ConfigFilePath, 
                                config['Boundary conditions']['SeaLevel'])
    SeaLevelTs = pd.read_csv(SeaLevelFile, index_col=0)
    
    # Read the initial shoreline position
    ShoreShpFile = os.path.join(ConfigFilePath, 
                                config['Spatial inputs']['Shoreline'])
    ShoreShp = shapefile.Reader(ShoreShpFile)
    # check it is a polyline and there is only one line
    assert ShoreShp.shapeType==3, 'Shoreline shapefile must be a polyline.'
    assert len(ShoreShp.shapes())==1, 'multiple polylines in Shoreline shapefile. There should only be 1.'
    # extract coordinates
    IniShoreCoords = np.asarray(ShoreShp.shape(0).points[:])
    
    # Read the river inflow location
    RiverShpFile = os.path.join(ConfigFilePath, 
                                config['Spatial inputs']['RiverLocation'])
    RiverShp = shapefile.Reader(RiverShpFile)
    # check it is a single point
    assert RiverShp.shapeType==1, 'Shoreline shapefile must be a point.'
    assert len(RiverShp.shapes())==1, 'multiple points in RiverLocation shapefile. There should only be 1.'
    InflowCoord = np.asarray(RiverShp.shape(0).points[:])
    
    # Make a straight reference baseline
    Baseline = np.polyfit(IniShoreCoords[:,0], IniShoreCoords[:,1], 1)
    
    # correct shoreline orientation: land (and river inflow) should be to left of line
        
    # Interpolate points at intervals along shoreline
    
    
    
    # Produce a map showing the spatial inputs
    plt.plot(IniShoreCoords[:,0], IniShoreCoords[:,1],'bx')
    plt.plot(IniShoreCoords[:,0], IniShoreCoords[:,0] * Baseline[0] + Baseline[1],'k:')
    plt.plot(InflowCoord[0,0], InflowCoord[0,1],'ro')
    plt.axis('equal')
    
    return (FlowTs, WaveTs, SeaLevelTs)