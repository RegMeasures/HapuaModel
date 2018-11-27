#import matplotlib.pyplot as plt
import pandas as pd
#import numpy as np
#import calendar
from configobj import ConfigObj
import os

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
    return (FlowTs, WaveTs, SeaLevelTs)