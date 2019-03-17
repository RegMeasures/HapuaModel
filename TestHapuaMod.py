import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import hapuamod as hm

# Set up logging (will be moved into main wrapper code)
try:
    RootLogger # check to see if code is being re-run to prevent dulicate outputs
except NameError:
    RootLogger = logging.getLogger()
    RootLogger.setLevel(logging.DEBUG)
    
    ConsoleHandler = logging.StreamHandler()
    ConsoleHandler.setLevel(logging.DEBUG)
    RootLogger.addHandler(ConsoleHandler)

# Test model load (particularly geometry processing)
ModelConfigFile = 'inputs\HurunuiModel.cnf'
Config = hm.load.readConfig(ModelConfigFile)
(FlowTs, WaveTs, SeaLevelTs, Origin, BaseShoreNormDir, ShoreX, ShoreY, LagoonY,
 Dx, Dt, SimTime, PhysicalPars) = hm.load.loadModel(Config)

hm.visualise.mapView(ShoreX, ShoreY, LagoonY, Origin, BaseShoreNormDir)
hm.visualise.modelView(ShoreX, ShoreY, LagoonY)

# Test longshore transport routine
EAngle_h = WaveTs.EAngle_h[0]
WavePower = WaveTs.WavePower[0]
WavePeriod = WaveTs.WavePeriod[0]
Wlen_h = WaveTs.Wlen_h[0]

LST = hm.sim.longShoreTransport(ShoreY, Dx, WavePower, WavePeriod, Wlen_h, 
                                EAngle_h, PhysicalPars)

Dy = hm.sim.shoreChange(LST, Dx, Dt, PhysicalPars)
