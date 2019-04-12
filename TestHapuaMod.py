import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation

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

#%% Test model load (particularly geometry processing)
ModelConfigFile = 'inputs\HurunuiModel.cnf'
Config = hm.load.readConfig(ModelConfigFile)
(FlowTs, WaveTs, SeaLevelTs, Origin, BaseShoreNormDir, ShoreX, ShoreY, LagoonY,
 LagoonElev, RiverElev, OutletX, OutletY, OutletElev, OutletWidth, 
 Dx, Dt, SimTime, PhysicalPars) = hm.load.loadModel(Config)

#plt.figure()
#hm.visualise.mapView(ShoreX, ShoreY, LagoonY, Origin, BaseShoreNormDir)    

#plt.figure(figsize=(12,5))
#hm.visualise.modelView(ShoreX, ShoreY, LagoonY, OutletX, OutletY)

#%% Test longshore transport routine
EDir_h = WaveTs.EDir_h[0]
WavePower = WaveTs.WavePower[0]
WavePeriod = WaveTs.WavePeriod[0]
Wlen_h = WaveTs.Wlen_h[0]

plt.figure()
plt.subplot(3, 1, 1)
plt.plot(ShoreX, ShoreY)

LST = hm.coast.longShoreTransport(ShoreY, Dx, WavePower, WavePeriod, Wlen_h, 
                                EDir_h, PhysicalPars)
plt.subplot(3, 1, 2)
plt.plot((ShoreX[0:-1]+ShoreX[1:])/2, LST)

Dy = hm.coast.shoreChange(LST, Dx, Dt, PhysicalPars)
plt.subplot(3, 1, 3)
plt.plot(ShoreX, Dy)

#%% Test river routines
# Join river and outlet through lagoon
(ChanDx, ChanElev, ChanWidth, ChanArea) = \
    hm.riv.assembleChannel(RiverElev, ShoreX, LagoonY, LagoonElev, 
                           OutletX, OutletY, OutletElev, OutletWidth, 
                           PhysicalPars['RiverWidth'], Dx)

# Steady state hydraulics
RivFlow = FlowTs[0]
SeaLevel = SeaLevelTs[0]
(ChanDep, ChanVel) = hm.riv.solveSteady(ChanDx, ChanElev, ChanWidth, 
                                        PhysicalPars['Roughness'], 
                                        RivFlow, SeaLevel)

hm.visualise.longSection(RiverElev, ShoreX, LagoonY, LagoonElev, 
                         OutletElev, OutletWidth, OutletX, OutletY,
                         PhysicalPars['RiverWidth'], Dx)

ChanDist = np.insert(np.cumsum(ChanDx),0,0)
plt.plot(ChanDist, ChanDep+ChanElev)

#%% Test core timestepping
Time = SimTime[0]
CurrentDay = Time.day
plt.figure(figsize=(12,5))
hm.visualise.modelView(ShoreX, ShoreY, LagoonY, OutletX, OutletY)
while Time <= SimTime[1]:
    hm.core.timestep(Time, Dx, Dt, WaveTs, SeaLevelTs, FlowTs, PhysicalPars,
                     ShoreX, ShoreY, LagoonY, LagoonElev, RiverElev, 
                     OutletX, OutletY, OutletElev, OutletWidth)
    Time += Dt
hm.visualise.modelView(ShoreX, ShoreY, LagoonY, OutletX, OutletY)
