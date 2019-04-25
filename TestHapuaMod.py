import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import hapuamod as hm

#%% Test model load (particularly geometry processing)
ModelConfigFile = 'inputs\HurunuiModel.cnf'
Config = hm.loadmod.readConfig(ModelConfigFile)
(FlowTs, WaveTs, SeaLevelTs, Origin, BaseShoreNormDir, ShoreX, ShoreY, LagoonY,
 LagoonElev, RiverElev, OutletX, OutletY, OutletElev, OutletWidth, TimePars, 
 PhysicalPars, NumericalPars, OutputOpts) = hm.loadmod.loadModel(Config)

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

LST = hm.coast.longShoreTransport(ShoreY, NumericalPars['Dx'], WavePower, 
                                  WavePeriod, Wlen_h, EDir_h, PhysicalPars)
plt.subplot(3, 1, 2)
plt.plot((ShoreX[0:-1]+ShoreX[1:])/2, LST)

Dy = hm.coast.shoreChange(LST, NumericalPars['Dx'], TimePars['MorDt'], 
                          PhysicalPars)
plt.subplot(3, 1, 3)
plt.plot(ShoreX, Dy)

#%% Test river routines
# Join river and outlet through lagoon
(ChanDx, ChanElev, ChanWidth, LagArea) = \
    hm.riv.assembleChannel(RiverElev, ShoreX, LagoonY, LagoonElev, 
                           OutletX, OutletY, OutletElev, OutletWidth, 
                           PhysicalPars['RiverWidth'], NumericalPars['Dx'])

# Steady state hydraulics
RivFlow = hm.core.interpolate_at(FlowTs, pd.DatetimeIndex([TimePars['StartTime']])).values
SeaLevel = hm.core.interpolate_at(SeaLevelTs, pd.DatetimeIndex([TimePars['StartTime']])).values
(ChanDep, ChanVel) = hm.riv.solveSteady(ChanDx, ChanElev, ChanWidth, 
                                        PhysicalPars['Roughness'], 
                                        RivFlow, SeaLevel)

hm.visualise.longSection(ChanDx, ChanElev, ChanDep, ChanVel)

ChanDist = np.insert(np.cumsum(ChanDx),0,0)
plt.plot(ChanDist, ChanDep+ChanElev, 'b-')

# Unsteady hydraulics
(ChanDep, ChanVel) = hm.riv.solveFullPreissmann(ChanElev, ChanWidth, LagArea, ChanDep, 
                                                ChanVel, ChanDx, TimePars['HydDt'], 
                                                PhysicalPars['Roughness'], 
                                                RivFlow, SeaLevel, NumericalPars)
plt.plot(ChanDist, ChanDep+ChanElev, 'r:')

#%% Test core timestepping
ModelConfigFile = 'inputs\HurunuiModel.cnf'
OutputTs = hm.core.run(ModelConfigFile)
