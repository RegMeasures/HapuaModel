import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import hapuamod as hm

#%% Test model load (particularly geometry processing)
ModelConfigFile = 'inputs\HurunuiModel.cnf'
Config = hm.loadmod.readConfig(ModelConfigFile)
(FlowTs, WaveTs, SeaLevelTs, Origin, ShoreNormDir, 
 ShoreX, ShoreY, LagoonElev, BarrierElev, OutletElev, 
 RiverElev, OutletEndX, OutletEndWidth, OutletEndElev,
 TimePars, PhysicalPars, NumericalPars, OutputOpts) = hm.loadmod.loadModel(Config)

plt.figure()
hm.visualise.mapView(ShoreX, ShoreY, Origin, ShoreNormDir)

#%% Test longshore transport routine
EDir_h = WaveTs.EDir_h[0]
WavePower = WaveTs.WavePower[0]
WavePeriod = WaveTs.WavePeriod[0]
Wlen_h = WaveTs.Wlen_h[0]

plt.figure()
plt.subplot(2, 1, 1)
plt.plot(ShoreX, ShoreY[:,0])

LST = hm.coast.longShoreTransport(ShoreY, NumericalPars['Dx'], WavePower, 
                                  WavePeriod, Wlen_h, EDir_h, PhysicalPars)
plt.subplot(2, 1, 2)
plt.plot((ShoreX[0:-1]+ShoreX[1:])/2, LST)

#%% Test river routines
# Join river and outlet through lagoon
(ChanDx, ChanElev, ChanWidth, LagArea, OnlineLagoon, OutletChanIx) = \
    hm.mor.assembleChannel(ShoreX, ShoreY, LagoonElev, OutletElev,
                           OutletEndX, OutletEndWidth, OutletEndElev, 
                           RiverElev, PhysicalPars['RiverWidth'], 
                           NumericalPars['Dx'])
hm.visualise.modelView(ShoreX, ShoreY, OutletEndX, OutletChanIx)
    
# Steady state hydraulics
RivFlow = hm.core.interpolate_at(FlowTs, pd.DatetimeIndex([TimePars['StartTime']])).values
SeaLevel = hm.core.interpolate_at(SeaLevelTs, pd.DatetimeIndex([TimePars['StartTime']])).values
(ChanDep, ChanVel) = hm.riv.solveSteady(ChanDx, ChanElev, ChanWidth, 
                                        PhysicalPars['Roughness'], 
                                        NumericalPars['Beta'], 
                                        RivFlow, SeaLevel)

# Bedload
Bedload = hm.riv.calcBedload(ChanElev, ChanWidth, ChanDep, ChanVel, ChanDx, PhysicalPars)

hm.visualise.longSection(ChanDx, ChanElev, ChanWidth, ChanDep, ChanVel, Bedload)

ChanDist = np.insert(np.cumsum(ChanDx),0,0)
plt.figure()
plt.plot(ChanDist, ChanDep+ChanElev, 'b-')
# Unsteady hydraulics
(ChanDep, ChanVel) = hm.riv.solveFullPreissmann(ChanElev, ChanWidth, LagArea, ChanDep, 
                                                ChanVel, ChanDx, TimePars['HydDt'], 
                                                PhysicalPars['Roughness'], 
                                                RivFlow, SeaLevel, NumericalPars)
plt.plot(ChanDist, ChanDep+ChanElev, 'r:')

#%% Morphology updating
# Bed updating
OldShoreY = ShoreY.copy()
hm.mor.updateMorphology(ShoreX, ShoreY, LagoonElev, OutletElev, BarrierElev,
                        OutletEndX, OutletEndWidth, OutletEndElev, 
                        RiverElev, PhysicalPars['RiverWidth'], OnlineLagoon, 
                        OutletChanIx, ChanWidth, ChanDep, ChanDx,
                        LST, Bedload, NumericalPars['Dx'], TimePars['MorDt'], 
                        PhysicalPars)
plt.plot(ShoreX, (ShoreY[:,0]-OldShoreY[:,0]))

#%% Test core timestepping
ModelConfigFile = 'inputs\HurunuiModel.cnf'
OutputTs = hm.core.run(ModelConfigFile)
