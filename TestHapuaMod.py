import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from hapuamod import loadmod, visualise, coast, riv, core, mor, out

#%% Test model load (particularly geometry processing)
ModelConfigFile = 'inputs\HurunuiModel.cnf'
Config = loadmod.readConfig(ModelConfigFile)
(FlowTs, WaveTs, SeaLevelTs, Origin, ShoreNormDir, 
 ShoreX, ShoreY, ShoreZ, RiverElev, 
 OutletEndX, OutletEndWidth, OutletEndElev,
 TimePars, PhysicalPars, NumericalPars, OutputOpts) = loadmod.loadModel(Config)

plt.figure()
visualise.mapView(ShoreX, ShoreY, Origin, ShoreNormDir)

#%% Test longshore transport routine
EDir_h = WaveTs.EDir_h[0]
WavePower = WaveTs.WavePower[0]
WavePeriod = WaveTs.WavePeriod[0]
Wlen_h = WaveTs.Wlen_h[0]

plt.figure()
plt.subplot(2, 1, 1)
plt.plot(ShoreX, ShoreY[:,0])

LST = coast.longShoreTransport(ShoreY, NumericalPars['Dx'], WavePower, 
                               WavePeriod, Wlen_h, EDir_h, PhysicalPars)
plt.subplot(2, 1, 2)
plt.plot((ShoreX[0:-1]+ShoreX[1:])/2, LST)

#%% Test river routines
# Join river and outlet through lagoon
(ChanDx, ChanElev, ChanWidth, LagArea, ChanDep, ChanVel, 
 OnlineLagoon, OutletChanIx, ChanFlag, Closed) = \
    riv.assembleChannel(ShoreX, ShoreY, ShoreZ,
                        OutletEndX, OutletEndWidth, OutletEndElev, 
                        RiverElev, PhysicalPars['RiverWidth'], 
                        np.zeros(RiverElev.size), np.zeros(RiverElev.size),
                        np.zeros(ShoreX.size), np.zeros(ShoreX.size), 
                        np.zeros(ShoreX.size), np.zeros(ShoreX.size),
                        np.zeros(2), np.zeros(2), NumericalPars['Dx'],
                        PhysicalPars)
visualise.modelView(ShoreX, ShoreY, OutletEndX, OutletEndWidth, OutletChanIx)
    
# Steady state hydraulics
RivFlow = core.interpolate_at(FlowTs, pd.DatetimeIndex([TimePars['StartTime']])).values
SeaLevel = core.interpolate_at(SeaLevelTs, pd.DatetimeIndex([TimePars['StartTime']])).values
(ChanDep, ChanVel) = riv.solveSteady(ChanDx, ChanElev, ChanWidth, 
                                     PhysicalPars['Roughness'], 
                                     RivFlow[0], SeaLevel[0], NumericalPars)

# Store hydraulics and re-generate
(LagoonWL, LagoonVel, OutletDep, OutletVel, OutletEndDep, OutletEndVel) = \
    riv.storeHydraulics(ChanDep, ChanVel, OnlineLagoon, OutletChanIx, 
                        ChanFlag, ShoreZ[:,3], Closed)
(ChanDx, ChanElev, ChanWidth, LagArea, ChanDep, ChanVel, 
 OnlineLagoon, OutletChanIx, ChanFlag, Closed) = \
    riv.assembleChannel(ShoreX, ShoreY, ShoreZ, 
                        OutletEndX, OutletEndWidth, OutletEndElev, 
                        RiverElev, PhysicalPars['RiverWidth'], 
                        ChanDep[ChanFlag==0], ChanVel[ChanFlag==0], 
                        LagoonWL, LagoonVel, OutletDep, OutletVel,
                        OutletEndDep, OutletEndVel, 
                        NumericalPars['Dx'], PhysicalPars)

# Bedload
Bedload = riv.calcBedload(ChanElev, ChanWidth, ChanDep, ChanVel, ChanDx, 
                          PhysicalPars, NumericalPars['Psi'])

visualise.longSection(ChanDx, ChanElev, ChanWidth, ChanDep, ChanVel, Bedload)

ChanDist = np.insert(np.cumsum(ChanDx),0,0)
plt.figure()
plt.plot(ChanDist, ChanDep+ChanElev, 'b-')
# Unsteady hydraulics
riv.solveFullPreissmann(ChanElev, ChanWidth, LagArea, Closed, 
                        ChanDep, ChanVel, ChanDx,
                        TimePars['HydDt'], PhysicalPars['Roughness'], 
                        RivFlow, SeaLevel, NumericalPars)
plt.plot(ChanDist, ChanDep+ChanElev, 'r:')

#%% Runup/overtopping
WavePeriod = WaveTs.WavePeriod[0]
Hs_offshore = WaveTs.Hsig_Offshore[0]

Runup = coast.runup(WavePeriod, Hs_offshore, PhysicalPars['BeachSlope'])

(CST_tot, OverwashProp) = coast.overtopping(Runup, SeaLevel, ShoreY, 
                                               ShoreZ, PhysicalPars)

#%% Morphology updating
OldShoreY = ShoreY.copy()
mor.updateMorphology(ShoreX, ShoreY, ShoreZ,
                     OutletEndX, OutletEndWidth, OutletEndElev, 
                     RiverElev, PhysicalPars['RiverWidth'], OnlineLagoon, 
                     OutletChanIx, LagoonWL, OutletDep,
                     ChanWidth, ChanDep, ChanDx, ChanFlag, 
                     Closed, LST, Bedload, CST_tot, OverwashProp,
                     NumericalPars['Dx'], TimePars['MorDt'], PhysicalPars)
plt.plot(ShoreX, (ShoreY[:,0]-OldShoreY[:,0]))

#%% Create output netcdf file and write initial condition
out.newOutFile('test.nc', Config['ModelName'], TimePars['StartTime'], 
               ShoreX, NumericalPars['Dx'], RiverElev, True)

out.writeCurrent('test.nc', TimePars['StartTime'], 
                 ShoreY, ShoreZ, LagoonWL, LagoonVel, OutletDep, OutletVel,
                 np.zeros(ShoreX.size-1), 
                 RiverElev, ChanDep[ChanFlag==0], ChanVel[ChanFlag==0],
                 OutletEndX, OutletEndElev, OutletEndWidth, 
                 OutletEndDep, OutletEndVel)

#%% Test core timestepping
ModelConfigFile = 'inputs\HurunuiModel.cnf'
OutputTs = core.run(ModelConfigFile)
