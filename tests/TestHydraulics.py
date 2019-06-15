# -*- coding: utf-8 -*-
"""
Tests for the 1D fluvial hydraulics model component
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging
from timeit import default_timer as timer
import timeit

import hapuamod.riv as riv
import hapuamod.visualise as vis

#%% Set up logging
RootLogger = logging.getLogger()
RootLogger.setLevel(logging.INFO)

ConsoleHandler = logging.StreamHandler()
ConsoleHandler.setLevel(logging.INFO)

FileHandler = logging.FileHandler('TestLog.txt')
FileHandler.setLevel(logging.INFO)

RootLogger.addHandler(ConsoleHandler)
RootLogger.addHandler(FileHandler)

#%% Test 1
logging.info('-----------------------------------------------------------------------')
logging.info('Test 1: steady flow through very simple channel with no offline storage')

# Setup
ChanDx = np.full(50, 20.0)
ChanDx2 = np.concatenate([[ChanDx[1]], (ChanDx[:-1]+ChanDx[1:])/2, [ChanDx[-1]]])
ChanElev = np.linspace(0.003*np.sum(ChanDx) - 2, -2, ChanDx.size+1)
ChanWidth = np.full(ChanDx.size+1, 70.0)
LagArea = np.full(ChanDx.size+1, 0.0)

Roughness = 0.03
PhysicalPars = {'RiverSlope': 0.003,
                'Gravity': 9.81,
                'GrainSize': 0.032,
                'VoidRatio': 0.4,
                'RhoRiv': 1000.0,
                'RhoSed': 2650.0}
NumericalPars = {'Beta':1.1,
                 'Theta':0.7,
                 'ErrTol':0.001,
                 'MaxIt':20,
                 'WarnTol':0.1}
HydDt = pd.Timedelta(seconds=5)

Qin = 70
Q_Ts = np.full(100, Qin)
DsWL = 0.0
DsWl_Ts = np.full(100, DsWL)

# Steady solution
(ChanDep, ChanVel) = riv.solveSteady(ChanDx, ChanElev, ChanWidth, 
                                     Roughness, NumericalPars['Beta'], Qin, DsWL)
SteadyDep = ChanDep.copy()
SteadyVel = ChanVel.copy()

SteadyTime = timeit.timeit(stmt='riv.solveSteady(ChanDx, ChanElev, ChanWidth, Roughness, NumericalPars["Beta"], Qin, DsWL)', 
                           globals=globals(), number=20)
logging.info('Steady solution took %f s' % SteadyTime)

# Unsteady solution
#riv.solveFullPreissmann(ChanElev, ChanWidth, LagArea, ChanDep, ChanVel, ChanDx, 
#                        HydDt, Roughness, Q_Ts, DsWl_Ts, NumericalPars)

UnsteadyTime = timeit.timeit(stmt='riv.solveFullPreissmann(ChanElev, ChanWidth, LagArea, ChanDep, ChanVel, ChanDx, HydDt, Roughness, Q_Ts, DsWl_Ts, NumericalPars)', 
                             globals=globals(), number=10)
logging.info('Unsteady solution of %i timesteps took %f s' % (Q_Ts.size, UnsteadyTime))

# Reporting
DepErr = ChanDep - SteadyDep
VelErr = ChanVel - SteadyVel
# vis.longSection(ChanDx, ChanElev, ChanWidth, ChanDep, ChanVel)
logging.info('Maximum difference between steady and unsteady solutions is: Depth %f m and velocity %f m/s' % (np.max(np.abs(DepErr)), np.max(np.abs(VelErr))))

#%% Test2
logging.info('-----------------------------------------------------------------------')
logging.info('Test 2: unsteady flow reverting to steady')

# Setup
Q_Ts = np.concatenate([np.linspace(Qin, Qin*2, 50),
                       np.linspace(Qin*2, Qin, 50),
                       np.full(150, Qin)])
DsWl_Ts = np.full(Q_Ts.size, DsWL)

# Unsteady solution
UnsteadyTime = timeit.timeit(stmt='riv.solveFullPreissmann(ChanElev, ChanWidth, LagArea, ChanDep, ChanVel, ChanDx, HydDt, Roughness, Q_Ts, DsWl_Ts, NumericalPars)', 
                             globals=globals(), number=10)
logging.info('Unsteady solution of %i timesteps took %f s' % (Q_Ts.size, UnsteadyTime))

# Reporting
DepErr = ChanDep - SteadyDep
VelErr = ChanVel - SteadyVel
vis.longSection(ChanDx, ChanElev, ChanWidth, ChanDep, ChanVel)
logging.info('Maximum difference between steady and unsteady solutions is: Depth %f m and velocity %f m/s' % (np.max(np.abs(DepErr)), np.max(np.abs(VelErr))))

#%% Test 3
logging.info('-----------------------------------------------------------------------')
logging.info('Test 3: unsteady d/s boundary')

# Setup
DsWl_Ts = np.concatenate([np.linspace(DsWL, DsWL-1, 500),
                          np.linspace(DsWL-1, DsWL+1, 1000)])
Q_Ts = np.full(DsWl_Ts.size, Qin)
ChanDep = SteadyDep.copy()
ChanVel = SteadyVel.copy()

# Initialise outputs
VolumeInModel = np.sum(ChanWidth * ChanDep * ChanDx2)
VolErr = 0.0
VolErrPerc = 0.0
CumVolErr = 0.0
OutputTs3 = pd.DataFrame(list(zip([Qin],[Qin],[DsWL],[VolumeInModel],[VolErr],[VolErrPerc],[CumVolErr])),
                         columns=['Qin','Qout','SeaLevel','Volume','VolErr','VolErrPerc','CumVolErr'],
                         index=[0])

# Unsteady solution without lagoon storage
#LongSecFig = vis.longSection(ChanDx, ChanElev, ChanWidth, ChanDep, ChanVel)
#LongSecFig[0].suptitle('Test 3: Unsteady downstream boundary')
StepSize = 20 
for ii in range(0, DsWl_Ts.size, StepSize):
    TimesToRun = np.arange(ii, min(ii+StepSize, DsWl_Ts.size))
    riv.solveFullPreissmann(ChanElev, ChanWidth, LagArea, ChanDep, ChanVel, ChanDx, 
                            HydDt, Roughness, Q_Ts[TimesToRun], DsWl_Ts[TimesToRun], 
                            NumericalPars)
    VolumeInModel = np.sum(ChanWidth * ChanDep * ChanDx2)
    VolIn = (ChanDep[0]*ChanVel[0]*ChanWidth[0] + OutputTs3['Qin'].iloc[-1]) * HydDt.seconds * TimesToRun.size / 2
    VolOut = (ChanDep[-1]*ChanVel[-1]*ChanWidth[-1] + OutputTs3['Qout'].iloc[-1]) * HydDt.seconds * TimesToRun.size / 2
    # Mass balance error = DeltaVol + VolOut - VolIn
    VolErr = (VolumeInModel - OutputTs3['Volume'].iloc[-1]) + VolOut - VolIn
    VolErrPerc = VolErr/VolIn
    CumVolErr += VolErr
    OutputTs3 = OutputTs3.append(pd.DataFrame(list(zip([ChanDep[0]*ChanVel[0]*ChanWidth[0]],
                                                       [ChanDep[-1]*ChanVel[-1]*ChanWidth[-1]],
                                                       [ChanDep[-1]+ChanElev[-1]],
                                                       [VolumeInModel], [VolErr], [VolErrPerc], [CumVolErr])),
                                              columns=['Qin','Qout','SeaLevel','Volume','VolErr','VolErrPerc','CumVolErr'],
                                              index=[TimesToRun[-1]]))
#    vis.updateLongSection(LongSecFig, ChanDx, ChanElev, ChanWidth, ChanDep, ChanVel)
BdyFig = vis.BdyCndFig(OutputTs3)
BdyFig[0].suptitle('Test 3: Unsteady downstream boundary')
logging.info('Maximum mass balance error = %f%% (of inflow)' % np.max(np.abs(OutputTs3.VolErrPerc)))
logging.info('Cumulative volumetric error over simulation = %f m3 (%f%% of total inflow)' % 
             (OutputTs3['CumVolErr'].iloc[-1], 
              OutputTs3['CumVolErr'].iloc[-1] / (np.sum(OutputTs3['Qin'])*StepSize*HydDt.seconds)))

#%% Test 4
logging.info('-----------------------------------------------------------------------')
logging.info('Test 4: Lagoon storage')

# Setup
LagArea[40] = 10000
ChanDep = SteadyDep.copy()
ChanVel = SteadyVel.copy()

# Initialise outputs
VolumeInModel = np.sum(ChanWidth * ChanDep * ChanDx2 + LagArea * ChanDep)
VolErr = 0.0
VolErrPerc = 0.0
CumVolErr = 0.0
OutputTs4 = pd.DataFrame(list(zip([Qin],[Qin],[DsWL],[VolumeInModel],[VolErr],[VolErrPerc],[CumVolErr])),
                         columns=['Qin','Qout','SeaLevel','Volume','VolErr','VolErrPerc','CumVolErr'],
                         index=[0])

# Unsteady solution with lagoon storage
#LongSecFig = vis.longSection(ChanDx, ChanElev, ChanWidth, ChanDep, ChanVel)
#LongSecFig[0].suptitle('Test 4: Lagoon storage')
StepSize = 20 
for ii in range(0, DsWl_Ts.size, StepSize):
    TimesToRun = np.arange(ii, min(ii+StepSize, DsWl_Ts.size))
    riv.solveFullPreissmann(ChanElev, ChanWidth, LagArea, ChanDep, ChanVel, ChanDx, 
                            HydDt, Roughness, Q_Ts[TimesToRun], DsWl_Ts[TimesToRun], 
                            NumericalPars)
    VolumeInModel = np.sum(ChanWidth * ChanDep * ChanDx2 + LagArea * ChanDep)
    VolIn = (ChanDep[0]*ChanVel[0]*ChanWidth[0] + OutputTs4['Qin'].iloc[-1]) * HydDt.seconds * TimesToRun.size / 2
    VolOut = (ChanDep[-1]*ChanVel[-1]*ChanWidth[-1] + OutputTs4['Qout'].iloc[-1]) * HydDt.seconds * TimesToRun.size / 2
    # Mass balance error = DeltaVol + VolOut - VolIn
    VolErr = (VolumeInModel - OutputTs4['Volume'].iloc[-1]) + VolOut - VolIn
    VolErrPerc = VolErr/VolIn
    CumVolErr += VolErr
    OutputTs4 = OutputTs4.append(pd.DataFrame(list(zip([ChanDep[0]*ChanVel[0]*ChanWidth[0]],
                                                       [ChanDep[-1]*ChanVel[-1]*ChanWidth[-1]],
                                                       [ChanDep[-1]+ChanElev[-1]],
                                                       [VolumeInModel], [VolErr], [VolErrPerc], [CumVolErr])),
                                              columns=['Qin','Qout','SeaLevel','Volume','VolErr','VolErrPerc','CumVolErr'],
                                              index=[TimesToRun[-1]]))
#    vis.updateLongSection(LongSecFig, ChanDx, ChanElev, ChanWidth, ChanDep, ChanVel)
BdyFig = vis.BdyCndFig(OutputTs4)
BdyFig[0].suptitle('Test 4: Lagoon Storage')
logging.info('Maximum mass balance error = %f%% (of inflow)' % np.max(np.abs(OutputTs4.VolErrPerc)))
logging.info('Cumulative volumetric error over simulation = %f m3 (%f%% of total inflow)' % 
             (OutputTs4['CumVolErr'].iloc[-1], 
              OutputTs4['CumVolErr'].iloc[-1] / (np.sum(OutputTs4['Qin'])*StepSize*HydDt.seconds)))
    
#%% Test 5
logging.info('-----------------------------------------------------------------------')
logging.info('Test 5: Reverse flow')

# Setup
LagArea[40] = 250000
ChanDep = SteadyDep.copy()
ChanVel = SteadyVel.copy()

# Initialise outputs
VolumeInModel = np.sum(ChanWidth * ChanDep * ChanDx2 + LagArea * ChanDep)
VolErr = 0.0
VolErrPerc = 0.0
CumVolErr = 0.0
OutputTs5 = pd.DataFrame(list(zip([Qin],[Qin],[DsWL],[VolumeInModel],[VolErr],[VolErrPerc],[CumVolErr])),
                         columns=['Qin','Qout','SeaLevel','Volume','VolErr','VolErrPerc','CumVolErr'],
                         index=[0])

# Unsteady solution with lagoon storage
#LongSecFig = vis.longSection(ChanDx, ChanElev, ChanWidth, ChanDep, ChanVel)
#LongSecFig[0].suptitle('Test 5: Reverse flow')
StepSize = 20 
for ii in range(0, DsWl_Ts.size, StepSize):
    TimesToRun = np.arange(ii, min(ii+StepSize, DsWl_Ts.size))
    riv.solveFullPreissmann(ChanElev, ChanWidth, LagArea, ChanDep, ChanVel, ChanDx, 
                            HydDt, Roughness, Q_Ts[TimesToRun], DsWl_Ts[TimesToRun], 
                            NumericalPars)
    VolumeInModel = np.sum(ChanWidth * ChanDep * ChanDx2 + LagArea * ChanDep)
    VolIn = (ChanDep[0]*ChanVel[0]*ChanWidth[0] + OutputTs5['Qin'].iloc[-1]) * HydDt.seconds * TimesToRun.size / 2
    VolOut = (ChanDep[-1]*ChanVel[-1]*ChanWidth[-1] + OutputTs5['Qout'].iloc[-1]) * HydDt.seconds * TimesToRun.size / 2
    # Mass balance error = DeltaVol + VolOut - VolIn
    VolErr = (VolumeInModel - OutputTs5['Volume'].iloc[-1]) + VolOut - VolIn
    VolErrPerc = VolErr/VolIn
    CumVolErr += VolErr
    OutputTs5 = OutputTs5.append(pd.DataFrame(list(zip([ChanDep[0]*ChanVel[0]*ChanWidth[0]],
                                                       [ChanDep[-1]*ChanVel[-1]*ChanWidth[-1]],
                                                       [ChanDep[-1]+ChanElev[-1]],
                                                       [VolumeInModel], [VolErr], [VolErrPerc], [CumVolErr])),
                                              columns=['Qin','Qout','SeaLevel','Volume','VolErr','VolErrPerc','CumVolErr'],
                                              index=[TimesToRun[-1]]))
#    vis.updateLongSection(LongSecFig, ChanDx, ChanElev, ChanWidth, ChanDep, ChanVel)
BdyFig = vis.BdyCndFig(OutputTs5)
BdyFig[0].suptitle('Test 5: Reverse flow')
logging.info('Maximum mass balance error = %f%% (of inflow)' % np.max(np.abs(OutputTs5.VolErrPerc)))
logging.info('Cumulative volumetric error over simulation = %f m3 (%f%% of total inflow)' % 
             (OutputTs5['CumVolErr'].iloc[-1], 
              OutputTs5['CumVolErr'].iloc[-1] / (np.sum(OutputTs5['Qin'])*StepSize*HydDt.seconds)))

#%% If needed
dx = ChanDx
dt = HydDt
n = Roughness
z = ChanElev
B = ChanWidth
h = ChanDep
V = ChanVel