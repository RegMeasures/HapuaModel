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
Dist = np.insert(np.cumsum(ChanDx),0,0)

Roughness = 0.03
PhysicalPars = {'RiverSlope': 0.003,
                'Gravity': 9.81,
                'GrainSize': 0.032,
                'VoidRatio': 0.4,
                'RhoRiv': 1000.0,
                'RhoSed': 2650.0}
NumericalPars = {'Beta':1.1,
                 'Theta':0.7,
                 'FrRelax1':0.75,
                 'FrRelax2': 0.9,
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

try:
    # Unsteady solution
    UnsteadyTime = timeit.timeit(stmt='riv.solveFullPreissmann(ChanElev, ChanWidth, LagArea, ChanDep, ChanVel, ChanDx, HydDt, Roughness, Q_Ts, DsWl_Ts, NumericalPars)', 
                                 globals=globals(), number=10)
    logging.info('Unsteady solution of %i timesteps took %f s' % (Q_Ts.size, UnsteadyTime))
    
    # Reporting
    DepErr = ChanDep - SteadyDep
    VelErr = ChanVel - SteadyVel
    # vis.longSection(ChanDx, ChanElev, ChanWidth, ChanDep, ChanVel)
    logging.info('Maximum difference between steady and unsteady solutions is: Depth %f m and velocity %f m/s' % (np.max(np.abs(DepErr)), np.max(np.abs(VelErr))))
except:
    logging.exception('Unsteady solution failed during test 1')

#%% Test2
logging.info('-----------------------------------------------------------------------')
logging.info('Test 2: unsteady flow reverting to steady')

# Setup
Q_Ts = np.concatenate([np.linspace(Qin, Qin*2, 50),
                       np.linspace(Qin*2, Qin, 50),
                       np.full(150, Qin)])
DsWl_Ts = np.full(Q_Ts.size, DsWL)

try:
    # Unsteady solution
    UnsteadyTime = timeit.timeit(stmt='riv.solveFullPreissmann(ChanElev, ChanWidth, LagArea, ChanDep, ChanVel, ChanDx, HydDt, Roughness, Q_Ts, DsWl_Ts, NumericalPars)', 
                                 globals=globals(), number=10)
    logging.info('Unsteady solution of %i timesteps took %f s' % (Q_Ts.size, UnsteadyTime))
    
    # Reporting
    DepErr = ChanDep - SteadyDep
    VelErr = ChanVel - SteadyVel
    LongSecFig = vis.longSection(ChanDx, ChanElev, ChanWidth, ChanDep, ChanVel)
    LongSecFig[0].suptitle('Test 2: unsteady flow reverting to steady')
    logging.info('Maximum difference between steady and unsteady solutions is: Depth %f m and velocity %f m/s' % (np.max(np.abs(DepErr)), np.max(np.abs(VelErr))))
except:
    logging.exception('Unsteady solution failed during test 2')
    
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

try:
    # Unsteady solution
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
except:
    logging.exception('Unsteady solution for test 3 failed after %i timesteps' % OutputTs3.index[-1]) 
    
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

try:
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
except:
    logging.exception('Unsteady solution for test 4 failed after %i timesteps' % OutputTs4.index[-1]) 

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

try:
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
except:
    logging.exception('Unsteady solution for test 5 failed after %i timesteps' % OutputTs5.index[-1]) 

#    vis.updateLongSection(LongSecFig, ChanDx, ChanElev, ChanWidth, ChanDep, ChanVel)
BdyFig = vis.BdyCndFig(OutputTs5)
BdyFig[0].suptitle('Test 5: Reverse flow')
logging.info('Maximum mass balance error = %f%% (of inflow)' % np.max(np.abs(OutputTs5.VolErrPerc)))
logging.info('Cumulative volumetric error over simulation = %f m3 (%f%% of total inflow)' % 
             (OutputTs5['CumVolErr'].iloc[-1], 
              OutputTs5['CumVolErr'].iloc[-1] / (np.sum(OutputTs5['Qin'])*StepSize*HydDt.seconds)))

#%% Test 6
logging.info('-----------------------------------------------------------------------')
logging.info('Test 6: Flow profile through constriction')

# Setup
ChanWidth[-30:-26] = 55
LagArea = 0
Q_Ts = np.full(300, Qin)
DsWl_Ts = np.full(300, DsWL)
ChanDep = SteadyDep.copy()
ChanVel = SteadyVel.copy()

# Initialise outputs
VolumeInModel = np.sum(ChanWidth * ChanDep * ChanDx2 + LagArea * ChanDep)
VolErr = 0.0
VolErrPerc = 0.0
CumVolErr = 0.0
OutputTs6 = pd.DataFrame(list(zip([Qin],[Qin],[DsWL],[VolumeInModel],[VolErr],[VolErrPerc],[CumVolErr])),
                         columns=['Qin','Qout','SeaLevel','Volume','VolErr','VolErrPerc','CumVolErr'],
                         index=[0])

try:
    # Unsteady solution with lagoon storage
    LongSecFig = vis.longSection(ChanDx, ChanElev, ChanWidth, ChanDep, ChanVel)
    LongSecFig[0].suptitle('Test 6: Flow profile through constriction - unsteady solution')
    StepSize = 5 
    for ii in range(0, DsWl_Ts.size, StepSize):
        TimesToRun = np.arange(ii, min(ii+StepSize, DsWl_Ts.size))
        riv.solveFullPreissmann(ChanElev, ChanWidth, LagArea, ChanDep, ChanVel, ChanDx, 
                                HydDt, Roughness, Q_Ts[TimesToRun], DsWl_Ts[TimesToRun], 
                                NumericalPars)
        VolumeInModel = np.sum(ChanWidth * ChanDep * ChanDx2 + LagArea * ChanDep)
        VolIn = (ChanDep[0]*ChanVel[0]*ChanWidth[0] + OutputTs6['Qin'].iloc[-1]) * HydDt.seconds * TimesToRun.size / 2
        VolOut = (ChanDep[-1]*ChanVel[-1]*ChanWidth[-1] + OutputTs6['Qout'].iloc[-1]) * HydDt.seconds * TimesToRun.size / 2
        # Mass balance error = DeltaVol + VolOut - VolIn
        VolErr = (VolumeInModel - OutputTs6['Volume'].iloc[-1]) + VolOut - VolIn
        VolErrPerc = VolErr/VolIn
        CumVolErr += VolErr
        OutputTs6 = OutputTs6.append(pd.DataFrame(list(zip([ChanDep[0]*ChanVel[0]*ChanWidth[0]],
                                                           [ChanDep[-1]*ChanVel[-1]*ChanWidth[-1]],
                                                           [ChanDep[-1]+ChanElev[-1]],
                                                           [VolumeInModel], [VolErr], [VolErrPerc], [CumVolErr])),
                                                  columns=['Qin','Qout','SeaLevel','Volume','VolErr','VolErrPerc','CumVolErr'],
                                                  index=[TimesToRun[-1]]))
        vis.updateLongSection(LongSecFig, ChanDx, ChanElev, ChanWidth, ChanDep, ChanVel)
except:
    logging.exception('Unsteady solution for test 6 failed after %i timesteps' % OutputTs6.index[-1]) 

# Reporting
BdyFig = vis.BdyCndFig(OutputTs6)
BdyFig[0].suptitle('Test 6: Flow profile through constriction')
logging.info('Maximum mass balance error = %f%% (of inflow)' % np.max(np.abs(OutputTs6.VolErrPerc)))
logging.info('Cumulative volumetric error over simulation = %f m3 (%f%% of total inflow)' % 
             (OutputTs6['CumVolErr'].iloc[-1], 
              OutputTs6['CumVolErr'].iloc[-1] / (np.sum(OutputTs6['Qin'])*StepSize*HydDt.seconds)))

(SteadyDep6, SteadyVel6) = riv.solveSteady(ChanDx, ChanElev, ChanWidth, 
                                       Roughness, NumericalPars['Beta'], Qin, DsWL)
DepErr = ChanDep - SteadyDep6
VelErr = ChanVel - SteadyVel6
LongSecFig[1].plot(Dist, ChanElev + SteadyDep6, 'm--', label='Steady WL')
LongSecFig[4].plot(Dist, SteadyVel6, 'm--', label='Steady vel')
LongSecFig[5].plot(Dist, (abs(SteadyVel6)/np.sqrt(9.81*SteadyDep6)), 'm--', label='Steady Fr')
logging.info('Maximum difference between steady and unsteady solutions is: Depth %f m and velocity %f m/s' % (np.max(np.abs(DepErr)), np.max(np.abs(VelErr))))

#%% Test 7
logging.info('-----------------------------------------------------------------------')
logging.info('Test 7: trans-critical flow')

# Setup
ChanWidth[-11:-9] = 25
LagArea = 0
Q_Ts = np.full(300, Qin)
DsWl_Ts = np.full(300, DsWL)
ChanDep = SteadyDep.copy()
ChanVel = SteadyVel.copy()

# Initialise outputs
VolumeInModel = np.sum(ChanWidth * ChanDep * ChanDx2 + LagArea * ChanDep)
VolErr = 0.0
VolErrPerc = 0.0
CumVolErr = 0.0
OutputTs7 = pd.DataFrame(list(zip([Qin],[Qin],[DsWL],[VolumeInModel],[VolErr],[VolErrPerc],[CumVolErr])),
                         columns=['Qin','Qout','SeaLevel','Volume','VolErr','VolErrPerc','CumVolErr'],
                         index=[0])

try:
    # Unsteady solution with lagoon storage
    LongSecFig = vis.longSection(ChanDx, ChanElev, ChanWidth, ChanDep, ChanVel)
    LongSecFig[0].suptitle('Test 7: Trans-critical flow through constriction - unsteady solution')
    StepSize = 5 
    for ii in range(0, DsWl_Ts.size, StepSize):
        TimesToRun = np.arange(ii, min(ii+StepSize, DsWl_Ts.size))
        riv.solveFullPreissmann(ChanElev, ChanWidth, LagArea, ChanDep, ChanVel, ChanDx, 
                                HydDt, Roughness, Q_Ts[TimesToRun], DsWl_Ts[TimesToRun], 
                                NumericalPars)
        VolumeInModel = np.sum(ChanWidth * ChanDep * ChanDx2 + LagArea * ChanDep)
        VolIn = (ChanDep[0]*ChanVel[0]*ChanWidth[0] + OutputTs7['Qin'].iloc[-1]) * HydDt.seconds * TimesToRun.size / 2
        VolOut = (ChanDep[-1]*ChanVel[-1]*ChanWidth[-1] + OutputTs7['Qout'].iloc[-1]) * HydDt.seconds * TimesToRun.size / 2
        # Mass balance error = DeltaVol + VolOut - VolIn
        VolErr = (VolumeInModel - OutputTs7['Volume'].iloc[-1]) + VolOut - VolIn
        VolErrPerc = VolErr/VolIn
        CumVolErr += VolErr
        OutputTs7 = OutputTs7.append(pd.DataFrame(list(zip([ChanDep[0]*ChanVel[0]*ChanWidth[0]],
                                                           [ChanDep[-1]*ChanVel[-1]*ChanWidth[-1]],
                                                           [ChanDep[-1]+ChanElev[-1]],
                                                           [VolumeInModel], [VolErr], [VolErrPerc], [CumVolErr])),
                                                  columns=['Qin','Qout','SeaLevel','Volume','VolErr','VolErrPerc','CumVolErr'],
                                                  index=[TimesToRun[-1]]))
        vis.updateLongSection(LongSecFig, ChanDx, ChanElev, ChanWidth, ChanDep, ChanVel)
except:
    logging.exception('Unsteady solution for test 7 failed after %i timesteps' % OutputTs7.index[-1]) 

# Reporting
BdyFig = vis.BdyCndFig(OutputTs7)
BdyFig[0].suptitle('Test 7: Trans-critical flow through constriction')
logging.info('Maximum mass balance error = %f%% (of inflow)' % np.max(np.abs(OutputTs7.VolErrPerc)))
logging.info('Cumulative volumetric error over simulation = %f m3 (%f%% of total inflow)' % 
             (OutputTs7['CumVolErr'].iloc[-1], 
              OutputTs7['CumVolErr'].iloc[-1] / (np.sum(OutputTs7['Qin'])*StepSize*HydDt.seconds)))

(SteadyDep7, SteadyVel7) = riv.solveSteady(ChanDx, ChanElev, ChanWidth, 
                                       Roughness, NumericalPars['Beta'], Qin, DsWL)
DepErr = ChanDep - SteadyDep7
VelErr = ChanVel - SteadyVel7
LongSecFig[1].plot(Dist, ChanElev + SteadyDep7, 'm--', label='Steady WL')
LongSecFig[4].plot(Dist, SteadyVel7, 'm--', label='Steady vel')
LongSecFig[5].plot(Dist, abs(SteadyVel7)/np.sqrt(9.81*SteadyDep7), 'm--', label='Steady Fr')
logging.info('Maximum difference between steady and unsteady solutions is: Depth %f m and velocity %f m/s' % (np.max(np.abs(DepErr)), np.max(np.abs(VelErr))))

#%% Test 8
logging.info('-----------------------------------------------------------------------')
logging.info('Test 8: Dynamic supercritical flow')

# Setup
ChanWidth[-11:-9] = 20
LagArea = 0
DsWl_Ts = np.concatenate([np.full(100, DsWL),
                          np.linspace(DsWL, DsWL-1, 200)])
Q_Ts = np.full(DsWl_Ts.size, Qin)
ChanDep = SteadyDep.copy()
ChanVel = SteadyVel.copy()

# Initialise outputs
VolumeInModel = np.sum(ChanWidth * ChanDep * ChanDx2 + LagArea * ChanDep)
VolErr = 0.0
VolErrPerc = 0.0
CumVolErr = 0.0
OutputTs8 = pd.DataFrame(list(zip([Qin],[Qin],[DsWL],[VolumeInModel],[VolErr],[VolErrPerc],[CumVolErr])),
                         columns=['Qin','Qout','SeaLevel','Volume','VolErr','VolErrPerc','CumVolErr'],
                         index=[0])

try:
    # Unsteady solution with lagoon storage
    LongSecFig = vis.longSection(ChanDx, ChanElev, ChanWidth, ChanDep, ChanVel)
    LongSecFig[0].suptitle('Test 8: Dynamic supercritical flow')
    StepSize = 5 
    for ii in range(0, DsWl_Ts.size, StepSize):
        TimesToRun = np.arange(ii, min(ii+StepSize, DsWl_Ts.size))
        riv.solveFullPreissmann(ChanElev, ChanWidth, LagArea, ChanDep, ChanVel, ChanDx, 
                                HydDt, Roughness, Q_Ts[TimesToRun], DsWl_Ts[TimesToRun], 
                                NumericalPars)
        VolumeInModel = np.sum(ChanWidth * ChanDep * ChanDx2 + LagArea * ChanDep)
        VolIn = (ChanDep[0]*ChanVel[0]*ChanWidth[0] + OutputTs8['Qin'].iloc[-1]) * HydDt.seconds * TimesToRun.size / 2
        VolOut = (ChanDep[-1]*ChanVel[-1]*ChanWidth[-1] + OutputTs8['Qout'].iloc[-1]) * HydDt.seconds * TimesToRun.size / 2
        # Mass balance error = DeltaVol + VolOut - VolIn
        VolErr = (VolumeInModel - OutputTs8['Volume'].iloc[-1]) + VolOut - VolIn
        VolErrPerc = VolErr/VolIn
        CumVolErr += VolErr
        OutputTs8 = OutputTs8.append(pd.DataFrame(list(zip([ChanDep[0]*ChanVel[0]*ChanWidth[0]],
                                                           [ChanDep[-1]*ChanVel[-1]*ChanWidth[-1]],
                                                           [ChanDep[-1]+ChanElev[-1]],
                                                           [VolumeInModel], [VolErr], [VolErrPerc], [CumVolErr])),
                                                  columns=['Qin','Qout','SeaLevel','Volume','VolErr','VolErrPerc','CumVolErr'],
                                                  index=[TimesToRun[-1]]))
        vis.updateLongSection(LongSecFig, ChanDx, ChanElev, ChanWidth, ChanDep, ChanVel)
except:
    logging.exception('Unsteady solution for test 8 failed after %i timesteps' % OutputTs8.index[-1]) 

# Reporting
BdyFig = vis.BdyCndFig(OutputTs8)
BdyFig[0].suptitle('Test 8: Dynamic supercritical flow')
logging.info('Maximum mass balance error = %f%% (of inflow)' % np.max(np.abs(OutputTs8.VolErrPerc)))
logging.info('Cumulative volumetric error over simulation = %f m3 (%f%% of total inflow)' % 
             (OutputTs8['CumVolErr'].iloc[-1], 
              OutputTs8['CumVolErr'].iloc[-1] / (np.sum(OutputTs8['Qin'])*StepSize*HydDt.seconds)))

#%% If needed
dx = ChanDx
dt = HydDt
n = Roughness
z = ChanElev
B = ChanWidth
h = ChanDep
V = ChanVel