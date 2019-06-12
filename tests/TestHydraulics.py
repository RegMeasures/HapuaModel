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
ChanElev = np.linspace(0.003*np.sum(ChanDx), 0, ChanDx.size+1)
ChanWidth = np.full(ChanDx.size+1, 100.0)
LagArea = np.full(ChanDx.size+1, 0.0)

Roughness = 0.03
PhysicalPars = {'RiverSlope': 0.003,
                'Gravity': 9.81,
                'GrainSize': 0.032,
                'VoidRatio': 0.4,
                'RhoRiv': 1000.0,
                'RhoSed': 2650.0}
NumericalPars = {'Theta':0.7,
                 'ErrTol':0.001,
                 'MaxIt':20,
                 'WarnTol':0.1}
HydDt = pd.Timedelta(seconds=5)

Qin = 45
Q_Ts = np.full(100, Qin)
DsWL = 1.5
DsWl_Ts = np.full(100, DsWL)

# Steady solution
(ChanDep, ChanVel) = riv.solveSteady(ChanDx, ChanElev, ChanWidth, 
                                     Roughness, Qin, DsWL)
SteadyDep = ChanDep.copy()
SteadyVel = ChanVel.copy()

SteadyTime = timeit.timeit(stmt='riv.solveSteady(ChanDx, ChanElev, ChanWidth, Roughness, Qin, DsWL)', 
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

#%% If needed
dx = ChanDx
dt = HydDt
n = Roughness
z = ChanElev
B = ChanWidth
h = ChanDep
V = ChanVel