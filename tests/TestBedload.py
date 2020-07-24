import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging
import timeit

import hapuamod.riv as riv
import hapuamod.visualise as vis

#%% Set up logging
LogFile = 'BedloadTestLog.html'
if os.path.isfile(LogFile):
    os.remove(LogFile)

RootLogger = logging.getLogger()
RootLogger.setLevel(logging.INFO)

ConsoleHandler = logging.StreamHandler()
ConsoleHandler.setLevel(logging.INFO)

FileHandler = logging.FileHandler(LogFile)
FileHandler.setLevel(logging.INFO)
LogFormatter = logging.Formatter('<p> %(message)s </p>')
FileHandler.setFormatter(LogFormatter)

RootLogger.addHandler(ConsoleHandler)
RootLogger.addHandler(FileHandler)

#%% Test 1
logging.info('<hr>')
logging.info('<h1>Test 1: steady 200m3/s flow through 100m wide channel</h1>')

# Setup
Width = 100.0
Slope = 0.003
Roughness = 0.04

ChanDx = np.full(50, 20.0)
ChanDx2 = np.concatenate([[ChanDx[1]], (ChanDx[:-1]+ChanDx[1:])/2, [ChanDx[-1]]])
ChanElev = np.linspace(Slope*np.sum(ChanDx) - 2, -2, ChanDx.size+1)
ChanWidth = np.full(ChanDx.size+1, Width)
LagArea = np.full(ChanDx.size+1, 0.0)
Dist = np.insert(np.cumsum(ChanDx),0,0)
Closed = False

PhysicalPars = {'Gravity': 9.81,
                'GrainSize': 0.032,
                'VoidRatio': 0.4,
                'RhoRiv': 1000.0,
                'RhoSed': 2650.0,
                'CritShieldsStress': 0.0495,
                'MPM_coef': 3.97,
                'MPM_exp': 1.5,
                'Roughness': Roughness}
NumericalPars = {'Beta':1.1,
                 'Theta':0.7,
                 'FrRelax1':0.75,
                 'FrRelax2': 0.9,
                 'ErrTol':0.0001,
                 'MaxIt':20,
                 'WarnTol':0.1,
                 'Psi':0.4}
HydDt = pd.Timedelta(seconds=5)

Qin = 200
Q_Ts = np.full(100, Qin)
DsWL = 0.0
DsWl_Ts = np.full(100, DsWL)

# Steady solution
(ChanDep, ChanVel) = riv.solveSteady(ChanDx, ChanElev, ChanWidth, 
                                     Roughness, Qin, DsWL, NumericalPars)

# Calc bedload
Bedload = riv.calcBedload(ChanElev, ChanWidth, ChanDep, ChanVel, 
                          ChanDx, PhysicalPars, NumericalPars['Psi'])

LongSecFig = vis.longSection(ChanDx, ChanElev, ChanWidth, ChanDep, ChanVel, Bedload)

logging.info('Bedload at 200m3/s = %.1f m3(bulk)/s. According to spreadhseet calc it should equal 176.8', Bedload[0] * 3600)

