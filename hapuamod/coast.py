# -*- coding: utf-8 -*-

# import standard packages
import numpy as np

def longShoreTransport(ShoreY, Dx, WavePower, WavePeriod, Wlen_h, EDir_h, PhysicalPars):
    """ Calculates longshore transport rate for each shore segment
    
    Calculation performed for whole model for a single timestep.
    
    Parameters:
        ShoreX
        Dx
        WavePower
        EAngleOffshore
    
    Returns:
        LST (np.ndarray(float64)): Longshore transport rate at each shoreline
            segment between 2 nodes (m^3/s)
    """
    
    # Calculate offshore wave angle relative to each shoreline segment
    LocalRelShoreDir = np.arctan((ShoreY[0:-1] - ShoreY[1:])/Dx)
    LocalEDir_h = EDir_h - LocalRelShoreDir
    
    # Calculate breaking wave depth
    BreakerDepth = (PhysicalPars['BreakerCoef'] * WavePower)**0.2
    
    # Calculate breaking wave angle
    C_b = (PhysicalPars['Gravity'] * BreakerDepth)**0.5
    C_h = ((PhysicalPars['Gravity'] * WavePeriod / (2.0 * np.pi)) * 
           np.tanh(2.0 * np.pi * PhysicalPars['WaveDataDepth'] / Wlen_h))
    # sin(alpha_b)/sin(alpha_h) = C_b/C_h
    BreakerAngle = np.arcsin(np.sin(LocalEDir_h) * (C_b / C_h))
    
    LST = - PhysicalPars['K2coef'] * WavePower * np.cos(LocalEDir_h) * np.sin(BreakerAngle)
    
    return LST

def shoreChange(LST, Dx, Dt, PhysicalPars):
    """ Calculate shoreline shift for a single timestep
    
    Dy = shoreChange(LST, Dx, Dt, PhysicalPars)
    Calculate shoreline shift for each shoreline node for a single timestep.
    
    Parameters:
        LST (np.ndarray(float)):
        Dx (float):
        Dt (float):
        PhysicalPars (dict):
    
    Returns:
        Dy (float): Shift in shoreline position for each node. +ve = accretion,
            -ve = erosion (m)
    """
    
    # TODO: handle shoreline boundary conditions
    
    Dy = np.zeros([LST.size + 1])
    Dy[1:-1] = (LST[0:-1] - LST[1:]) * Dt.seconds / (PhysicalPars['ClosureDepth'] * Dx)
    
    return Dy
    
def runup(WavePeriod, BeachSlope):
    """ Calculates wave runup height """
    print('test')