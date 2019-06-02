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
            LST (np.ndarray(float64)): Longshore transport rate at each 
                shoreline segment between 2 nodes (m^3/s)
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

def runup(WavePeriod, Hs_offshore, BeachSlope):
    """ Calculates wave runup height using Poate et al 2016.
        
        Parameters:
            WavePeriod (Tz) = mean wave period [s]
            Hs_offshore = significant deeep water wave height [m]
            BeachSlope = beach slope (0 < BeachSlope < 1) [m/m]
            
        Returns:
            Runup = 2% exceedance runup above stillwater level of the sea
                    (R_2%) [m]
        
        Reference:
            Poate T.G., McCall R.T., Masselink G. (2016) A new parameterisation
                for runup on gravel beaches. Coast Eng 117:176–190.
    """
    Runup = 0.49 * BeachSlope**0.5 * WavePeriod * Hs_offshore
    return Runup