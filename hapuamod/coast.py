# -*- coding: utf-8 -*-

# import standard packages
import numpy as np

def longShoreTransport(ShoreY, Dx, WavePower, WavePeriod, Wlen_h, EDir_h, PhysicalPars):
    """ Calculates longshore transport rate for each shore segment
    
        Longshore transport rate is calculated using the CERC sediment 
        transport formula (CERC 1984). The calculation is performed for the 
        whole model for a single timestep.
        
        Parameters:
            ShoreY (np.ndarray(float64)): The first column of ShoreY specifies 
                the shoreline position at each shore profile relative to a 
                straight line through the initial shoreline position (m).
            Dx (float): Alongshore discretisation distance between shore 
                profiles in ShoreY (m).
            WavePower (float): Wave power (W/m wave crest length) 
            Wlen_h: (float): Wavelength (m) at depth given by
                PhysicalPars['WaveDataDepth'] 
            EDir_h (float): Net direction of wave energy flux (in radians,
                relative to shore normal, -ve = waves arriving from left of 
                shore-normal, +ve = waves arriving from right of shore-normal)
                 at depth given by PhysicalPars['WaveDataDepth'].
            PhysicalPars: Physical parameters including:
                BreakerCoef (float): see notes below.
                K2coef (float): see notes below.
                Gravity (float): (ms^-2)
                WaveDataDepth (float): depth at which input wave data is 
                    specified (m)
        
        Returns:
            LST (np.ndarray(float64)): Longshore transport rate at each 
                shoreline segment between 2 nodes (m^3/s)
        
        Notes: 
            The calculation uses PhysicalPars['K2coef'] as a transport 
                coefficient, but this has already been calculated in 
                loadmod.loadModel as:
                    K2 = K / (RhoSed - RhoSea) * g * (1 - VoidRatio))
            PhysicalPars['BreakerCoef'] is calculated in loadmod.loadModel as:
                BreakerCoef = 8 / (RhoSea * Gravity^1.5 * GammaRatio^2)
            EDir_h is required in radians. Conversion from degrees to radians
                is done by loadmod.loadModel when the wave timeseries are read
                during model initialisation.
        
        Reference:
            Coastal Engineering Research Center (1984) Shore protection manual. 
            US Army Corps of Engineers, Vicksburg, Mississippi. 
            https://doi.org/10.5962/bhl.title.47830
    """
    
    # Calculate offshore wave angle relative to each shoreline segment
    LocalRelShoreDir = np.arctan((ShoreY[0:-1,0] - ShoreY[1:,0])/Dx)
    LocalEDir_h = EDir_h - LocalRelShoreDir
    
    # Set wave power to zero for segments where waves are directed offshore
    LocalWavePower = np.full(LocalEDir_h.size, WavePower)
    LocalWavePower[LocalEDir_h < (-np.pi/2)] = 0
    LocalWavePower[LocalEDir_h > (np.pi/2)] = 0
    
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

def overtopping(Runup, SeaLevel, ShoreY, ShoreZ, PhysicalPars):
    """ Overtopping/overwashing sediment fluxes to barrier crest and backshore
        
        (CST_tot, OverwashProp) = overtopping(Runup, SeaLevel, ShoreY, ShoreZ, 
                                              PhysicalPars)
        Parameters:
            Runup
            SeaLevel
            ShoreY
            ShoreZ
            
        Returns:
            CST_tot = total cross-shore sediment flux at each transect (m3/s/m)
            OverwashProp = proportion of CST_tot overwashing at each transect
        
    """
    # Overtopping potential
    OP = np.maximum(Runup + SeaLevel - ShoreZ[:,0], 0.0)
    
    # Total sed flux
    CST_tot = PhysicalPars['OT_coef'] * OP**PhysicalPars['OT_exp']
    
    # Barrier width
    OutletPresent = ~np.isnan(ShoreY[:,1])
    CrestWidth = np.maximum(ShoreY[:,0] - ShoreY[:,3], 0.001)
    CrestWidth[OutletPresent] = ShoreY[OutletPresent,0] - ShoreY[OutletPresent,1]
    
    # Split sed flux between crest and backshore (except where no lagoon)
    NoLagoon = ShoreY[:,3] <= ShoreY[:,4]
    OverwashProp = np.minimum(PhysicalPars['OwProp_coef'] * OP / CrestWidth, 1.)
    OverwashProp[NoLagoon] = 0.0
    
    # Checks
    assert np.all(np.logical_and(OverwashProp>=0.0, OverwashProp<=1.0)), 'OverwashProp outside of range 0-1 in overtopping function'
    assert np.all(CST_tot >= 0), 'Negative cross-shore transport in overtopping function!'
    
    return(CST_tot, OverwashProp)
    
    

    
    