# -*- coding: utf-8 -*-
"""
morphological updating and linking of different model components
"""

# import standard packages
import numpy as np
import logging

def assembleChannel(RiverElev, ShoreX, LagoonY, LagoonElev, OutletX, OutletY,
                    OutletElev, OutletWidth, RiverWidth, Dx):
    """ Combine river, lagoon and outlet into single channel for hyd-calcs
    
    (ChanDx, ChanElev, ChanWidth, LagArea) = \
        riv.assembleChannel(RiverElev, ShoreX, LagoonY, LagoonElev, OutletDx, 
                            OutletElev, OutletWidth, OutletX, RiverWidth, Dx)
    """
    LagoonWidth = LagoonY[:,1] - LagoonY[:,0]
    OutletDx = np.sqrt((OutletX[1:]-OutletX[0:-1])**2 + 
                       (OutletY[1:]-OutletY[0:-1])**2)
    # TODO: Account for potentially dry parts of the lagoon when 
    #       calculating ChanArea
    
    # To identify the 'online' section of lagoon the river flows through we 
    # first need to know which direction the offset is
    if OutletX[0] > 0:
        # Outlet channel to right of river
        OnlineLagoon = np.where(np.logical_and(0 <= ShoreX, ShoreX <= OutletX[0]))[0]
        StartArea = np.nansum(LagoonWidth[ShoreX < 0] * Dx)
        EndArea = np.nansum(LagoonWidth[ShoreX > OutletX[0]] * Dx)
    else:
        # Outlet channel to left of river
        OnlineLagoon = np.flipud(np.where(np.logical_and(0 >= ShoreX, ShoreX >= OutletX[0]))[0])
        StartArea = np.nansum(LagoonWidth[ShoreX > 0] * Dx)
        EndArea = np.nansum(LagoonWidth[ShoreX < OutletX[0]] * Dx)
    
    ChanDx = np.concatenate([np.tile(Dx, RiverElev.size + OnlineLagoon.size),
                             OutletDx])
    ChanElev = np.concatenate([RiverElev, LagoonElev[OnlineLagoon], OutletElev])
    ChanWidth = np.concatenate([np.tile(RiverWidth, RiverElev.size), 
                                LagoonWidth[OnlineLagoon], OutletWidth])
    LagArea = np.zeros(ChanElev.size)
    LagArea[RiverElev.size] = StartArea
    LagArea[-(OutletX.size+1)] = EndArea
    
    return (ChanDx, ChanElev, ChanWidth, LagArea)

def riverMorphology(Qs, B, h, z, BankElev, dx, dt, PhysicalPars):
    
    # Change in volume at each cross-section (except the upstream Bdy)
    dVol = (Qs[:-1]-Qs[1:]) * dt
    
    # Some current physical properties of each XS
    BedArea = dx * B[1:] 
    AspectRatio = B[1:]/h[1:]
    TooWide = AspectRatio > PhysicalPars['WidthRatio']
    
    # Update bed elevation
    ErosionVol = np.minimum(dVol, 0.0)
    z[1:] += (np.maximum(dVol, 0) + ErosionVol * TooWide)/BedArea
    
    # Update channel width
    B[1:] += (-ErosionVol * np.logical_not(TooWide)) / ((BankElev[1:]-z[1:])*dx)
    # TODO split L and R bank calculations and account for differences in bank height

#def updateMorphology(LST, Bedload, ShoreX, ShoreY, LagoonY, OutletX, OutletWidth, Dx, Dt):
#    """
#    """
#    
#    
#    
#    return (ShoreY)