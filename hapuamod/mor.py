# -*- coding: utf-8 -*-
"""
morphological updating and linking of different model components
"""

# import standard packages
import numpy as np
import logging

# import local modules
from hapuamod import geom

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
    
    return (ChanDx, ChanElev, ChanWidth, LagArea, OnlineLagoon)

def updateMorphology(LST, Bedload, 
                     ChanWidth, ChanDep, OnlineLagoon, RiverElev, 
                     OutletWidth, OutletElev, OutletX, OutletY, 
                     ShoreX, ShoreY, LagoonY, LagoonElev, BarrierElev,
                     Dx, Dt, PhysicalPars):
    
    #%% Pre-calculate some useful parameters
    
    # Find location outlet channel intersects shoreline
    OutletRbShoreIx = np.where(OutletX[-1]<ShoreX)[0][0]
    
    #%% 1D River model morphology
    # Change in volume at each cross-section (except the upstream Bdy)
    dVol = (Bedload[:-1]-Bedload[1:]) * Dt.seconds
    
    # Some current physical properties of each XS
    #BedArea = ChanDx * ChanWidth[1:] 
    AspectRatio = ChanWidth[1:]/ChanDep[1:]
    TooWide = AspectRatio > PhysicalPars['WidthRatio']
    OutletDx1 = np.sqrt((OutletX[1:]-OutletX[:-1])**2 + 
                        (OutletY[1:]-OutletY[:-1])**2)
    OutletDx2 = np.zeros(OutletElev.size)
    OutletDx2[[0,-1]] = OutletDx1[[0,-1]]
    OutletDx2[1:-1] = (OutletDx1[1:] + OutletDx1[:-1])/2
    
    # Update river bed elevation
    NRiv = RiverElev.size-1         # No of river cross-sections for updating
    NOut = OutletElev.size          # No of outlet channel cross-sections for updating
    EroVol = np.minimum(dVol, 0.0)  # Total erosion volume
    BedEro = EroVol * TooWide       # Bed erosion volume
    BankEro = EroVol * np.logical_not(TooWide) # Bank erosion volume
    BedDep = np.maximum(dVol, 0.0)  # Bed deposition volume (=total)
    # note that += updates variables in place so no need to explicitly return them!
    RiverElev[1:] += (BedDep[:NRiv] + BedEro[:NRiv]) / (PhysicalPars['RiverWidth'] * Dx)
    LagoonElev[OnlineLagoon] += ((BedDep[NRiv:-NOut] + BedEro[NRiv:-NOut])
                                 / ((LagoonY[OnlineLagoon,1] - LagoonY[OnlineLagoon,0]) * Dx))
    OutletElev += (BedDep[-NOut:] + BedEro[-NOut:]) / (OutletWidth * OutletDx2)
    
    # Update lagoon bank positions
    # Note: River upstream of lagoon has fixed width - all morpho change is on bed
    LagoonY[OnlineLagoon,0] += (BankEro[NRiv:-NOut]/2) / ((BarrierElev[OnlineLagoon] - LagoonElev[OnlineLagoon]) * Dx)
    LagoonY[OnlineLagoon,1] += (BankEro[NRiv:-NOut]/2) / ((PhysicalPars['BackshoreElev'] - LagoonElev[OnlineLagoon]) * Dx)
    
    # Track sed vol for outlet channel bank adjustment
    OutletLbEro = BankEro[-NOut:] / 2
    OutletRbEro = BankEro[-NOut:] / 2
    
    # Put sediment discharged from outlet onto shoreline 
    # TODO improve sediment distribution...
    ShoreY[OutletRbShoreIx-1:OutletRbShoreIx] += (Bedload[-1] / 2) * Dt.seconds / (PhysicalPars['ClosureDepth'] * Dx)
    
    #%% 1-Line shoreline model morphology
    
    # Update shoreline position
    # TODO add shoreline boundary conditions here (github issue #10)
    ShoreY[1:-1] += (LST[:-1] - LST[1:]) * Dt.seconds / (PhysicalPars['ClosureDepth'] * Dx)
    
    # Remove LST driven sed supply out of outlet channel and put on outlet channel bank instead
    if LST[OutletRbShoreIx-1]>0:
        # Transport from L to R
        ShoreY[OutletRbShoreIx] -= LST[OutletRbShoreIx-1] * Dt.seconds / (PhysicalPars['ClosureDepth'] * Dx)
        OutletLbEro[-1] -= (LST[OutletRbShoreIx-1] * Dt.seconds)
    else:
        ShoreY[OutletRbShoreIx-1] += LST[OutletRbShoreIx-1] * Dt.seconds / (PhysicalPars['ClosureDepth'] * Dx)
        OutletRbEro[-1] += (LST[OutletRbShoreIx-1] * Dt.seconds)
    
    #%% Update outlet channel width and position
    # TODO use actual barrier height, split L and R bank calculations and account for differences in bank height and movement of channel centerline!
    OutletBankElev = 3.0
    OutletWidth += (OutletLbEro / ((OutletBankElev-OutletElev) * OutletDx2) 
                    + OutletRbEro / ((OutletBankElev-OutletElev) * OutletDx2)) 
    geom.shiftLineSideways(OutletX, OutletY, (OutletRbEro-OutletLbEro)/2)
