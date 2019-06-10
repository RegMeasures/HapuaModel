# -*- coding: utf-8 -*-
"""
morphological updating and linking of different model components
"""

# import standard packages
import numpy as np
import logging

# import local modules
from hapuamod import geom

def assembleChannel(ShoreX, ShoreY, LagoonElev, OutletElev, 
                    OutletEndX, OutletEndWidth, OutletEndElev, 
                    RiverElev, RiverWidth, Dx):
    """ Combine river, lagoon and outlet into single channel for hyd-calcs
    
    (ChanDx, ChanElev, ChanWidth, LagArea, OnlineLagoon) = \
        riv.assembleChannel(ShoreX, ShoreY, LagoonElev, OutletElev, 
                            OutletEndX, OutletEndWidth, OutletEndElev, 
                            RiverElev, RiverWidth, Dx)
    """
    
    # Find location and orientation of outlet channel
    if OutletEndX[0] < OutletEndX[1]:
        # Outlet angles from L to R
        OutletChanIx = np.where(np.logical_and(OutletEndX[0] < ShoreX, 
                                               ShoreX < OutletEndX[1]))[0]
    else:
        # Outlet from R to L
        OutletChanIx = np.flipud(np.where(OutletEndX[1] < ShoreX < OutletEndX[0])[0])
    OutletWidth = ShoreY[OutletChanIx,1] - ShoreY[OutletChanIx,2]
    
    # TODO: Account for potentially dry parts of the lagoon when 
    #       calculating ChanArea
    
    # Calculate properties for the 'online' section of lagoon
    LagoonWidth = ShoreY[:,3] - ShoreY[:,4]
    if OutletEndX[0] > 0:
        # Outlet channel to right of river
        OnlineLagoon = np.where(np.logical_and(0 <= ShoreX, ShoreX <= OutletEndX[0]))[0]
        StartArea = np.nansum(LagoonWidth[ShoreX < 0] * Dx)
        EndArea = np.nansum(LagoonWidth[ShoreX > OutletEndX[0]] * Dx)
    else:
        # Outlet channel to left of river
        OnlineLagoon = np.flipud(np.where(np.logical_and(0 >= ShoreX, ShoreX >= OutletEndX[0]))[0])
        StartArea = np.nansum(LagoonWidth[ShoreX > 0] * Dx)
        EndArea = np.nansum(LagoonWidth[ShoreX < OutletEndX[0]] * Dx)
    
    # Assemble the complete channel
    ChanDx = np.tile(Dx, RiverElev.size + OnlineLagoon.size + OutletChanIx.size + 1)
    if OutletEndX[0]//Dx == OutletEndX[1]//Dx:
        ChanDx[-1] += Dx + abs(OutletEndX[1] - OutletEndX[0])
    elif OutletEndX[0] < OutletEndX[1]:
        ChanDx[RiverElev.size + OnlineLagoon.size] += Dx - (OutletEndX[0] % Dx)
        ChanDx[-1] += OutletEndX[1] % Dx
    else:
        ChanDx[RiverElev.size + OnlineLagoon.size] += OutletEndX[0] % Dx
        ChanDx[-1] += Dx - (OutletEndX[1] % Dx)
    ChanElev = np.concatenate([RiverElev, LagoonElev[OnlineLagoon], 
                               [OutletEndElev[0]], OutletElev[OutletChanIx], 
                               [OutletEndElev[1]]])
    ChanWidth = np.concatenate([np.tile(RiverWidth, RiverElev.size), 
                                LagoonWidth[OnlineLagoon], [OutletEndWidth[0]],
                                OutletWidth, [OutletEndWidth[1]]])
    LagArea = np.zeros(ChanElev.size)
    LagArea[RiverElev.size] = StartArea
    LagArea[-(OutletChanIx.size+3)] = EndArea
    
    return (ChanDx, ChanElev, ChanWidth, LagArea, OnlineLagoon, OutletChanIx)

def updateMorphology(ShoreX, ShoreY, LagoonElev, OutletElev, BarrierElev,
                    OutletEndX, OutletEndWidth, OutletEndElev, 
                    RiverElev, RiverWidth, OnlineLagoon, OutletChanIx, 
                    ChanWidth, ChanDep, ChanDx,
                    LST, Bedload, Dx, Dt, PhysicalPars):
    
    #%% Pre-calculate some useful parameters
    
    # Find location outlet channel intersects shoreline
    OutletRbShoreIx = np.where(OutletEndX[1] < ShoreX)[0][0]
    
    #%% 1D River model morphology
    # Change in volume at each cross-section (except the upstream Bdy)
    dVol = (Bedload[:-1]-Bedload[1:]) * Dt.seconds
    
    # Some current physical properties of each XS
    #BedArea = ChanDx * ChanWidth[1:] 
    AspectRatio = ChanWidth[1:]/ChanDep[1:]
    TooWide = AspectRatio > PhysicalPars['WidthRatio']
    OutletDx2 = np.zeros(OutletChanIx.size + 2)
    OutletDx2[-1] = ChanDx[-1]
    OutletDx2[0:-1] = (ChanDx[-(OutletChanIx.size+2):-1] + ChanDx[-(OutletChanIx.size+1):-1])/2
    
    # Update channel bed elevation
    NRiv = RiverElev.size-1         # No of river cross-sections for updating
    NOut = OutletChanIx.size + 2    # No of outlet channel cross-sections for updating
    EroVol = np.minimum(dVol, 0.0)  # Total erosion volume
    BedEro = EroVol * TooWide       # Bed erosion volume
    BankEro = EroVol * np.logical_not(TooWide) # Bank erosion volume
    BedDep = np.maximum(dVol, 0.0)  # Bed deposition volume (=total)
    # note that += modifies variables in place so no need to explicitly return them!
    RiverElev[1:] += (BedDep[:NRiv] + BedEro[:NRiv]) / (PhysicalPars['RiverWidth'] * Dx)
    LagoonElev[OnlineLagoon] += ((BedDep[NRiv:-NOut] + BedEro[NRiv:-NOut])
                                 / ((ShoreY[OnlineLagoon, 3] - ShoreY[OnlineLagoon, 4]) * Dx))
    OutletElev[OutletChanIx] += (BedDep[-NOut:] + BedEro[-NOut:]) / (ChanWidth[-NOut:] * OutletDx2)
    
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
    ShoreY[1:-1] += ((LST[:-1] - LST[1:]) * Dt.seconds 
                     / ((PhysicalPars['ClosureDepth'] + BarrierElev[1:-1]) * Dx))
    
    # Remove LST driven sed supply out of outlet channel and put on outlet channel bank instead
    if LST[OutletRbShoreIx-1]>0:
        # Transport from L to R
        ShoreY[OutletRbShoreIx] -= (LST[OutletRbShoreIx-1] * Dt.seconds 
                                    / ((PhysicalPars['ClosureDepth'] 
                                        + BarrierElev[OutletRbShoreIx]) * Dx))
        OutletLbEro[-1] -= (LST[OutletRbShoreIx-1] * Dt.seconds)
    else:
        ShoreY[OutletRbShoreIx-1] += (LST[OutletRbShoreIx-1] * Dt.seconds 
                                      / ((PhysicalPars['ClosureDepth'] 
                                          + BarrierElev[OutletRbShoreIx-1]) * Dx))
        OutletRbEro[-1] += (LST[OutletRbShoreIx-1] * Dt.seconds)
        
    #%% Cross-shore morphology (overtopping etc)
    
    # Calculate barrier width of seaward patr of barrier 
    # (including the influence of the outlet channel)
    
    # Calculate volume changes to barrier width and height
    
    # Apply volume changes
    
    
    #%% Update outlet channel width and position
    # TODO use actual barrier height, split L and R bank calculations and account for differences in bank height and movement of channel centerline!
    OutletBankElev = 3.0
    OutletWidth += (OutletLbEro / ((OutletBankElev-OutletElev) * OutletDx2) 
                    + OutletRbEro / ((OutletBankElev-OutletElev) * OutletDx2))
    geom.shiftLineSideways(OutletX, OutletY, (OutletRbEro-OutletLbEro)/2)
    
    # trim/extend outlet channel ends as necessary
    geom.trimLine(OutletX, OutletY, ShoreX, LagoonY[:,1], ShoreY)
    
    # adjust outlet channel segmentation as necessary
    (OutletX, OutletY, OutletElev, OutletWidth) = \
        geom.adjustLineDx(OutletX, OutletY, Dx, OutletElev, OutletWidth)
