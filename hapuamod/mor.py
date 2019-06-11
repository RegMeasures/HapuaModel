# -*- coding: utf-8 -*-
"""
morphological updating and linking of different model components
"""

# import standard packages
import numpy as np
import logging

# import local modules

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
    if OutletEndX[0]//Dx == OutletEndX[1]//Dx:
        # Outlet doesn't cross any transects
        if OutletEndX[0] < OutletEndX[1]:
            OutletChanIx = np.where(np.logical_and(OutletEndX[0] < ShoreX, 
                                                   ShoreX < OutletEndX[1]+Dx))[0]
        else:
            OutletChanIx = np.where(np.logical_and(OutletEndX[1]-Dx < ShoreX,
                                                   ShoreX < OutletEndX[0]))[0]
    elif OutletEndX[0] < OutletEndX[1]:
        # Outlet angles from L to R
        OutletChanIx = np.where(np.logical_and(OutletEndX[0] < ShoreX, 
                                               ShoreX < OutletEndX[1]))[0]
    else:
        # Outlet from R to L
        OutletChanIx = np.flipud(np.where(np.logical_and(OutletEndX[1] < ShoreX,
                                                         ShoreX < OutletEndX[0]))[0])
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
        # Outlet is straight (i.e. doesn't cross any transects)
        ChanDx[-2] += abs(OutletEndX[1] - OutletEndX[0])
    elif OutletEndX[0] < OutletEndX[1]:
        # Outlet angles from L to R
        ChanDx[RiverElev.size + OnlineLagoon.size] += Dx - (OutletEndX[0] % Dx)
        ChanDx[-1] += OutletEndX[1] % Dx
    else:
        # Outlet from R to L
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
    OutletElev[OutletChanIx] += (BedDep[-NOut+1:-1] + BedEro[-NOut+1:-1]) / (ChanWidth[-NOut+1:-1] * Dx)
    OutletEndElev += (BedDep[[-NOut,-1]] + BedEro[[-NOut,-1]]) / (OutletEndWidth * ChanDx[[-NOut+1,-1]])
    
    # Update lagoon and outlet channel bank positions
    # Note: River upstream of lagoon has fixed width - all morpho change is on bed
    # TODO: Account for differences in bank height either side of outlet channel
    ShoreY[OutletChanIx,1] += (BankEro[-NOut+1:-1]/2) / ((BarrierElev[OutletChanIx] - OutletElev[OutletChanIx]) * Dx)
    ShoreY[OutletChanIx,2] += (BankEro[-NOut+1:-1]/2) / ((BarrierElev[OutletChanIx] - OutletElev[OutletChanIx]) * Dx)
    ShoreY[OnlineLagoon,3] += (BankEro[NRiv:-NOut]/2) / ((BarrierElev[OnlineLagoon] - LagoonElev[OnlineLagoon]) * Dx)
    ShoreY[OnlineLagoon,4] += (BankEro[NRiv:-NOut]/2) / ((PhysicalPars['BackshoreElev'] - LagoonElev[OnlineLagoon]) * Dx)
    
    OutletEndWidth += BankEro[[-NOut,-1]] / ((BarrierElev[OutletChanIx[[0,-1]]] - OutletEndElev) * ChanDx[[-NOut+1,-1]])
    
    # Put sediment discharged from outlet onto shoreline 
    # TODO improve sediment distribution...
    ShoreY[OutletRbShoreIx-1:OutletRbShoreIx,0] += (Bedload[-1] / 2) * Dt.seconds / ((PhysicalPars['ClosureDepth'] + BarrierElev[OutletRbShoreIx-1:OutletRbShoreIx]) * Dx)
    
    #%% 1-Line shoreline model morphology
    
    # Update shoreline position
    # TODO add shoreline boundary conditions here (github issue #10)
    ShoreY[1:-1,0] += ((LST[:-1] - LST[1:]) * Dt.seconds 
                       / ((PhysicalPars['ClosureDepth'] + BarrierElev[1:-1]) * Dx))
    
    # Remove LST driven sed supply out of outlet channel and put on outlet channel bank instead
    if LST[OutletRbShoreIx-1]>0:
        # Transport from L to R
        ShoreY[OutletRbShoreIx,0] -= (LST[OutletRbShoreIx-1] * Dt.seconds 
                                      / ((PhysicalPars['ClosureDepth'] 
                                          + BarrierElev[OutletRbShoreIx]) * Dx))
        WidthReduction = (LST[OutletRbShoreIx-1] * Dt.seconds) / ((BarrierElev[OutletChanIx[-1]] - OutletEndElev[1]) * ChanDx[-1])
        OutletEndWidth[1] -= WidthReduction
        OutletEndX[1] += WidthReduction/2
    else:
        # Transport from R to L
        ShoreY[OutletRbShoreIx-1,0] += (LST[OutletRbShoreIx-1] * Dt.seconds 
                                        / ((PhysicalPars['ClosureDepth'] 
                                            + BarrierElev[OutletRbShoreIx-1]) * Dx))
        WidthReduction = (-LST[OutletRbShoreIx-1] * Dt.seconds) / ((BarrierElev[OutletChanIx[-1]] - OutletEndElev[1]) * ChanDx[-1])
        OutletEndWidth[1] -= WidthReduction
        OutletEndX[1] -= WidthReduction/2
        
    #%% Cross-shore morphology (overtopping etc)
    
    # Calculate barrier width of seaward patr of barrier 
    # (including the influence of the outlet channel)
    
    # Calculate volume changes to barrier width and height
    
    # Apply volume changes
    
    
    #%% Check d/s end of outlet channel hasn't migrated across a transect line and adjust as necessary...
    
    if OutletEndX[0] < OutletEndX[1]:
        # Outlet angles from L to R
        if ShoreX[OutletChanIx[-1]+1] <= OutletEndX[1]:
            logging.debug('Outlet channel elongated rightwards across transect line')
            Extend = True
            ExtendMask = np.logical_and(ShoreX[OutletChanIx[-1]] < ShoreX,
                                        ShoreX <= OutletEndX[1])
        else:
            Extend = False    
    else:
        # Outlet angles from R to L
        if ShoreX[OutletChanIx[-1]-1] >= OutletEndX[1]:
            logging.debug('Outlet channel elongated leftwards across transect line')
            Extend = True
            ExtendMask = np.logical_and(ShoreX[OutletChanIx[-1]] > ShoreX,
                                        ShoreX >= OutletEndX[1])
        else:
            Extend = False
    
    if Extend:
        # Width of new outlet section = end width
        # Bed level of new outlet section  = end bed level
        # Dist from new outlet section to shoreline = 1/2 distance of last outlet section
        ShoreY[ExtendMask,1] = ShoreY[ExtendMask,0] - (ShoreY[OutletChanIx[-1],0] - ShoreY[OutletChanIx[-1],1])/2
        ShoreY[ExtendMask,2] = ShoreY[ExtendMask,1] - OutletEndWidth[1]
        OutletElev[ExtendMask] = OutletEndElev[1]
    
    