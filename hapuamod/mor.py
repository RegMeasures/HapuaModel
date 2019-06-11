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
    
    #%% Check if d/s end of outlet channel has migrated across a transect line and adjust as necessary...
    
    if OutletEndX[0] < OutletEndX[1]:
        # Outlet angles from L to R
        if ShoreX[OutletChanIx[-1]+1] <= OutletEndX[1]:
            logging.info('Outlet channel elongated rightwards across transect line')
            Extend = True
            ExtendMask = np.logical_and(ShoreX[OutletChanIx[-1]] < ShoreX,
                                        ShoreX <= OutletEndX[1])
        else:
            Extend = False    
    else:
        # Outlet angles from R to L
        if ShoreX[OutletChanIx[-1]-1] >= OutletEndX[1]:
            logging.info('Outlet channel elongated leftwards across transect line')
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
    
    #%% Check for outlet channel truncation
    # Note: have to be careful to leave at least 1 transect in outlet channel
    
    # Check for truncation of online outlet channel and move ends of channel if required
    # (Only check for truncation if outlet channel crosses >1 transect)
    if OutletChanIx.size > 1:
        
        # Check for truncation of lagoonward end of outlet channel 
        # (don't check last transect as trucation here would leave 0 transects)
        if np.any(ShoreY[OutletChanIx[1:],2] <= ShoreY[OutletChanIx[1:],3]):
            logging.info('Truncating lagoon end of outlet channel')
            TruncationIx = OutletChanIx[ShoreY[OutletChanIx,1] <= ShoreY[OutletChanIx,0]]
            if OutletEndX[0] < OutletEndX[1]:
                # Outlet angles from L to R
                OutletEndX[0] = ShoreX[TruncationIx[-1]] + Dx/2
                if ShoreY[TruncationIx[-1], 3] <= ShoreY[TruncationIx[-1], 4]:
                    logging.info('Extending R end of lagoon via outletchannel to cliffline collision.')
                    Extend = True
                    CurLagEndIx = np.where(ShoreY[:,3] > ShoreY[:,4])[0][-1]
                    LagExtension = np.arange(CurLagEndIx+1, TruncationIx[-1])
                else:
                    Extend = False
            else:
                # Outlet angles from R to L
                OutletEndX[0] = ShoreX[TruncationIx[0]] - Dx/2 
                if ShoreY[TruncationIx[0], 3] <= ShoreY[TruncationIx[0], 4]:
                    logging.info('Extending L end of lagoon via outletchannel to cliffline collision.')
                    Extend = True
                    CurLagEndIx = np.where(ShoreY[:,3] > ShoreY[:,4])[0][0]
                    LagExtension = np.arange(TruncationIx[0], CurLagEndIx-1)
                else:
                    Extend = False
            
            if Extend:
                logging.info('Schematisation requires removal of any remaining gravel between new lagoon and cliff-toe as part of extension')
                # Convert outlet channel to lagoon
                ShoreY[LagExtension,3] = ShoreY[LagExtension,1]
                LagoonElev[LagExtension] = OutletElev[LagExtension]
                # Remove outlet channel
                ShoreY[LagExtension,1] = np.nan
                ShoreY[LagExtension,2] = np.nan
                OutletElev[LagExtension] = np.nan
        
        # Check seaward end
        # Not sure what happens if truncation of both ends happens in same timestep???
        # (don't check first transect as trucation here would leave 0 transects)
        if np.any(ShoreY[OutletChanIx[:-1],1] >= ShoreY[OutletChanIx[:-1],0]):
            logging.info('Truncating seaward end of outlet channel')
            TruncationIx = OutletChanIx[ShoreY[OutletChanIx,1] >= ShoreY[OutletChanIx,0]]
            if OutletEndX[0] < OutletEndX[1]:
                # Outlet angles from L to R
                OutletEndX[1] = ShoreX[TruncationIx[0]] - Dx/2 
            else:
                # Outlet angles from R to L
                OutletEndX[1] = ShoreX[TruncationIx[-1]] + Dx/2 
    
    # if outlet has only 1 transect then adjust shoreline/lagoonline to preserve it's width when we adjust ShoreY in the next step
    if abs(OutletEndX[0]//Dx - OutletEndX[1]//Dx) <= 1:
        if OutletEndX[0]//Dx == OutletEndX[1]//Dx:
            if OutletEndX[0] < OutletEndX[1]:
                OutletChanIx = np.where(np.logical_and(OutletEndX[0] < ShoreX, 
                                                       ShoreX < OutletEndX[1]+Dx))[0]
            else:
                OutletChanIx = np.where(np.logical_and(OutletEndX[1]-Dx < ShoreX,
                                                       ShoreX < OutletEndX[0]))[0]
        else:
            OutletChanIx = np.where(np.logical_and(np.min(OutletEndX) < ShoreX,
                                                   ShoreX < np.max(OutletEndX[0])))[0]
        # Preserve channel width by extending into lagoon as required 
        # (not extending into sea as this would mess with LST)
        if ShoreY[OutletChanIx,1] > ShoreY[OutletChanIx,0]:
            ShoreY[OutletChanIx,2] -= ShoreY[OutletChanIx,1] - ShoreY[OutletChanIx,0]
            ShoreY[OutletChanIx,1] = ShoreY[OutletChanIx,0]
        ShoreY[OutletChanIx,3] = min(ShoreY[OutletChanIx,3], ShoreY[OutletChanIx,2])
            
    
    # Adjust ShoreY where outlet banks intersects coast or lagoon
    # Note this can include offline/disconnected bits of outlet as well as online bits
    OutletExists = ~np.isnan(ShoreY[:,1])
    ShoreIntersect = np.less(ShoreY[:,0], ShoreY[:,1], 
                             out = OutletExists, where=OutletExists)
    if np.any(ShoreIntersect):
        ShoreY[ShoreIntersect, 0] -= ((ShoreY[ShoreIntersect, 1] - ShoreY[ShoreIntersect, 2]) 
                                      * (BarrierElev[ShoreIntersect] - OutletElev[ShoreIntersect]) 
                                      / (BarrierElev[ShoreIntersect] + PhysicalPars['ClosureDepth']))
        ShoreY[ShoreIntersect, 1] = np.nan
        ShoreY[ShoreIntersect, 2] = np.nan
        OutletElev[ShoreIntersect] = np.nan
    
    LagoonIntersect = np.less(ShoreY[:,2], ShoreY[:,3], 
                              out = OutletExists, where=OutletExists)
    if np.any(LagoonIntersect):
        ShoreY[LagoonIntersect, 3] = ShoreY[LagoonIntersect, 1]
        ShoreY[LagoonIntersect, 1] = np.nan
        ShoreY[LagoonIntersect, 2] = np.nan
        OutletElev[LagoonIntersect] = np.nan
    