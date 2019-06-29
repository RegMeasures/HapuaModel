# -*- coding: utf-8 -*-
"""
morphological updating and linking of different model components
"""

# import standard packages
import numpy as np
import logging

# import local modules

def updateMorphology(ShoreX, ShoreY, LagoonElev, OutletElev, BarrierElev,
                    OutletEndX, OutletEndWidth, OutletEndElev, 
                    RiverElev, RiverWidth, OnlineLagoon, OutletChanIx, 
                    ChanWidth, ChanDep, ChanDx,
                    LST, Bedload, Dx, Dt, PhysicalPars):
    """ Update river, lagoon, outlet, barrier and shoreline morphology
    """
    
    #%% Pre-calculate some useful parameters
    
    # Find location outlet channel intersects shoreline
    OutletRbShoreIx = np.where(OutletEndX[1] < ShoreX)[0][0]
    
    #%% 1D River model morphology
    # Change in volume at each cross-section (except the upstream Bdy and dummy downstream XS)
    dVol = (Bedload[:-1]-Bedload[1:]) * Dt.seconds
    
    # Some current physical properties of each XS which will be updated
    AspectRatio = ChanWidth[1:-1]/ChanDep[1:-1]
    TooWide = AspectRatio > PhysicalPars['WidthRatio']
    
    # Update channel bed elevation
    NRiv = RiverElev.size-1         # No of river cross-sections for updating
    NOut = OutletChanIx.size + 2    # No of outlet channel cross-sections for updating
    EroVol = np.minimum(dVol, 0.0)  # Total erosion volume (-ve)
    BedEro = EroVol * TooWide       # Bed erosion volume (-ve)
    BankEro = EroVol * np.logical_not(TooWide) # Bank erosion volume (-ve)
    BedDep = np.maximum(dVol, 0.0)  # Bed deposition volume (+ve)
    # note that += modifies variables in place so no need to explicitly return them!
    # No change in river width allowed so all erosion applied to bed.
    RiverElev[1:] += (BedDep[:NRiv] + EroVol[:NRiv]) / (PhysicalPars['RiverWidth'] * Dx)
    LagoonElev[OnlineLagoon] += ((BedDep[NRiv:-NOut] + BedEro[NRiv:-NOut])
                                 / ((ShoreY[OnlineLagoon, 3] - ShoreY[OnlineLagoon, 4]) * Dx))
    OutletElev[OutletChanIx] += (BedDep[-NOut+1:-1] + BedEro[-NOut+1:-1]) / (ChanWidth[-NOut+1:-1] * Dx)
    OutletEndElev += (BedDep[[-NOut,-1]] + BedEro[[-NOut,-1]]) / (OutletEndWidth * ChanDx[[-NOut+1,-1]])
    
    # Update lagoon and outlet channel bank positions
    # Note: River upstream of lagoon has fixed width - all morpho change is on bed
    # TODO: Account for differences in bank height either side of outlet channel
    # Lagoon
    ShoreY[OnlineLagoon,3] -= (BankEro[NRiv:-NOut]/2) / ((BarrierElev[OnlineLagoon] - LagoonElev[OnlineLagoon]) * Dx)
    ShoreY[OnlineLagoon,4] += (BankEro[NRiv:-NOut]/2) / ((PhysicalPars['BackshoreElev'] - LagoonElev[OnlineLagoon]) * Dx)
    # Outlet channel
    ShoreY[OutletChanIx,1] -= (BankEro[-NOut+1:-1]/2) / ((BarrierElev[OutletChanIx] - OutletElev[OutletChanIx]) * Dx)
    ShoreY[OutletChanIx,2] += (BankEro[-NOut+1:-1]/2) / ((BarrierElev[OutletChanIx] - OutletElev[OutletChanIx]) * Dx)
    OutletEndWidth -= BankEro[[-NOut,-1]] / ((BarrierElev[OutletChanIx[[0,-1]]] - OutletEndElev) * ChanDx[[-NOut+1,-1]])
    
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
        if ShoreX[OutletChanIx[-1]+1] < OutletEndX[1]:
            Extend = True
            ExtendMask = np.logical_and(ShoreX[OutletChanIx[-1]] <= ShoreX,
                                        ShoreX <= OutletEndX[1])
            logging.info('Outlet channel elongated rightwards across transect line x=%f' % ShoreX[ExtendMask][-1])
        else:
            Extend = False    
    else:
        # Outlet angles from R to L
        if ShoreX[OutletChanIx[-1]-1] > OutletEndX[1]:
            Extend = True
            ExtendMask = np.logical_and(ShoreX[OutletChanIx[-1]] > ShoreX,
                                        ShoreX >= OutletEndX[1])
            logging.info('Outlet channel elongated leftwards across transect line x=%f' % ShoreX[ExtendMask][0])
        else:
            Extend = False
    
    if Extend:
        # Width of new outlet section = end width
        # Bed level of new outlet section  = end bed level
        # Dist from new outlet section to shoreline = 1/2 distance of last outlet section
        ShoreY[ExtendMask,1] = ShoreY[ExtendMask,0] - (ShoreY[OutletChanIx[-1],0] - ShoreY[OutletChanIx[-1],1])/2
        ShoreY[ExtendMask,2] = ShoreY[ExtendMask,1] - OutletEndWidth[1]
        OutletElev[ExtendMask] = OutletEndElev[1]
        # Update OutletChanIx as it has changed and its used later in this function
        if OutletEndX[0] < OutletEndX[1]:
            OutletChanIx = np.where(np.logical_and(OutletEndX[0] < ShoreX, 
                                                   ShoreX < OutletEndX[1]))[0]
        else:
            OutletChanIx = np.flipud(np.where(np.logical_and(OutletEndX[1] < ShoreX,
                                                             ShoreX < OutletEndX[0]))[0])
        
    
    #%% Check for outlet channel truncation
    # Note: have to be careful to leave at least 1 transect in outlet channel
    
    # Check for truncation of online outlet channel and move ends of channel if required
    # (Only check for truncation if outlet channel crosses >1 transect)
    if OutletChanIx.size > 1:
        
        # Check for truncation of lagoonward end of outlet channel 
        # (don't check last transect as trucation here would leave 0 transects)
        if np.any(ShoreY[OutletChanIx[:-1],2] <= ShoreY[OutletChanIx[:-1],3]):
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
        if np.any(ShoreY[OutletChanIx[1:],1] >= ShoreY[OutletChanIx[1:],0]):
            logging.info('Truncating seaward end of outlet channel')
            TruncationIx = OutletChanIx[ShoreY[OutletChanIx,1] >= ShoreY[OutletChanIx,0]]
            if OutletEndX[0] < OutletEndX[1]:
                # Outlet angles from L to R
                OutletEndX[1] = ShoreX[TruncationIx[0]] - Dx/2 
            else:
                # Outlet angles from R to L
                OutletEndX[1] = ShoreX[TruncationIx[-1]] + Dx/2 
    
    # if outlet has only 1 transect then adjust shoreline/lagoonline to preserve it's width when we adjust ShoreY in the next step
    else:
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
            ShoreY[OutletChanIx,1] = ShoreY[OutletChanIx,0] - 0.0001
        ShoreY[OutletChanIx,3] = min(ShoreY[OutletChanIx,3], ShoreY[OutletChanIx,2]-0.0001)
            
    
    # Adjust ShoreY where outlet banks intersects coast or lagoon
    # Note this can include offline/disconnected bits of outlet as well as online bits
    OutletExists = np.where(~np.isnan(ShoreY[:,1]))[0]
    ShoreIntersect = ShoreY[OutletExists,0] < ShoreY[OutletExists,1]
    if np.any(ShoreIntersect):
        logging.info('Outlet intersects shoreline at %i transects - filling outlet with sediment from shoreface' % np.sum(ShoreIntersect))
        IntTsects = OutletExists[ShoreIntersect]
        ShoreY[IntTsects, 0] -= ((ShoreY[IntTsects, 1] - ShoreY[IntTsects, 2]) 
                                 * (BarrierElev[IntTsects] - OutletElev[IntTsects]) 
                                 / (BarrierElev[IntTsects] + PhysicalPars['ClosureDepth']))
        ShoreY[IntTsects, 1] = np.nan
        ShoreY[IntTsects, 2] = np.nan
        OutletElev[IntTsects] = np.nan
    
    LagoonIntersect = ShoreY[OutletExists,2] < ShoreY[OutletExists,3]
    if np.any(LagoonIntersect):
        logging.info('Outlet intersects lagoon at %i transects - adding channel width into lagoon' % np.sum(LagoonIntersect))
        IntTsects = OutletExists[LagoonIntersect]
        # TODO: handle sediment balance properly here based on difference in bed elevation between lagoon and channel?
        ShoreY[IntTsects, 3] = ShoreY[IntTsects, 1]
        ShoreY[IntTsects, 1] = np.nan
        ShoreY[IntTsects, 2] = np.nan
        OutletElev[IntTsects] = np.nan
    