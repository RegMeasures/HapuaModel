# -*- coding: utf-8 -*-
"""
morphological updating and linking of different model components
"""

# import standard packages
import numpy as np
import logging

# import local modules

def updateMorphology(ShoreX, ShoreY, ShoreZ,
                     OutletEndX, OutletEndWidth, OutletEndElev, 
                     RiverElev, RiverWidth, OnlineLagoon, OutletChanIx, 
                     LagoonWL, OutletWL,
                     ChanWidth, ChanDep, ChanDx, ChanFlag, Closed,
                     LST, Bedload, CST_tot, OverwashProp,
                     Dx, Dt, PhysicalPars):
    """ Update river, lagoon, outlet, barrier and shoreline morphology
    """
    
    #%% Pre-calculate some useful parameters
    
    # Find location outlet channel intersects shoreline
    if not Closed:
        OutletRbShoreIx = np.where(OutletEndX[1] < ShoreX)[0][0]
    
    # Shoreface height for converting volumetric change into advance retreat of shoreline
    ShorefaceHeight = PhysicalPars['ClosureDepth'] + ShoreZ[:,0]
    
    # Locations where an outlet channel exists in the barrier
    # Note not all "OutletPresent" are online/connected to sea...
    OutletPresent = np.where(~np.isnan(ShoreY[:,1]))[0]
    
    #%% 1D River model morphology
    
    if Closed:
        # Change in volume at each cross-section (except the upstream Bdy)
        dVol = (Bedload-np.concatenate([Bedload[1:],[0]])) * Dt.seconds
        AspectRatio = ChanWidth[1:]/ChanDep[1:]
        RivXS = ChanFlag[1:]==0
        LagXS = ChanFlag[1:]==1
    else:
        # Change in volume at each cross-section (except the upstream Bdy and dummy downstream XS)
        dVol = (Bedload[:-1]-Bedload[1:]) * Dt.seconds
        AspectRatio = ChanWidth[1:-1]/ChanDep[1:-1]
        RivXS = ChanFlag[1:-1]==0
        LagXS = ChanFlag[1:-1]==1
        OutXS = ChanFlag==3
        OutEndXS = np.where(ChanFlag==2)[0]
    TooWide = AspectRatio > PhysicalPars['WidthRatio']
    
    # Update channel bed elevation
    EroVol = np.minimum(dVol, 0.0)  # Total erosion volume (-ve)
    BedEro = EroVol * TooWide       # Bed erosion volume (-ve)
    BankEro = EroVol * np.logical_not(TooWide) # Bank erosion volume (-ve)
    BedDep = np.maximum(dVol, 0.0)  # Bed deposition volume (+ve)
    # note that += modifies variables in place so no need to explicitly return them!
    # No change in river width allowed so all erosion applied to bed.
    RiverElev[1:] += (BedDep[RivXS] + EroVol[RivXS]) / (PhysicalPars['RiverWidth'] * Dx)
    ShoreZ[OnlineLagoon, 3] += ((BedDep[LagXS] + BedEro[LagXS])
                                / ((ShoreY[OnlineLagoon, 3] - ShoreY[OnlineLagoon, 4]) * Dx))
    if not Closed:
        ShoreZ[OutletChanIx, 1] += (BedDep[OutXS[1:-1]] + BedEro[OutXS[1:-1]]) / (ChanWidth[OutXS] * Dx)
        OutletEndElev += (BedDep[OutEndXS-1] + BedEro[OutEndXS-1]) / (OutletEndWidth * Dx)
    
    # Update lagoon and outlet channel bank positions
    # Note: River upstream of lagoon has fixed width - all morpho change is on bed
    LagoonBankElev = ShoreZ[:,0]
    LagoonBankElev[OutletPresent] = ShoreZ[OutletPresent, 2]
    ShoreY[OnlineLagoon,3] -= (BankEro[LagXS]/2) / ((LagoonBankElev[OnlineLagoon] - ShoreZ[OnlineLagoon,3]) * Dx)
    ShoreY[OnlineLagoon,4] += (BankEro[LagXS]/2) / ((PhysicalPars['BackshoreElev'] - ShoreZ[OnlineLagoon,3]) * Dx)
    # Outlet channel
    if not Closed:
        ShoreY[OutletChanIx,1] -= (BankEro[OutXS[1:-1]]/2) / ((ShoreZ[OutletChanIx,0] - ShoreZ[OutletChanIx,1]) * Dx)
        ShoreY[OutletChanIx,2] += (BankEro[OutXS[1:-1]]/2) / ((ShoreZ[OutletChanIx,2] - ShoreZ[OutletChanIx,1]) * Dx)
        
        OutletEndWidth[0] -= BankEro[OutEndXS[0]-1] / ((ShoreZ[OutletChanIx[0],0] - OutletEndElev[0]) * Dx)
        OutletEndWidth[1] -= BankEro[OutEndXS[1]-1] / ((PhysicalPars['SpitHeight'] - OutletEndElev[1]) * PhysicalPars['SpitWidth'])
        
        # Put sediment discharged from outlet onto shoreline 
        # TODO improve sediment distribution...
        ShoreY[[OutletRbShoreIx-1,OutletRbShoreIx],0] += ((Bedload[-1] / 2) * Dt.seconds 
                                                        / (ShorefaceHeight[[OutletRbShoreIx-1,OutletRbShoreIx]] * Dx))
    
    #%% 1-Line shoreline model morphology
    
    # Update shoreline position
    # TODO add shoreline boundary conditions here (github issue #10)
    ShoreY[1:-1,0] += ((LST[:-1] - LST[1:]) * Dt.seconds 
                       / (ShorefaceHeight[1:-1] * Dx))
    
    # Remove LST driven sed supply out of outlet channel and put on outlet channel bank instead
    if not Closed:
        if LST[OutletRbShoreIx-1]>0:
            # Transport from L to R
            ShoreY[OutletRbShoreIx,0] -= (LST[OutletRbShoreIx-1] * Dt.seconds 
                                          / (ShorefaceHeight[OutletRbShoreIx] * Dx))
            WidthReduction = ((LST[OutletRbShoreIx-1] * Dt.seconds) 
                              / ((PhysicalPars['SpitHeight'] - OutletEndElev[1]) * PhysicalPars['SpitWidth']))
            OutletEndWidth[1] -= WidthReduction
            OutletEndX[1] += WidthReduction/2
        else:
            # Transport from R to L
            ShoreY[OutletRbShoreIx-1,0] += (LST[OutletRbShoreIx-1] * Dt.seconds 
                                            / (ShorefaceHeight[OutletRbShoreIx-1] * Dx))
            WidthReduction = ((-LST[OutletRbShoreIx-1] * Dt.seconds) 
                              / ((PhysicalPars['SpitHeight'] - OutletEndElev[1]) * PhysicalPars['SpitWidth']))
            OutletEndWidth[1] -= WidthReduction
            OutletEndX[1] -= WidthReduction/2
        
    #%% Cross-shore morphology (overtopping etc)
    
    CrestWidth = ShoreY[:,0] - ShoreY[:,3]
    CrestWidth[OutletPresent] = ShoreY[OutletPresent,0] - ShoreY[OutletPresent,1]
    
    BackBarHeight = ShoreZ[:,0] - ShoreZ[:,3]
    BackBarHeight[OutletPresent] = ShoreZ[OutletPresent,0] - ShoreZ[OutletPresent,1]
    
    # Accumulation of sediment on top of the barrier
    ShoreZ[:,0] += (1-OverwashProp) * CST_tot * Dt.seconds / CrestWidth
    
    # Accumulation of sediment on the back of the barrier
    ShoreY[~OutletPresent,3] -= OverwashProp[~OutletPresent] * CST_tot[~OutletPresent] * Dt.seconds / BackBarHeight[~OutletPresent]
    ShoreY[OutletPresent,1] -= OverwashProp[OutletPresent] * CST_tot[OutletPresent] * Dt.seconds / BackBarHeight[OutletPresent]
    
    # Erosion of sediment off the shoreface (don't move ends)
    ShoreY[1:-1,0] -= CST_tot[1:-1] * Dt.seconds / ShorefaceHeight[1:-1]
    
    #%% Check if d/s end of outlet channel has migrated across a transect line and adjust as necessary...
    if not Closed:
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
            # Dist from new outlet section to shoreline = PhysicalPars['SpitWidth']
            ShoreY[ExtendMask,1] = ShoreY[ExtendMask,0] - PhysicalPars['SpitWidth']
            # Width of new outlet section = end width
            ShoreY[ExtendMask,2] = ShoreY[ExtendMask,1] - OutletEndWidth[1]
            # Bed level of new outlet section  = end bed level
            ShoreZ[ExtendMask,1] = OutletEndElev[1]
            # Barrier height of inner barrier = PhysicalPars['SpitHeight']
            ShoreZ[ExtendMask,2] = PhysicalPars['SpitHeight']
            # TODO: Modify Barrier height of outer barrier ?????
            
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
                ShoreZ[LagExtension,3] = ShoreZ[LagExtension,1]
                # Remove outlet channel
                ShoreY[LagExtension,1] = np.nan
                ShoreY[LagExtension,2] = np.nan
                ShoreZ[LagExtension,1] = np.nan
                ShoreZ[LagExtension,2] = np.nan
        
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
    
    # if outlet is open but has only 1 transect then adjust shoreline/lagoonline to preserve it's width when we adjust ShoreY in the next step
    elif not Closed:
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
    ShoreIntersect = ShoreY[OutletPresent,0] < ShoreY[OutletPresent,1]
    if np.any(ShoreIntersect):
        logging.info('Outlet intersects shoreline at %i transects - filling outlet with sediment from shoreface' % np.sum(ShoreIntersect))
        IntTsects = OutletPresent[ShoreIntersect]
        ShoreY[IntTsects, 0] -= ((ShoreY[IntTsects, 1] - ShoreY[IntTsects, 2]) 
                                 * (ShoreZ[IntTsects, 2] - ShoreZ[IntTsects, 1]) 
                                 / (ShoreZ[IntTsects, 2] + PhysicalPars['ClosureDepth']))
        # Remove outlet channel from transect now it has been dissolved into shoreline
        ShoreY[IntTsects, 1] = np.nan
        ShoreY[IntTsects, 2] = np.nan
        ShoreZ[IntTsects, 1] = np.nan
        # Remove inner barrier now there's no outlet in the transect
        ShoreZ[IntTsects, 0] = ShoreZ[IntTsects, 2]
        ShoreZ[IntTsects, 2] = np.nan
    
    LagoonIntersect = ShoreY[OutletPresent,2] < ShoreY[OutletPresent,3]
    if np.any(LagoonIntersect):
        logging.info('Outlet intersects lagoon at %i transects - adding channel width into lagoon' % np.sum(LagoonIntersect))
        IntTsects = OutletPresent[LagoonIntersect]
        ShoreY[IntTsects, 3] += ((ShoreY[IntTsects, 1] - ShoreY[IntTsects, 2]) 
                                 * (ShoreZ[IntTsects, 2] - ShoreZ[IntTsects, 1]) 
                                 / (ShoreZ[IntTsects, 0] - ShoreZ[IntTsects, 3]))
        # Remove outlet channel from transect now it has been dissolved into shoreline
        ShoreY[IntTsects, 1] = np.nan
        ShoreY[IntTsects, 2] = np.nan
        ShoreZ[IntTsects, 1] = np.nan
        
    #%% Check for breach
    WaterLevel = LagoonWL.copy()
    WaterLevel[OutletChanIx] = OutletWL[OutletChanIx]
    if np.any(ShoreZ[:,0]<WaterLevel):
        Deepest = np.argmax(WaterLevel-ShoreZ[:,0])
        np.info('Lagoon overtopping barrier at X = %f - potential breach' % ShoreX[Deepest])
        