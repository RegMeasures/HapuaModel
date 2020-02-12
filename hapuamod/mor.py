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
                     LagoonWL, OutletDep,
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
    # Note: 
    #   - Not all "OutletPresent" are online/connected to sea...
    #   - Some locations where "OutletPresent" have a zero widht outlet (i.e. 
    #     just a discontinuity between front and back barrier height where
    #     there used to  be an outlet channel).
    OutletPresent = ~np.isnan(ShoreY[:,1])
    
    # Locations where lagoon width > 0
    LagoonPresent = ShoreY[:,3] > ShoreY[:,4]
    
    # Lagoon/outlet water level (to check for breach)
    # Calculated before updating bed levels incase this results in temporarily unrealistic water levels
    WaterLevel = np.full(ShoreX.size, -9999.9)
    WaterLevel[LagoonPresent] = LagoonWL[LagoonPresent]
    if not Closed:
        WaterLevel[OutletChanIx] = (OutletDep[OutletChanIx] + ShoreZ[OutletChanIx,1])
    
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
    EroVol = - np.minimum(dVol, 0.0)  # Total erosion volume (+ve = erosion)
    BedEro = EroVol * TooWide       # Bed erosion volume (+ve)
    BankEro = EroVol * np.logical_not(TooWide) # Bank erosion volume (+ve)
    BedDep = np.maximum(dVol, 0.0)  # Bed deposition volume (+ve)
    # note that += modifies variables in place so no need to explicitly return them!
    # No change in river width allowed so all erosion applied to bed.
    RiverElev[1:] += (BedDep[RivXS] - EroVol[RivXS]) / (PhysicalPars['RiverWidth'] * Dx)
    ShoreZ[OnlineLagoon, 3] += ((BedDep[LagXS] - BedEro[LagXS])
                                / ((ShoreY[OnlineLagoon, 3] - ShoreY[OnlineLagoon, 4]) * Dx))
    if not Closed:
        ShoreZ[OutletChanIx, 1] += (BedDep[OutXS[1:-1]] - BedEro[OutXS[1:-1]]) / (ChanWidth[OutXS] * Dx)
        OutletEndElev += (BedDep[OutEndXS-1] - BedEro[OutEndXS-1]) / (OutletEndWidth * Dx)
    
    # Update lagoon and outlet channel bank positions
    # Note: River upstream of lagoon has fixed width - all morpho change is on bed
    LagoonBankElev = ShoreZ[:,0].copy()
    LagoonBankElev[OutletPresent] = ShoreZ[OutletPresent, 2]
    ShoreY[OnlineLagoon,3] += (BankEro[LagXS]/2) / ((LagoonBankElev[OnlineLagoon] - ShoreZ[OnlineLagoon,3]) * Dx)
    ShoreY[OnlineLagoon,4] -= (BankEro[LagXS]/2) / ((PhysicalPars['BackshoreElev'] - ShoreZ[OnlineLagoon,3]) * Dx)
    # Outlet channel
    if not Closed:
        ShoreY[OutletChanIx,1] += (BankEro[OutXS[1:-1]]/2) / ((ShoreZ[OutletChanIx,0] - ShoreZ[OutletChanIx,1]) * Dx)
        ShoreY[OutletChanIx,2] -= (BankEro[OutXS[1:-1]]/2) / ((ShoreZ[OutletChanIx,2] - ShoreZ[OutletChanIx,1]) * Dx)
        
        OutletEndWidth[0] += BankEro[OutEndXS[0]-1] / ((ShoreZ[OutletChanIx[0],0] - OutletEndElev[0]) * Dx)
        OutletEndWidth[1] += BankEro[OutEndXS[1]-1] / ((PhysicalPars['BeachTopElev'] - OutletEndElev[1]) * PhysicalPars['SpitWidth'])
        
        # Put sediment discharged from outlet onto shoreline 
        # TODO improve sediment distribution (github issue #46)
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
                              / ((PhysicalPars['BeachTopElev'] - OutletEndElev[1]) * PhysicalPars['SpitWidth']))
            OutletEndWidth[1] -= WidthReduction
            OutletEndX[1] += WidthReduction/2
        else:
            # Transport from R to L
            ShoreY[OutletRbShoreIx-1,0] += (LST[OutletRbShoreIx-1] * Dt.seconds 
                                            / (ShorefaceHeight[OutletRbShoreIx-1] * Dx))
            WidthReduction = ((-LST[OutletRbShoreIx-1] * Dt.seconds) 
                              / ((PhysicalPars['BeachTopElev'] - OutletEndElev[1]) * PhysicalPars['SpitWidth']))
            OutletEndWidth[1] -= WidthReduction
            OutletEndX[1] -= WidthReduction/2
        
    #%% Cross-shore morphology (overtopping etc)
    
    CrestWidth = ShoreY[:,0] - ShoreY[:,3]
    CrestWidth[OutletPresent] = ShoreY[OutletPresent,0] - ShoreY[OutletPresent,1]
    
    BackBarHeight = (ShoreZ[:,0] - ShoreZ[:,3])
    BackBarHeight[OutletPresent] = ShoreZ[OutletPresent,0] - ShoreZ[OutletPresent,1]
    
    # Accumulation of sediment on top of the barrier
    ShoreZ[:,0] += (1-OverwashProp) * CST_tot * Dt.seconds / CrestWidth
    
    # Accumulation of sediment on the back of the barrier
    ShoreY[~OutletPresent,3] -= OverwashProp[~OutletPresent] * CST_tot[~OutletPresent] * Dt.seconds / BackBarHeight[~OutletPresent]
    ShoreY[OutletPresent,1] -= OverwashProp[OutletPresent] * CST_tot[OutletPresent] * Dt.seconds / BackBarHeight[OutletPresent]
    
    # Erosion of sediment off the shoreface (don't move ends)
    ShoreY[1:-1,0] -= CST_tot[1:-1] * Dt.seconds / ShorefaceHeight[1:-1]
    
    #%% Check if outlet channel (online or offline) has closed due to overwash and adjust to prevent negative width channel
    if np.any(ShoreY[OutletPresent,1] < ShoreY[OutletPresent,2]):
        logging.info('Overwashing occuring into closed channel in barrier - redistributing overwash onto barrier top')
        # Find locations where the outlet channel width is negative after applying overwash
#        NegativeOutletWidth = np.zeros(ShoreX.size, dtype=bool)
#        NegativeOutletWidth[OutletPresent] = ShoreY[OutletPresent,1] < ShoreY[OutletPresent,2]
        NegativeOutletWidth = np.where(OutletPresent)[0][ShoreY[OutletPresent,1] < ShoreY[OutletPresent,2]]
        
        # Quantify how much we need to move to fix things while retaining mass balance
        WidthToMove = ShoreY[NegativeOutletWidth,2] - ShoreY[NegativeOutletWidth,1]
        VolToMove = WidthToMove * BackBarHeight[NegativeOutletWidth]
        assert np.all(VolToMove > 0), "VolToMove should not be negative"
        
        # Where to move it depends on whether the barrier is higher in front or behind the closed outlet channel
        VolUntilLevel = ShoreZ[NegativeOutletWidth,2] - ShoreZ[NegativeOutletWidth,0]
        PutSedAtFront = VolUntilLevel > 0.0
        PutSedAtBack = VolUntilLevel < 0.0
        VolUntilLevel[PutSedAtFront] *= CrestWidth[NegativeOutletWidth][PutSedAtFront]
        VolUntilLevel[PutSedAtBack] *= -(ShoreY[NegativeOutletWidth,2][PutSedAtBack] - 
                                         ShoreY[NegativeOutletWidth,3][PutSedAtBack])
        assert np.all(VolUntilLevel > 0.0), "Vol until level should not be negative"
        LeveledSections = VolToMove > VolUntilLevel
        
        # Update the barrier morphology
        ShoreY[NegativeOutletWidth,1] += WidthToMove
        ShoreZ[NegativeOutletWidth[PutSedAtFront],0] += VolToMove[PutSedAtFront] / CrestWidth[NegativeOutletWidth[PutSedAtFront]]
        ShoreZ[NegativeOutletWidth[PutSedAtBack],2] += VolToMove[PutSedAtBack] / (ShoreY[NegativeOutletWidth[PutSedAtBack],2] - 
                                                                                  ShoreY[NegativeOutletWidth[PutSedAtBack],3])
        
        if np.any(LeveledSections):
            print('weird')
            for XcoordOfSec in ShoreX[NegativeOutletWidth[LeveledSections]]:
                logging.info('Overwash into closed channel has leveled barrier at X = %f. Removing channel from model.' % XcoordOfSec)
            ShoreY[NegativeOutletWidth[LeveledSections],1] = np.nan
            ShoreY[NegativeOutletWidth[LeveledSections],2] = np.nan
            ShoreZ[NegativeOutletWidth[LeveledSections],0] = (np.maximum(ShoreZ[NegativeOutletWidth[LeveledSections],0],
                                                                         ShoreZ[NegativeOutletWidth[LeveledSections],2]) + 
                                                              (VolToMove[LeveledSections] - VolUntilLevel[LeveledSections]) /
                                                              (ShoreY[NegativeOutletWidth[LeveledSections],0] - 
                                                               ShoreY[NegativeOutletWidth[LeveledSections],3]))
            ShoreZ[NegativeOutletWidth[LeveledSections],1] = np.nan
            ShoreZ[NegativeOutletWidth[LeveledSections],2] = np.nan
    
    
    #%% Check if d/s end of outlet channel has migrated across a transect line and adjust as necessary...
    if not Closed:
        if OutletEndX[0] < OutletEndX[1]:
            # Outlet angles from L to R
            if ShoreX[OutletChanIx[-1]+1] <= OutletEndX[1]:
                Extend = True
                ExtendMask = np.logical_and(ShoreX[OutletChanIx[-1]] < ShoreX,
                                            ShoreX <= OutletEndX[1])
                logging.info('Outlet channel elongated rightwards across transect line x=%f' % ShoreX[ExtendMask][-1])
            else:
                Extend = False    
        else:
            # Outlet angles from R to L
            if ShoreX[OutletChanIx[-1]-1] >= OutletEndX[1]:
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
            # Barrier height of inner barrier = Barrier height of old outer barrier
            ShoreZ[ExtendMask,2] = ShoreZ[ExtendMask,0]
            # Barrier height of outer barrier = PhysicalPars['BeachTopElev']
            ShoreZ[ExtendMask,0] = PhysicalPars['BeachTopElev']
            # TODO: Modify Barrier height of outer barrier ?????
            
            # Update OutletChanIx as it has changed and its used later in this function
            if OutletEndX[0] < OutletEndX[1]:
                OutletChanIx = np.where(np.logical_and(OutletEndX[0] <= ShoreX, 
                                                       ShoreX <= OutletEndX[1]))[0]
            else:
                OutletChanIx = np.flipud(np.where(np.logical_and(OutletEndX[1] <= ShoreX,
                                                                 ShoreX <= OutletEndX[0]))[0])
        
    
    #%% Check for outlet channel truncation
    # Note: have to be careful to leave at least 1 transect in outlet channel
    
    # Check for truncation of online outlet channel and move ends of channel if required
    # (Perform each check seperately incase multiple triggered in the same timestep...)
    
    # Check for truncation of lagoonward end of outlet channel 
    # (Only check for truncation if outlet channel crosses >1 transect)
    if OutletChanIx.size > 1:
        # (don't check last transect as trucation here would leave 0 transects)
        if np.any(ShoreY[OutletChanIx[:-1],2] <= ShoreY[OutletChanIx[:-1],3]):
            logging.info('Truncating lagoon end of outlet channel (due to erosion)')
            TruncationIx = OutletChanIx[:-1][ShoreY[OutletChanIx[:-1],2] <= ShoreY[OutletChanIx[:-1],3]]
            for AffectedX in TruncationIx:
                logging.info('Truncation tiggered at X = %f' % ShoreX[AffectedX])
                
            if OutletEndX[0] < OutletEndX[1]:
                # Outlet angles from L to R
                OutletEndX[0] = ShoreX[TruncationIx[0] + 1]
                OutletChanIx = np.where(np.logical_and(OutletEndX[0] <= ShoreX, 
                                                       ShoreX <= OutletEndX[1]))[0]
                if ShoreY[TruncationIx[0], 2] <= ShoreY[TruncationIx[0], 4]:
                    logging.info('Extending R end of lagoon via outletchannel to cliffline collision.')
                    Extend = True
                    CurLagEndIx = np.where(ShoreY[:,3] > ShoreY[:,4])[0][-1]
                    LagExtension = np.arange(CurLagEndIx + 1, TruncationIx[0] + 1)
                else:
                    Extend = False
            else:
                # Outlet angles from R to L
                OutletEndX[0] = ShoreX[TruncationIx[0] - 1]
                OutletChanIx = np.flipud(np.where(np.logical_and(OutletEndX[1] <= ShoreX,
                                                                 ShoreX <= OutletEndX[0]))[0])
                if ShoreY[TruncationIx[0], 2] <= ShoreY[TruncationIx[0], 4]:
                    logging.info('Extending L end of lagoon via outletchannel to cliffline collision.')
                    Extend = True
                    CurLagEndIx = np.where(ShoreY[:,3] > ShoreY[:,4])[0][0]
                    LagExtension = np.arange(TruncationIx[0], CurLagEndIx)
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
    
    # Check for truncation of seaward end of outlet channel
    if OutletChanIx.size > 1:
        # (don't check first transect as trucation here would leave 0 transects)
        if np.any(ShoreY[OutletChanIx[1:],1] >= ShoreY[OutletChanIx[1:],0]):
            logging.info('Truncating seaward end of outlet channel (due to erosion)')
            TruncationIx = OutletChanIx[ShoreY[OutletChanIx,1] >= ShoreY[OutletChanIx,0]]
            if OutletEndX[0] < OutletEndX[1]:
                # Outlet angles from L to R
                OutletEndX[1] = ShoreX[TruncationIx[0]] - Dx/2 
                OutletChanIx = np.where(np.logical_and(OutletEndX[0] <= ShoreX, 
                                                       ShoreX <= OutletEndX[1]))[0]
            else:
                # Outlet angles from R to L
                OutletEndX[1] = ShoreX[TruncationIx[-1]] + Dx/2 
                OutletChanIx = np.flipud(np.where(np.logical_and(OutletEndX[1] <= ShoreX,
                                                                 ShoreX <= OutletEndX[0]))[0])
    
    # if outlet is open but truncation would result in a outlet channel with no transects... 
    # then adjust shoreline/lagoonline to preserve it's width when we adjust ShoreY in the next step
    if not Closed:
        # Preserve channel width by extending into lagoon as required 
        # (not extending into sea as this would mess with LST)
        if ShoreY[OutletChanIx[0],1] >= ShoreY[OutletChanIx[0],0]:
            logging.info('Preventing seaward truncation at X = %.1f as it would result in a length 0 outlet channel' % 
                         ShoreX[OutletChanIx[0]])
            ShoreY[OutletChanIx[0],2] -= ShoreY[OutletChanIx[0],1] - ShoreY[OutletChanIx[0],0]
            ShoreY[OutletChanIx[0],1] = ShoreY[OutletChanIx[0],0] - 0.0001
        if ShoreY[OutletChanIx[-1],3] >= ShoreY[OutletChanIx[-1],2]:
            logging.info('Preventing lagoonward truncation at X = %.1f as it would result in a length 0 outlet channel' % 
                         ShoreX[OutletChanIx[-1]])
            ShoreY[OutletChanIx[-1],3] = ShoreY[OutletChanIx[-1],2] - 0.0001
            # if this pushes into cliff then assume cliff erodes into outlet channel...
            if ShoreY[OutletChanIx[-1],3] < ShoreY[OutletChanIx[-1],4]:
                logging.info('Preventing lagoonward truncation means outlet channel pushed into cliff at X = %.1f' % 
                             ShoreX[OutletChanIx[-1]])
                CliffRetDist = ShoreY[OutletChanIx[-1],4] - ShoreY[OutletChanIx[-1],3]
                ShoreY[OutletChanIx[-1], 4] = ShoreY[OutletChanIx[-1], 3]
                ShoreZ[OutletChanIx[-1], 1] += (CliffRetDist * (PhysicalPars['BackshoreElev'] - ShoreZ[OutletChanIx[-1],1]) / 
                                                (ShoreY[OutletChanIx[-1], 1] - ShoreY[OutletChanIx[-1], 2]))
                logging.debug('CliffRetDist = %f, OutletWidth = %f, NewOutletBedLevel = %f' % 
                              (CliffRetDist, (ShoreY[OutletChanIx[-1], 1] - ShoreY[OutletChanIx[-1], 2]), 
                               ShoreZ[OutletChanIx[-1], 1]))
    
    # Adjust ShoreY where outlet banks intersects coast or lagoon
    # Note this can include offline/disconnected bits of outlet as well as online bits
    ShoreIntersect = ~np.greater(ShoreY[:,0], ShoreY[:,1], where=~np.isnan(ShoreY[:,1]))
    if np.any(ShoreIntersect):
        logging.info('Outlet intersects shoreline at %i transects - filling outlet with sediment from shoreface' % 
                     np.sum(ShoreIntersect))
        ShoreY[ShoreIntersect, 0] -= ((ShoreY[ShoreIntersect, 1] - ShoreY[ShoreIntersect, 2]) 
                                      * (ShoreZ[ShoreIntersect, 2] - ShoreZ[ShoreIntersect, 1]) 
                                      / (ShoreZ[ShoreIntersect, 2] + PhysicalPars['ClosureDepth']))
        # Remove outlet channel from transect now it has been dissolved into shoreline
        ShoreY[ShoreIntersect, 1] = np.nan
        ShoreY[ShoreIntersect, 2] = np.nan
        ShoreZ[ShoreIntersect, 1] = np.nan
        # Remove inner barrier now there's no outlet in the transect
        ShoreZ[ShoreIntersect, 0] = ShoreZ[ShoreIntersect, 2]
        ShoreZ[ShoreIntersect, 2] = np.nan
    
    LagoonIntersect = ~np.greater(ShoreY[:,2], ShoreY[:,3], where=~np.isnan(ShoreY[:,1]))
    if np.any(LagoonIntersect):
        for IntersectIx in np.where(LagoonIntersect)[0]:
            logging.info('Outlet intersects lagoon at X = %.1f: adding channel width into lagoon' % 
                         ShoreX[IntersectIx])
        ShoreZ[LagoonIntersect, 3] = (((ShoreY[LagoonIntersect, 1] - ShoreY[LagoonIntersect, 2]) * ShoreZ[LagoonIntersect, 1] 
                                      + (ShoreY[LagoonIntersect, 3] - ShoreY[LagoonIntersect, 4]) * ShoreZ[LagoonIntersect, 3]) 
                                      / (ShoreY[LagoonIntersect, 1] - ShoreY[LagoonIntersect, 4]))
        ShoreY[LagoonIntersect, 3] += (ShoreY[LagoonIntersect, 1] - ShoreY[LagoonIntersect, 2]) 
        # Remove outlet channel from transect now it has been dissolved into shoreline
        ShoreY[LagoonIntersect, 1] = np.nan
        ShoreY[LagoonIntersect, 2] = np.nan
        ShoreZ[LagoonIntersect, 1] = np.nan
        
    #%% Breaching
    
    # Check for breach
    Breach = False
    if np.any(np.logical_and(ShoreY[:,3]>=ShoreY[:,0], ShoreY[:,3]>ShoreY[:,4])):
        EroTsects = np.where(np.logical_and(ShoreY[:,3]>=ShoreY[:,0], ShoreY[:,3]>ShoreY[:,4]))[0]
        BreachIx = EroTsects[np.argmin(np.abs(ShoreX[EroTsects]))]
        Breach = True
        logging.info('Barrier completely eroded at X = %i' % ShoreX[BreachIx])
    elif np.any(ShoreZ[:,0] < WaterLevel):
        if Closed:
            # If lagoon closed then assume any overtopping causes breach, 
            # and that breach occurs where overtopping is deepest
            BreachIx = np.argmax(WaterLevel-ShoreZ[:,0])
            Breach = True
            logging.info('Closed lagoon overtopping at X = %f' % ShoreX[BreachIx])
        else:
            # If lagoon open then assume overtopping only causes breach if it 
            # is closer to where river enters lagoon than existing outlet 
            # (and not on the first transect of the outlet channel as this
            # would leave a 0-transect outlet)
            CloserToRiv = np.abs(ShoreX) < np.max(np.abs(OutletEndX))
            CloserToRiv[OutletChanIx[0]] = False
            if np.any(ShoreZ[CloserToRiv,0] < WaterLevel[CloserToRiv]):
                BreachIx = np.argmax((WaterLevel-ShoreZ[:,0]) * CloserToRiv)
                Breach = True
                logging.info('Lagoon overtopping barrier at X = %f' % ShoreX[BreachIx])
    
    # Create breach
    if Breach:
        if BreachIx in OutletChanIx:
            # Outlet truncation breach
            logging.info('Outlet truncation due to breach at X = %f' % ShoreX[BreachIx])
            if OutletEndX[0] < OutletEndX[1]:
                # Outlet angles from L to R
                OutletEndX[1] = ShoreX[BreachIx] - Dx/2
                OutletEndWidth[1] = Dx
                OutletEndElev[1] = (ShoreZ[BreachIx-1,1] + PhysicalPars['MaxOutletElev'])/2
            else:
                # Outlet angles from R to L 
                OutletEndX[1] = ShoreX[BreachIx] + Dx/2
                OutletEndWidth[1] = Dx
                OutletEndElev[1] = (ShoreZ[BreachIx+1,1] + PhysicalPars['MaxOutletElev'])/2
            # TODO: close sediment balance by putting breach eroded sed onto shore
        else:
            # Lagoon breach (i.e. new outlet channel)
            logging.info('Breach/creation of new outlet channel at X = %f' % ShoreX[BreachIx])
            # Assume breach of width Dx with bed level linearly interpolated 
            # between lagoon level at upstream end and PhysicalPars['MaxOutletElev']
            OutletEndWidth[:] = Dx
            ShoreY[BreachIx,1] = min(ShoreY[BreachIx,0]-PhysicalPars['SpitWidth'],
                                     (ShoreY[BreachIx,0]+ShoreY[BreachIx,3])/2 + Dx/2)
            ShoreY[BreachIx,2] = ShoreY[BreachIx,1] - Dx
            OutletEndX[:] = ShoreX[BreachIx]
            
            ShoreZ[BreachIx, 2] = ShoreZ[BreachIx, 0]
            
            OutletEndElev[0] = 0.25 * PhysicalPars['MaxOutletElev'] + 0.75 * ShoreZ[BreachIx,3]
            ShoreZ[BreachIx,1] = 0.5 * PhysicalPars['MaxOutletElev'] + 0.5 * ShoreZ[BreachIx,3]
            OutletEndElev[1] = 0.75 * PhysicalPars['MaxOutletElev'] + 0.25 * ShoreZ[BreachIx,3]
            
        
        
        