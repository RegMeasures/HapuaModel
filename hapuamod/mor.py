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
                     RiverElev, OnlineLagoon, OutletChanIx,
                     LagoonWL, OutletDep,
                     ChanWidth, ChanDep, ChanDx, ChanFlag, Closed,
                     LST, Bedload, CST_tot, OverwashProp,
                     MorDt, PhysicalPars, TimePars, NumericalPars):
    """ Update river, lagoon, outlet, barrier and shoreline morphology
        
        Returns:
            MorDt = Current value of the adaptive morphological timestep
            Breach = Flag for whether a breach has occured this timestep
    """
    
    #%% Pre-calculate some useful parameters
    
    Dx = NumericalPars['Dx']
    X0Ix = np.where(ShoreX==0)[0][0]
    
    # Find location outlet channel intersects shoreline
    if not Closed:
        OutletRbShoreIx = np.where(OutletEndX[1] < ShoreX)[0][0]
    
    # Shoreface height for converting volumetric change into advance retreat of shoreline
    ShorefaceHeight = PhysicalPars['ClosureDepth'] + ShoreZ[:,0]
    
    # Locations where an outlet channel exists in the barrier
    # Note: 
    #   - Not all "OutletPresent" are online/connected to sea...
    #   - Some locations where "OutletPresent" have a zero width outlet (i.e. 
    #     just a discontinuity between front and back barrier height where
    #     there used to  be an outlet channel).
    OutletPresent = ~np.isnan(ShoreY[:,1])
    
    # Locations where lagoon width > 0
    LagoonPresent = ShoreY[:,3] > ShoreY[:,4]
    
    # Locations where lagoon width wide enough to pass flow
    LagoonOpen = (ShoreY[:,3] - ShoreY[:,4]) > PhysicalPars['MinOutletWidth']
    
    # Lagoon/outlet water level (to check for breach)
    # Calculated before updating bed levels incase this results in temporarily unrealistic water levels
    WaterLevel = np.full(ShoreX.size, -9999.9)
    WaterLevel[LagoonPresent] = LagoonWL[LagoonPresent]
    if not Closed:
        WaterLevel[OutletChanIx] = (OutletDep[OutletChanIx] + ShoreZ[OutletChanIx,1])
    assert not np.any(np.isnan(WaterLevel)), 'NaN value(s) in WaterLevel at X = %s' % ShoreX[np.isnan(WaterLevel)]
    
    # Width of barrier crest (the part of the barrier backing the shoreface)
    CrestWidth = ShoreY[:,0] - ShoreY[:,3]
    CrestWidth[OutletPresent] = ShoreY[OutletPresent,0] - ShoreY[OutletPresent,1]
    assert np.all(CrestWidth>0), 'Negative CrestWidth at the start of updateMorphology. Negative at X = %s' % ShoreX[CrestWidth<=0]
    
    # Height of barrier backshore (the part of hte barrier backing the shoreface)
    BackBarHeight = (ShoreZ[:,0] - ShoreZ[:,3])
    BackBarHeight[OutletPresent] = ShoreZ[OutletPresent,0] - ShoreZ[OutletPresent,1]
    assert np.all(CrestWidth>0), 'Negative BackBarHeight at the start of updateMorphology'
    
    #%% Calculate rates of movement of morphology due to river forcing
    ShoreYChangeRate = np.zeros(ShoreY.shape)
    ShoreZChangeRate = np.zeros(ShoreZ.shape)
    OutletEndWideningRate = 0.0
    OutletEndXMoveRate = 0.0
    
    if Closed:
        # Rate of change in volume at each cross-section (except the upstream Bdy)
        DVolRate = (Bedload-np.concatenate([Bedload[1:],[0]]))
        AspectRatio = ChanWidth[1:]/ChanDep[1:]
        RivXS = ChanFlag[1:]==0
        LagXS = ChanFlag[1:]==1
    else:
        # Rate of change in volume at each cross-section (except the upstream Bdy and dummy downstream XS)
        DVolRate = (Bedload[:-1]-Bedload[1:])
        AspectRatio = ChanWidth[1:-1]/ChanDep[1:-1]
        RivXS = ChanFlag[1:-1]==0
        LagXS = ChanFlag[1:-1]==1
        OutXS = ChanFlag[1:-1]==2
        OutEndXS = ChanFlag[1:-1]==3
    TooWide = AspectRatio > PhysicalPars['WidthDepthRatio']
    
    EroVolRate = - np.minimum(DVolRate, 0.0)  # Total volumetric erosion rate (+ve = erosion)
    BedEroRate = EroVolRate * TooWide       # Bed erosion volumetric rate (+ve)
    BankEroRate = EroVolRate * np.logical_not(TooWide) # Bank erosion volumetric rate (+ve)
    BedDepRate = np.maximum(DVolRate, 0.0)  # Bed deposition volumetric rate (+ve)
    
    LagoonBankElev = ShoreZ[:,0].copy()
    LagoonBankElev[OutletPresent] = ShoreZ[OutletPresent, 2]
    
    # Bed level change rates
    # Note: No change in river width allowed so all erosion applied to bed.
    RivBedAggRate = (BedDepRate[RivXS] - EroVolRate[RivXS]) / (PhysicalPars['RiverWidth'] * Dx)
    ShoreZChangeRate[OnlineLagoon, 3] += ((BedDepRate[LagXS] - BedEroRate[LagXS])
                                          / ((ShoreY[OnlineLagoon, 3] - ShoreY[OnlineLagoon, 4]) * Dx))
    if not Closed:
        ShoreZChangeRate[OutletChanIx, 1] += ((BedDepRate[OutXS] - BedEroRate[OutXS]) / 
                                              ((ShoreY[OutletChanIx,1] - ShoreY[OutletChanIx,2]) * Dx))
        OutletEndAggRate = (BedDepRate[OutEndXS] - BedEroRate[OutEndXS]) / (OutletEndWidth * Dx)
    else:
        OutletEndAggRate = 0.0
    
    # Bank erosion change rates
    # Lagoon bank erosion
    ShoreYChangeRate[OnlineLagoon,3] += (BankEroRate[LagXS]/2) / ((LagoonBankElev[OnlineLagoon] - ShoreZ[OnlineLagoon,3]) * Dx)
    ShoreYChangeRate[OnlineLagoon,4] -= (BankEroRate[LagXS]/2) / ((PhysicalPars['BackshoreElev'] - ShoreZ[OnlineLagoon,3]) * Dx)
    if not Closed:
        # Outlet channel bank erosion
        ShoreYChangeRate[OutletChanIx,1] += (BankEroRate[OutXS]/2) / ((ShoreZ[OutletChanIx,0] - ShoreZ[OutletChanIx,1]) * Dx)
        ShoreYChangeRate[OutletChanIx,2] -= (BankEroRate[OutXS]/2) / ((ShoreZ[OutletChanIx,2] - ShoreZ[OutletChanIx,1]) * Dx)
        
        # Outlet end bank erosion
        OutletEndWideningRate += BankEroRate[OutEndXS] / ((PhysicalPars['BeachTopElev'] - OutletEndElev) * PhysicalPars['SpitWidth'])
        if OutletEndX[0] < OutletEndX[1]:
            # Outlet angles from L to R
            OutletEndXMoveRate += (OutletEndWideningRate / 2) * PhysicalPars['OutletBankEroFac']
        elif OutletEndX[0] > OutletEndX[1]:
            # Outlet angles from R to L
            OutletEndXMoveRate -= (OutletEndWideningRate / 2) * PhysicalPars['OutletBankEroFac']
        
        # Distribute sediment discharged from outlet onto shoreline 
        BedloadToShoreRate = ((Dx * Bedload[-1] / PhysicalPars['OutletSedSpreadDist']) * 
                              np.maximum(1 - np.abs(ShoreX - OutletEndX[1])/PhysicalPars['OutletSedSpreadDist'], 0.0))
        ShoreYChangeRate[:,0] += BedloadToShoreRate / (ShorefaceHeight * Dx)    
    
    #%% Calculate rate of change of shoreline position due to long shore transport
        
    ShoreYChangeRate[1:-1,0] += (LST[:-1] - LST[1:]) / (ShorefaceHeight[1:-1] * Dx)
    # TODO add shoreline boundary conditions here (github issue #10)
    
    # Remove LST driven sed supply out of outlet channel and put on outlet channel bank instead
    if not Closed:
        if LST[OutletRbShoreIx-1]>0:
            # Transport from L to R
            ShoreYChangeRate[OutletRbShoreIx,0] -= (LST[OutletRbShoreIx-1]
                                                    / (ShorefaceHeight[OutletRbShoreIx] * Dx))
            WidthReductionRate = (LST[OutletRbShoreIx-1] 
                                  / ((PhysicalPars['BeachTopElev'] - OutletEndElev) * PhysicalPars['SpitWidth']))
            OutletEndWideningRate -= WidthReductionRate
            OutletEndXMoveRate += WidthReductionRate/2
        else:
            # Transport from R to L
            ShoreY[OutletRbShoreIx-1,0] += (LST[OutletRbShoreIx-1] 
                                            / (ShorefaceHeight[OutletRbShoreIx-1] * Dx))
            WidthReductionRate = (-LST[OutletRbShoreIx-1]
                                  / ((PhysicalPars['BeachTopElev'] - OutletEndElev) * PhysicalPars['SpitWidth']))
            OutletEndWideningRate -= WidthReductionRate
            OutletEndXMoveRate -= WidthReductionRate/2
    
    #%% Rates of change of morphology due to cross-shore morphology (overtopping etc)
    
    # Accumulation of sediment on top of the barrier
    ShoreZChangeRate[:,0] += (1-OverwashProp) * CST_tot / CrestWidth
    
    # Accumulation of sediment on the back of the barrier
    ShoreYChangeRate[~OutletPresent,3] -= OverwashProp[~OutletPresent] * CST_tot[~OutletPresent] / BackBarHeight[~OutletPresent]
    ShoreYChangeRate[OutletPresent,1] -= OverwashProp[OutletPresent] * CST_tot[OutletPresent] / BackBarHeight[OutletPresent]
    
    # Erosion of sediment off the shoreface (don't move ends)
    ShoreYChangeRate[1:-1,0] -= CST_tot[1:-1] / ShorefaceHeight[1:-1]
    
    #%% Enforce shoreline end erosion rate
    ShoreYChangeRate[[0,-1],0] = -PhysicalPars['ShorelineErosionRate']/(365.25*86400)
    
    #%% Set adaptive timestep
    
    MaxMorChangeRate = max(np.max(np.abs(RivBedAggRate)),
                           np.max(np.abs(ShoreYChangeRate)),
                           np.max(np.abs(ShoreZChangeRate)),
                           abs(OutletEndWideningRate),
                           abs(OutletEndAggRate),
                           abs(OutletEndXMoveRate))
    
    while MorDt.seconds > TimePars['MorDtMin'].seconds and MaxMorChangeRate * MorDt.seconds > NumericalPars['MaxMorChange']:
        MorDt /= 2
        logging.debug('Decreasing morphological timestep to %.1fs', MorDt.seconds)
    
    while MorDt.seconds < TimePars['MorDtMax'].seconds and MaxMorChangeRate * MorDt.seconds < NumericalPars['MaxMorChange']/2:
        MorDt *= 2
        logging.debug('Increasing morphological timestep to %.1fs', MorDt.seconds)
    
    # If morphological change is going to exceed MaxMorChange despite adaptive timestepping then:
    #  1. issue warning
    #  2. limit change (while preserving sign) to prevent model blowing up (even though this may be at the expense of mass balance)
    if MaxMorChangeRate * MorDt.seconds > NumericalPars['MaxMorChange']:
        assert MorDt.seconds <= TimePars['MorDtMin'].seconds
        logging.warning('Max morphological change (%.3fm) exceeds "MaxMorChange (%.3fm)" but MorDt already at minimum. Limiting change to +/- %.3fm',
                        MaxMorChangeRate * MorDt.seconds, NumericalPars['MaxMorChange'], NumericalPars['MaxMorChange'])
        MorChangeRateLimit = NumericalPars['MaxMorChange'] / MorDt.seconds
        RivBedAggRate = np.minimum(np.maximum(RivBedAggRate, -MorChangeRateLimit), MorChangeRateLimit)
        ShoreYChangeRate = np.minimum(np.maximum(ShoreYChangeRate, -MorChangeRateLimit), MorChangeRateLimit)
        ShoreZChangeRate = np.minimum(np.maximum(ShoreZChangeRate, -MorChangeRateLimit), MorChangeRateLimit)
        OutletEndWideningRate = np.minimum(np.maximum(OutletEndWideningRate, -MorChangeRateLimit), MorChangeRateLimit)
        OutletEndAggRate = min(max(OutletEndAggRate, -MorChangeRateLimit), MorChangeRateLimit)
        OutletEndXMoveRate = min(max(OutletEndXMoveRate, -MorChangeRateLimit), MorChangeRateLimit)
    
    #%% Update morphology
        
    # note that += modifies variables in place so no need to explicitly return them!
    RiverElev[1:] += RivBedAggRate * MorDt.seconds
    ShoreZ += ShoreZChangeRate * MorDt.seconds
    ShoreY += ShoreYChangeRate * MorDt.seconds
    if not Closed:
        OutletEndWidth.flat[0] += OutletEndWideningRate * MorDt.seconds
        OutletEndElev.flat[0] += OutletEndAggRate * MorDt.seconds
        OutletEndX[1] += OutletEndXMoveRate * MorDt.seconds
    
    #%% 1. Check if outlet has extended too close to the edge of the domain
    if OutletEndX[1] < ShoreX[1]:
        logging.warning('Left (negative X) end of outlet channel too close to edge of domain - preventing further migration')
        OutletEndX[1] = ShoreX[1]
    if OutletEndX[1] > ShoreX[-2]:
        logging.warning('Right (negative X) end of outlet channel too close to edge of domain - preventing further migration')
        OutletEndX[1] = ShoreX[1]
    
    #%% 2. Check if shoreline has eroded into cliff anywhere
    CliffCollision = ShoreY[:,0] <= ShoreY[:,4]
    if np.any(CliffCollision):
        logging.info('Shoreline/cliff collision - retreating cliff at X = %s', ShoreX[CliffCollision])
        CliffOverlapDist = (ShoreY[CliffCollision,4] - ShoreY[CliffCollision,0]) + 0.001
        CliffRetDist = (CliffOverlapDist / 
                        (1 + ((PhysicalPars['BackshoreElev'] - ShoreZ[CliffCollision,0]) / 
                              ShorefaceHeight[CliffCollision])))
        ShoreY[CliffCollision, 4] -= CliffRetDist
        ShoreY[CliffCollision, 3] = ShoreY[CliffCollision, 4]
        ShoreY[CliffCollision, 0] += (CliffOverlapDist - CliffRetDist)
    
    #%% 3. Check if outlet channel (online or offline) has negative width due to overwash and adjust to prevent negative width channel
    if np.any(ShoreY[OutletPresent,1] < ShoreY[OutletPresent,2]):
        logging.debug('Overwashing occuring into closed channel in barrier - redistributing overwash onto barrier top')
        # Find locations where the outlet channel width is negative after applying overwash
        NegativeOutletWidth = np.where(OutletPresent)[0][ShoreY[OutletPresent,1] < ShoreY[OutletPresent,2]]
        
        # Quantify how much we need to move to fix things while retaining mass balance
        WidthToMove = ShoreY[NegativeOutletWidth,2] - ShoreY[NegativeOutletWidth,1]
        VolToMove = WidthToMove * BackBarHeight[NegativeOutletWidth]
        assert np.all(VolToMove > 0), "VolToMove should not be negative"
        
        # Where to move it depends on whether the barrier is higher in front or behind the closed outlet channel
        VolUntilLevel = ShoreZ[NegativeOutletWidth,2] - ShoreZ[NegativeOutletWidth,0]
        PutSedAtFront = VolUntilLevel >= 0.0
        PutSedAtBack = VolUntilLevel < 0.0
        VolUntilLevel[PutSedAtFront] *= CrestWidth[NegativeOutletWidth][PutSedAtFront]
        VolUntilLevel[PutSedAtBack] *= -(ShoreY[NegativeOutletWidth,2][PutSedAtBack] - 
                                         ShoreY[NegativeOutletWidth,3][PutSedAtBack])
        assert np.all(VolUntilLevel >= 0.0), "Vol until level should not be negative"
        LeveledSections = VolToMove > VolUntilLevel
        PutSedAtFront[LeveledSections] = False
        PutSedAtBack[LeveledSections] = False
        
        # Update the barrier morphology
        ShoreY[NegativeOutletWidth,1] += WidthToMove
        ShoreZ[NegativeOutletWidth[PutSedAtFront],0] += (VolToMove[PutSedAtFront] / 
                                                         CrestWidth[NegativeOutletWidth[PutSedAtFront]])
        ShoreZ[NegativeOutletWidth[PutSedAtBack],2] += (VolToMove[PutSedAtBack] / 
                                                        (ShoreY[NegativeOutletWidth[PutSedAtBack],2] - 
                                                                                   ShoreY[NegativeOutletWidth[PutSedAtBack],3]))
        
        if np.any(LeveledSections):
            for XcoordOfSec in ShoreX[NegativeOutletWidth[LeveledSections]]:
                logging.info('Overwash into closed channel has leveled barrier at X = %.1fm. Removing channel from model.' % XcoordOfSec)
            ShoreY[NegativeOutletWidth[LeveledSections],1] = np.nan
            ShoreY[NegativeOutletWidth[LeveledSections],2] = np.nan
            ShoreZ[NegativeOutletWidth[LeveledSections],0] = (np.maximum(ShoreZ[NegativeOutletWidth[LeveledSections],0],
                                                                         ShoreZ[NegativeOutletWidth[LeveledSections],2]) + 
                                                              (VolToMove[LeveledSections] - VolUntilLevel[LeveledSections]) /
                                                              (ShoreY[NegativeOutletWidth[LeveledSections],0] - 
                                                               ShoreY[NegativeOutletWidth[LeveledSections],3]))
            ShoreZ[NegativeOutletWidth[LeveledSections],1] = np.nan
            ShoreZ[NegativeOutletWidth[LeveledSections],2] = np.nan
            # Update OutletPresent variable to reflect removed section of channel
            OutletPresent = ~np.isnan(ShoreY[:,1])
    
    #%% 4. Check if d/s end of outlet channel has elongated across a transect line and adjust as necessary...
    if not Closed:
        if OutletEndX[0] < OutletEndX[1]:
            # Outlet angles from L to R
            if ShoreX[OutletChanIx[-1]+1] <= OutletEndX[1]:
                Extend = True
                ExtendMask = np.logical_and(ShoreX[OutletChanIx[-1]] < ShoreX,
                                            ShoreX <= OutletEndX[1])
                logging.info('Outlet channel elongated rightwards across transect line X = %.1fm' % ShoreX[ExtendMask][-1])
            else:
                Extend = False    
        else:
            # Outlet angles from R to L
            if ShoreX[OutletChanIx[-1]-1] >= OutletEndX[1]:
                Extend = True
                ExtendMask = np.logical_and(ShoreX[OutletChanIx[-1]] > ShoreX,
                                            ShoreX >= OutletEndX[1])
                logging.info('Outlet channel elongated leftwards across transect line X = %.1fm' % ShoreX[ExtendMask][0])
            else:
                Extend = False
        
        if Extend:
            # Dist from new outlet section to shoreline = PhysicalPars['SpitWidth']
            ShoreY[ExtendMask,1] = ShoreY[ExtendMask,0] - PhysicalPars['SpitWidth']
            # Width of new outlet section = end width
            ShoreY[ExtendMask,2] = ShoreY[ExtendMask,1] - OutletEndWidth
            # Bed level of new outlet section  = end bed level
            ShoreZ[ExtendMask,1] = OutletEndElev
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
    # Check for truncation of online outlet channel and move ends of channel if required
    # (Several checks performed in specific order)
    # Note: have to be careful to leave at least 1 transect in outlet channel
    
    # 5. First check for collision between outlet channel and sea in upstream most transect of online outlet
    # Truncation is not possible here so we adjust the outlet channel position to prevent it...
    if not Closed:
        if ShoreY[OutletChanIx[0],1] >= ShoreY[OutletChanIx[0],0]:
            logging.info('Preventing seaward truncation at X = %.1fm as it would result in a length 0 outlet channel' % 
                         ShoreX[OutletChanIx[0]])
            ShoreShiftDist = 0.001 + (ShoreY[OutletChanIx[0],1] - ShoreY[OutletChanIx[0],0])
            ShoreY[OutletChanIx[0],1] -= ShoreShiftDist
            ShoreY[OutletChanIx[0],2] -= (ShoreShiftDist * 
                                          (ShoreZ[OutletChanIx[0],0] - ShoreZ[OutletChanIx[0],1]) /
                                          (ShoreZ[OutletChanIx[0],2] - ShoreZ[OutletChanIx[0],1]))
            # Note: this may have caused a new conflict between hte lagoonward 
            #       edge of the outlet channel and the lagoon or cliff, but 
            #       this will be delt with later.
    
    # 6. Check for collision between outlet channel and lagoon causing truncation 
    # of lagoonward end of outlet channel 
    # (Only check for truncation if outlet channel crosses >1 transect)
    if OutletChanIx.size > 1:
        # (don't check last transect as trucation here would leave 0 transects)
        if np.any(ShoreY[OutletChanIx[:-1],2] <= ShoreY[OutletChanIx[:-1],3]):
            TruncationIx = OutletChanIx[:-1][ShoreY[OutletChanIx[:-1],2] <= ShoreY[OutletChanIx[:-1],3]]
            for AffectedX in TruncationIx:
                logging.info('Truncating lagoon end of outlet channel (due to erosion) at X = %.1fm' % ShoreX[AffectedX])
            
            # Move the ustream end of the outlet channel
            if OutletEndX[0] < OutletEndX[1]:
                # Outlet angles from L to R
                OutletEndX[0] = ShoreX[TruncationIx[0] + 1]
            else:
                # Outlet angles from R to L
                OutletEndX[0] = ShoreX[TruncationIx[0] - 1]
            
            # Recompute OutletChanIx
            if OutletEndX[0] < OutletEndX[1]:
                # Outlet angles from L to R
                OutletChanIx = np.where(np.logical_and(OutletEndX[0] <= ShoreX, 
                                                       ShoreX <= OutletEndX[1]))[0]
            else:
                # Outlet angles from R to L
                OutletChanIx = np.flipud(np.where(np.logical_and(OutletEndX[1] <= ShoreX,
                                                                 ShoreX <= OutletEndX[0]))[0])
            assert OutletChanIx.size > 0, 'zero length outlet channel after lagoonward truncation'
            
            # Check if truncation is a collision between outlet channel and cliff
            # i.e. where there is no lagoon. If so then extend lagoon.
            if ShoreY[TruncationIx[0], 2] <= ShoreY[TruncationIx[0], 4]:    
                if ShoreX[TruncationIx[0]] > 0:
                    # Extend right end of lagoon
                    logging.info('Extending R end of lagoon via outletchannel to cliffline collision.')
                    Extend = True
                    CurLagEndIx = np.where(ShoreY[:,3] > ShoreY[:,4])[0][-1]
                    LagExtension = np.arange(CurLagEndIx + 1, TruncationIx[0] + 1)
                else:
                    # Extend left end of lagoon
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
                OutletPresent[LagExtension] = False
    
    # 7. Check for truncation of seaward end of outlet channel
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
    
    # 8. Check for collision between outlet channel and lagoon at downstream end
    # Truncation is not possible here (as it would result in an outlet channel 
    # with no transects) so we adjust the outlet channel position to prevent it...
    if not Closed:
        if ShoreY[OutletChanIx[-1],3] >= ShoreY[OutletChanIx[-1],2]:
            logging.info('Preventing lagoonward truncation at X = %.1fm as it would result in a length 0 outlet channel' % 
                         ShoreX[OutletChanIx[-1]])
            ShoreY[OutletChanIx[-1],3] = ShoreY[OutletChanIx[-1],2] - 0.0001
            
            # if this pushes into cliff then assume cliff erodes depositing sed into outlet channel...
            if ShoreY[OutletChanIx[-1],3] < ShoreY[OutletChanIx[-1],4]:
                logging.info('Preventing lagoonward truncation means outlet channel pushed into cliff at X = %.1fm' % 
                             ShoreX[OutletChanIx[-1]])
                CliffOverlapDist = ShoreY[OutletChanIx[-1],4] - ShoreY[OutletChanIx[-1],3]
                
                CliffRetDist = (CliffOverlapDist / 
                                (1 + ((PhysicalPars['BackshoreElev'] - ShoreZ[OutletChanIx[-1],2]) / 
                                      (ShoreZ[OutletChanIx[-1],2] - ShoreZ[OutletChanIx[-1],1]))))
                ShoreY[OutletChanIx[-1], 4] -= CliffRetDist
                ShoreY[OutletChanIx[-1], [2,3]] += (CliffOverlapDist - CliffRetDist)
    
    #%% Check for complete erosion of barrier between offline outlet channel and lagoon
    # This *should* only include offline/disconnected bits of outlet as online 
    # bits have already been dealt with...
    #
    # Remove old outlet channel and preserve sediment mass balance
    
    LagoonIntersect = ~np.greater(ShoreY[:,2], ShoreY[:,3], where=~np.isnan(ShoreY[:,1]))
    if np.any(LagoonIntersect):
        for IntersectIx in np.where(LagoonIntersect)[0]:
            logging.info('Outlet intersects lagoon at X = %.1fm - adding channel width into lagoon' % 
                         ShoreX[IntersectIx])
        ShoreZ[LagoonIntersect, 3] = (((ShoreY[LagoonIntersect, 1] - ShoreY[LagoonIntersect, 2]) * ShoreZ[LagoonIntersect, 1] 
                                      + (ShoreY[LagoonIntersect, 3] - ShoreY[LagoonIntersect, 4]) * ShoreZ[LagoonIntersect, 3]) 
                                      / (ShoreY[LagoonIntersect, 1] - ShoreY[LagoonIntersect, 4]))
        ShoreY[LagoonIntersect, 3] += (ShoreY[LagoonIntersect, 1] - ShoreY[LagoonIntersect, 2]) 
        # Remove outlet channel from transect now it has been dissolved into lagoon
        ShoreY[LagoonIntersect, 1] = np.nan
        ShoreY[LagoonIntersect, 2] = np.nan
        ShoreZ[LagoonIntersect, 1] = np.nan
    
    #%% ID locations where existing or relic outlet channel is connected to lagoon
    # Also update WaterLevel variable to include outlet channel water level 
    # (rather than lagoon WL) for connected bits of outlet channel.
    
    WideEnoughChannel = np.full(ShoreX.shape, False)
    WideEnoughChannel[OutletPresent] = (ShoreY[OutletPresent,1]-ShoreY[OutletPresent,2]) >= PhysicalPars['MinOutletWidth']
    OutletUsIx = np.argmin(np.abs(ShoreX-OutletEndX[0]))
    
    if not Closed:
        # Lagoon open - use online outlet to start search for connected channel
        ConnectedChanMinMax = np.array([np.min(OutletChanIx), np.max(OutletChanIx)])
        
    elif not np.isnan(OutletEndX[0]):
        # Lagoon closed but potentially still connected to relic outlet chanel
        
        # check if relic outlet channel has become disconnected
        if not WideEnoughChannel[OutletUsIx]:
            OutletEndX[0] = np.nan
            ConnectedChanMinMax = [np.nan, np.nan]
        elif not np.all(LagoonOpen[min(OutletUsIx,X0Ix):max(OutletUsIx,X0Ix)+1]):
            OutletEndX[0] = np.nan
            ConnectedChanMinMax = [np.nan, np.nan]
        else:
            # relic outlet channel still connected at upstream end
            ConnectedChanMinMax = [OutletUsIx, OutletUsIx]
    else:
        # Lagoon closed and no connected relic channel
        ConnectedChanMinMax = [np.nan, np.nan]
        
    # Find additional connected bits of relic outlet channel
    if not np.isnan(ConnectedChanMinMax[0]):
        while WideEnoughChannel[ConnectedChanMinMax[0]-1]:
            WaterLevel[ConnectedChanMinMax[0]-1] = WaterLevel[ConnectedChanMinMax[0]]
            ConnectedChanMinMax[0] -= 1
    
    if not np.isnan(ConnectedChanMinMax[1]):
        while WideEnoughChannel[ConnectedChanMinMax[1]+1]:
            WaterLevel[ConnectedChanMinMax[1]+1] = WaterLevel[ConnectedChanMinMax[1]]
            ConnectedChanMinMax[1] += 1
            
    ConnectedChan = np.logical_and(np.arange(0,ShoreX.size) >= ConnectedChanMinMax[0],
                                   np.arange(0,ShoreX.size) <= ConnectedChanMinMax[1])
    assert not np.any(np.isnan(WaterLevel)), 'NaN value(s) in WaterLevel at X = %s' % ShoreX[np.isnan(WaterLevel)]
    
    #%% Check for complete erosion of barrier between offline relic outlet channel and sea.
    # This *should* only include offline/disconnected bits of outlet as online 
    # bits have already been dealt with...
    #
    # In most cases complete erosion of the barrier between offline outlet 
    # channel and sea just requires adjusting ShoreY to remove outlet channel 
    # and preserve mass balance. However it can also trigger breaching if the 
    # lagoon is cloased and the relic outlet channel is connected to the lagoon.
    
    Breach = False
    BreachIx = np.nan
    
    ShoreIntersect = ~np.greater(ShoreY[:,0], ShoreY[:,1], where=~np.isnan(ShoreY[:,1]))
    if np.any(ShoreIntersect):
        # If lagoon closed then check if channel at any eroded location(s) is connected to lagoon via relic channel
        if Closed and np.any(ShoreIntersect[ConnectedChan]):
            # if so trigger a breach
            Breach = True
            # at the eroded location closest to the upstream end of the relic outlet channel
            PossBreachTsects = np.where(np.logical_and(ShoreIntersect, ConnectedChan))[0]
            ChanLength = np.abs(ShoreX[PossBreachTsects] - OutletEndX[0])
            BreachIx = PossBreachTsects[np.argmin(ChanLength)]
            ShoreIntersect = ShoreIntersect[ShoreIntersect!=BreachIx]
    
        # Adjust ShoreY where outlet banks intersects coast or lagoon    
        for IntersectIx in np.where(ShoreIntersect)[0]:
            logging.info('Outlet intersects shoreline at X = %.1fm - filling outlet with sediment from shoreface' % 
                         ShoreX[IntersectIx])
        ShoreY[ShoreIntersect, 0] -= ((ShoreY[ShoreIntersect, 1] - ShoreY[ShoreIntersect, 2]) 
                                      * (ShoreZ[ShoreIntersect, 2] - ShoreZ[ShoreIntersect, 1]) 
                                      / (ShoreZ[ShoreIntersect, 2] + PhysicalPars['ClosureDepth']))
        # Remove outlet channel from transect now it has been dissolved into shoreline
        ShoreY[ShoreIntersect, 1] = np.nan
        ShoreY[ShoreIntersect, 2] = np.nan
        ShoreZ[ShoreIntersect, 1] = np.nan
        OutletPresent[ShoreIntersect] = False
        # Remove inner barrier now there's no outlet in the transect
        ShoreZ[ShoreIntersect, 0] = ShoreZ[ShoreIntersect, 2]
        ShoreZ[ShoreIntersect, 2] = np.nan
    
    #%% ID locations where barrier completely eroded (and there is a lagoon)
    # should only apply to locations where there is no channel in barrier
    EroTsects = np.where(np.logical_and(ShoreY[:,3]>=ShoreY[:,0], ShoreY[:,3]>ShoreY[:,4]))[0]
    
    if EroTsects.size > 0:
        # Check which eroded locations are hydraulically connected to river
        PossBreachTsects = []
        for TsectIx in EroTsects:
            logging.info('Barrier completely eroded at X = %.1fm' % ShoreX[TsectIx])
            if TsectIx > X0Ix:
                if np.all(LagoonOpen[np.arange(X0Ix+1, TsectIx+1)]):
                    PossBreachTsects.append(TsectIx)
            elif TsectIx < X0Ix:
                if np.all(LagoonOpen[np.arange(TsectIx, X0Ix)]):
                    PossBreachTsects.append(TsectIx)
            else: # TsectIx == X0Ix
                PossBreachTsects.append(TsectIx)
        
        # Decide whether erosion causes breach
        if Breach:
            logging.info('Complete barrier erosion does not cause breach as relic channel has already re-opened')
        elif len(PossBreachTsects) == 0:
            logging.info('Not possible for breach to occur as eroded barrier is on disconnected part of lagoon')
        elif Closed:
            # If closed then any (connected) erosion of barrier causes breach
            logging.info('Erosion breach of previously closed lagoon')
            Breach = True
        elif np.min(np.abs(ShoreX[PossBreachTsects])) < np.abs(OutletEndX[0]):
            # if open then barrier erosion only causes breach if it is closer to the river than the existing outlet
            # (as the model cannot handle multiple outlets)
            logging.info('Erosion breach in open lagoon')
            Breach = True
        else:
            # barrier eroded but further from river than existing outlet 
            logging.info('Preventing breach as eroded barrier is further from river than existing outlet')
        
        # ID which eroded transect is the breach (if any) and which should not breach (if any)
        if Breach and np.isnan(BreachIx):
            BreachIx = PossBreachTsects[np.argmin(np.abs(ShoreX[PossBreachTsects]))]
            NotBreachIx = EroTsects[EroTsects != BreachIx]
        else:
            NotBreachIx = EroTsects
        
        # Adjust any transects which are not breaching (in a mass conservative way) to restore barrier
        if NotBreachIx.size > 0:
            OverlapDist = 0.001 + (ShoreY[NotBreachIx,3] - ShoreY[NotBreachIx,0])
            ShoreShiftDist = OverlapDist / (1 + ((ShorefaceHeight[NotBreachIx]) / 
                                                 (ShoreZ[NotBreachIx, 0] - ShoreZ[NotBreachIx, 3])))
            ShoreY[NotBreachIx,0] += ShoreShiftDist
            ShoreY[NotBreachIx,3] -= (OverlapDist - ShoreShiftDist)
    
    #%% Check for overtopping breach 
    
    # Find the barrier height which has to be overtopped to cause a breach:
    CrestHeight = ShoreZ[:,0] # default
    CrestHeight[OutletPresent] = np.amax(ShoreZ[OutletPresent,:][:,[0,2]], axis=1) # transects with active or relic channels (could be diconnected)
    CrestHeight[ConnectedChan] = ShoreZ[ConnectedChan,0] # transects with connected channels (active or relic)
    assert not np.any(np.isnan(CrestHeight)), 'NaN value(s) in CrestHeight at X = %s' % ShoreX[np.isnan(CrestHeight)]
    
    if not Breach and np.any(CrestHeight < WaterLevel):
        # Note: only allowed if an erosion breach has not already occured in the same timestep
        
        SpillTsects = np.where(CrestHeight < WaterLevel)[0]
        # Check which overtopped locations are hydraulically connected to river
        PossBreachTsects = []
        LagoonOpen = (ShoreY[:,3] - ShoreY[:,4]) > PhysicalPars['MinOutletWidth']
        for TsectIx in SpillTsects:
            logging.info('Potential overtopping breach at X = %.1f' % ShoreX[TsectIx])
            if ConnectedChan[TsectIx]:
                # Potential breach on connected channel
                PossBreachTsects.append(TsectIx)
            if TsectIx > X0Ix:
                # Potential breach to right of river
                if np.all(LagoonOpen[np.arange(X0Ix+1, TsectIx+1)]):
                    PossBreachTsects.append(TsectIx)
            elif TsectIx < X0Ix:
                # Potential breach to left of river
                if np.all(LagoonOpen[np.arange(TsectIx, X0Ix)]):
                    PossBreachTsects.append(TsectIx)
            else: # TsectIx == X0Ix
                PossBreachTsects.append(TsectIx)
        PossBreachTsects = np.array(PossBreachTsects)
        
        if PossBreachTsects.size == 0 and not Closed:
            # If lagoon open then assume overtopping can only cause breach if it 
            # is closer to where river enters lagoon than existing outlet
            CloserToRiv = np.abs(ShoreX) < np.min(np.abs(OutletEndX))
            CloserToRiv[ConnectedChan] = np.abs(ShoreX[ConnectedChan] - OutletEndX[0]) < np.abs(OutletEndX[1]-OutletEndX[0])
            PossBreachTsects = PossBreachTsects[CloserToRiv[PossBreachTsects]]
        
        if PossBreachTsects.size == 0:
            logging.info('Not possible for breach to occur as spill is on disconnected part of lagoon or further from river than existing outlet')
        else: 
            # Overtopping breach occurs - we have to work out where incase 
            # there is still multiple possible locations...
            # -> Assume that breach occurs where overtopping is deepest
            BreachIx = PossBreachTsects[np.argmax(WaterLevel[PossBreachTsects] -
                                                  CrestHeight[PossBreachTsects])]
            Breach = True
            if Closed:
                logging.info('Spill breach of closed lagoon at X = %.1fm' % ShoreX[BreachIx])
            else:
                logging.info('Spill breach of open lagoon at X = %.1fm' % ShoreX[BreachIx])
            
    
    #%% Create breach if one has been triggered
    if Breach:
        if BreachIx in ConnectedChan:
            # Outlet truncation breach
            logging.info('Breach on pre-existing outlet channel at X = %.1fm' % ShoreX[BreachIx])
            OutletEndX[1] = ShoreX[BreachIx]
            # TODO: close sediment balance by putting breach eroded sed onto shore
        else:
            # Lagoon breach (i.e. new outlet channel)
            logging.info('Breach/creation of new outlet channel at X = %.1fm' % ShoreX[BreachIx])
            
            OutletEndX[:] = ShoreX[BreachIx]
            ShoreZ[BreachIx, 2] = ShoreZ[BreachIx, 0]
            
            # Assume breach width equals the bigger of current outlet end width and 2 times the outlet width for closure
            OutletEndWidth.flat[0] = max(OutletEndWidth, 2*PhysicalPars['MinOutletWidth'])
            ShoreY[BreachIx,1] = min(ShoreY[BreachIx,0]-PhysicalPars['SpitWidth'],
                                     (ShoreY[BreachIx,0]+ShoreY[BreachIx,3])/2 + OutletEndWidth/2)
            ShoreY[BreachIx,2] = ShoreY[BreachIx,1] - OutletEndWidth
            
            # Assume outlet end bed level is the lower of the current outlet end bed level, or the MaxOutletElev
            OutletEndElev.flat[0] = min(OutletEndElev, PhysicalPars['MaxOutletElev'])
            ShoreZ[BreachIx,1] = 0.5 * OutletEndElev + 0.5 * ShoreZ[BreachIx,3]
            
            # Check the newly created outlet channel fits into the barrier...
            if ShoreY[BreachIx,3] >= ShoreY[BreachIx,2]:
                logging.warning('barrier widened to fit breach outlet channel')
                ShoreY[BreachIx,3] = ShoreY[BreachIx,2] - 0.001
            if ShoreY[BreachIx,4] >= ShoreY[BreachIx,2]:
                logging.warning('cliff retreated to fit breach outlet channel')
                ShoreY[BreachIx,4] = ShoreY[BreachIx,2] - 0.001
    
    #%% Prevent lagoon closure at X=0 (where river enters)
    # This ensures there is always one viable cross-section in the lagoon
    if (ShoreY[X0Ix,3] - ShoreY[X0Ix,4]) < PhysicalPars['MinOutletWidth']:
        logging.info('Modifying "backshore" location at X=0 to ensure at least 1 lagoon XS is maintained')
        ShoreY[X0Ix,4] = ShoreY[X0Ix,3] - PhysicalPars['MinOutletWidth'] - 0.001
    
    assert np.all(~np.isnan(ShoreY[:,0])), 'NaN value(s) in ShoreY[:,0] at X = %s' % ShoreX[np.isnan(ShoreY[:,0])]
    assert np.all(~np.isnan(ShoreY[:,3])), 'NaN value(s) in ShoreY[:,3] at X = %s' % ShoreX[np.isnan(ShoreY[:,3])]
    assert np.all(~np.isnan(ShoreY[:,4])), 'NaN value(s) in ShoreY[:,4] at X = %s' % ShoreX[np.isnan(ShoreY[:,4])]
    
    #%% Return updated MorDt (adaptive timestepping) and Closed/open status (opening happens in mor, closing in riv.assembleChannel)
    return (MorDt, Breach)
