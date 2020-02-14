# -*- coding: utf-8 -*-
""" Sub-module of hapuamod containing functions to plot aspects of hapuamod.
"""

# import standard packages
import matplotlib.pyplot as plt
import numpy as np
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

# import local packages
from hapuamod import geom

def mapView(ShoreX, ShoreY, Origin, ShoreNormDir):
    """ Map the current model state in real world coordinates
    """
    
    # convert the coordinates to real world
    (ShoreXreal, ShoreYreal) = geom.mod2real(np.transpose(np.tile(ShoreX, [5,1])), 
                                               ShoreY, Origin, ShoreNormDir)
    
    # calculate the (idealised) river location
    RiverX = np.asarray([0,0])
    RiverY = np.asarray([np.min(ShoreY[:,4])*3,
                         geom.intersectPolyline(np.stack([ShoreX, ShoreY[:,4]], axis=1), 0)[0]])
    (RiverXreal, RiverYreal) = geom.mod2real(RiverX, RiverY, Origin, ShoreNormDir)
    
    # draw the main features
    plt.plot(ShoreXreal[:,0], ShoreYreal[:,0], 'g-', label='Shore')    
    #plt.plot(ShoreXreal[:,1], ShoreYreal[:,1], 'r-', label='Outlet')
    plt.plot(ShoreXreal[:,2], ShoreYreal[:,2], 'r-')
    plt.plot(ShoreXreal[:,3], ShoreYreal[:,3], 'c-', label='Lagoon')
    plt.plot(ShoreXreal[:,4], ShoreYreal[:,4], 'k-', label='Cliff')
    plt.plot(RiverXreal, RiverYreal, 'b-', label='River')
    
    EndTransects = np.where(np.isnan(ShoreY[:,3])==False)[0][[0,-1]]
    plt.plot(ShoreXreal[EndTransects[0],[3,4]], ShoreYreal[EndTransects[0],[3,4]], 'b-')
    plt.plot(ShoreXreal[EndTransects[1],[3,4]], ShoreYreal[EndTransects[1],[3,4]], 'b-')
    
    # plot the origin and axes
    plt.plot(Origin[0], Origin[1], 'ko', label='Origin', zorder=99)
    (BaseXreal, BaseYreal) = geom.mod2real(np.array([ShoreX[0],0,ShoreX[-1]]), 
                                           np.array([0,0,0]), 
                                           Origin, ShoreNormDir)
    ArrowX = (BaseXreal[-1]-BaseXreal[0])/20
    ArrowY = (BaseYreal[-1]-BaseYreal[0])/20
    plt.arrow(BaseXreal[1], BaseYreal[1], ArrowX, ArrowY, 
              width=30, zorder=98, facecolor='white')
    plt.arrow(BaseXreal[1], BaseYreal[1], -ArrowY, ArrowX, 
              width=30, zorder=97, facecolor='white')
    XLab = plt.annotate('X', (BaseXreal[1] + ArrowX*2.4, BaseYreal[1] + ArrowY*2.4), 
                        horizontalalignment='center', verticalalignment='center')
    YLab = plt.annotate('Y', (BaseXreal[1] - ArrowY*2.4, BaseYreal[1] + ArrowX*2.4), 
                        horizontalalignment='center', verticalalignment='center')
    XLab.set_bbox(dict(facecolor='white', edgecolor='none', boxstyle='Square, pad=0.1'))
    YLab.set_bbox(dict(facecolor='white', edgecolor='none', boxstyle='Square, pad=0.1'))
    
    # add labels
    plt.legend()
    plt.xlabel('Easting (m)')
    plt.ylabel('Northing (m)')
    
    # tidy up the plot
    plt.axis('equal')

def modelView(ShoreX, ShoreY, OutletEndX, OutletEndWidth, OutletChanIx, RiverWidth,
              ShoreZ=None, WavePower=None, EDir_h=0, LST=None, CST=None, 
              WaveScaling=0.01, CstScaling=0.00005, LstScaling=0.0001,
              QuiverWidth=0.002, AreaOfInterest=None, PlotTime=None):
    """ Map the current model state in model coordinates
    
        Parameters:
            ShoreX
            ShoreY
            OutletEndX
            OutletChanIx
            RiverWidth
            ShoreZ (optional)
            WavePower (optional)
            EDir_h (optional)
            LST (optional)
            CST (optional)
            WaveScaling (optional, defalt = 0.01)
            CstScaling (optional, default = 0.00005)
            LstScaling (optional, default = 0.0001)
            QuiverWidth (optional, default = 0.002)
            AreaOfInterest (optional)
            
        Returns:
            ModelFig (dict): handles for the various parts of the plot. Used as
                input to updateModelView.
        
        Note: smaller 'scale' makes arrows longer
    """
    
    # Create a new figure
    if not ShoreZ is None:
        PlanFig = plt.figure(figsize=[10,6])
        PlanAx = plt.subplot2grid((3,1), (0,0), rowspan=2)
        VertAx = plt.subplot2grid((3,1), (2,0), sharex=PlanAx)
        VertAx.set_ylim([1,5])
    else:
        PlanFig, PlanAx = plt.subplots(figsize=[10,5])
    
    # Set the axis backgrounjd to grey (gravel!)
    PlanAx.set_facecolor('lightgrey')
    
    # Set the field of view
    PlanAx.axis('equal')
    if not AreaOfInterest is None:
        assert len(AreaOfInterest) == 4, 'AreaOfInterest must be given as four parameters (Xmin, Xmax, Ymin, Ymax)'
        PlanAx.set_xlim(AreaOfInterest[0], AreaOfInterest[1])
        PlanAx.set_ylim(AreaOfInterest[2], AreaOfInterest[3])
    else:
        PlanAx.set_xlim(ShoreX[np.logical_not(np.isnan(ShoreY[:,3]))][[0,-1]])
        PlanAx.set_ylim(np.nanmin(ShoreY), np.nanmax(ShoreY))
    
    # Create dummy lines
    WaterFill,   = PlanAx.fill(ShoreX, ShoreY[:,3], 'lightskyblue', label='Water', zorder=0)
    CliffFill,   = PlanAx.fill(ShoreX, ShoreY[:,4], fill=False, hatch = '/', label='Cliff', zorder=1)
    ShoreLine,   = PlanAx.plot(ShoreX, ShoreY[:,0], 'k-', label='Shore', zorder=4)
    ShoreDots,   = PlanAx.plot(ShoreX, ShoreY[:,0], 'k.', label='Shore', color='grey', zorder=3)
    OutletLine,  = PlanAx.plot(ShoreX, ShoreY[:,1], 'k-', label='Outlet bank', zorder=5)
    ChannelLine, = PlanAx.plot(ShoreX, ShoreY[:,1], '-x', label='Channel', color='grey', zorder=2)
    LagoonLine,  = PlanAx.plot(ShoreX, ShoreY[:,3], 'k-', label='Lagoon', zorder=6)
    CliffLine,   = PlanAx.plot(ShoreX, ShoreY[:,4], 'k-', label='Cliff', zorder=7)
    RiverLine,   = PlanAx.plot([0, 0], [-1000, 0], 'k-', label='River bank', zorder=8)
    if not ShoreZ is None:
        CrestLine, = VertAx.plot(ShoreX, ShoreZ[:,0], 'k-', label='Barrier crest')
    
    if not WavePower is None:
        WaveArrow = PlanAx.arrow(0,200,100,100, zorder=11)
    
    if not LST is None:
        LstQuiver = PlanAx.quiver((ShoreX[:-1]+ShoreX[1:])/2, 
                                  (ShoreY[:-1,0]+ShoreY[1:,0])/2, 
                                  LST, np.zeros(LST.size),
                                  scale=LstScaling, width=QuiverWidth,
                                  scale_units='x', units='width', 
                                  color='red', zorder=9)
    
    if not CST is None:
        CstQuiver = PlanAx.quiver(ShoreX, ShoreY[:,0], 
                                  np.zeros(CST.size), -CST, 
                                  scale=CstScaling, width=QuiverWidth,
                                  scale_units='x', units='width', 
                                  color='red', zorder=10)
    
    # Add some labels
    #PlanAx.legend()
    PlanAx.set_xlabel('Alongshore distance (m)')
    PlanAx.set_ylabel('Cross-shore distance (m)')
    if not ShoreZ is None:
        VertAx.set_ylabel('Elevation (m)')
    
    # Compile output variable
    ModelFig = {'PlanFig':PlanFig, 'PlanAx':PlanAx, 'ShoreLine':ShoreLine, 
                'OutletLine':OutletLine, 'RiverLine':RiverLine,
                'LagoonLine':LagoonLine, 'CliffLine':CliffLine, 
                'WaterFill':WaterFill, 'RiverWidth':RiverWidth,
                'ShoreDots':ShoreDots, 'ChannelLine':ChannelLine,
                'CliffFill':CliffFill}
    if not ShoreZ is None:
        ModelFig['CrestLine'] = CrestLine
    if not WavePower is None:
        ModelFig['WaveArrow'] = WaveArrow
        ModelFig['WaveScaling'] = WaveScaling
    if not LST is None:
        ModelFig['LstQuiver'] = LstQuiver
    if not CST is None:
        ModelFig['CstQuiver'] = CstQuiver
    
    # Update plot with correct data
    updateModelView(ModelFig, ShoreX, ShoreY, OutletEndX, OutletEndWidth, 
                    OutletChanIx, ShoreZ=ShoreZ, WavePower=WavePower, 
                    EDir_h=EDir_h, LST=LST, CST=CST, PlotTime=PlotTime)
    
    return ModelFig

def updateModelView(ModelFig, ShoreX, ShoreY, OutletEndX, OutletEndWidth, 
                    OutletChanIx, Closed=False, 
                    ShoreZ=None, WavePower=None, EDir_h=0, LST=None, CST=None,
                    PlotTime=None):
    
    RiverWidth = ModelFig['RiverWidth']
    
    # Prep the bits of the plotting which are the same whether the outlet is closed or not:
    
    # Channel (online and offline) plotting position
    ChannelX = np.tile(ShoreX, 2)
    ChannelY = np.asarray([ShoreY[:,1], ShoreY[:,2]])
    
    # Cliff
    CliffToL = ShoreX <= (-RiverWidth/2)
    CliffToR = ShoreX >= (RiverWidth/2)
    CliffToLX  = np.hstack([ShoreX[CliffToL], -RiverWidth/2])
    CliffToRX  = np.hstack([RiverWidth/2, ShoreX[CliffToR]])
    CliffLineX = np.hstack([CliffToLX, np.nan, CliffToRX])
    CliffToLY  = np.hstack([ShoreY[CliffToL, 4], np.interp(-RiverWidth/2, ShoreX, ShoreY[:,4])])
    CliffToRY  = np.hstack([np.interp(RiverWidth/2, ShoreX, ShoreY[:,4]), ShoreY[CliffToR, 4]])
    CliffLineY = np.hstack([CliffToLY, np.nan, CliffToRY])
    
    # River
    RiverLbX = np.ones(2) * -RiverWidth/2
    RiverRbX = np.ones(2) * RiverWidth/2
    RiverLineX = np.hstack([RiverLbX, np.nan, RiverRbX])
    RiverLbY = np.hstack([-5000.0, CliffToLY[-1]])
    RiverRbY = np.hstack([-5000.0, CliffToRY[0]])
    RiverLineY = np.hstack([RiverLbY, np.nan, RiverRbY])
    
    # Cliff hatch polygon
    CliffHatchX = np.hstack([CliffToLX,             # Cliff LHS
                             np.flipud(RiverLbX),   # River LB
                             CliffToLX[[0,0]],      # LH Edge of domain
                             np.nan,                # Break to separate L and R polygons
                             CliffToRX[[-1,-1]],    # RH Edge of domain
                             RiverRbX,              # River RB
                             CliffToRX])            # Cliff RHS
    CliffHatchY = np.hstack([CliffToLY,             # Cliff LHS
                             np.flipud(RiverLbY),   # River LB
                             -5000.0, CliffToLY[0], # LH Edge of domain
                             np.nan,                # Break to separate L and R polygons
                             CliffToRX[-1],-5000.0, # RH Edge of domain
                             RiverRbY,              # River RB
                             CliffToRY])            # Cliff RHS
    
    # Wave arrow plotting position
    if not WavePower is None:
        if (-np.pi/2) < EDir_h < (np.pi/2):
            ArrLength = WavePower
        else:
            ArrLength = 0.01
        ArrWidth = np.maximum((ArrLength * ModelFig['WaveScaling'])/10, 0.001)
        WaveX = -np.sin(EDir_h) * ArrLength * ModelFig['WaveScaling']
        WaveY = -np.cos(EDir_h) * ArrLength * ModelFig['WaveScaling']
    
    # Prep the bits of the plotting which depend on whether the outlet is closed or not...
    
    if Closed:
        # Outlet plotting position
        OutletX = np.nan
        OutletY = np.nan
        
        # Shoreline
        ShoreLineX = ShoreX
        ShoreLineY = ShoreY[:,0]
        ShoreDotsX = np.nan
        ShoreDotsY = np.nan
        
        # BarrierBackshore
        LagoonLineX = ShoreX
        LagoonLineY = ShoreY[:,3]
        
        # Water polygon
        WaterX = np.hstack([LagoonLineX,            # Lagoon backshore
                            np.flipud(CliffToRX),   # Cliff RHS
                            np.flipud(RiverRbX),    # River RB
                            RiverLbX,               # River LB
                            np.flipud(CliffToLX),   # Cliff LHS
                            LagoonLineX[0],         # Close the loop
                            np.nan,                 # Break to separate sea and lagoon polygons
                            ShoreLineX,             # Coast
                            ShoreX[[-1,0]],         # Offshore
                            ShoreLineX[0]])         # Close the loop
    
        WaterY = np.hstack([LagoonLineY,            # Lagoon backshore
                            np.flipud(CliffToRY),   # Cliff RHS
                            np.flipud(RiverRbY),    # River RB
                            RiverLbY,               # River LB
                            np.flipud(CliffToLY),   # Cliff LHS
                            LagoonLineY[0],         # Close the loop
                            np.nan,                 # Break to separate sea and lagoon polygons
                            ShoreLineY,             # Coast
                            5000.0, 5000.0,         # Offshore
                            ShoreLineY[0]])         # Close the loop
        
    else: # Outlet open
        
        LagoonPresent = ShoreY[:,3] > ShoreY[:,4]
        Dx = ShoreX[1] - ShoreX[0]
        LagoonLeftEndX = np.min(ShoreX[LagoonPresent]) - Dx
        LagoonRightEndX = np.max(ShoreX[LagoonPresent]) + Dx
        
        OutletUsLbPlotX = max(OutletEndX[0] - OutletEndWidth[0]/2, LagoonLeftEndX)
        OutletUsRbPlotX = min(OutletEndX[0] + OutletEndWidth[0]/2, LagoonRightEndX)
        
        OutletDsLbX = OutletEndX[1] - OutletEndWidth[1]/2
        OutletDsRbX = OutletEndX[1] + OutletEndWidth[1]/2
        
        # Outlet plotting position
        if OutletEndX[0] < OutletEndX[1]:
            # Outlet angkles L to R
            L_Ok = ShoreX[OutletChanIx] < OutletDsLbX
            R_Ok = ShoreX[OutletChanIx] > OutletUsRbPlotX
            OutletLbY = np.hstack([np.interp(OutletUsLbPlotX, ShoreX, ShoreY[:,3]),
                                   ShoreY[OutletChanIx[0],1], 
                                   ShoreY[OutletChanIx[L_Ok],1], 
                                   np.interp(OutletDsLbX, ShoreX[OutletChanIx], 
                                             ShoreY[OutletChanIx, 1]),
                                   np.interp(OutletDsLbX, ShoreX, ShoreY[:,0])])
            OutletRbY = np.hstack([np.interp(OutletUsRbPlotX, ShoreX, ShoreY[:,3]),
                                   np.interp(OutletUsRbPlotX, ShoreX[OutletChanIx], 
                                             ShoreY[OutletChanIx, 2]),
                                   ShoreY[OutletChanIx[R_Ok],2],
                                   ShoreY[OutletChanIx[-1],2],
                                   np.interp(OutletDsRbX, ShoreX, ShoreY[:,0])])
        else:
            # Outlet angkles R to L
            L_Ok = ShoreX[OutletChanIx] < OutletUsLbPlotX
            R_Ok = ShoreX[OutletChanIx] > OutletDsRbX
            OutletLbY = np.hstack([np.interp(OutletUsLbPlotX, ShoreX, ShoreY[:,3]),
                                   np.interp(OutletUsLbPlotX, ShoreX[np.flip(OutletChanIx)], 
                                             ShoreY[np.flip(OutletChanIx), 2]),
                                   ShoreY[OutletChanIx[L_Ok],2],
                                   ShoreY[OutletChanIx[-1],2],
                                   np.interp(OutletDsLbX, ShoreX, ShoreY[:,0])])
            OutletRbY = np.hstack([np.interp(OutletUsRbPlotX, ShoreX, ShoreY[:,3]),
                                   ShoreY[OutletChanIx[0],1],
                                   ShoreY[OutletChanIx[R_Ok],1],
                                   np.interp(OutletDsRbX, ShoreX[np.flip(OutletChanIx)], 
                                             ShoreY[np.flip(OutletChanIx), 1]),
                                   np.interp(OutletDsRbX, ShoreX, ShoreY[:,0])])
        
        OutletLbY[1] = np.max(OutletLbY[:2])
        OutletRbY[1] = np.max(OutletRbY[:2])
        OutletLbY[-2] = np.min(OutletLbY[-2:])
        OutletRbY[-2] = np.min(OutletRbY[-2:])
        
        OutletY = np.hstack([OutletLbY, np.nan, OutletRbY])
        OutletLbX = np.hstack([OutletUsLbPlotX, OutletUsLbPlotX,
                               ShoreX[OutletChanIx[L_Ok]],
                               OutletDsLbX, OutletDsLbX])
        OutletRbX = np.hstack([OutletUsRbPlotX, OutletUsRbPlotX,
                               ShoreX[OutletChanIx[R_Ok]],
                               OutletDsRbX, OutletDsRbX])
        OutletX = np.hstack([OutletLbX, np.nan, OutletRbX])
    
        # Some useful masks for the transects
        LagoonToL = ShoreX <= OutletUsLbPlotX
        LagoonToR = ShoreX >= OutletUsRbPlotX
        ShoreToL = ShoreX <= OutletDsLbX
        ShoreToR = ShoreX >= OutletDsRbX
        ShoreInChan = ~np.logical_or(ShoreToL, ShoreToR)
    
        # Shoreline
        ShoreToLX  = np.hstack([ShoreX[ShoreToL], OutletLbX[-1]])
        ShoreToRX  = np.hstack([OutletRbX[-1], ShoreX[ShoreToR]])
        ShoreLineX = np.hstack([ShoreToLX, np.nan, ShoreToRX])
        ShoreToLY  = np.hstack([ShoreY[ShoreToL,0], OutletLbY[-1]])
        ShoreToRY  = np.hstack([OutletRbY[-1], ShoreY[ShoreToR,0]])
        ShoreLineY = np.hstack([ShoreToLY, np.nan, ShoreToRY])
        
        ShoreDotsX = ShoreX[ShoreInChan]
        ShoreDotsY = ShoreY[ShoreInChan,0]
        
        # BarrierBackshore
        LagoonToLX  = np.hstack([ShoreX[LagoonToL], OutletLbX[0]])
        LagoonToRX  = np.hstack([OutletRbX[0], ShoreX[LagoonToR]])
        LagoonLineX = np.hstack([LagoonToLX, np.nan, LagoonToRX])
        LagoonToLY  = np.hstack([ShoreY[LagoonToL, 3], OutletLbY[0]])
        LagoonToRY  = np.hstack([OutletRbY[0], ShoreY[LagoonToR, 3]])
        LagoonLineY = np.hstack([LagoonToLY, np.nan, LagoonToRY])
    
        # Water polygon
        WaterX = np.hstack([LagoonToRX,             # Lagoon backshore RHS
                            np.flipud(CliffToRX),   # Cliff RHS
                            np.flipud(RiverRbX),    # River RB
                            RiverLbX,               # River LB
                            np.flipud(CliffToLX),   # Cliff LHS
                            LagoonToLX,             # Lagoon backshore LHS
                            OutletLbX,              # Outlet channel LB
                            np.flipud(ShoreToLX),   # Coast LHS
                            ShoreX[[0,-1]],         # Offshore
                            np.flipud(ShoreToRX),   # Coast RHS
                            np.flipud(OutletRbX)])  # Outlet channel RB
    
        WaterY = np.hstack([LagoonToRY,             # Lagoon backshore RHS
                            np.flipud(CliffToRY),   # Cliff RHS
                            np.flipud(RiverRbY),    # River RB
                            RiverLbY,               # River LB
                            np.flipud(CliffToLY),   # Cliff LHS
                            LagoonToLY,             # Lagoon backshore LHS
                            OutletLbY,              # Outlet channel LB
                            np.flipud(ShoreToLY),   # Coast LHS
                            5000.0, 5000.0,         # Offshore
                            np.flipud(ShoreToRY),   # Coast RHS
                            np.flipud(OutletRbY)])  # Outlet channel RB
    
    # Update the lines etc
    ModelFig['ShoreLine'].set_data(ShoreLineX, ShoreLineY)
    ModelFig['ShoreDots'].set_data(ShoreDotsX, ShoreDotsY)
    ModelFig['ChannelLine'].set_data(ChannelX, ChannelY)
    ModelFig['OutletLine'].set_data(OutletX, OutletY)
    ModelFig['LagoonLine'].set_data(LagoonLineX, LagoonLineY)
    ModelFig['CliffLine'].set_data(CliffLineX, CliffLineY)
    ModelFig['RiverLine'].set_data(RiverLineX, RiverLineY)
    ModelFig['WaterFill'].set_xy(np.vstack([WaterX, WaterY]).T)
    ModelFig['CliffFill'].set_xy(np.vstack([CliffHatchX, CliffHatchY]).T)
    
    if not ShoreZ is None:
        ModelFig['CrestLine'].set_data(ShoreX, ShoreZ[:,0])
    
    if not WavePower is None:
        ModelFig['WaveArrow'].remove()
        ModelFig['WaveArrow'] = ModelFig['PlanAx'].arrow(0, 200, WaveX, WaveY,
                                                         width=ArrWidth, 
                                                         zorder=11)
    if not LST is None:
        LstScale = ModelFig['LstQuiver'].scale
        LstWidth = ModelFig['LstQuiver'].width
        ModelFig['LstQuiver'].remove()
        ModelFig['LstQuiver'] = ModelFig['PlanAx'].quiver((ShoreX[:-1]+ShoreX[1:])/2, 
                                                          (ShoreY[:-1,0]+ShoreY[1:,0])/2, 
                                                          LST, np.zeros(LST.size),
                                                          scale=LstScale, width=LstWidth,
                                                          scale_units='x', units='width',
                                                          color='red', zorder=9)
    
    if not CST is None:
        CstScale = ModelFig['CstQuiver'].scale
        CstWidth = ModelFig['CstQuiver'].width
        ModelFig['CstQuiver'].remove()
        ModelFig['CstQuiver'] = ModelFig['PlanAx'].quiver(ShoreX, ShoreY[:,0], 
                                                          np.zeros(ShoreX.size), -CST, 
                                                          scale=CstScale, width=CstWidth,
                                                          scale_units='x', units='width',
                                                          color='red', zorder=10)
    
    if not PlotTime is None:
        ModelFig['PlanAx'].set_title(PlotTime.strftime('%d/%m/%y %H:%M'), loc='right')
    
    # Redraw
    ModelFig['PlanFig'].canvas.draw()
    ModelFig['PlanFig'].canvas.flush_events()
    
def longSection(ChanDx, ChanElev, ChanWidth, ChanDep, ChanVel, Bedload=None,
                PlotTime=None, AreaOfInterest=None):
    """ Create a long section of the river to the lagoon outlet
    
    LongSecFig = longSection(ChanDx, ChanElev, ChanWidth, ChanDep, ChanVel, 
                             Bedload)
    
    Parameters:
        ChanDx
        ChanElev
        ChanWidth
        ChanDep
        ChanVel
        Bedload (optional)
    
    Returns:
        LongSecFig = (RivFig, BedLine, WaterLine, EnergyLine, WidthLine, 
                      VelLine, FrLine, QsLine)
    """
    g = 9.81
    Dist = np.insert(np.cumsum(ChanDx),0,0)
    D2 = (Dist[:-1]+Dist[1:])/2
    WL = ChanElev + ChanDep
    WL2 = (WL[:-1]+WL[1:])/2
    Vel2 = (ChanVel[:-1]+ChanVel[1:])/2
    Energy = WL2 + (Vel2**2) / (2*g)
    Fr = abs(ChanVel)/np.sqrt(g*ChanDep)
    Q = ChanVel * ChanDep * ChanWidth
    
    # Create new figure with sub plots
    RivFig = plt.figure(figsize=(9,9))
    if Bedload is None:
        NoOfPlots = 3
    else:
        NoOfPlots = 4
    ElevAx = RivFig.add_subplot(NoOfPlots,1,1)
    WidthAx = RivFig.add_subplot(NoOfPlots,1,2, sharex=ElevAx)
    FlowAx = WidthAx.twinx()
    VelAx = RivFig.add_subplot(NoOfPlots,1,3, sharex=ElevAx)
    FrAx = VelAx.twinx()
    if not Bedload is None:
        QsAx = RivFig.add_subplot(4,1,4, sharex=ElevAx)
    else:
        QsAx = None
    
    # Plot the river bed level, water surface and energy line
    BedLine, = ElevAx.plot(Dist, ChanElev, 'k-')
    WaterLine, = ElevAx.plot(Dist, WL, 'b-')
    EnergyLine, = ElevAx.plot(D2, Energy, 'b:')
    ElevAx.set_ylabel('Elevation [m]')
    ElevAx.grid(axis='x', which='both', linestyle=':')
    
    if not AreaOfInterest is None:
        assert len(AreaOfInterest) == 4, 'AreaOfInterest must be given as four parameters (Xmin, Xmax, Ymin, Ymax)'
        ElevAx.set_xlim(AreaOfInterest[0], AreaOfInterest[1])
        ElevAx.set_ylim(AreaOfInterest[2], AreaOfInterest[3])
    
    # Plot the river width and flow
    WidthLine, = WidthAx.plot(Dist, ChanWidth, 'k-')
    WidthAx.set_ylabel('Width [m]')
    WidthAx.set_ylim([0,np.amax(ChanWidth)+10])
    WidthAx.grid(axis='x', which='both', linestyle=':')
    
    FlowLine, = FlowAx.plot(Dist, Q, 'c-')
    FlowAx.set_ylabel('Flow [$\mathrm{m^3/s}$]')
    FlowAx.autoscale_view(tight = False)
    
    # Plot velocity and Froude number
    VelLine, = VelAx.plot(Dist, ChanVel, 'r-')
    VelAx.set_ylabel('Velocity [m/s]', color='red')
    VelAx.tick_params(axis='y', colors='red')
    VelAx.set_ylim([0,3])
    VelAx.grid(axis='x', which='both', linestyle=':')
    
    FrLine, = FrAx.plot(Dist, Fr, 'g-')
    FrAx.set_ylabel('Froude No.', color='green')
    FrAx.tick_params(axis='y', colors='green')
    FrAx.set_ylim([0,1.6])
    
    # Plot bedload
    if not Bedload is None:
        QsLine, = QsAx.plot(D2, Bedload*3600, 'k-')
        QsAx.set_ylabel(r'Bedload [$\mathrm{m^3/hr}$]')
        QsAx.set_xlabel('Distance downstream [m]')
        QsAx.set_ylim([0,50])
        QsAx.grid(axis='x', which='both', linestyle=':')
    else:
        QsLine=None
    
    # Add timestamp
    if not PlotTime is None:
        RivFig.suptitle(PlotTime.strftime('%d/%m/%y %H:%M'))
    
    # Compile outputs
    LongSecFig = {'RivFig':RivFig, 'ElevAx':ElevAx, 'WidthAx':WidthAx, 
                  'FlowAx':FlowAx, 'VelAx':VelAx, 'FrAx':FrAx, 'QsAx':QsAx,
                  'BedLine':BedLine, 'WaterLine':WaterLine, 
                  'EnergyLine':EnergyLine, 'WidthLine':WidthLine, 
                  'FlowLine':FlowLine, 'VelLine':VelLine, 'FrLine':FrLine,
                  'QsLine':QsLine}
    
    return(LongSecFig)

def updateLongSection(LongSecFig, ChanDx, ChanElev, ChanWidth, ChanDep, 
                      ChanVel, Bedload=None, PlotTime=None):
    
    # Calculate required variables to plot
    g = 9.81
    Dist = np.insert(np.cumsum(ChanDx),0,0)
    D2 = (Dist[:-1]+Dist[1:])/2
    WL = ChanElev + ChanDep
    WL2 = (WL[:-1] + WL[1:])/2
    Vel2 = (ChanVel[:-1] + ChanVel[1:]) / 2
    Energy = WL2 + (Vel2**2) / (2*g)
    Fr = abs(ChanVel) / np.sqrt(g * ChanDep)
    Q = ChanVel * ChanDep * ChanWidth
    
    # Update the lines
    LongSecFig['BedLine'].set_data(Dist, ChanElev)
    LongSecFig['WaterLine'].set_data(Dist, WL)
    LongSecFig['EnergyLine'].set_data(D2, Energy)
    LongSecFig['WidthLine'].set_data(Dist, ChanWidth)
    LongSecFig['FlowLine'].set_data(Dist, Q)
    LongSecFig['VelLine'].set_data(Dist, ChanVel)
    LongSecFig['FrLine'].set_data(Dist, Fr)
    if not Bedload is None:
        LongSecFig['QsLine'].set_data(D2, Bedload*3600)
    
    # Add timestamp
    if not PlotTime is None:
        LongSecFig['RivFig'].suptitle(PlotTime.strftime('%d/%m/%y %H:%M'))
    
    # Update flow axis scaling
    Qmin = min(0.,np.nanmin(Q))
    Qmax = np.nanmax(Q)
    LongSecFig['FlowAx'].set_ylim(Qmin, Qmax + 0.1*(Qmax-Qmin))
    
    # Redraw
    LongSecFig['RivFig'].canvas.draw()
    LongSecFig['RivFig'].canvas.flush_events()

def newTransectFig(ShoreX, ShoreY, ShoreZ, LagoonWL, OutletWL, SeaLevel, 
                   BeachSlope, BackshoreElev, ClosureDepth, BeachTopElev,
                   TransectX, PlotTime=None, AreaOfInterest=None):
    """ Create a plot of a specified transect line through the hapua
        
        TransectFig = newTransectFig(ShoreX, ShoreY, ShoreZ, LagoonWL, OutletWL, 
                                     SeaLevel, BeachSlope, BackshoreElev, 
                                     ClosureDepth, BeachTopElev, TransectX)
        
        Note: for this function SeaLevel should be a float rather than an 
              array.
    """
    
    # Find index of closest transect to TransectX
    assert ShoreX[0] <= TransectX <= ShoreX[-1], 'TransectX must be within the range of X in the model domain (%.1f to %.1f)' % (ShoreX[0], ShoreX[-1])
    TransectIx = np.argmin(np.abs(ShoreX - TransectX))
    
    # Create a new figure window
    TransFig, TransAx = plt.subplots(figsize=[10,5])
    if TransectX < 0:
        LeftRight = 'Transect through hapua %.0fm to left of river' % abs(ShoreX[TransectIx])
    elif TransectX == 0:
        LeftRight = 'Transect through hapua in line with river'
    else:
        LeftRight = 'Transect through hapua %.0fm to right of river' % abs(ShoreX[TransectIx])
    TransAx.set_title(LeftRight, loc='left')
    
    # Set the field of view
    if not AreaOfInterest is None:
        assert len(AreaOfInterest) == 4, 'AreaOfInterest must be given as four parameters (Xmin, Xmax, Ymin, Ymax)'
        TransAx.set_xlim(AreaOfInterest[0], AreaOfInterest[1])
        TransAx.set_ylim(AreaOfInterest[2], AreaOfInterest[3])
    
    # Add some dummy lines
    GroundLine, = TransAx.plot([ShoreY[TransectIx,4], ShoreY[TransectIx,0] + 
                                (BeachTopElev + ClosureDepth) / BeachSlope], 
                               [BackshoreElev, -ClosureDepth], 'k-', zorder=3)
    GroundFill, = TransAx.fill([0.,1.,1.,0.], [0.,1.,0.,0.], 'lightgrey', label='Gravel', zorder=2)
    WaterLine,  = TransAx.plot([0.,0.5],[0.,0.5], '-', color='steelblue', zorder=1)
    WaterFill,  = TransAx.fill([0.,0.,0.5,0.], [0.,0.5,0.5,0.], 'lightskyblue', label='Water', zorder=0)
    
    # Compile output variable
    TransectFig = {'TransFig':TransFig, 'TransAx':TransAx, 
                   'GroundLine':GroundLine, 'GroundFill':GroundFill, 
                   'WaterLine':WaterLine, 'WaterFill':WaterFill,
                   'BeachSlope': BeachSlope, 'BackshoreElev':BackshoreElev, 
                   'ClosureDepth':ClosureDepth, 'BeachTopElev':BeachTopElev,
                   'TransectIx':TransectIx}
    
    # Update lines
    updateTransectFig(TransectFig, ShoreY, ShoreZ, 
                      LagoonWL, OutletWL, SeaLevel)
    
    return TransectFig

def updateTransectFig(TransectFig, ShoreY, ShoreZ, 
                      LagoonWL, OutletWL, SeaLevel, PlotTime=None):
    """ Update an existing TransectFig with new data
    """
    
    TransectIx = TransectFig['TransectIx']
    ClosureDepth = TransectFig['ClosureDepth']
    BackshoreElev = TransectFig['BackshoreElev']
    BeachSlope = TransectFig['BeachSlope']
    BeachTopElev = TransectFig['BeachTopElev']
    
    # Crop to specified transect
    TransY = ShoreY[TransectIx,:].squeeze()
    TransZ = ShoreZ[TransectIx,:].squeeze()
    TransLagWL = LagoonWL[TransectIx]
    TransOutletWL = OutletWL[TransectIx]
    
    # Build GroundLine and WaterFill data from sea to cliff...
    
    # Seabed to toe of beach
    GroundLineX = [1000.0, TransY[0] + (BeachTopElev + ClosureDepth) / BeachSlope]
    GroundLineY = [-ClosureDepth, -ClosureDepth]
    WaterLineX = [1000.0]
    WaterLineY = [SeaLevel]
    
    # Beach face and front edge of barrier
    # TODO: if aligned with seaward end of outlet channel...?
    GroundLineX.extend(TransY[[0,0]])
    GroundLineY.extend([BeachTopElev, TransZ[0]])
    WaterLineX.append(TransY[0])
    WaterLineY.append(SeaLevel)
    
    # Top of barrier
    # TODO: if aligned with lagoon end of outlet channel...?
    if np.isnan(TransY[2]):
        GroundLineX.append(TransY[3])
        GroundLineY.append(TransZ[0])
    else:
        GroundLineX.extend(TransY[[1,1,2,2,3]])
        GroundLineY.extend(TransZ[[0,1,1,2,2]])
        WaterLineX.extend(TransY[[1,2]])
        WaterLineY.extend([TransOutletWL, TransOutletWL])
    
    # Lagoon
    if not TransY[3] == TransY[4]:
        GroundLineX.extend(TransY[[3,4]])
        GroundLineY.extend(TransZ[[3,3]])
        WaterLineX.extend(TransY[[3,4]])
        WaterLineY.extend([TransLagWL, TransLagWL])
    
    # Cliff
    # TODO: if aligned with river?
    GroundLineX.extend([TransY[4], -1000.0])
    GroundLineY.extend([BackshoreElev, BackshoreElev])
    
    # GroundFill data
    GroundFillX = GroundLineX + [GroundLineX[-1], GroundLineX[0], GroundLineX[0]]
    GroundFillY = GroundLineY + [-100.0, -100.0, GroundLineY[0]]
    
    # WaterFill data
    WaterFillX = WaterLineX + [WaterLineX[-1], WaterLineX[0], WaterLineX[0]]
    WaterFillY = WaterLineY + [-100.0, -100.0, WaterLineY[0]]
    
    # Update the lines etc
    TransectFig['GroundLine'].set_data(GroundLineX, GroundLineY)
    TransectFig['GroundFill'].set_xy(np.vstack([GroundFillX, GroundFillY]).T)
    TransectFig['WaterLine'].set_data(WaterLineX, WaterLineY)
    TransectFig['WaterFill'].set_xy(np.vstack([WaterFillX, WaterFillY]).T)
    
    if not PlotTime is None:
        TransectFig['TransAx'].set_title(PlotTime.strftime('%d/%m/%y %H:%M'), loc='right')
    
    # Redraw
    TransectFig['TransFig'].canvas.draw()
    TransectFig['TransFig'].canvas.flush_events()

def bdyCndFig(OutputTs):
    Fig = plt.figure(figsize=(9,3))
    
    # Flow plots
    QAx = Fig.subplots()
    QInLine, = QAx.plot(OutputTs.index.to_numpy(), OutputTs.Qin, 'b-')
    QOutLine, = QAx.plot(OutputTs.index.to_numpy(), OutputTs.Qout, 'r-')
    QAx.autoscale_view(tight = False)
    QAx.set_ylabel('Flow [$\mathrm{m^3/s}$]')
    
    # Sea level plot
    WlAx = QAx.twinx()
    DsWlLine, = WlAx.plot(OutputTs.index.to_numpy(), OutputTs.SeaLevel, 'g-')
    WlAx.set_ylim([-1,3])
    WlAx.set_ylabel('Water level [m]')
    
    # Add legend
    QAx.legend([QInLine, QOutLine, DsWlLine], 
               ['Inflow', 'Outflow', 'Downstream WL'], loc=0)
    
    BdyFig = (Fig, QAx, WlAx, QInLine, QOutLine, DsWlLine)
    return BdyFig

def updateBdyCndFig(BdyFig, OutputTs):
    # update lines
    BdyFig[3].set_data(OutputTs.index, OutputTs.Qin)
    BdyFig[4].set_data(OutputTs.index, OutputTs.Qout)
    BdyFig[5].set_data(OutputTs.index, OutputTs.SeaLevel)
    
    # extend x-axis
    BdyFig[1].set_xlim(OutputTs.index[[0,-1]])
    
    # rescale flow axis
    BdyFig[1].relim()
    BdyFig[1].autoscale_view(tight = False)
    
    # Redraw
    BdyFig[0].canvas.draw()
    BdyFig[0].canvas.flush_events()
    
    