# -*- coding: utf-8 -*-

# import standard packages
import matplotlib.pyplot as plt
import numpy as np

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

def modelView(ShoreX, ShoreY, OutletEndX, OutletEndWidth, OutletChanIx, 
              ShoreZ=None, WavePower=None, EDir_h=0, LST=None, CST=None, 
              WaveScaling=0.01, CstScaling=0.00005, LstScaling=0.0001,
              QuiverWidth=0.002):
    """ Map the current model state in model coordinates
    
        Parameters:
            ShoreX
            ShoreY
            OutletEndX
            OutletChanIx
            WavePower (optional)
            LST (Optional)
            
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
    
    PlanAx.axis('equal')
    
    # Create dummy lines
    ShoreLine,  = PlanAx.plot(ShoreX, ShoreY[:,0], 'g-', label='Shore')
    OutletLine, = PlanAx.plot(ShoreX, ShoreY[:,1], 'r-', label='Outlet', zorder = 10)
    ChannelLine, = PlanAx.plot(ShoreX, ShoreY[:,1], '-x', label='Channel', color='grey')
    LagoonLine, = PlanAx.plot(ShoreX, ShoreY[:,3], 'c-', label='Lagoon')
    CliffLine,  = PlanAx.plot(ShoreX, ShoreY[:,4], 'k-', label='Cliff')
    RiverLine,  = PlanAx.plot([0,0], [-100,-300], 'b-', label='River')
    if not ShoreZ is None:
        CrestLine, = VertAx.plot(ShoreX, ShoreZ[:,0], 'k-', label='Barrier crest')
    
    if not WavePower is None:
        WaveArrow = PlanAx.arrow(0,200,100,100)
    
    if not LST is None:
        LstQuiver = PlanAx.quiver((ShoreX[:-1]+ShoreX[1:])/2, 
                                  (ShoreY[:-1,0]+ShoreY[1:,0])/2, 
                                  LST, np.zeros(LST.size),
                                  scale=LstScaling, width=QuiverWidth,
                                  scale_units='x', units='width')
    
    if not CST is None:
        CstQuiver = PlanAx.quiver(ShoreX, ShoreY[:,0], 
                                  np.zeros(CST.size), -CST, 
                                  scale=CstScaling, width=QuiverWidth,
                                  scale_units='x', units='width')
    
    # Add some labels
    PlanAx.legend()
    PlanAx.set_xlabel('Model X-coordinate (m)')
    PlanAx.set_ylabel('Model Y-coordinate (m)')
    if not ShoreZ is None:
        VertAx.set_ylabel('Elevation (m)')
    
    # Compile output variable
    ModelFig = {'PlanFig':PlanFig, 'PlanAx':PlanAx, 'ShoreLine':ShoreLine, 
                'OutletLine':OutletLine, 'ChannelLine':ChannelLine,
                'LagoonLine':LagoonLine, 'CliffLine':CliffLine, 
                'RiverLine':RiverLine}
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
                    EDir_h=EDir_h, LST=LST, CST=CST)
    
    return ModelFig

def updateModelView(ModelFig, ShoreX, ShoreY, OutletEndX, OutletEndWidth, 
                    OutletChanIx, Closed=False, 
                    ShoreZ=None, WavePower=None, EDir_h=0, LST=None, CST=None):
    
    # Channel (online and offline) plotting position
    ChannelX = np.tile(ShoreX, 2)
    ChannelY = np.asarray([ShoreY[:,1], ShoreY[:,2]])
    
    # Calculate online outlet plotting position
    if Closed:
        OutletX = np.nan
        OutletY = np.nan
    else:
        
        if OutletEndX[0] < OutletEndX[1]:
            # Outlet angkles L to R
            L_Ok = ShoreX[OutletChanIx] < (OutletEndX[1] - OutletEndWidth[1]/2)
            R_Ok = ShoreX[OutletChanIx] > (OutletEndX[0] + OutletEndWidth[0]/2)
            OutletY = np.concatenate([[np.interp(OutletEndX[0] - OutletEndWidth[0]/2, ShoreX, ShoreY[:,3])],
                                      ShoreY[OutletChanIx[L_Ok],1], 
                                      [np.interp(OutletEndX[1] - OutletEndWidth[1]/2, ShoreX, ShoreY[:,0]),
                                       np.nan,
                                       np.interp(OutletEndX[0] + OutletEndWidth[0]/2, ShoreX, ShoreY[:,3])],
                                      ShoreY[OutletChanIx[R_Ok],2], 
                                      [np.interp(OutletEndX[1] + OutletEndWidth[1]/2, ShoreX, ShoreY[:,0])]])
        else:
            # Outlet angkles R to L
            L_Ok = ShoreX[OutletChanIx] < (OutletEndX[0] - OutletEndWidth[0]/2)
            R_Ok = ShoreX[OutletChanIx] > (OutletEndX[1] + OutletEndWidth[1]/2)
            OutletY = np.concatenate([[np.interp(OutletEndX[0] - OutletEndWidth[0]/2, ShoreX, ShoreY[:,3])],
                                      ShoreY[OutletChanIx[L_Ok],2], 
                                      [np.interp(OutletEndX[1] - OutletEndWidth[1]/2, ShoreX, ShoreY[:,0]),
                                       np.nan,
                                       np.interp(OutletEndX[0] + OutletEndWidth[0]/2, ShoreX, ShoreY[:,3])],
                                      ShoreY[OutletChanIx[R_Ok],1], 
                                      [np.interp(OutletEndX[1] + OutletEndWidth[1]/2, ShoreX, ShoreY[:,0])]])
        OutletX = np.concatenate([[OutletEndX[0] - OutletEndWidth[0]/2],
                                  ShoreX[OutletChanIx[L_Ok]],
                                  [OutletEndX[1] - OutletEndWidth[1]/2,
                                   np.nan,
                                   OutletEndX[0] + OutletEndWidth[0]/2],
                                  ShoreX[OutletChanIx[R_Ok]],
                                  [OutletEndX[1] + OutletEndWidth[1]/2]])
        
    # Calculate river plotting position
    RiverY = ShoreY[ShoreX==0,4]
    
    if not WavePower is None:
        if (-np.pi/2) < EDir_h < (np.pi/2):
            ArrLength = WavePower
        else:
            ArrLength = 0.01
        ArrWidth = np.maximum((ArrLength * ModelFig['WaveScaling'])/10, 0.001)
        WaveX = -np.sin(EDir_h) * ArrLength * ModelFig['WaveScaling']
        WaveY = -np.cos(EDir_h) * ArrLength * ModelFig['WaveScaling']
    
    # Update the lines
    ModelFig['ShoreLine'].set_data(ShoreX, ShoreY[:,0])
    ModelFig['ChannelLine'].set_data(ChannelX, ChannelY)
    ModelFig['OutletLine'].set_data(OutletX, OutletY)
    ModelFig['LagoonLine'].set_data(ShoreX, ShoreY[:,3])
    ModelFig['CliffLine'].set_data(ShoreX, ShoreY[:,4])
    ModelFig['RiverLine'].set_data([0,0], [RiverY, RiverY-300])
    
    if not ShoreZ is None:
        ModelFig['CrestLine'].set_data(ShoreX, ShoreZ[:,0])
    
    if not WavePower is None:
        ModelFig['WaveArrow'].remove()
        ModelFig['WaveArrow'] = ModelFig['PlanAx'].arrow(0, 200, WaveX, WaveY,
                                                         width=ArrWidth)
    if not LST is None:
        LstScale = ModelFig['LstQuiver'].scale
        LstWidth = ModelFig['LstQuiver'].width
        ModelFig['LstQuiver'].remove()
        ModelFig['LstQuiver'] = ModelFig['PlanAx'].quiver((ShoreX[:-1]+ShoreX[1:])/2, 
                                                          (ShoreY[:-1,0]+ShoreY[1:,0])/2, 
                                                          LST, np.zeros(LST.size),
                                                          scale=LstScale, width=LstWidth,
                                                          scale_units='x', units='width')
    
    if not CST is None:
        CstScale = ModelFig['CstQuiver'].scale
        CstWidth = ModelFig['CstQuiver'].width
        ModelFig['CstQuiver'].remove()
        ModelFig['CstQuiver'] = ModelFig['PlanAx'].quiver(ShoreX, ShoreY[:,0], 
                                                          np.zeros(ShoreX.size), -CST, 
                                                          scale=CstScale, width=CstWidth,
                                                          scale_units='x', units='width')
    
    # Redraw
    ModelFig['PlanFig'].canvas.draw()
    ModelFig['PlanFig'].canvas.flush_events()
    
def longSection(ChanDx, ChanElev, ChanWidth, ChanDep, ChanVel, Bedload=None):
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
    
    # Plot the river bed level, water surface and energy line
    BedLine, = ElevAx.plot(Dist, ChanElev, 'k-')
    WaterLine, = ElevAx.plot(Dist, WL, 'b-')
    EnergyLine, = ElevAx.plot(D2, Energy, 'b:')
    ElevAx.set_ylabel('Elevation [m]')
    ElevAx.grid(axis='x', which='both', linestyle=':')
    
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
    
    # Compile outputs
    if Bedload is None:
        LongSecFig = (RivFig, ElevAx, WidthAx, FlowAx, VelAx, FrAx, 
                      BedLine, WaterLine, EnergyLine, WidthLine, 
                      FlowLine, VelLine, FrLine)
    else:
        LongSecFig = (RivFig, ElevAx, WidthAx, FlowAx, VelAx, FrAx, 
                      BedLine, WaterLine, EnergyLine, WidthLine, 
                      FlowLine, VelLine, FrLine, QsLine)
    
    return(LongSecFig)

def updateLongSection(LongSecFig, ChanDx, ChanElev, ChanWidth, ChanDep, 
                      ChanVel, Bedload=None):
    
    # Calculate required variables to plot
    g = 9.81
    Dist = np.insert(np.cumsum(ChanDx),0,0)
    D2 = (Dist[:-1]+Dist[1:])/2
    WL = ChanElev + ChanDep
    WL2 = (WL[:-1]+WL[1:])/2
    Vel2 = (ChanVel[:-1]+ChanVel[1:])/2
    Energy = WL2 + (Vel2**2) / (2*g)
    Fr = abs(ChanVel)/np.sqrt(g*ChanDep)
    Q = ChanVel * ChanDep * ChanWidth
    
    # Update the lines
    LongSecFig[6].set_data(Dist, ChanElev)
    LongSecFig[7].set_data(Dist, WL)
    LongSecFig[8].set_data(D2, Energy)
    LongSecFig[9].set_data(Dist, ChanWidth)
    LongSecFig[10].set_data(Dist, Q)
    LongSecFig[11].set_data(Dist, ChanVel)
    LongSecFig[12].set_data(Dist, Fr)
    if not Bedload is None:
        LongSecFig[13].set_data(D2, Bedload*3600)
    
    # Update flow axis scaling
    LongSecFig[3].relim()
    LongSecFig[3].autoscale_view(tight = False)
    
    # Redraw
    LongSecFig[0].canvas.draw()
    LongSecFig[0].canvas.flush_events()

def BdyCndFig(OutputTs):
    Fig = plt.figure(figsize=(9,3))
    
    # Flow plots
    QAx = Fig.subplots()
    QInLine, = QAx.plot(OutputTs.index, OutputTs.Qin, 'b-')
    QOutLine, = QAx.plot(OutputTs.index, OutputTs.Qout, 'r-')
    QAx.autoscale_view(tight = False)
    QAx.set_ylabel('Flow [$\mathrm{m^3/s}$]')
    
    # Sea level plot
    WlAx = QAx.twinx()
    DsWlLine, = WlAx.plot(OutputTs.index, OutputTs.SeaLevel, 'g-')
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
    
    