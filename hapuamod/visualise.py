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

def modelView(ShoreX, ShoreY, OutletEndX, OutletChanIx):
    """ Map the current model state in model coordinates
    
    Parameters:
        ShoreX
        ShoreY
        
    Returns:
        ModelFig = (PlanFig, PlanAx, ShoreLine, OutletLine, LagoonLine, CliffLine)
    """
    
    # Create a new figure
    PlanFig, PlanAx = plt.subplots(figsize=[10,5])
    PlanAx.axis('equal')
    PlanAx.set_xlabel('Alongshore distance [m]')
    PlanAx.set_ylabel('Crossshore distance [m]')
    
    # Create dummy lines
    ShoreLine,  = PlanAx.plot(ShoreX, ShoreY[:,0], 'g-', label='Shore')
    OutletLine, = PlanAx.plot(ShoreX, ShoreY[:,1], 'r-', label='Outlet')
    LagoonLine, = PlanAx.plot(ShoreX, ShoreY[:,3], 'c-', label='Lagoon')
    CliffLine,  = PlanAx.plot(ShoreX, ShoreY[:,4], 'k-', label='Cliff')
    RiverLine,  = PlanAx.plot([0,0], [-100,-300], 'b-', label='River')
    
    # Add some labels
    plt.legend()
    plt.xlabel('Model X-coordinate (m)')
    plt.ylabel('Model Y-coordinate (m)')
    
    ModelFig = (PlanFig, PlanAx, ShoreLine, OutletLine, LagoonLine, CliffLine, 
                RiverLine)
    
    # Replace with correct lines
    updateModelView(ModelFig, ShoreX, ShoreY, OutletEndX, OutletChanIx)
    
    return ModelFig

def updateModelView(ModelFig, ShoreX, ShoreY, OutletEndX, OutletChanIx):
    
    # Calculate outlet plotting position
    OutletX = np.tile(ShoreX,[2,1]).flatten()
    OutletY = ShoreY[:,[1,2]].transpose().flatten()
    # Join the end of the (online) outlet channel to the shore/lagoon line
    OutletX = np.append(OutletX,
                        [np.nan, 
                         ShoreX[OutletChanIx[0]], OutletEndX[0], ShoreX[OutletChanIx[0]],
                         np.nan,
                         ShoreX[OutletChanIx[-1]], OutletEndX[1], ShoreX[OutletChanIx[-1]]])
    OutletY = np.append(OutletY,
                        [np.nan, 
                         ShoreY[OutletChanIx[0],1], np.interp(OutletEndX[0],ShoreX,ShoreY[:,3]), ShoreY[OutletChanIx[0],2],
                         np.nan,
                         ShoreY[OutletChanIx[-1],1], np.interp(OutletEndX[1],ShoreX,ShoreY[:,0]), ShoreY[OutletChanIx[-1],2]])
    
    # Calculate river plotting position
    RiverY = ShoreY[ShoreX==0,4]
    
    # Update the lines
    ModelFig[2].set_data(ShoreX, ShoreY[:,0])
    ModelFig[3].set_data(OutletX, OutletY)
    ModelFig[4].set_data(ShoreX, ShoreY[:,3])
    ModelFig[5].set_data(ShoreX, ShoreY[:,4])
    ModelFig[6].set_data([0,0], [RiverY, RiverY-300])
    
    
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
    WL = ChanElev + ChanDep
    Energy = WL + ChanVel**2 / (2*g)
    Fr = abs(ChanVel)/np.sqrt(g*ChanDep)
    Q = ChanVel * ChanDep * ChanWidth
    
    # Create new figure with sub plots
    RivFig = plt.figure(figsize=(9,9))
    if Bedload is None:
        NoOfPlots = 3
    else:
        NoOfPlots = 4
    ElevAx = RivFig.add_subplot(NoOfPlots,1,1)
    WidthAx = RivFig.add_subplot(NoOfPlots,1,2)
    FlowAx = WidthAx.twinx()
    VelAx = RivFig.add_subplot(NoOfPlots,1,3, sharex=ElevAx)
    FrAx = VelAx.twinx()
    if not Bedload is None:
        QsAx = RivFig.add_subplot(4,1,4, sharex=ElevAx)
    
    # Plot the river bed level, water surface and energy line
    BedLine, = ElevAx.plot(Dist, ChanElev, 'k-')
    WaterLine, = ElevAx.plot(Dist, WL, 'b-')
    EnergyLine, = ElevAx.plot(Dist, Energy, 'b:')
    ElevAx.set_ylabel('Elevation [m]')
    
    # Plot the river width and flow
    WidthLine, = WidthAx.plot(Dist, ChanWidth, 'k-')
    WidthAx.set_ylabel('Width [m]')
    WidthAx.set_ylim([0,np.amax(ChanWidth)+10])
    
    FlowLine, = FlowAx.plot(Dist, Q, 'c-')
    FlowAx.set_ylabel('Flow [$\mathrm{m^3/s}$]')
    FlowAx.autoscale_view(tight = False)
    
    # Plot velocity and Froude number
    VelLine, = VelAx.plot(Dist, ChanVel, 'r-')
    VelAx.set_ylabel('Velocity [m/s]', color='red')
    VelAx.tick_params(axis='y', colors='red')
    VelAx.set_ylim([0,2])
    
    FrLine, = FrAx.plot(Dist, Fr, 'g-')
    FrAx.set_ylabel('Froude No.', color='green')
    FrAx.tick_params(axis='y', colors='green')
    FrAx.set_ylim([0,1.3])
    
    # Plot bedload
    if not Bedload is None:
        QsLine, = QsAx.plot(Dist, Bedload*3600, 'k-')
        QsAx.set_ylabel(r'Bedload [$\mathrm{m^3/hr}$]')
        QsAx.set_xlabel('Distance downstream [m]')
        QsAx.set_ylim([0,10])
    
    # Compile outputs
    if Bedload is None:
        LongSecFig = (RivFig, BedLine, WaterLine, EnergyLine, WidthLine, 
                      FlowAx, FlowLine, VelLine, FrLine)
    else:
        LongSecFig = (RivFig, BedLine, WaterLine, EnergyLine, WidthLine, 
                      FlowAx, FlowLine, VelLine, FrLine, QsLine)
    
    return(LongSecFig)

def updateLongSection(LongSecFig, ChanDx, ChanElev, ChanWidth, ChanDep, 
                      ChanVel, Bedload=None):
    
    # Calculate required variables to plot
    g = 9.81
    Dist = np.insert(np.cumsum(ChanDx),0,0)
    WL = ChanElev + ChanDep
    Energy = WL + ChanVel**2 / (2*g)
    Fr = abs(ChanVel)/np.sqrt(g*ChanDep)
    Q = ChanVel * ChanDep * ChanWidth
    
    # Update the lines
    LongSecFig[1].set_data(Dist, ChanElev)
    LongSecFig[2].set_data(Dist, WL)
    LongSecFig[3].set_data(Dist, Energy)
    LongSecFig[4].set_data(Dist, ChanWidth)
    LongSecFig[6].set_data(Dist, Q)
    LongSecFig[7].set_data(Dist, ChanVel)
    LongSecFig[8].set_data(Dist, Fr)
    if not Bedload is None:
        LongSecFig[9].set_data(Dist, Bedload*3600)
    
    # Update flow axis scaling
    LongSecFig[5].relim()
    LongSecFig[5].autoscale_view(tight = False)
    
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
    
    