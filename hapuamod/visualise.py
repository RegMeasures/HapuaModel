# -*- coding: utf-8 -*-

# import standard packages
import matplotlib.pyplot as plt
import numpy as np

# import local packages
from hapuamod import geom

def mapView(ShoreX, ShoreY, LagoonY, Origin, ShoreNormalDir):
    """ Map the current model state in real world coordinates
    """
    
    # Plot the shoreline
    (ShoreXreal, ShoreYreal) = geom.mod2real(ShoreX, ShoreY, Origin, ShoreNormalDir)
    plt.plot(ShoreXreal, ShoreYreal, 'g-')
    
    # Plot the lagoon
    LagoonXmodel = np.transpose(np.tile(ShoreX, [2,1]))
    (LagoonXreal, LagoonYreal) = geom.mod2real(LagoonXmodel, LagoonY, Origin, ShoreNormalDir)
    plt.plot(LagoonXreal[:,0], LagoonYreal[:,0], 'b-')
    plt.plot(LagoonXreal[:,1], LagoonYreal[:,1], 'b-')
    EndTransects = np.where(np.isnan(LagoonY[:,0])==False)[0][[0,-1]]
    plt.plot(LagoonXreal[EndTransects[0],:], LagoonYreal[EndTransects[0],:], 'b-')
    plt.plot(LagoonXreal[EndTransects[1],:], LagoonYreal[EndTransects[1],:], 'b-')
    
    # plot the origin and baseline
    plt.plot(Origin[0], Origin[1], 'ko')
    (BaseXreal, BaseYreal) = geom.mod2real(ShoreX[[1,-1]], np.array([0,0]), Origin, ShoreNormalDir)
    plt.plot(BaseXreal, BaseYreal, 'k--')
    
    # tidy up the plot
    plt.axis('equal')

def modelView(ShoreX, ShoreY, LagoonY, OutletX, OutletY):
    """ Map the current model state in model coordinates
    
    Parameters:
        ShoreX
        ShoreY
        LagoonY
        OutletX
        OutletY
        
    Returns:
        ModelFig = (PlanFig, PlanAx, ShoreLine, LagoonLine, OutletLine)
    """
    
    # Create a new figure
    PlanFig, PlanAx = plt.subplots()
    PlanAx.axis('equal')
    PlanAx.set_xlabel('Alongshore distance [m]')
    PlanAx.set_ylabel('Crossshore distance [m]')
    
    # Plot shoreline
    ShoreLine = PlanAx.plot(ShoreX, ShoreY, 'k-')
    
    # Plot lagoon (inc closing ends)
    LagoonMask = np.isnan(LagoonY[:,1])==False
    PlotLagoonX = np.concatenate(([ShoreX[np.where(LagoonMask)[0][0]]],
                                  ShoreX[LagoonMask], 
                                  np.flipud(ShoreX[LagoonMask])))
    PlotLagoonY = np.concatenate(([LagoonY[np.where(LagoonMask)[0][0],0]],
                                  LagoonY[:,1][LagoonMask], 
                                  np.flipud(LagoonY[:,0][LagoonMask])))
    LagoonLine = plt.plot(PlotLagoonX, PlotLagoonY, 'b-')
    
    # Plot Outlet channel
    OutletLine = plt.plot(OutletX, OutletY, 'r-x')
    
    ModelFig = (PlanFig, ShoreLine, LagoonLine, OutletLine)
    
    return ModelFig

def updateModelView(ModelFig, ShoreX, ShoreY, LagoonY, OutletX, OutletY):
    
    # Calculate required variables to plot
    LagoonMask = np.isnan(LagoonY[:,1])==False
    PlotLagoonX = np.concatenate(([ShoreX[np.where(LagoonMask)[0][0]]],
                                  ShoreX[LagoonMask], 
                                  np.flipud(ShoreX[LagoonMask])))
    PlotLagoonY = np.concatenate(([LagoonY[np.where(LagoonMask)[0][0],0]],
                                  LagoonY[:,1][LagoonMask], 
                                  np.flipud(LagoonY[:,0][LagoonMask])))
    
    # Update the lines
    ModelFig[1][0].set_data(ShoreX, ShoreY)
    ModelFig[2][0].set_data(PlotLagoonX, PlotLagoonY)
    ModelFig[3][0].set_data(OutletX, OutletY)
    
    
def longSection(ChanDx, ChanElev, ChanWidth, ChanDep, ChanVel, Bedload):
    """ Create a long section of the river to the lagoon outlet
    
    LongSecFig = longSection(ChanDx, ChanElev, ChanWidth, ChanDep, ChanVel, 
                             Bedload)
    
    Parameters:
        ChanDx
        ChanElev
        ChanWidth
        ChanDep
        ChanVel
        Bedload
    
    Returns:
        LongSecFig = (RivFig, BedLine, WaterLine, EnergyLine, WidthLine, 
                      VelLine, FrLine, QsLine)
    """
    g = 9.81
    Dist = np.insert(np.cumsum(ChanDx),0,0)
    WL = ChanElev + ChanDep
    Energy = WL + ChanVel**2 / (2*g)
    Fr = ChanVel/np.sqrt(g*ChanDep)
    
    # Create new figure with sub plots
    RivFig = plt.figure(figsize=(9,9))
    ElevAx = RivFig.add_subplot(4,1,1)
    WidthAx = RivFig.add_subplot(4,1,2)
    VelAx = RivFig.add_subplot(4,1,3, sharex=ElevAx)
    FrAx = VelAx.twinx()
    QsAx = RivFig.add_subplot(4,1,4, sharex=ElevAx)
    
    # Plot the river bed level, water surface and energy line
    BedLine = ElevAx.plot(Dist, ChanElev, 'k-')
    WaterLine = ElevAx.plot(Dist, WL, 'b-')
    EnergyLine = ElevAx.plot(Dist, Energy, 'b:')
    ElevAx.set_ylabel('Elevation [m]')
    
    # Plot the river width
    WidthLine = WidthAx.plot(Dist, ChanWidth, 'k-')
    WidthAx.set_ylabel('Width [m]')
    WidthAx.set_ylim([0,np.amax(ChanWidth)+10])
    
    # Plot velocity and Froude number
    VelLine = VelAx.plot(Dist, ChanVel, 'r-')
    VelAx.set_ylabel('Velocity [m/s]', color='red')
    VelAx.tick_params(axis='y', colors='red')
    VelAx.set_ylim([0,2])
    
    FrLine = FrAx.plot(Dist, Fr, 'g-')
    FrAx.set_ylabel('Froude No.', color='green')
    FrAx.tick_params(axis='y', colors='green')
    FrAx.set_ylim([0,1.3])
    
    # Plot bedload
    QsLine = QsAx.plot(Dist, Bedload*3600, 'k-')
    QsAx.set_ylabel(r'Bedload [$\mathrm{m^3/hr}$]')
    QsAx.set_xlabel('Distance downstream [m]')
    QsAx.set_ylim([0,10])
    
    LongSecFig = (RivFig, BedLine, WaterLine, EnergyLine, WidthLine, 
                  VelLine, FrLine, QsLine)
    
    return(LongSecFig)

def updateLongSection(LongSecFig, ChanDx, ChanElev, ChanWidth, ChanDep, 
                      ChanVel, Bedload):
    
    # Calculate required variables to plot
    g = 9.81
    Dist = np.insert(np.cumsum(ChanDx),0,0)
    WL = ChanElev + ChanDep
    Energy = WL + ChanVel**2 / (2*g)
    Fr = ChanVel/np.sqrt(g*ChanDep)
    
    # Update the lines
    LongSecFig[1][0].set_data(Dist, ChanElev)
    LongSecFig[2][0].set_data(Dist, WL)
    LongSecFig[3][0].set_data(Dist, Energy)
    LongSecFig[4][0].set_data(Dist, ChanWidth)
    LongSecFig[5][0].set_data(Dist, ChanVel)
    LongSecFig[6][0].set_data(Dist, Fr)
    LongSecFig[7][0].set_data(Dist, Bedload*3600)
    
    # Redraw
    LongSecFig[0].canvas.draw()
    LongSecFig[0].canvas.flush_events()

def BdyCndFig(OutputTs):
    Fig = plt.figure(figsize=(9,3))
    
    # Flow plots
    QAx = Fig.subplots()
    QInLine = QAx.plot(OutputTs.index, OutputTs.Qin, 'b-')
    QOutLine = QAx.plot(OutputTs.index, OutputTs.Qout, 'r-')
    QAx.set_ylim([0,200])
    
    # Sea level plot
    WlAx = QAx.twinx()
    DsWlLine = WlAx.plot(OutputTs.index, OutputTs.SeaLevel, 'g-')
    WlAx.set_ylim([-1,3])
    
    BdyFig = (Fig, QAx, WlAx, QInLine, QOutLine, DsWlLine)
    return BdyFig

def updateBdyCndFig(BdyFig, OutputTs):
    # update lines
    BdyFig[3][0].set_data(OutputTs.index, OutputTs.Qin)
    BdyFig[4][0].set_data(OutputTs.index, OutputTs.Qout)
    BdyFig[5][0].set_data(OutputTs.index, OutputTs.SeaLevel)
    
    # extend x-axis
    BdyFig[1].set_xlim(OutputTs.index[[0,-1]])
    
    # Redraw
    BdyFig[0].canvas.draw()
    BdyFig[0].canvas.flush_events()
    
    