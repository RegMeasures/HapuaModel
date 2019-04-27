# -*- coding: utf-8 -*-

# import standard packages
import matplotlib.pyplot as plt
import numpy as np

# import local packages
from hapuamod import geom
from hapuamod import riv

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
    """
    
    # Plot shoreline
    plt.plot(ShoreX, ShoreY, 'k-')
    
    # Plot lagoon (inc closing ends)
    plt.plot(ShoreX, LagoonY[:,0], 'b-')
    plt.plot(ShoreX, LagoonY[:,1], 'b-')
    EndTransects = np.where(np.isnan(LagoonY[:,0])==False)[0][[0,-1]]
    plt.plot([ShoreX[EndTransects[0]],ShoreX[EndTransects[0]]], LagoonY[EndTransects[0],:], 'b-')
    plt.plot([ShoreX[EndTransects[1]],ShoreX[EndTransects[1]]], LagoonY[EndTransects[1],:], 'b-')
    
    # Plot Outlet channel
    plt.plot(OutletX, OutletY, 'r-x')
    
    plt.axis('equal')
    
def longSection(ChanDx, ChanElev, ChanDep, ChanVel, Bedload):
    """ View a long section of the river to the lagoon outlet
    
    Parameters:
        ChanDx
        ChanElev
        ChanDep
        ChanVel
    
    Returns:
        WaterLine
        EnergyLine
        VelLine
        FrLine
    """
    g = 9.81
    Dist = np.insert(np.cumsum(ChanDx),0,0)
    ReachDist = (Dist[:-1]+Dist[1:])/2
    WL = ChanElev + ChanDep
    Energy = WL + ChanVel**2 / (2*g)
    Fr = ChanVel/np.sqrt(g*ChanDep)
    
    # Create new figure with sub plots
    RivFig = plt.figure()
    ElevAx = RivFig.add_subplot(3,1,1)
    VelAx = RivFig.add_subplot(3,1,2, sharex=ElevAx)
    FrAx = VelAx.twinx()
    QsAx = RivFig.add_subplot(3,1,3, sharex=ElevAx)
    
    # Plot the river bed level, water surface and energy line
    ElevAx.plot(Dist, ChanElev, 'k-')
    WaterLine = ElevAx.plot(Dist, WL, 'b-')
    EnergyLine = ElevAx.plot(Dist, Energy, 'b:')
    ElevAx.set_ylabel('Elevation [m]')
    
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
    QsLine = QsAx.plot(ReachDist, Bedload*3600, 'k-')
    QsAx.set_ylabel(r'Bedload [$\mathrm{m^3/hr}$]')
    QsAx.set_xlabel('Distance downstream [m]')
    QsAx.set_ylim([0,10])
    
    LongSecFig = (RivFig, WaterLine, EnergyLine, VelLine, FrLine, QsLine)
    
    return(LongSecFig)

def updateLongSection(LongSecFig, ChanDx, ChanElev, ChanDep, ChanVel, Bedload):
    
    # Calculate required variables to plot
    g = 9.81
    Dist = np.insert(np.cumsum(ChanDx),0,0)
    ReachDist = (Dist[:-1]+Dist[1:])/2
    WL = ChanElev + ChanDep
    Energy = WL + ChanVel**2 / (2*g)
    Fr = ChanVel/np.sqrt(g*ChanDep)
    
    # Update the lines
    LongSecFig[1][0].set_data(Dist, WL)
    LongSecFig[2][0].set_data(Dist, Energy)
    LongSecFig[3][0].set_data(Dist, ChanVel)
    LongSecFig[4][0].set_data(Dist, Fr)
    LongSecFig[5][0].set_data(ReachDist, Bedload*3600)
    
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
    
    