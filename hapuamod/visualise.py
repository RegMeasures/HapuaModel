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
    
def longSection(ChanDx, ChanElev, ChanDep, ChanVel):
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
    WL = ChanElev + ChanDep
    Energy = WL + ChanVel**2 / (2*g)
    Fr = ChanVel/np.sqrt(g*ChanDep)
    
    # Create new figure with sub plots
    RivFig = plt.figure()
    RivTopAx = RivFig.add_subplot(2,1,1)
    RivBotAx = RivFig.add_subplot(2,1,2, sharex=RivTopAx)
    
    # Plot the river bed level
    RivTopAx.plot(Dist, ChanElev, 'k-')
    
    # Plot water surface and energy line
    WaterLine = RivTopAx.plot(Dist, WL, 'b-')
    EnergyLine = RivTopAx.plot(Dist, Energy, 'b:')
    
    # Plot velocity and Froude number
    VelLine = RivBotAx.plot(Dist, ChanVel, 'r-')
    FrLine = RivBotAx.plot(Dist, Fr, 'g-')
    
    LongSecFig = (RivFig, WaterLine, EnergyLine, VelLine, FrLine)
    
    return(LongSecFig)

def updateLongSection(LongSecFig, ChanDx, ChanElev, ChanDep, ChanVel):
    
    # Calculate required variables to plot
    g = 9.81
    Dist = np.insert(np.cumsum(ChanDx),0,0)
    WL = ChanElev + ChanDep
    Energy = WL + ChanVel**2 / (2*g)
    Fr = ChanVel/np.sqrt(g*ChanDep)
    
    # Update the lines
    LongSecFig[1][0].set_data(Dist, WL)
    LongSecFig[2][0].set_data(Dist, Energy)
    LongSecFig[3][0].set_data(Dist, ChanVel)
    LongSecFig[4][0].set_data(Dist, Fr)
    
    # Redraw
    LongSecFig[0].canvas.draw()
    LongSecFig[0].canvas.flush_events()
