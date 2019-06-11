# -*- coding: utf-8 -*-

# import standard packages
import matplotlib.pyplot as plt
import numpy as np

# import local packages
from hapuamod import geom

def mapView(ShoreX, ShoreY, Origin, ShoreNormalDir):
    """ Map the current model state in real world coordinates
    """
    
    # convert the coordinates to real world
    (ShoreXreal, ShoreYreal) = geom.mod2real(np.transpose(np.tile(ShoreX, [5,1])), 
                                               ShoreY, Origin, ShoreNormalDir)
    plt.plot(ShoreXreal[:,0], ShoreYreal[:,0], 'g-')    
    plt.plot(ShoreXreal[:,1], ShoreYreal[:,1], 'r-')
    plt.plot(ShoreXreal[:,2], ShoreYreal[:,2], 'r-')
    plt.plot(ShoreXreal[:,3], ShoreYreal[:,3], 'b-')
    plt.plot(ShoreXreal[:,4], ShoreYreal[:,4], 'k-')
    
    EndTransects = np.where(np.isnan(ShoreY[:,3])==False)[0][[0,-1]]
    plt.plot(ShoreXreal[EndTransects[0],[3,4]], ShoreYreal[EndTransects[0],[3,4]], 'b-')
    plt.plot(ShoreXreal[EndTransects[1],[3,4]], ShoreYreal[EndTransects[1],[3,4]], 'b-')
    
    # plot the origin and baseline
    plt.plot(Origin[0], Origin[1], 'ko')
    (BaseXreal, BaseYreal) = geom.mod2real(ShoreX[[1,-1]], np.array([0,0]), Origin, ShoreNormalDir)
    plt.plot(BaseXreal, BaseYreal, 'k--')
    
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
    PlanFig, PlanAx = plt.subplots()
    PlanAx.axis('equal')
    PlanAx.set_xlabel('Alongshore distance [m]')
    PlanAx.set_ylabel('Crossshore distance [m]')
    
    # Create dummy lines
    ShoreLine  = PlanAx.plot(ShoreX, ShoreY[:,0], 'k-')
    OutletLine = PlanAx.plot(ShoreX, ShoreY[:,1], 'r-')
    LagoonLine = PlanAx.plot(ShoreX, ShoreY[:,3], 'b-')
    CliffLine  = PlanAx.plot(ShoreX, ShoreY[:,4], 'g-')
    
    ModelFig = (PlanFig, PlanAx, ShoreLine, OutletLine, LagoonLine, CliffLine)
    
    # Replace with correct lines
    updateModelView(ModelFig, ShoreX, ShoreY, OutletEndX, OutletChanIx)
    
    return ModelFig

def updateModelView(ModelFig, ShoreX, ShoreY, OutletEndX, OutletChanIx):
    
    # Calculate lagoon plotting position
    LagoonMask = np.isnan(ShoreY[:,3])==False
    LagoonX = np.concatenate(([ShoreX[np.where(LagoonMask)[0][0]]],
                              ShoreX[LagoonMask], 
                              [ShoreX[np.where(LagoonMask)[0][-1]]]))
    LagoonY = np.concatenate(([ShoreY[np.where(LagoonMask)[0][0],4]],
                              ShoreY[:,3][LagoonMask], 
                              [ShoreY[np.where(LagoonMask)[0][-1],4]]))
    
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
    
    # Update the lines
    ModelFig[2][0].set_data(ShoreX, ShoreY[:,0])
    ModelFig[3][0].set_data(OutletX, OutletY)
    ModelFig[4][0].set_data(LagoonX, LagoonY)
    ModelFig[5][0].set_data(ShoreX, ShoreY[:,4])
    
    
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
    
    