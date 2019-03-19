# -*- coding: utf-8 -*-

# import standard packages
import matplotlib.pyplot as plt
import numpy as np

# import local packages
import hapuamod.geom as geom

def mapView(ShoreX, ShoreY, LagoonY, Origin, ShoreNormalDir):
    """ Map the current model state in real world coordinates
    """
    # Create new figure
    plt.figure()
    
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
    # Create new figure
    plt.figure(figsize=(12,5))
    
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
    
def riverLongSection(RiverElev, Dx):
    """ View a long section of the river to the lagoon outlet
    """
    # Create new figure
    plt.figure()
    
    # Calculatethe distance along the channel
    RivDist = np.arange(0, RiverElev.size * Dx, Dx)
    
    # Plot the river bed level
    plt.plot(RivDist, RiverElev)