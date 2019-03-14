# -*- coding: utf-8 -*-

# import standard packages
import matplotlib.pyplot as plt
import numpy as np

# import local packages
import hapuamod.geom as geom

def mapView(ShoreX, ShoreY, Origin, ShoreNormalDir):
    # Produce a map showing the spatial inputs
    
    # Plot the shoreline
    (ShoreXreal, ShoreYreal) = geom.mod2real(ShoreX, ShoreY, Origin, ShoreNormalDir)
    plt.plot(ShoreXreal, ShoreYreal, 'g.')
    
    # plot the origin and baseline
    plt.plot(Origin[0], Origin[1], 'ko')
    (BaseXreal, BaseYreal) = geom.mod2real(ShoreX[[1,-1]], np.array([0,0]), Origin, ShoreNormalDir)
    plt.plot(BaseXreal, BaseYreal, 'k--')
    
    # tidy up the plot
    plt.axis('equal')
