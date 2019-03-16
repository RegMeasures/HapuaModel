# -*- coding: utf-8 -*-

# import standard packages
import matplotlib.pyplot as plt
import numpy as np

# import local packages
import hapuamod.geom as geom

def mapView(ShoreX, ShoreY, Origin, ShoreNormalDir):
    """ Map the current model state in real world coordinates
    """
    
    # Plot the shoreline
    (ShoreXreal, ShoreYreal) = geom.mod2real(ShoreX, ShoreY, Origin, ShoreNormalDir)
    plt.plot(ShoreXreal, ShoreYreal, 'g.')
    
    # plot the origin and baseline
    plt.plot(Origin[0], Origin[1], 'ko')
    (BaseXreal, BaseYreal) = geom.mod2real(ShoreX[[1,-1]], np.array([0,0]), Origin, ShoreNormalDir)
    plt.plot(BaseXreal, BaseYreal, 'k--')
    
    # tidy up the plot
    plt.axis('equal')

def modelView(ShoreX, ShoreY):
    """ Map the current model state in model coordinates
    """
    
    plt.figure(figsize=(12,5))
    plt.plot(ShoreX, ShoreY, 'k-')
    
    plt.axis('equal')
