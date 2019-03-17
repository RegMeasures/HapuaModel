""" Functions for converting between model and real world coordinate systems"""

import numpy as np

def mod2real(Xmod, Ymod, Origin, ShoreNormalDir):
    """ Converts from model coordinates to real world coordinates
    
    (Xreal, Yreal) = mod2real(Xmod, Ymod, Origin, ShoreNormalDir)
    """
    Xreal = (Origin[0] + 
             Xmod * np.sin(ShoreNormalDir+np.pi/2) - 
             Ymod * np.cos(ShoreNormalDir+np.pi/2))
    Yreal = (Origin[1] + 
             Xmod * np.cos(ShoreNormalDir+np.pi/2) + 
             Ymod * np.sin(ShoreNormalDir+np.pi/2))
    return (Xreal, Yreal)

def real2mod(Xreal, Yreal, Origin, ShoreNormalDir):
    """ Converts from real word to model coordinates
    
    (Xmod, Ymod) = real2mod(Xreal, Yreal, Origin, ShoreNormalDir)
    """
    Xrelative = Xreal - Origin[0]
    Yrelative = Yreal - Origin[1]
    Dist = np.sqrt(Xrelative**2 + Yrelative**2)
    Dir = np.arctan2(Xrelative, Yrelative)
    Xmod = Dist * np.cos(ShoreNormalDir - Dir + np.pi/2)
    Ymod = Dist * np.sin(ShoreNormalDir - Dir + np.pi/2)
    return (Xmod, Ymod)

def intersectPolygon(Polygon, Xcoord):
    """ Identifies points where a polygon intersects a given x coordinate
    
    YIntersects = intersectPolygon(Polygon, Xcoord)
    
    Parameters:
        Polygon (np.ndarry(float)): Two-column numpy array giving X and Y
            coordinates of points definiing a polygon (first and last points 
            are identical)
        Xcoord (float): X coordinate at which to intersect the polygon
    
    Returns:
        YIntersects (np.ndarray(float)): 1d array listing the Y coordinate of 
            all the locations the polygon intersects the specified X coordinate
    """
    # Find polygon points to left of X coordinate
    LeftOfX = Polygon[:,0] < Xcoord
    
    # Identify points where polygon crosses X coordinate
    IntPoints = np.where(LeftOfX[0:-2] != LeftOfX[1:-1])[0]
    
    # Loop over each crossing and interpolate Y coord of crossing
    YIntersects = np.zeros(IntPoints.size)
    for IntNo in range(IntPoints.size):
        YIntersects[IntNo] = (Polygon[IntPoints[IntNo],1] +
                              (Xcoord-Polygon[IntPoints[IntNo],0]) *
                              (Polygon[IntPoints[IntNo]+1,1] - Polygon[IntPoints[IntNo],1]) / 
                              (Polygon[IntPoints[IntNo]+1,0] - Polygon[IntPoints[IntNo],0]))
    
    return YIntersects