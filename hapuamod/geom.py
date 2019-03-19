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
    IntPoints = np.where(LeftOfX[0:-1] != LeftOfX[1:])[0]
    
    # Loop over each crossing and interpolate Y coord of crossing
    YIntersects = np.zeros(IntPoints.size)
    for IntNo in range(IntPoints.size):
        YIntersects[IntNo] = (Polygon[IntPoints[IntNo],1] +
                              (Xcoord-Polygon[IntPoints[IntNo],0]) *
                              (Polygon[IntPoints[IntNo]+1,1] - Polygon[IntPoints[IntNo],1]) / 
                              (Polygon[IntPoints[IntNo]+1,0] - Polygon[IntPoints[IntNo],0]))
    
    return YIntersects

def trimSegment(LineX, LineY, TrimLineX, TrimLineY):
    """ trim/extend line to trimline
    
    Find the first location Line crosses TrimLine and trim off Line at this 
    point. If Line does not cross TrimLine then extend the last segment of Line 
    until it does.
    
    (NewLineX, NewLineY) = trimSegment(LineX, LineY, TrimLineX, TrimLineY)
    """
    # Loop over line from start until crossing found then trim to intersection
    for ii in range(LineX.size-1):
        # Find TrimLine sections which might intersect
        MaxX = np.amax(LineX[[ii,ii+1]])
        MinX = np.amin(LineX[[ii,ii+1]])
        PossSections = np.where(np.logical_and(MinX < TrimLineX[1:], 
                                               TrimLineX[0:-1] < MaxX))[0]
        
        for SecNo in PossSections:
            # Check for intersections
            # https://en.wikipedia.org/wiki/Line%E2%80%93line_intersection
            Denominator = ((LineX[ii]-LineX[ii+1]) * (TrimLineY[SecNo]-TrimLineY[SecNo+1]) - 
                           (LineY[ii]-LineY[ii+1]) * (TrimLineX[SecNo]-TrimLineX[SecNo+1]))
            if Denominator == 0: # make sure lines aren't parallel
                continue
            TNumerator = ((LineX[ii]-TrimLineX[SecNo]) * (TrimLineY[SecNo]-TrimLineY[SecNo+1]) - 
                          (LineY[ii]-TrimLineY[SecNo]) * (TrimLineX[SecNo]-TrimLineX[SecNo+1]))
            UNumerator = ((LineX[ii]-LineX[ii+1]) * (LineY[ii]-TrimLineY[SecNo]) - 
                          (LineY[ii]-LineY[ii+1]) * (LineX[ii]-TrimLineX[SecNo]))
            if (0 < (TNumerator/Denominator) < 1) and (0 < (UNumerator/Denominator) < 1):
                # line segments intersect - delete further line segments
                break
        else:
            continue
        break
    
    # find intersection
    assert Denominator!=0, 'Attempting to extend parrallel lines!'
    
    XIntersect = ((LineX[ii] * LineY[ii+1] - LineY[ii] * LineX[ii+1]) * 
                  (TrimLineX[SecNo] - TrimLineX[SecNo+1]) - 
                  (LineX[ii] - LineX[ii+1]) * 
                  (TrimLineX[SecNo] * TrimLineY[SecNo+1] - 
                   TrimLineY[SecNo] * TrimLineX[SecNo+1])) / Denominator
    
    YIntersect = ((LineX[ii] * LineY[ii+1] - LineY[ii] * LineX[ii+1]) * 
                  (TrimLineY[SecNo] - TrimLineY[SecNo+1]) - 
                  (LineY[ii] - LineY[ii+1]) * 
                  (TrimLineX[SecNo]*TrimLineY[SecNo+1] - 
                   TrimLineY[SecNo]*TrimLineX[SecNo+1])) / Denominator
                
    NewLineX = np.append(LineX[0:ii+1], XIntersect)
    NewLineY = np.append(LineY[0:ii+1], YIntersect)
    
    return (NewLineX, NewLineY)
    
def adjustLineDx(LineX, LineY, MaxDx):
    """ Move points on line to maintain Dx within target range
    
    (NewLineX, NewLineY) = adjustLineDx(LineX, LineY, MaxDx)
    """
    MinDx = 0.4 * MaxDx
    DefaultDx = 0.7 * MaxDx
    
    # Calc current length of each line segment
    SegLen = np.sqrt((LineX[1:]-LineX[0:-1])**2 + (LineY[1:]-LineY[0:-1])**2)
    
    # Cumulative length of segments in input line (to use for interpolation of 
    # line properties)
    CumSegLen = np.cumsum(SegLen)
    CumSegLen = np.insert(CumSegLen, 0, 0.0)
    
    # Delete nodes on segments that are too short - do this iteratively.
    # On each iteratio find the shortest segment and delete the node between it 
    # and it's shortest neighbouring segment.
    while np.any(SegLen < MinDx):
        Shortest = np.which(SegLen == np.amin(SegLen))[0]
        if Shortest == 0:
            RemNode = 1
        elif Shortest == SegLen.size-1:
            RemNode = Shortest
        elif SegLen[Shortest-1] <= SegLen[Shortest+1]:
            RemNode = Shortest
        else:
            RemNode = Shortest + 1
        
        SegLen = np.concatenate([SegLen[:RemNode-1],
                                 SegLen[RemNode-1]+SegLen[RemNode],
                                 SegLen[RemNode:]])
    
    # Split segments which are too long
    TooLong = np.where(SegLen > MaxDx)[0]
    for Seg in np.nditer(TooLong):
        SplitInto = int(np.ceil(SegLen[Seg]/DefaultDx))
        SegLen = np.concatenate((SegLen[:Seg],
                                 np.tile(SegLen[Seg] / SplitInto, SplitInto),
                                 SegLen[Seg+1:]))
    
    # Re-interpolate line properties
    NewCumSegLen = np.cumsum(SegLen)
    NewLineX = np.concatenate([[LineX[0]], 
                               np.interp(NewCumSegLen[0:-1], CumSegLen, LineX),
                               [LineX[-1]]])
    NewLineY = np.concatenate([[LineY[0]], 
                               np.interp(NewCumSegLen[0:-1], CumSegLen, LineY),
                               [LineY[-1]]])
    
    return (NewLineX, NewLineY)
    
    # TODO: Extend function to interpolate line properties (supplied as new optional input(S?))
    
    