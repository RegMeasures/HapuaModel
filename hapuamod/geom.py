""" HapuaModel Geometry module

    Includes functions for :
        Converting between model and real world coordinate systems.
        Intersecting different parts of the model.
        Trimming/extending lines.
        Interpolating/deleting nodes to ensure line discretisation is within 
            desired tolerance.
"""

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

def intersectPolyline(Polyline, Xcoord):
    """ Identifies points where a polyline intersects a given x coordinate
    
        YIntersects = intersectPolyline(Polyline, Xcoord)
        
        Parameters:
            Polyline (np.ndarry(float)): Two-column numpy array giving X and Y
                coordinates of points defining a polyline (or polygon if first
                and last points are the same).
            Xcoord (float): X coordinate at which to intersect the polyline
        
        Returns:
            YIntersects (np.ndarray(float)): 1d array listing the Y coordinate 
                of all the locations the polyline intersects the specified X 
                coordinate.
    """
    # Find polyline points to left of X coordinate
    LeftOfX = Polyline[:,0] < Xcoord
    
    # Identify points where polyline crosses X coordinate
    IntPoints = np.where(LeftOfX[0:-1] != LeftOfX[1:])[0]
    
    # Loop over each crossing and interpolate Y coord of crossing
    YIntersects = np.zeros(IntPoints.size)
    for IntNo in range(IntPoints.size):
        YIntersects[IntNo] = (Polyline[IntPoints[IntNo],1] +
                              (Xcoord-Polyline[IntPoints[IntNo],0]) *
                              (Polyline[IntPoints[IntNo]+1,1] - Polyline[IntPoints[IntNo],1]) / 
                              (Polyline[IntPoints[IntNo]+1,0] - Polyline[IntPoints[IntNo],0]))
    
    return YIntersects

def trimLine(LineX, LineY, TrimLineX, StartTrimLineY, EndTrimLineY):
    """ trim/extend line to trimlines
    
        Trim/extemd the start and end of Line to specified trim lines by moving
        the end nodes (i.e. no node deletion or new node creation).
        
        trimSegment(LineX, LineY, TrimLineX, StartTrimLineY, EndTrimLineY)        
        
        Notes: 
            TrimLineX must be increasing
            Both StartTrimLineY and EndTrimLineY are paired with TrimLineX
            LineY must be increasing
            LineX and LineY are modified in-place
    """
    
    #%% Trim Start
    SearchingForTrimSeg = True
    TrimSeg = np.where(np.logical_and(LineX[0] <= TrimLineX[1:], 
                                      TrimLineX[:-1] < LineX[0]))[0][0]
    while SearchingForTrimSeg:
        # find intersection and trim/extend
        Denominator = ((LineX[0]-LineX[1]) * (StartTrimLineY[TrimSeg]-StartTrimLineY[TrimSeg+1]) - 
                       (LineY[0]-LineY[1]) * (TrimLineX[TrimSeg]-TrimLineX[TrimSeg+1]))
#        if Denominator == 0: # make sure lines aren't parallel
#            continue
        XIntersect = ((LineX[0] * LineY[1] - LineY[0] * LineX[1]) * 
                      (TrimLineX[TrimSeg] - TrimLineX[TrimSeg+1]) - 
                      (LineX[0] - LineX[1]) * 
                      (TrimLineX[TrimSeg] * StartTrimLineY[TrimSeg+1] - 
                       StartTrimLineY[TrimSeg] * TrimLineX[TrimSeg+1])) / Denominator
        
        if TrimLineX[TrimSeg] > XIntersect:
            TrimSeg -= 1
        elif TrimLineX[TrimSeg+1] < XIntersect:
            TrimSeg += 1
        else:
            SearchingForTrimSeg = False
            
    
    YIntersect = ((LineX[0] * LineY[1] - LineY[0] * LineX[1]) * 
                  (StartTrimLineY[TrimSeg] - StartTrimLineY[TrimSeg+1]) - 
                  (LineY[0] - LineY[1]) * 
                  (TrimLineX[TrimSeg]*StartTrimLineY[TrimSeg+1] - 
                   StartTrimLineY[TrimSeg]*TrimLineX[TrimSeg+1])) / Denominator
            
    LineX[0] = XIntersect
    LineY[0] = YIntersect
    
    #%% Trim End
    SearchingForTrimSeg = True
    TrimSeg = np.where(np.logical_and(LineX[-1] <= TrimLineX[1:], 
                                      TrimLineX[:-1] < LineX[-1]))[0][0]
    while SearchingForTrimSeg:
        # find intersection and trim/extend
        Denominator = ((LineX[-2]-LineX[-1]) * (EndTrimLineY[TrimSeg]-EndTrimLineY[TrimSeg+1]) - 
                       (LineY[-2]-LineY[-1]) * (TrimLineX[TrimSeg]-TrimLineX[TrimSeg+1]))
#        if Denominator == 0: # make sure lines aren't parallel
#            continue
        XIntersect = ((LineX[-2] * LineY[-1] - LineY[-2] * LineX[-1]) * 
                      (TrimLineX[TrimSeg] - TrimLineX[TrimSeg+1]) - 
                      (LineX[-2] - LineX[-1]) * 
                      (TrimLineX[TrimSeg] * EndTrimLineY[TrimSeg+1] - 
                       EndTrimLineY[TrimSeg] * TrimLineX[TrimSeg+1])) / Denominator
        
        if TrimLineX[TrimSeg] > XIntersect:
            TrimSeg -= 1
        elif TrimLineX[TrimSeg+1] < XIntersect:
            TrimSeg += 1
        else:
            SearchingForTrimSeg = False
            
    
    YIntersect = ((LineX[-2] * LineY[-1] - LineY[-2] * LineX[-1]) * 
                  (EndTrimLineY[TrimSeg] - EndTrimLineY[TrimSeg+1]) - 
                  (LineY[-2] - LineY[-1]) * 
                  (TrimLineX[TrimSeg]*EndTrimLineY[TrimSeg+1] - 
                   EndTrimLineY[TrimSeg]*TrimLineX[TrimSeg+1])) / Denominator
            
    LineX[-1] = XIntersect
    LineY[-1] = YIntersect
    
def adjustLineDx(LineX, LineY, MaxDx, *args):
    """ Move points on line to maintain Dx within target range
    
        (NewLineX, NewLineY) = adjustLineDx(LineX, LineY, MaxDx)
            or
        (NewLineX, NewLineY, NewLineProperties)
            = adjustLineDx(LineX, LineY, MaxDx, LineProperties)
    """
    LineProperties = list(args)
    
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
            RemoveNode = 1
        elif Shortest == SegLen.size-1:
            RemoveNode = Shortest
        elif SegLen[Shortest-1] <= SegLen[Shortest+1]:
            RemoveNode = Shortest
        else:
            RemoveNode = Shortest + 1
        
        SegLen[RemoveNode-1] = SegLen[RemoveNode-1] + SegLen[RemoveNode]
        SegLen = np.delete(SegLen, RemoveNode)
        LineX = np.delete(LineX, RemoveNode)
        LineY = np.delete(LineY, RemoveNode)
        for Property in LineProperties:
            Property = np.delete(Property, RemoveNode)
    
    # Split segments which are too long
    TooLong = np.where(SegLen > MaxDx)[0]
    for SegNo in TooLong:
        SplitInto = int(np.ceil(SegLen[SegNo]/DefaultDx))
        LineX = np.insert(LineX, SegNo+1, 
                          LineX[SegNo] + (LineX[SegNo+1] - LineX[SegNo]) 
                                           * (np.linspace(0,1,SplitInto,False)[1:]))
        LineY = np.insert(LineY, SegNo+1, 
                          LineY[SegNo] + (LineY[SegNo+1] - LineY[SegNo]) 
                                           * (np.linspace(0,1,SplitInto,False)[1:]))
        for Property in LineProperties:
            Property = np.insert(Property, SegNo+1, 
                                 Property[SegNo] + (Property[SegNo+1] - Property[SegNo]) 
                                                     * (np.linspace(0,1,SplitInto,False)[1:]))

    return tuple([LineX, LineY] + LineProperties)

def shiftLineSideways(LineX, LineY, Shift):
    """ Apply lateral shift to a line by moving XY node coordinates
    
        shiftLineSideways(LineX, LineY, Shift)
        
        Notes: 
            LineX and LineY are edited in-place so no return parameters are 
                required.
            Shift is positive to right (in direction of line)
    """
    Dx = np.zeros(LineX.size)
    Dx[0] = LineX[1] - LineX[0]
    Dx[1:-1] = LineX[2:] - LineX[:-2]
    Dx[-1] = LineX[-1] - LineX[-2]
    
    Dy = np.zeros(LineY.size)
    Dy[0] = LineY[1] - LineY[0]
    Dy[1:-1] = LineY[2:] - LineY[:-2]
    Dy[-1] = LineY[-1] - LineY[-2]
    
    SlopeLength = np.sqrt(Dx**2 + Dy**2)
    Dx /= SlopeLength
    Dy /= SlopeLength
    
    LineX += Shift * Dy
    LineY -= Shift * Dx
    
    