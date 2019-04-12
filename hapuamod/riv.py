# -*- coding: utf-8 -*-
""" Fluvial hydraulics calculations """

# import standard packages
import numpy as np
import logging

def assembleChannel(RiverElev, ShoreX, LagoonY, LagoonElev, OutletX, OutletY,
                    OutletElev, OutletWidth, RiverWidth, Dx):
    """ Combine river, lagoon and outlet into single channel for hyd-calcs
    
    (ChanDx, ChanElev, ChanWidth, ChanArea) = \
        riv.assembleChannel(RiverElev, ShoreX, LagoonY, LagoonElev, OutletDx, 
                            OutletElev, OutletWidth, OutletX, RiverWidth, Dx)
    """
    LagoonWidth = LagoonY[:,1] - LagoonY[:,0]
    OutletDx = np.sqrt((OutletX[1:]-OutletX[0:-1])**2 + 
                       (OutletY[1:]-OutletY[0:-1])**2)
    # TODO: Account for potentially dry parts of the lagoon when 
    #       calculating ChanArea
    
    # To identify the 'online' section of lagoon the river flows through we 
    # first need to know which direction the offset is
    RivToRight = OutletX[0] > 0
    
    if RivToRight:
        OnlineLagoon = np.logical_and(0 <= ShoreX, ShoreX <= OutletX[0])
        OnlineElev = LagoonElev[OnlineLagoon]
        OnlineWidth = LagoonWidth[OnlineLagoon]
        StartArea = np.nansum(LagoonWidth[ShoreX < 0] * Dx)
        EndArea = np.nansum(LagoonWidth[ShoreX > OutletX[0]] * Dx)
    else:
        OnlineLagoon = np.logical_and(0 >= ShoreX, ShoreX >= OutletX[0])
        OnlineElev = np.flipud(LagoonElev[OnlineLagoon])
        OnlineWidth = np.flipud(LagoonWidth[OnlineLagoon])
        StartArea = np.nansum(LagoonWidth[ShoreX > 0] * Dx)
        EndArea = np.nansum(LagoonWidth[ShoreX < OutletX[0]] * Dx)
    
    ChanDx = np.concatenate([np.tile(Dx, RiverElev.size + np.sum(OnlineLagoon)),
                             OutletDx])
    ChanElev = np.concatenate([RiverElev, OnlineElev, OutletElev])
    ChanWidth = np.concatenate([np.tile(RiverWidth, RiverElev.size), 
                                OnlineWidth, OutletWidth])
    ChanArea = np.insert(ChanDx, 0, 0) * ChanWidth
    ChanArea[RiverElev.size] += StartArea
    ChanArea[-(OutletX.size+1)] += EndArea
    
    return (ChanDx, ChanElev, ChanWidth, ChanArea)

def solveSteady(ChanDx, ChanElev, ChanWidth, Roughness, Qin, DsWL):
    """ Solve steady state river hydraulics
    
    (ChanDep, ChanVel) = solveSteady(ChanDx, ChanElev, ChanWidth, 
                                     Roughness, Qin, DsWL)
    """
    Grav = 9.81
    Tol = 0.005
    MaxIter = 10
    
    # Find critical depth
    # Fr = Vel/sqrt(Grav*Dep) = 1 i.e. Vel^2 = Grav*Dep
    # Vel = Q/(Width*Dep)
    # Q^2/(Width^2*Dep^2) = Grav*Dep
    # Dep^3 = Q^2/(Width^2*Grav)
    CritDep = (Qin**2 / (ChanWidth**2 * Grav))**(1/3)
    
    ChanVel = np.zeros(ChanElev.size)
    ChanDep = np.zeros(ChanElev.size)
    Energy = np.zeros(ChanElev.size)
    S_f = np.zeros(ChanElev.size)
    ChanDep[-1] = np.maximum(DsWL, CritDep[-1])
    ChanVel[-1] = Qin / (ChanWidth[-1] * ChanDep[-1])
    Energy[-1] = ChanElev[-1] + ChanDep[-1] + ChanVel[-1]**2 / (2*Grav)
    S_f[-1] = ChanVel[-1]**2 * Roughness**2 / ChanDep[-1]**(4/3)
    # iterate from d/s end
    for XS in range(ChanDep.size-2, -1, -1):
        # Manning: Vel = R^(2/3)*Sf^(1/2) / n
        # Wide channel: R = h
        # Sf = Vel^2 * n^2 / h^(4/3)
        # Rectangular channel: Vel = Q / (Width*Dep)
        # Sf = Q^2 * n^2 / (Width^2 * Dep^(10/3))
        # Bernoulli: Energy = Zb + Dep + Vel^2/(2g)
        
        # iterate solution
        ChanDep[XS] = ChanDep[XS+1] + ChanElev[XS+1] - ChanElev[XS] + S_f[XS+1]*ChanDx[XS] # initial estimate
        
        Acoef = (Qin**2 / (2*Grav*ChanWidth[XS]**2))
        Bcoef = (ChanDx[XS] * Qin**2 * Roughness**2 / (2 * ChanWidth[XS]**2))
        Cconst = ChanElev[XS] - S_f[XS+1]*ChanDx[XS]/2 - Energy[XS+1] 
        DepErr = ChanDep[XS] + Acoef*ChanDep[XS]**(-2) - Bcoef*ChanDep[XS]**(-10/3) + Cconst
        CheckCount = 0
        while DepErr > Tol:
            Gradient = 1 - 2*Acoef*ChanDep[XS]**(-3) + (10/3)*Bcoef*ChanDep[XS]**(-13/3)
            ChanDep[XS] -= DepErr / Gradient
            DepErr = ChanDep[XS] + Acoef*ChanDep[XS]**(-2) - Bcoef*ChanDep[XS]**(-10/3) + Cconst
            CheckCount += 1
            assert CheckCount < MaxIter, 'Maximum iterations exceeded solving steady state water level'
        
        # Check for supercritical
        if ChanDep[XS] < CritDep[XS]:
            ChanDep[XS] = CritDep[XS]
            logging.warning('Steady state solution results in critical depth at XS%i' % XS)
        ChanVel[XS] = Qin / (ChanWidth[XS] * ChanDep[XS])
        Energy[XS] = ChanElev[XS] + ChanDep[XS] + ChanVel[XS]**2 / (2*Grav)
        S_f[XS] = ChanVel[XS]**2 * Roughness**2 / ChanDep[XS]**(4/3)
    
    return ChanDep, ChanVel