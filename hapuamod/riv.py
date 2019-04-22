# -*- coding: utf-8 -*-
""" Fluvial hydraulics calculations """

# import standard packages
import numpy as np
from scipy import linalg
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
    """ Solve steady state river hydraulics for a rectangular channel
    
    (ChanDep, ChanVel) = solveSteady(ChanDx, ChanElev, ChanWidth, 
                                     Roughness, Qin, DsWL)
    """
    Grav = 9.81
    Tol = 0.0005
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
    ChanDep[-1] = np.maximum(DsWL - ChanElev[-1], CritDep[-1])
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

def solveFullPreissmann(z, B, y, V, dx, dt, n, Q, DsWl, alpha, Tol, MaxIt, g):
    """ Solve full S-V eqns for a rectangular channel using preissmann scheme
        Uses a newton raphson solution to the preissmann discretisation of the 
        Saint-Venant equations.
        
        (y, V) = solveFullPreissmann(z, B, y, V, dx, dt, n, Q, DsWl, 
                                     alpha, Tol, MaxIt, g)
        
        Parameters:
            z (np.ndarray(float64)): Bed elevation at each node (m)
            B (np.ndarray(float64)): Channel width at each node (m)
            y (np.ndarray(float64)): Depth at each node at the last timestep (m)
            V (np.ndarray(float64)): Mean velocity at each node at the last timestep (m)
            dx (np.ndarray(float64)): Length of each reach between nodes (m)
                note the dx array should be one shorter than for z, B, etc
            dt (pd.timedelta): timestep
            n (float): mannings 'n' roughness
            Q (float): inflow at upstream end of channel (m^3/s)
            DsWl (float): downstream water level (m)
            alpha (float): temporal weighting coefficient for preissmann scheme
            Tol (float): error tolerance (both m and m/s)
            MaxIt (integer): maximum number of iterations to find solution
            g (float): gravity (m/s^2)
        
        Returns:
            y
            V
        
    """
    
    dt = dt.seconds
    
    N = z.size
    S_0 = (z[:-1]-z[1:])/dx
    A = y*B
    Sf = V**2 * n**2 / y**(4/3)
    
    V_old = V
    y_old = y
    Sf_old = Sf
    A_old = A
    
    # Constant parts of the S-V Equations which can be computed outside the loop
    # For continuity equation
    C1 = A_old[:-1] + A_old[1:] - 2*(dt/dx)*(1-alpha) * (V_old[1:]*A_old[1:] - V_old[:-1]*A_old[:-1])
    
    # For momentum equation
    C2 = (g*dt*(1-alpha) * (A_old[1:]*(S_0 - Sf_old[1:]) 
                            + A_old[:-1]*(S_0 - Sf_old[:-1]))
          + V_old[:-1]*A_old[:-1]
          + V_old[1:]*A_old[1:]
          -2*(dt/dx)*(1-alpha) * (V_old[1:]**2*A_old[1:] - V_old[:-1]**2*A_old[:-1]))
    
    # Iterative solution
    ItCount = 0
    Err = np.zeros(2*N)
    while ItCount < MaxIt:
        
        # Error in Us Bdy
        Err[0] = A[0]*V[0]-Q
        
        # Error in continuity equation
        Err[np.arange(1,2*N-1,2)] = (A[:-1] + A[1:]
                                     + 2*(dt/dx)*alpha * (V[1:]*A[1:] - V[:-1]*A[:-1])) - C1
        
        # Error in momentum equation
        Err[np.arange(2,2*N-1,2)] = (V[:-1]*A[:-1] + V[1:]*A[1:]
                                     + 2*(dt/dx)*alpha * (V[1:]**2*A[1:] - V[:-1]**2*A[:-1]) 
                                     - g*dt*alpha * (A[1:]*(S_0-Sf[1:]) + A[:-1]*(S_0-Sf[:-1]))
                                     + g*(dt/dx)*(alpha*(A[1:]+A[:-1]) + (1-alpha)*(A_old[1:]+A_old[:-1]))
                                                *(alpha*(y[1:]-y[:-1]) + (1-alpha)*(y_old[1:]-y_old[:-1]))
                                    ) - C2
        
        # Error in Ds Bdy
        Err[2*N-1] = z[N-1] + y[N-1] - DsWl
        #Err[2*N-1] = z[N-1] + y[N-1] + V[N-1]**2/(2*g) - DsWl
        
        # Solve errors using Newton Raphson
        # a Delta = Err
        # where Delta = [dy[0],dV[0],dy[1],dV[1],...,...,dy[N-1],dV[N-1]]
        # a_banded = sparse 5 banded matrix see https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.solve_banded.html
        a_banded = np.zeros([5,2*N])
        
        # Us Bdy condition derivatives
        a_banded[1,1] = A[0]
        a_banded[2,0] = V[0]*B[0]
        
        # Continuity equation derivatives
        # d/dy[0]
        a_banded[3,np.arange(0,2*(N)-2,2)] = B[:-1] - 2*dt/dx*alpha*V[:-1]*B[:-1]
        # d/dV[0]
        a_banded[2,np.arange(1,2*(N)-2,2)] = -2*dt/dx*alpha*A[:-1]
        # d/dy[1]
        a_banded[1,np.arange(2,2*(N),2)] = B[1:] + 2*dt/dx*alpha*V[1:]*B[1:]
        # d/dV[1]
        a_banded[0,np.arange(3,2*(N),2)] = 2*dt/dx*alpha*A[1:]
        
        # Momentum equation derivatives
        # d/dy[0]
        a_banded[4,np.arange(0,2*(N)-2,2)] = (V[:-1]*B[:-1] 
                                              - 2*(dt/dx)*alpha*V[:-1]**2*B[:-1]
                                              - g*dt*alpha*(B[:-1]*(S_0+(1/3)*Sf[:-1]))
                                              + alpha*g*(dt/dx)*(B[:-1]*(alpha*(y[1:]-y[:-1]) + (1-alpha)*(y_old[1:]-y_old[:-1]))
                                                                 - (alpha*(A[1:]+A[:-1]) + (1-alpha)*(A_old[1:]+A_old[:-1]))))
        # d/dV[0]
        a_banded[3,np.arange(1,2*(N)-2,2)] = (A[:-1] 
                                              - 4*(dt/dx)*alpha*V[:-1]*A[:-1] 
                                              + g*dt*alpha*A[:-1]*2*Sf[:-1]/V[:-1])
        # d/dy[1]
        a_banded[2,np.arange(2,2*(N),2)] = (V[1:]*B[1:] 
                                            + 2*(dt/dx)*alpha*(V[1:]**2*B[1:] + g*A[1:]) 
                                            - g*dt*alpha*(B[1:]*(S_0-(1/3)*Sf[1:]))
                                            + alpha*g*(dt/dx)*(B[1:]*(alpha*(y[1:]-y[:-1]) + (1-alpha)*(y_old[1:]-y_old[:-1]))
                                                               + alpha*(A[1:]+A[:-1]) + (1-alpha)*(A_old[1:]+A_old[:-1])))
        # d/dV[1]
        a_banded[1,np.arange(3,2*(N),2)] = (A[1:] 
                                            + 4*dt/dx*alpha*V[1:]*A[1:] 
                                            + g*dt*alpha*A[1:]*2*Sf[1:]/V[1:])
        
        # Ds Bdy condition derivatives
        a_banded[2,2*N-1] = 0
        #a_banded[2,2*N-1] = V[N-1]/g
        a_banded[3,2*N-2] = 1
        
        # Solve the banded matrix
        Delta = linalg.solve_banded([2,2],a_banded,Err)
        
        # Update y & V
        y -= Delta[np.arange(0,2*N,2)]
        V -= Delta[np.arange(1,2*N,2)]
        
        # Update Sf and A
        Sf = V**2 * n**2 / y**(4/3)
        A = y*B
        
        # Check if solution is within tolerance
#        if np.sum(np.abs(Delta)) < Tol:
#            break
        if np.all(np.abs(Delta) < Tol):
            break
        
        ItCount += 1
    
    assert ItCount < MaxIt, 'Max iterations exceeded.'
    
    return(y, V)