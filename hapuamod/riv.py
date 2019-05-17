# -*- coding: utf-8 -*-
""" Fluvial hydraulics calculations """

# import standard packages
import numpy as np
from scipy import linalg
import logging

def assembleChannel(RiverElev, ShoreX, LagoonY, LagoonElev, OutletX, OutletY,
                    OutletElev, OutletWidth, RiverWidth, Dx):
    """ Combine river, lagoon and outlet into single channel for hyd-calcs
    
    (ChanDx, ChanElev, ChanWidth, LagArea) = \
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
    LagArea = np.zeros(ChanElev.size)
    LagArea[RiverElev.size] = StartArea
    LagArea[-(OutletX.size+1)] = EndArea
    
    return (ChanDx, ChanElev, ChanWidth, LagArea)

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
        # Rectangular channel: Vel = Q / (B*h)
        # Sf = Q^2 * n^2 / (B^2 * h^(10/3))
        # Bernoulli: Energy = Zb + Dep + Vel^2/(2g)
        # h[i] + (Q^2/(2*g*B[i]^2))*h^(-2) - (Dx*Q^2*n^2/(2*B[i]^2))*h^(-10/3) - Energy[i+1]+z[i]-(Dx/2)*Sf[i+1] = 0
        # h[i] + A*h[i]^(-2) - B*h[i]^(-10/3) + C = 0
        
        # initial estimate
        ChanDep[XS] = (ChanDep[XS+1] + ChanElev[XS+1]) - ChanElev[XS] + S_f[XS+1]*ChanDx[XS] 
        
        # iterate solution for h
        Acoef = (Qin**2 / (2*Grav*ChanWidth[XS]**2))
        Bcoef = (ChanDx[XS] * Qin**2 * Roughness**2 / (2 * ChanWidth[XS]**2))
        Cconst = ChanElev[XS] - S_f[XS+1]*ChanDx[XS]/2 - Energy[XS+1] 
        DepErr = ChanDep[XS] + Acoef*ChanDep[XS]**(-2) - Bcoef*ChanDep[XS]**(-10/3) + Cconst
        CheckCount = 0
        while np.abs(DepErr) > Tol:
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

def solveFullPreissmann(z, B, LagArea, h, V, dx, dt, n, Q_Ts, DsWl_Ts, NumericalPars):
    """ Solve full S-V eqns for a rectangular channel using preissmann scheme.
        Uses a newton raphson solution to the preissmann discretisation of the 
        Saint-Venant equations.
        
        (h, V) = solveFullPreissmann(z, B, h, V, dx, dt, n, Q, DsWl, 
                                     Theta, Tol, MaxIt)
        
        Parameters:
            z (np.ndarray(float64)): Bed elevation at each node (m)
            B (np.ndarray(float64)): Channel width at each node (m)
            LagArea
            h (np.ndarray(float64)): Depth at each node at the last timestep (m)
            V (np.ndarray(float64)): Mean velocity at each node at the last timestep (m)
            dx (np.ndarray(float64)): Length of each reach between nodes (m)
                note the dx array should be one shorter than for z, B, etc
            dt (pd.timedelta): timestep
            n (float): mannings 'n' roughness
            Qin (np.ndarray(float64)): List (as np.array) of inflows at 
                upstream end of channel corresponding to each timestep. 
                Function will loop over each inflow in turn (m^3/s)
            DsWl (np.ndarray(float64)): List (as np.array) of downstream 
                boundary water levels at upstream end of channel corresponding 
                to each timestep. Function will loop over each inflow in turn 
                (m^3/s)
            Theta (float): temporal weighting coefficient for preissmann scheme
            Tol (float): error tolerance (both m and m/s)
            MaxIt (integer): maximum number of iterations to find solution
            g (float): 
        
        Returns:
            h
            V
        
    """
    g = 9.81                    # gravity [m/s^2]
    
    Theta = NumericalPars['Theta']
    Tol = NumericalPars['ErrTol']
    MaxIt = NumericalPars['MaxIt']
    dt = dt.seconds             # timestep for hydraulics [s]
    N = z.size                  # number of cross-sections
    S_0 = (z[:-1]-z[1:])/dx     # bed slope in each reach between XS [m/m]
    
    # Calculate effective width for volume change i.e. total planform area/reach length. 
    dx2 = np.zeros(z.size)
    dx2[0] = dx[0]
    dx2[1:-1] = (dx[:-1]+dx[1:])/2
    dx2[-1] = dx[-1]
    Be = B + LagArea/dx2
    
    # Pre-compute some variables required within the loop
    A = h*B                     # area of flow at each XS [m^2]
    Sf = V*np.abs(V) * n**2 / h**(4/3) # friction slope at each XS [m/m]
    
    # Main timestepping loop
    for StepNo in range(Q_Ts.shape[0]):
        Q = Q_Ts[StepNo]
        DsWl = DsWl_Ts[StepNo]
        
        V_old = V
        h_old = h
        Sf_old = Sf
        A_old = A
        
        # Constant parts of the S-V Equations which can be computed outside the loop
        # For continuity equation
        C1 = (h_old[:-1]*Be[:-1] + h_old[1:]*Be[1:] 
              - 2*(dt/dx)*(1-Theta) * (V_old[1:]*A_old[1:] 
                                       - V_old[:-1]*A_old[:-1]))
        
        # For momentum equation
        C2 = (V_old[:-1]*A_old[:-1]
              + V_old[1:]*A_old[1:]
              -2*(dt/dx)*(1-Theta) * (V_old[1:]**2*A_old[1:]
                                      - V_old[:-1]**2*A_old[:-1])
              + g*dt*(1-Theta) * (A_old[1:]*(S_0 - Sf_old[1:]) 
                                  + A_old[:-1]*(S_0 - Sf_old[:-1])))
              
        
        # Iterative solution
        ItCount = 0
        Err = np.zeros(2*N)
        while ItCount < MaxIt:
            
            # Error in Us Bdy
            Err[0] = A[0]*V[0]-Q
            
            # Error in continuity equation
            Err[np.arange(1,2*N-1,2)] = (h[:-1]*Be[:-1] + h[1:]*Be[1:]
                                         + 2*(dt/dx)*Theta * (V[1:]*A[1:] - V[:-1]*A[:-1])) - C1
            
            # Error in momentum equation
            Err[np.arange(2,2*N-1,2)] = (V[:-1]*A[:-1] + V[1:]*A[1:]
                                         + 2*(dt/dx)*Theta * (V[1:]**2*A[1:]
                                                              - V[:-1]**2*A[:-1])
                                         + g*(dt/dx)*(Theta*(A[1:]+A[:-1])+(1-Theta)*(A_old[1:]+A_old[:-1]))
                                                    *(Theta*(h[1:]-h[:-1])+(1-Theta)*(h_old[1:]-h_old[:-1]))
                                         - g*dt*Theta * (A[1:]*(S_0-Sf[1:]) + A[:-1]*(S_0-Sf[:-1]))
                                        ) - C2
            
            # Error in Ds Bdy
            Err[2*N-1] = z[N-1] + h[N-1] - DsWl
            #Err[2*N-1] = z[N-1] + h[N-1] + V[N-1]**2/(2*g) - DsWl
            
            # Solve errors using Newton Raphson
            # a Delta = Err
            # where Delta = [dh[0],dV[0],dh[1],dV[1],...,...,dh[N-1],dV[N-1]]
            # a_banded = sparse 5 banded matrix see https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.solve_banded.html
            a_banded = np.zeros([5,2*N])
            
            # Us Bdy condition derivatives
            a_banded[1,1] = A[0]
            a_banded[2,0] = V[0]*B[0]
            
            # Continuity equation derivatives
            # d/dh[0]
            a_banded[3,np.arange(0,2*(N)-2,2)] = Be[:-1] - 2*dt/dx*Theta*V[:-1]*B[:-1]
            # d/dV[0]
            a_banded[2,np.arange(1,2*(N)-2,2)] = -2*dt/dx*Theta*A[:-1]
            # d/dh[1]
            a_banded[1,np.arange(2,2*(N),2)] = Be[1:] + 2*dt/dx*Theta*V[1:]*B[1:]
            # d/dV[1]
            a_banded[0,np.arange(3,2*(N),2)] = 2*dt/dx*Theta*A[1:]
            
            # Momentum equation derivatives
            # d/dh[0]
            a_banded[4,np.arange(0,2*(N)-2,2)] = (V[:-1]*B[:-1] 
                                                  - 2*(dt/dx)*Theta*V[:-1]**2*B[:-1]
                                                  + g*(dt/dx)*Theta*(-2*Theta*A[:-1]
                                                                     +B[:-1]*(Theta*h[1:]+(1-Theta)*(h_old[1:]-h_old[:-1]))
                                                                     -(Theta*A[1:]+(1-Theta)*(A_old[1:]+A_old[:-1])))
                                                  - g*dt*Theta*B[:-1]*(S_0+(1/3)*Sf[:-1]))
            # d/dV[0]
            a_banded[3,np.arange(1,2*(N)-2,2)] = (A[:-1] 
                                                  - 4*(dt/dx)*Theta*V[:-1]*A[:-1] 
                                                  + g*dt*Theta*A[:-1]*Sf[:-1]/V[:-1])
            # d/dh[1]
            a_banded[2,np.arange(2,2*(N),2)] = (V[1:]*B[1:] 
                                                + 2*(dt/dx)*Theta*V[1:]**2*B[1:]
                                                + g*(dt/dx)*Theta*(2*Theta*A[1:]
                                                                   +B[1:]*(-Theta*h[:-1]+(1-Theta)*(h_old[1:]-h_old[:-1]))
                                                                   +(Theta*A[:-1]+(1-Theta)*(A_old[1:]+A_old[:-1])))
                                                - g*dt*Theta*B[1:]*(S_0+(1/3)*Sf[1:]))
            # d/dV[1]
            a_banded[1,np.arange(3,2*(N),2)] = (A[1:] 
                                                + 4*dt/dx*Theta*V[1:]*A[1:] 
                                                + g*dt*Theta*A[1:]*Sf[1:]/V[1:])
            
            # Ds Bdy condition derivatives
            a_banded[2,2*N-1] = 0
            #a_banded[2,2*N-1] = V[N-1]/g
            a_banded[3,2*N-2] = 1
            
            # Solve the banded matrix
            Delta = linalg.solve_banded([2,2],a_banded,Err)
            
            # Update h & V
            h -= Delta[np.arange(0,2*N,2)]
            V -= Delta[np.arange(1,2*N,2)]
            
            # Update Sf and A
            Sf = V*np.abs(V) * n**2 / h**(4/3)
            A = h*B
            
            # Check if solution is within tolerance
    #        if np.sum(np.abs(Delta)) < Tol:
    #            break
            if np.all(np.abs(Delta) < Tol):
                break
            
            # Stability warnings
            if np.amax(Delta)>NumericalPars['WarnTol']:
                WarnNode = np.floor(np.argmax(Delta)/2)
                if np.argmax(Delta)%2 == 0:
                    WarnVar = 'Depth'
                else:
                    WarnVar = 'Velocity'
                logging.warning('%s change in cross-section %i equals %f (greater than WarnTol)',
                                WarnVar, WarnNode, np.max(Delta))            
            ItCount += 1
        
        assert ItCount < MaxIt, 'Max iterations exceeded.'
    
    return(h, V)
    
def calcBedload(z, B, h, V, dx, PhysicalPars):
    """ Calculate bedload transport using Bagnold 1980 streampower approach
    """
    
    Rho = PhysicalPars['RhoRiv']
    RhoS = PhysicalPars['RhoSed']
    g = PhysicalPars['Gravity']
    D = PhysicalPars['GrainSize']
    VoidRatio = PhysicalPars['VoidRatio']
    
    # Slope at each XS assumed to equal energy slope in reach upstream of XS
    TotHead = z + h + V**2/(2*g)
    S = np.empty(z.size)
    S[0] = PhysicalPars['RiverSlope']
    S[1:] = (TotHead[:-1]-TotHead[1:])/dx
    
    # Threshold streampower per unit width [kg/m/s]
    omega_0 = 5.75*(0.04*(RhoS - Rho))**(3/2) * (g/Rho)**(1/2) * D**(3/2) * np.log10(12*h/D)
    # TODO move constant part of above line into loadmod (i.e. out of loop)
    
    # streampower per unit width [kg/m/s]
    omega = Rho * S * V * h 
    
    # bedload transport rate (bulk volume accounting for voids) [m^3/s]
    Qs = RhoS/(RhoS-Rho) * 0.01 * (np.maximum(omega-omega_0,0.0)/0.5)**(3/2) * (h/0.1)**(-2/3) * (D/0.0011)**(-1/2) * B / (RhoS*(1-VoidRatio)) 
    # TODO move constant part of above line into loadmod (i.e. out of loop)
    
    return(Qs)
    
def riverMorphology(Qs, B, h, z, BankElev, dx, dt, PhysicalPars):
    
    # Change in volume at each cross-section (except the upstream Bdy)
    dVol = (Qs[:-1]-Qs[1:]) * dt
    
    # Some current physical properties of each XS
    BedArea = dx * B[1:] 
    AspectRatio = B[1:]/h[1:]
    TooWide = AspectRatio > PhysicalPars['WidthRatio']
    
    # Update bed elevation
    ErosionVol = np.minimum(dVol, 0.0)
    z[1:] += (np.maximum(dVol, 0) + ErosionVol * TooWide)/BedArea
    
    # Update channel width
    B[1:] += (-ErosionVol * np.logical_not(TooWide)) / ((BankElev[1:]-z[1:])*dx)
    # TODO split L and R bank calculations and account for differences in bank height
    