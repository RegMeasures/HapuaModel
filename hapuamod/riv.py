# -*- coding: utf-8 -*-
""" Fluvial hydraulics calculations """

# import standard packages
import numpy as np
from scipy import linalg
import logging

def solveSteady(ChanDx, ChanElev, ChanWidth, Roughness, Qin, DsWL, 
                NumericalPars):
    """ Solve steady state river hydraulics for a rectangular channel
    
    (ChanDep, ChanVel) = solveSteady(ChanDx, ChanElev, ChanWidth, 
                                     Roughness, Qin, DsWL, NumericalPars)
    """
    Grav = 9.81
    Beta = NumericalPars['Beta']
    Tol = NumericalPars['ErrTol']
    MaxIt = NumericalPars['MaxIt']
    
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
    Energy[-1] = ChanElev[-1] + ChanDep[-1] + Beta * ChanVel[-1]**2 / (2*Grav)
    S_f[-1] = ChanVel[-1]**2 * Roughness**2 / ChanDep[-1]**(4/3)
    # iterate from d/s end
    for XS in range(ChanDep.size-2, -1, -1):
        # Manning: Vel = R^(2/3)*Sf^(1/2) / n
        # Wide channel: R = h
        # Sf = Vel^2 * n^2 / h^(4/3)
        # Rectangular channel: Vel = Q / (B*h)
        # Sf = Q^2 * n^2 / (B^2 * h^(10/3))
        # Bernoulli: Energy = Zb + Dep + Btea*Vel^2/(2g)
        # h[i] + Beta*(Q^2/(2*g*B[i]^2))*h^(-2) - (Dx*Q^2*n^2/(2*B[i]^2))*h^(-10/3) - Energy[i+1]+z[i]-(Dx/2)*Sf[i+1] = 0
        # h[i] + A*h[i]^(-2) - B*h[i]^(-10/3) + C = 0
        
        # initial estimate
        ChanDep[XS] = (ChanDep[XS+1] + ChanElev[XS+1]) - ChanElev[XS] + S_f[XS+1]*ChanDx[XS] 
        
        # iterate solution for h
        Acoef = Beta * (Qin**2 / (2*Grav*ChanWidth[XS]**2))
        Bcoef = (ChanDx[XS] * Qin**2 * Roughness**2 / (2 * ChanWidth[XS]**2))
        Cconst = ChanElev[XS] - S_f[XS+1]*ChanDx[XS]/2 - Energy[XS+1] 
        DepErr = ChanDep[XS] + Acoef*ChanDep[XS]**(-2) - Bcoef*ChanDep[XS]**(-10/3) + Cconst
        CheckCount = 0
        while np.abs(DepErr) > Tol:
            Gradient = 1 - 2*Acoef*ChanDep[XS]**(-3) + (10/3)*Bcoef*ChanDep[XS]**(-13/3)
            ChanDep[XS] -= DepErr / Gradient
            DepErr = ChanDep[XS] + Acoef*ChanDep[XS]**(-2) - Bcoef*ChanDep[XS]**(-10/3) + Cconst
            CheckCount += 1
            assert CheckCount < MaxIt, 'Maximum iterations exceeded solving steady state water level'
        
        # Check for supercritical
        if ChanDep[XS] < CritDep[XS]:
            ChanDep[XS] = CritDep[XS]
            logging.warning('Steady state solution results in critical depth at XS%i' % XS)
        ChanVel[XS] = Qin / (ChanWidth[XS] * ChanDep[XS])
        Energy[XS] = ChanElev[XS] + ChanDep[XS] + Beta * ChanVel[XS]**2 / (2*Grav)
        S_f[XS] = ChanVel[XS]**2 * Roughness**2 / ChanDep[XS]**(4/3)
    
    return ChanDep, ChanVel

def solveFullPreissmann(z, B, LagArea, h, V, dx, dt, n, Q_Ts, DsWl_Ts, NumericalPars):
    """ Solve full S-V eqns for a rectangular channel using preissmann scheme.
        Uses a newton raphson solution to the preissmann discretisation of the 
        Saint-Venant equations.
        
        solveFullPreissmann(z, B, h, V, dx, dt, n, Q, DsWl, Theta, Tol, MaxIt)
        
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
        
        Modifies in place:
            h
            V
        
    """
    g = 9.81                    # gravity [m/s^2]
    
    Theta = NumericalPars['Theta']
    Beta = NumericalPars['Beta']
    Tol = NumericalPars['ErrTol']
    MaxIt = NumericalPars['MaxIt']
    dt = dt.seconds             # timestep for hydraulics [s]
    N = z.size                  # number of cross-sections
    S_0 = (z[:-1]-z[1:])/dx     # bed slope in each reach between XS [m/m]
    
    # Prevent unexpected errors
    h[h<=0] = 0.0001 # Prevent negative depths
    V[V==0] = 0.0001 # Prevent zero velocities
    
    # Calculate effective width for volume change i.e. total planform area/reach length. 
    dx2 = np.zeros(z.size)
    dx2[0] = dx[0]
    dx2[1:-1] = (dx[:-1]+dx[1:])/2
    dx2[-1] = dx[-1]
    Be = B + LagArea/dx2
    
    # Pre-compute some variables required within the loop
    A = h*B                     # area of flow at each XS [m^2]
    Sf = V*np.abs(V) * (n**2) / (h**(4/3)) # friction slope at each XS [m/m]
    
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
              - 2*(dt/dx)*(1-Theta) * (V_old[1:]*A_old[1:] - V_old[:-1]*A_old[:-1]))
        
        # For momentum equation
        C2 = (V_old[:-1]*A_old[:-1] + V_old[1:]*A_old[1:]
              -2*Beta*(dt/dx)*(1-Theta) * (V_old[1:]*np.abs(V_old[1:])*A_old[1:]
                                           - V_old[:-1]*np.abs(V_old[:-1])*A_old[:-1])
              - g*(dt/dx)*(1-Theta) * (A_old[1:] + A_old[:-1]) * (h_old[1:]-h_old[:-1])
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
                                         + 2*Beta*(dt/dx)*Theta * (V[1:]*np.abs(V[1:])*A[1:]
                                                                   - V[:-1]*np.abs(V[:-1])*A[:-1])
                                         + g*(dt/dx)*Theta * (A[1:] + A[:-1]) * (h[1:]-h[:-1])
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
                                                  - 2*Beta*(dt/dx)*Theta*V[:-1]**2*B[:-1]
                                                  + g*Theta*(dt/dx)*(-2*A[:-1] + B[:-1]*h[1:] - A[1:])
                                                  - g*dt*Theta*B[:-1]*(S_0+(1/3)*Sf[:-1]))
            # d/dV[0]
            a_banded[3,np.arange(1,2*(N)-2,2)] = (A[:-1] 
                                                  - 4*Beta*(dt/dx)*Theta*V[:-1]*A[:-1] 
                                                  + 2*g*dt*Theta*A[:-1]*Sf[:-1]/V[:-1])
            # d/dh[1]
            a_banded[2,np.arange(2,2*(N),2)] = (V[1:]*B[1:] 
                                                + 2*Beta*(dt/dx)*Theta*V[1:]**2*B[1:]
                                                + g*Theta*(dt/dx)*(2*A[1:] - B[1:]*h[:-1] + A[:-1])
                                                - g*dt*Theta*B[1:]*(S_0+(1/3)*Sf[1:]))
            # d/dV[1]
            a_banded[1,np.arange(3,2*(N),2)] = (A[1:] 
                                                + 4*Beta*dt/dx*Theta*V[1:]*A[1:] 
                                                + 2*g*dt*Theta*A[1:]*Sf[1:]/V[1:])
            
            # Ds Bdy condition derivatives
            a_banded[2,2*N-1] = 0
            #a_banded[2,2*N-1] = V[N-1]/g
            a_banded[3,2*N-2] = 1
            
            # Solve the banded matrix
            try:
                Delta = linalg.solve_banded([2,2],a_banded,Err)
            except linalg.LinAlgError as ErrMsg:
                logging.warning('LinAlgError %s - adding some noise to see if this fixes it...' % ErrMsg)
                a_banded += np.random.random(a_banded.shape) * 1e-7 - 5e-8
            
            # Update h & V
            h -= Delta[np.arange(0,2*N,2)]
            V -= Delta[np.arange(1,2*N,2)]
            
            # Prevent unexpected errors
            h[h<=0] = 0.0001 # Prevent negative depths
            V[V==0] = 0.0001 # Prevent zero velocities
            
            # Update Sf and A
            Sf = V*np.abs(V) * (n**2) / (h**(4/3))
            A = h*B
            
            # Check if solution is within tolerance
    #        if np.sum(np.abs(Delta)) < Tol:
    #            break
            if np.all(np.abs(Delta) < Tol):
                break
            
            # Stability warnings
            if np.amax(np.abs(Delta))>NumericalPars['WarnTol']:
                WarnIx = np.argmax(np.abs(Delta))
                WarnNode = np.floor(WarnIx/2)
                if WarnIx % 2 == 0:
                    WarnVar = 'Depth'
                else:
                    WarnVar = 'Velocity'
                logging.warning('%s change in cross-section %i equals %f (greater than WarnTol)',
                                WarnVar, WarnNode, Delta[WarnIx])            
            ItCount += 1
        
        assert ItCount < MaxIt, 'Max iterations exceeded.'

def assembleChannel(ShoreX, ShoreY, LagoonElev, OutletElev, 
                    OutletEndX, OutletEndWidth, OutletEndElev, 
                    RiverElev, RiverWidth, RivDep, RivVel, 
                    LagoonWL, LagoonVel, OutletDep, OutletVel,
                    OutletEndDep, OutletEndVel, Dx, MaxOutletElev):
    """ Combine river, lagoon and outlet into single channel for hyd-calcs
        
        (ChanDx, ChanElev, ChanWidth, LagArea, ChanDep, ChanVel, 
         OnlineLagoon, OutletChanIx, ChanFlag) = \
            riv.assembleChannel(ShoreX, ShoreY, LagoonElev, OutletElev, 
                                OutletEndX, OutletEndWidth, OutletEndElev, 
                                RiverElev, RiverWidth, RivDep, RivVel, 
                                LagoonWL, LagoonVel, OutletDep, OutletVel, 
                                OutletEndDep, OutletEndVel, Dx, MaxOutletElev)
        
        Returns:
            ChanDx = distance between each cross-section in combined channel\
            ChanElev = Bed level at each cross-section
            ChanWidth = Width of each cross-section
            LagArea = Surface area of offline lagoon storage attached to each 
                      node (all zero except for first and last 'lagoon'
                      cross-section).
            ChanDep = Water depth in each cross-section
            ChanVel = Water velocity in each cross-section
            OnlineLagoon = indices of lagoon transects which are 'online'. Note
                           that the indices are given in the order in which the 
                           lagoon sections appear in the combined channel. i.e.
                           if the outlet is offset to the left the lagoon 
                           transect indices in OnlineLagoon will be decreasing.
            OutletChanIx = indices of outlet channel transects which are 
                           'online' - similar to OnlineLagoon
            ChanFlag = Flags showing the origin of the different cross-sections
                       making up the outlet channel. 0 = river, 1 = lagoon,
                       2 = outlet channel ends, 3 = outlet channel.
    """
    
    # Find location and orientation of outlet channel
    if OutletEndX[0]//Dx == OutletEndX[1]//Dx:
        # Outlet doesn't cross any transects
        if OutletEndX[0] < OutletEndX[1]:
            OutletChanIx = np.where(np.logical_and(OutletEndX[0] < ShoreX, 
                                                   ShoreX < OutletEndX[1]+Dx))[0]
        else:
            OutletChanIx = np.where(np.logical_and(OutletEndX[1]-Dx < ShoreX,
                                                   ShoreX < OutletEndX[0]))[0]
    elif OutletEndX[0] < OutletEndX[1]:
        # Outlet angles from L to R
        OutletChanIx = np.where(np.logical_and(OutletEndX[0] < ShoreX, 
                                               ShoreX < OutletEndX[1]))[0]
    else:
        # Outlet from R to L
        OutletChanIx = np.flipud(np.where(np.logical_and(OutletEndX[1] < ShoreX,
                                                         ShoreX < OutletEndX[0]))[0])
    OutletWidth = ShoreY[OutletChanIx,1] - ShoreY[OutletChanIx,2]
    
    # TODO: Account for potentially dry parts of the lagoon when 
    #       calculating ChanArea
    
    # Calculate properties for the 'online' section of lagoon
    LagoonWidth = ShoreY[:,3] - ShoreY[:,4]
    if OutletEndX[0] > 0:
        # Outlet channel to right of river
        OnlineLagoon = np.where(np.logical_and(0 <= ShoreX, ShoreX <= OutletEndX[0]))[0]
        StartArea = np.nansum(LagoonWidth[ShoreX < 0] * Dx)
        EndArea = np.nansum(LagoonWidth[ShoreX > OutletEndX[0]] * Dx)
    else:
        # Outlet channel to left of river
        OnlineLagoon = np.flipud(np.where(np.logical_and(0 >= ShoreX, ShoreX >= OutletEndX[0]))[0])
        StartArea = np.nansum(LagoonWidth[ShoreX > 0] * Dx)
        EndArea = np.nansum(LagoonWidth[ShoreX < OutletEndX[0]] * Dx)
    
    # Assemble the complete channel
    ChanDx = np.tile(Dx, RiverElev.size + OnlineLagoon.size + OutletChanIx.size + 2)
    if OutletEndX[0]//Dx == OutletEndX[1]//Dx:
        # Outlet is straight (i.e. doesn't cross any transects)
        ChanDx[-3] += abs(OutletEndX[1] - OutletEndX[0])
    elif OutletEndX[0] < OutletEndX[1]:
        # Outlet angles from L to R
        ChanDx[RiverElev.size + OnlineLagoon.size] += Dx - (OutletEndX[0] % Dx)
        ChanDx[-2] += OutletEndX[1] % Dx
    else:
        # Outlet from R to L
        ChanDx[RiverElev.size + OnlineLagoon.size] += OutletEndX[0] % Dx
        ChanDx[-2] += Dx - (OutletEndX[1] % Dx)
    ChanFlag = np.concatenate([np.full(RiverElev.size, 0), 
                               np.full(OnlineLagoon.size, 1), 
                               [2], np.full(OutletChanIx.size, 3), [2,4]])
    ChanElev = np.concatenate([RiverElev, LagoonElev[OnlineLagoon], 
                               [OutletEndElev[0]], OutletElev[OutletChanIx], 
                               [OutletEndElev[1], min(MaxOutletElev, OutletEndElev[1])]])
    ChanWidth = np.concatenate([np.tile(RiverWidth, RiverElev.size), 
                                LagoonWidth[OnlineLagoon], [OutletEndWidth[0]],
                                OutletWidth, np.full(2, OutletEndWidth[1])])
    LagArea = np.zeros(ChanElev.size)
    LagArea[RiverElev.size] = StartArea
    LagArea[-(OutletChanIx.size+3)] = EndArea
    
    # Assemble the hydraulic initial conditions
    ChanDep = np.concatenate([RivDep,
                              LagoonWL[OnlineLagoon]-LagoonElev[OnlineLagoon],
                              [OutletEndDep[0]], 
                              OutletDep[OutletChanIx], 
                              OutletEndDep[-2:]])
    # If depth is missing then interpolate it
    DepNan = np.isnan(ChanDep)
    if np.any(DepNan):
        logging.debug('Interpolating %i missing channel depths' % np.sum(DepNan))
        ChanDep[DepNan] = np.interp(np.where(DepNan)[0], np.where(~DepNan)[0], ChanDep[~DepNan])
    
    ChanVel = np.concatenate([RivVel,
                              LagoonVel[OnlineLagoon],
                              [OutletEndVel[0]], 
                              OutletVel[OutletChanIx], 
                              OutletEndVel[-2:]])
    # If vel is missing then interpolate it (based on flow rather than vel)
    VelNan = np.isnan(ChanVel)
    if np.any(VelNan):
        logging.debug('Interpolating %i missing channel velocities' % np.sum(VelNan))
        ChanQ = ChanDep*ChanVel
        ChanVel[VelNan] = (np.interp(np.where(DepNan)[0], np.where(~DepNan)[0], ChanQ[~DepNan])
                           / ChanDep[VelNan])
    
    return (ChanDx, ChanElev, ChanWidth, LagArea, ChanDep, ChanVel, 
            OnlineLagoon, OutletChanIx, ChanFlag)

def storeHydraulics(ChanDep, ChanVel, OnlineLagoon, OutletChanIx, ChanFlag, 
                    LagoonElev):
    """ Extract lagoon/outlet hydraulic conditions.
        
        (LagoonWL, LagoonVel, OutletDep, OutletVel, 
         OutletEndDep, OutletEndVel) = \
            storeHydraulics(ChanDep, ChanVel, OnlineLagoon, OutletChanIx, 
                            ChanFlag, LagoonElev)
        
        All the hydraulics calculations are carried out on a merged channel
        which includes the river upstream of the lagoon, the onluine part of
        lagoon itself, and the outlet channel. This function extracts the 
        current hydraulics in the lagoon and outlet channel and stores them in 
        shore parrallel model schematisation. This is necessary to allow the 
        model to preserve initial conditions for the hydraulics solution
        while the outlet channel changes position/length etc.
    """
    
    # Lagoon WL
    LagoonWL = LagoonElev.copy()
    LagoonWL[OnlineLagoon] += ChanDep[ChanFlag==1]
    MinOnline = np.min(OnlineLagoon)
    MaxOnline = np.max(OnlineLagoon)
    LagoonWL[:MinOnline] = LagoonWL[MinOnline]
    LagoonWL[MaxOnline+1:] = LagoonWL[MaxOnline]
    
    # Lagoon Vel
    LagoonVel = np.full(LagoonElev.size, np.nan)
    LagoonVel[OnlineLagoon] = ChanVel[ChanFlag==1]
    
    # Outlet Depth
    OutletDep = np.full(LagoonElev.size, np.nan)
    OutletDep[OutletChanIx] = ChanDep[ChanFlag==3]
    
    # Outlet Vel
    OutletVel = np.full(LagoonElev.size, np.nan)
    OutletVel[OutletChanIx] = ChanVel[ChanFlag==3]
    
    # Depth and vel at ends of outlet channel
    EndNodes = np.logical_or(ChanFlag==2, ChanFlag==4)
    OutletEndDep = ChanDep[EndNodes]
    OutletEndVel = ChanVel[EndNodes]
    
    return (LagoonWL, LagoonVel, OutletDep, OutletVel, 
            OutletEndDep, OutletEndVel)
    
def calcBedload(z, B, h, V, dx, PhysicalPars):
    """ Calculate bedload transport using Bagnold 1980 streampower approach
    
        Uses a central weighting approach
        
        Parameters:
            z = bed level at each cross-section [m]
            B = cross-section width at each cross-section [m]
            h = water depth at each cross-section (m)
            V = velocity (average) at each cross-section [m/s]
            dx = distance-to-next at each cross section (note, the array dx is
                 one smaller than z, B, etc) [m]
            PhysicalPars = dict of model input parameters
        
        Returns:
            Qs = numpy array of bedload transport at each cross-section 
                 [m3(bulk including voids)/s]
        
        Bagnold R.A. (1980) An empirical correlation of bedload transport rates
        in flumes and natural rivers. Proc R Soc Lond A Math Phys Sci 
        372(October):453–473. http://www.jstor.org/stable/2397042
    """
    
    Rho = PhysicalPars['RhoRiv']
    RhoS = PhysicalPars['RhoSed']
    g = PhysicalPars['Gravity']
    D = PhysicalPars['GrainSize']
    VoidRatio = PhysicalPars['VoidRatio']
    
    # Use friction slope rather than actual energy or water surface slope as it is more reliable
    S = V**2 * PhysicalPars['Roughness']**2 / h**(4/3)
    
    # Threshold streampower per unit width [kg/m/s]
    omega_0 = 5.75*(0.04*(RhoS - Rho))**(3/2) * (g/Rho)**(1/2) * D**(3/2) * np.log10(12*h/D)
    # TODO move constant part of above line into loadmod (i.e. out of loop)
    
    # streampower per unit width [kg/m/s]
    omega = Rho * S * V * h 
    
    # bedload transport rate at each node (bulk volume accounting for voids) [m^3/s]
    Qs_node = RhoS/(RhoS-Rho) * 0.01 * (np.maximum(omega-omega_0,0.0)/0.5)**(3/2) * (h/0.1)**(-2/3) * (D/0.0011)**(-1/2) * B / (RhoS*(1-VoidRatio)) 
    # TODO move constant part of above line into loadmod (i.e. out of loop)
    Qs_node[V<0] *= -1
    
    # bedload transport in each reach (central weighting)
    Qs_reach = (Qs_node[:-1] + Qs_node[1:])/2
    
    return(Qs_reach)
    