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

def solveFullExplicitChar(z, B, LagArea, LagLen, Closed, h, V, 
                          dx, dt, Q_Ts, DsWl_Ts, PhysicalPars):
    g = PhysicalPars['Gravity']
    n = PhysicalPars['RoughnessManning']
    dt = dt.seconds          # timestep for hydraulics [s]
    
    S_0 = np.pad((z[:-2]-z[2:])/(dx[:-1]+dx[1:]), pad_width=1, mode='edge')  # bed slope at each cross-section
    
    DsWl = z[-1] + h[-1] # initial downstream WL
    C = np.sqrt(g*h)     # celerity
    
    import hapuamod.visualise as vis
    LongSecFig = vis.longSection(dx, z, B, h, V)
    
    # Main timestepping loop
    for StepNo in range(Q_Ts.shape[0]):
        V_old = V.copy()
        C_old = C.copy()
        h_old = h.copy()
        
        Q = Q_Ts[StepNo]
        DsWl = DsWl_Ts[StepNo]
        
        Sf = V*np.abs(V) * (n**2) / ((C**2/g)**(4/3)) # friction slope at each XS [m/m]
        
        DeltaX_L = dt*(V+C)
        DeltaX_R = -dt*(V-C)
        
        V_L = V[1:] - (DeltaX_L[1:] / dx) * (V[1:] - V[:-1])
        V_R = V[:-1] - (DeltaX_R[1:] / dx) * (V[:-1] - V[1:])
        
        C_L = C[1:] - (DeltaX_L[1:] / dx) * (C[1:] - C[:-1])
        C_R = C[:-1] - (DeltaX_R[1:] / dx) * (C[:-1] - C[1:])
        
        # Solve for model interior
        V[1:-1] = (V_L[:-1] + V_R[1:]) / 2 + (C_L[:-1] - C_R[1:]) + dt*g*(S_0[1:-1]-Sf[1:-1])
        C[1:-1] = (V_L[:-1] + V_R[1:]) / 4 + (C_L[:-1] + C_R[1:]) / 2
        
        # Add downstream bdy conditions
        C[-1] = np.sqrt((DsWl-z[-1])*g)
        V[-1] = V_L[-1] + 2*(C_L[-1] - C[-1] + dt*g*(S_0[-1]-Sf[-1]))
        
        # Add upstream boundary conditions (involves solving a polynomial)
        C_Roots = np.roots([2, V_R[0] - 2*C_R[0] + dt*g*(S_0[0]-Sf[0]), 0, Q*g/B[0]])
        C[0] = C_Roots.real[abs(C_Roots.imag)<1e-5][0]
        V[0] = Q*g / (B[0] * C[0]**2)
        
        h = C**2 / g
        vis.updateLongSection(LongSecFig, dx, z, B, h, V)

def solveFullExplicit(z, B, LagArea, LagLen, Closed, h, V, 
                      dx, dt, Q_Ts, DsWl_Ts, PhysicalPars):
    """ Solve full S-V eqns for a rectangular channel using explicit scheme.
        
        solveFullExplicit(z, B, LagArea, ChanFlag, Closed, h, V, dx, dt, 
                          Q_Ts, DsWl_Ts, NumericalPars, PhysicalPars))
        
        Parameters:
            z (np.ndarray(float64)): Bed elevation at each node (m)
            B (np.ndarray(float64)): Channel width at each node (m)
            LagArea(np.ndarray(float64)): Offline area connected to each node (m2)
            LagLen(np.ndarray(float64)): Length of lagoon connected to each 
                node - used for barrier seepage calculation (m)
            Closed (boolean): Flag indicating zero flow downstream boundary
                condition.
            h (np.ndarray(float64)): Depth at each node at the last timestep (m)
            V (np.ndarray(float64)): Mean velocity at each node at the last 
                timestep (m)
            dx (np.ndarray(float64)): Length of each reach between nodes (m)
                note the dx array should be one shorter than for z, B, etc
            dt (pd.timedelta): timestep
            
            Qin (np.ndarray(float64)): List (as np.array) of inflows at 
                upstream end of channel corresponding to each timestep. 
                Function will loop over each inflow in turn (m^3/s)
            DsWl (np.ndarray(float64)): List (as np.array) of downstream 
                boundary water levels at upstream end of channel corresponding 
                to each timestep. Function will loop over each inflow in turn 
                (m^3/s)
            PhysicalPars (dict): Physical parameters including:
                Gravity (float): gravity [m/s2]
                RoughnessManning (float): mannings 'n' roughness
                Barrier Permeability (float): permeability of the gravel 
                    barrier in m3/s, per m length of barrier, per m head 
                    difference accross barrier [m/s].
        
        Modifies in place:
            h
            V
        
    """
    
    # TODO: Handle closed condition
    # TODO: Handle barrier permeability
    # TODO: Handle effective width / offline storage
    
    g = PhysicalPars['Gravity']
    n = PhysicalPars['RoughnessManning']
    dt = dt.seconds          # timestep for hydraulics [s]
    # P = PhysicalPars['BarrierPermeability']
    
    dtdx = dt/dx
    
    S_0 = (z[:-1]-z[1:])/dx  # bed slope in each reach between XS [m/m]
    
    DsWl = z[-1] + h[-1] # initial downstream WL
    import hapuamod.visualise as vis
    LongSecFig = vis.longSection(dx, z, B, h, V)
    
    # Main timestepping loop
    for StepNo in range(Q_Ts.shape[0]):
        Q = Q_Ts[StepNo]
        DsWl = DsWl_Ts[StepNo]
        Sf = V*np.abs(V) * (n**2) / (h**(4/3)) # friction slope at each XS [m/m]
        
        V_old = V
        h_old = h
        
        h[-1] = DsWl - z[-1]
        h[:-1] = h_old[:-1] - dtdx  * (h_old[1:] * V_old[1:] * (B[1:]/B[:-1])  - h_old[:-1] * V_old[:-1])
        V[0] = Q / (B[0] * h[0])
        V[1:] = (V_old[1:] 
                 - dtdx/2 * (V_old[1:] - V_old[:-1]) 
                 - g*dtdx * (h_old[1:] - h_old[:-1])
                 + g*dt * (S_0 - Sf[1:]))
        
        
        
        vis.updateLongSection(LongSecFig, dx, z, B, h, V)

def solveFullPreissmann(z, B, LagArea, LagLen, Closed, h, V, 
                        dx, dt, Q_Ts, DsWl_Ts, NumericalPars, PhysicalPars):
    """ Solve full S-V eqns for a rectangular channel using preissmann scheme.
        Uses a newton raphson solution to the preissmann discretisation of the 
        Saint-Venant equations.
        
        solveFullPreissmann(z, B, LagArea, ChanFlag, Closed, h, V, dx, dt, 
                            Q_Ts, DsWl_Ts, NumericalPars, PhysicalPars))
        
        Parameters:
            z (np.ndarray(float64)): Bed elevation at each node (m)
            B (np.ndarray(float64)): Channel width at each node (m)
            LagArea(np.ndarray(float64)): Offline area connected to each node (m2)
            LagLen(np.ndarray(float64)): Length of lagoon connected to each 
                node - used for barrier seepage calculation (m)
            Closed (boolean): Flag indicating zero flow downstream boundary
                condition.
            h (np.ndarray(float64)): Depth at each node at the last timestep (m)
            V (np.ndarray(float64)): Mean velocity at each node at the last 
                timestep (m)
            dx (np.ndarray(float64)): Length of each reach between nodes (m)
                note the dx array should be one shorter than for z, B, etc
            dt (pd.timedelta): timestep
            
            Qin (np.ndarray(float64)): List (as np.array) of inflows at 
                upstream end of channel corresponding to each timestep. 
                Function will loop over each inflow in turn (m^3/s)
            DsWl (np.ndarray(float64)): List (as np.array) of downstream 
                boundary water levels at upstream end of channel corresponding 
                to each timestep. Function will loop over each inflow in turn 
                (m^3/s)
            NumericalPars (dict): Numerical parameters including:
                Theta (float): temporal weighting coefficient for preissmann scheme
                Tol (float): error tolerance (both m and m/s)
                MaxIt (integer): maximum number of iterations to find solution
            PhysicalPars (dict): Physical parameters including:
                Gravity (float): gravity [m/s2]
                RoughnessManning (float): mannings 'n' roughness
                Barrier Permeability (float): permeability of the gravel 
                    barrier in m3/s, per m length of barrier, per m head 
                    difference accross barrier [m/s].
        
        Modifies in place:
            h
            V
        
    """
    g = PhysicalPars['Gravity']
    n = PhysicalPars['RoughnessManning']
    P = PhysicalPars['BarrierPermeability']
    
    Theta = NumericalPars['Theta']
    Beta = NumericalPars['Beta']
    Tol = NumericalPars['ErrTol']
    MaxIt = NumericalPars['MaxIt']
    dt = dt.seconds          # timestep for hydraulics [s]
    N = z.size               # number of cross-sections
    S_0 = (z[:-1]-z[1:])/dx  # bed slope in each reach between XS [m/m]
    
    # Prevent unexpected errors
    h[h<=0] = 0.0001 # Prevent negative depths
    V[V==0] = 0.0001 # Prevent zero velocities
    
    # Calculate effective width of each XS for volume change i.e. total planform area/reach length. 
    dx2 = np.zeros(z.size)
    dx2[0] = dx[0]
    dx2[1:-1] = (dx[:-1]+dx[1:])/2
    dx2[-1] = dx[-1]
    Be = B + LagArea/dx2
    
    # Calculate ratio of length over which seepage occurs, to length of reach, for each reach
    SeepLen = (LagLen[:-1]+LagLen[1:]) / (2*dx)
    
    # Pre compute some constant values which get used a lot 
    # (to save having to calculate them in the loop)
    dt_P_Theta = dt*P*Theta
    g_dt_Theta = g*dt*Theta
    dtdx = dt/dx
    Theta1 = 1-Theta
    
    # Pre-compute some variables required for the start of the first timestep of the loop
    A = h*B                                # area of flow at each XS [m^2]
    Sf = V*np.abs(V) * (n**2) / (h**(4/3)) # friction slope at each XS [m/m]
    DsWl = z[-1] + h[-1]
    
    # Main timestepping loop
    for StepNo in range(Q_Ts.shape[0]):
        Q = Q_Ts[StepNo]
        DsWl_old = DsWl
        DsWl = DsWl_Ts[StepNo]
        
        V_old = V
        h_old = h
        Sf_old = Sf
        A_old = A
        
        # Constant parts of the S-V Equations which can be computed outside the loop
        # For continuity equation
        C1 = (h_old[1:]*Be[1:] + h_old[:-1]*Be[:-1]
              - 2*(dtdx)*(1-Theta) * (V_old[1:]*A_old[1:] - V_old[:-1]*A_old[:-1])
              - SeepLen*dt*P*(z[1:] + z[:-1] 
                              - 2*Theta * DsWl 
                              + Theta1 * (h_old[1:] + h_old[:-1] - 2*DsWl_old)))
        
        # For momentum equation
        C2 = (V_old[:-1]*A_old[:-1] + V_old[1:]*A_old[1:]
              -2*Beta*(dtdx)*Theta1 * (V_old[1:]*np.abs(V_old[1:])*A_old[1:]
                                       - V_old[:-1]*np.abs(V_old[:-1])*A_old[:-1])
              - g*(dtdx)*Theta1 * (A_old[1:] + A_old[:-1]) * (h_old[1:]-h_old[:-1])
              + g*dt*Theta1 * (A_old[1:]*(S_0 - Sf_old[1:]) 
                               + A_old[:-1]*(S_0 - Sf_old[:-1])))
        
        # Iterative solution
        ItCount = 0
        Err = np.zeros(2*N)
        while ItCount < MaxIt:
            
            # Error in Us Bdy
            Err[0] = A[0]*V[0]-Q
            
            # Error in continuity equation
            Err[np.arange(1,2*N-1,2)] = (h[1:]*Be[1:] + h[:-1]*Be[:-1]
                                         + 2*dtdx*Theta * (V[1:]*A[1:] - V[:-1]*A[:-1])
                                         + SeepLen * dt_P_Theta * (h[1:] + h[:-1]) 
                                         - C1)
            
            # Error in momentum equation
            Err[np.arange(2,2*N-1,2)] = (V[:-1]*A[:-1] + V[1:]*A[1:]
                                         + 2*Beta*dtdx*Theta * (V[1:]*np.abs(V[1:])*A[1:]
                                                                   - V[:-1]*np.abs(V[:-1])*A[:-1])
                                         + g*(dtdx)*Theta * (A[1:] + A[:-1]) * (h[1:]-h[:-1])
                                         - g_dt_Theta * (A[1:]*(S_0-Sf[1:]) + A[:-1]*(S_0-Sf[:-1]))
                                         - C2)
            
            # Error in Ds Bdy
            if Closed:
                Err[-1] = h[-1] * V[-1]
            else:
                Err[-1] = z[-1] + h[-1] - DsWl
            
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
            a_banded[3,np.arange(0,2*(N)-2,2)] = (Be[:-1] 
                                                  - 2*dtdx*Theta*V[:-1]*B[:-1] 
                                                  + SeepLen*dt_P_Theta)
            # d/dV[0]
            a_banded[2,np.arange(1,2*(N)-2,2)] = -2*dtdx*Theta*A[:-1]
            # d/dh[1]
            a_banded[1,np.arange(2,2*(N),2)]   = (Be[1:] 
                                                  + 2*dtdx*Theta*V[1:]*B[1:] 
                                                  + SeepLen*dt_P_Theta)
            # d/dV[1]
            a_banded[0,np.arange(3,2*(N),2)]   = 2*dtdx*Theta*A[1:]
            
            # Momentum equation derivatives
            # d/dh[0]
            a_banded[4,np.arange(0,2*(N)-2,2)] = (V[:-1]*B[:-1] 
                                                  - 2*Beta*dtdx*Theta*abs(V[:-1])*V[:-1]*B[:-1]
                                                  + g*Theta*dtdx*(-2*A[:-1] + B[:-1]*h[1:] - A[1:])
                                                  - g_dt_Theta*B[:-1]*(S_0+(1/3)*Sf[:-1]))
            # d/dV[0]
            a_banded[3,np.arange(1,2*(N)-2,2)] = (A[:-1] 
                                                  - 4*Beta*dtdx*Theta*V[:-1]*A[:-1] 
                                                  + 2*g_dt_Theta*A[:-1]*Sf[:-1]/V[:-1])
            # d/dh[1]
            a_banded[2,np.arange(2,2*(N),2)]   = (V[1:]*B[1:] 
                                                  + 2*Beta*dtdx*Theta*abs(V[1:])*V[1:]*B[1:]
                                                  + g*Theta*dtdx*(2*A[1:] - B[1:]*h[:-1] + A[:-1])
                                                  - g_dt_Theta*B[1:]*(S_0+(1/3)*Sf[1:]))
            # d/dV[1]
            a_banded[1,np.arange(3,2*(N),2)]   = (A[1:] 
                                                  + 4*Beta*dtdx*Theta*V[1:]*A[1:] 
                                                  + 2*g_dt_Theta*A[1:]*Sf[1:]/V[1:])
            
            # Ds Bdy condition derivatives
            if Closed:
                a_banded[2,2*N-1] = h[-1]
                a_banded[3,2*N-2] = V[-1]
            else:
                a_banded[2,2*N-1] = 0
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
        
        if ItCount >= MaxIt:
            logging.warning('Max iterations exceeded in unsteady river hydraulics - reverting to steady solution for this timestep')
            (h, V) = solveSteady(dx, z, B, n, Q_Ts[-1], DsWl_Ts[-1], NumericalPars)

def assembleChannel(ShoreX, ShoreY, ShoreZ, 
                    OutletEndX, OutletEndWidth, OutletEndElev, 
                    Closed, RiverElev, RivDep, RivVel, 
                    LagoonWL, LagoonVel, OutletDep, OutletVel,
                    OutletEndDep, OutletEndVel, Dx, PhysicalPars):
    """ Combine river, lagoon and outlet into single channel for hyd-calcs
        
        (ChanDx, ChanElev, ChanWidth, LagArea, LagLen, ChanDep, ChanVel, 
         OnlineLagoon, OutletChanIx, ChanFlag, Closed) = \
            riv.assembleChannel(ShoreX, ShoreY, ShoreZ, 
                                OutletEndX, OutletEndWidth, OutletEndElev, 
                                Closed, RiverElev, RivDep, RivVel, 
                                LagoonWL, LagoonVel, OutletDep, OutletVel, 
                                OutletEndDep, OutletEndVel, Dx, PhysicalPars)
        
        Returns:
            ChanDx: distance between each cross-section in combined channel (m)
            ChanElev: Bed level at each cross-section (m)
            ChanWidth: Width of each cross-section (m)
            LagArea: Surface area of offline lagoon storage attached to each 
                node - all zero except for first and last 'lagoon'
                cross-section (m2)
            LagLen: Length of lagoon connected to node - used to calculate 
                barrier seepage (m)
            ChanDep: Water depth in each cross-section (m)
            ChanVel: Water velocity in each cross-section (m/s)
            OnlineLagoon: indices of lagoon transects which are 'online'. Note
                that the indices are given in the order in which the lagoon 
                sections appear in the combined channel. i.e. if the outlet 
                is offset to the left the lagoon transect indices in 
                OnlineLagoon will be decreasing.
            OutletChanIx: indices of outlet channel transects which are 
                'online' - similar to OnlineLagoon
            ChanFlag: Flags showing the origin of the different cross-sections
                making up the outlet channel. 0 = river, 1 = lagoon,
                2 = outlet channel, 3 = outlet channel end, 4 = dummy XS in sea.
            Closed (boolean): Is the channel closed or open at its downstream 
                end?
    """
    
    X0Ix = np.where(ShoreX==0)[0][0] # Index of shore transect where X=0 (i.e. inline with river)
    
    # Handle the postprocessing situation when we don't have (or need) a dummy XS in the sea
    if OutletEndDep.size == 1:
        OutletEndDep = np.append(OutletEndDep, np.nan)
    if OutletEndVel.size == 1:
        OutletEndVel = np.append(OutletEndVel, np.nan)
    
    if not Closed:
        # Find location and orientation of outlet channel
        if OutletEndX[0] < OutletEndX[1]:
            # Outlet angles from L to R
            OutletChanIx = np.where(np.logical_and(OutletEndX[0] <= ShoreX, 
                                                   ShoreX <= OutletEndX[1]))[0]
        else:
            # Outlet from R to L
            OutletChanIx = np.flipud(np.where(np.logical_and(OutletEndX[1] <= ShoreX,
                                                             ShoreX <= OutletEndX[0]))[0])
        OutletWidth = ShoreY[OutletChanIx,1] - ShoreY[OutletChanIx,2]
        OutletWidth[np.isnan(OutletWidth)] = 0.
    
        # Check if closure has occured in outlet channel (only if not already closed from previous timestep).
        if np.any(OutletWidth<=PhysicalPars['MinOutletWidth']) | (OutletEndWidth<=PhysicalPars['MinOutletWidth']):
            Closed = True
            for ClosedIx in OutletChanIx[OutletWidth<=PhysicalPars['MinOutletWidth']]:
                logging.info('Outlet channel closed by wave washover at X = %f' % ShoreX[ClosedIx])
            if OutletEndWidth <= PhysicalPars['MinOutletWidth']:
                logging.info('Downstream end of outlet channel closed by longshore transport (EndX = %f)' % OutletEndX[1])
            OutletChanIx = np.empty(0)
        elif np.min(OutletWidth) < PhysicalPars['MinOutletWidth'] * 3:
            logging.debug('Narrow outlet, min outlet width = %f' % (np.min(OutletWidth)))
        
    else: # Outlet already closed in previous timestep and not reopened by mor
        OutletChanIx = np.empty(0)
    
    # TODO: Account for potentially dry parts of the lagoon when 
    #       calculating ChanArea
    
    # Calculate properties for the 'online' section of lagoon
    LagoonWidth = ShoreY[:,3] - ShoreY[:,4]
    assert LagoonWidth[X0Ix] > 0, 'Zero/negtive lagoon width at X=0'
    assert np.all(~np.isnan(LagoonWidth)), 'NaN values in lagoon width at X = %s' % ShoreX[np.isnan(LagoonWidth)]
    if OutletEndX[0] == 0:
        # Outlet directly inline with river (special case to ensure lagoon always has at least 1 XS)
        if Closed:
            OnlineLagoon = np.arange(X0Ix, np.where(LagoonWidth > 0)[0][-1] + 1)
            EndArea = 0
            EndLen = 0
        else:
            OnlineLagoon = np.array([X0Ix])
            EndArea = np.nansum(LagoonWidth[ShoreX > OutletEndX[0]] * Dx)
            EndLen = np.nansum((LagoonWidth[ShoreX > OutletEndX[0]] > 0) * Dx)
        StartArea = np.nansum(LagoonWidth[X0Ix+1:] * Dx)
        StartLen = np.nansum((LagoonWidth[X0Ix+1:] > 0) * Dx)
    elif OutletEndX[0] > 0:
        # Outlet channel to right of river
        if Closed:
            OnlineLagoon = np.arange(X0Ix, np.where(LagoonWidth > 0)[0][-1] + 1)
            EndArea = 0
            EndLen = 0
        else:
            OnlineLagoon = np.arange(X0Ix, np.where(ShoreX < OutletEndX[0])[0][-1] + 1)
            EndArea = np.nansum(LagoonWidth[ShoreX > OutletEndX[0]] * Dx)
            EndLen = np.nansum((LagoonWidth[ShoreX > OutletEndX[0]] > 0) * Dx)
        StartArea = np.nansum(LagoonWidth[:X0Ix] * Dx)
        StartLen = np.nansum((LagoonWidth[:X0Ix] > 0) * Dx)
    else:
        # Outlet channel to left of river
        if Closed:
            OnlineLagoon = np.arange(X0Ix, np.where(LagoonWidth > 0)[0][0] - 1, -1)
            EndArea = 0
            EndLen = 0
        else:
            OnlineLagoon = np.arange(X0Ix, np.where(ShoreX > OutletEndX[0])[0][0] - 1, -1)
            EndArea = np.nansum(LagoonWidth[ShoreX < OutletEndX[0]] * Dx)
            EndLen = np.nansum((LagoonWidth[ShoreX < OutletEndX[0]] > 0) * Dx)
        StartArea = np.nansum(LagoonWidth[X0Ix+1:] * Dx)
        StartLen = np.nansum((LagoonWidth[X0Ix+1:] > 0) * Dx)
    
    # Check there is at least 1 transect in lagoon
    # (zero length lagoon should now be prevented in mor)
    assert OnlineLagoon.size > 0, 'No online transects in lagoon'
    
    # Check the lagoon hasn't closed anywhere
    # note this can happen:
    #   - when the outlet is open (i.e. causing closure of the outlet)
    #   - or if the outlet is already closed somewhere else (it which case it just reduces the length of lagoon which is "online" in the closed lagoon...)
    if np.any(LagoonWidth[OnlineLagoon] < PhysicalPars['MinOutletWidth']):
        LagCloseIx = np.where(LagoonWidth[OnlineLagoon] < PhysicalPars['MinOutletWidth'])[0][0]
        if not Closed:
            logging.info('Closure occured in lagoon at X=%.0f', ShoreX[OnlineLagoon[LagCloseIx]])
            Closed = True
        EndArea += np.nansum(LagoonWidth[OnlineLagoon[LagCloseIx:]] * Dx)
        EndLen += np.nansum((LagoonWidth[OnlineLagoon[LagCloseIx:]] > 0) * Dx)
        OnlineLagoon = OnlineLagoon[:LagCloseIx]
    
    # These asserts *should* now be impossible to breach...
    assert np.abs(OnlineLagoon[-1]-OnlineLagoon[0]) == (OnlineLagoon.size-1), 'Gap in online lagoon (possibly split lagoon and closed?) need to handle this. OnlineLagoon = %s' % OnlineLagoon
    assert np.all(LagoonWidth[OnlineLagoon]>0), 'Lagoon closure in assembleChannel at X = %f' % ShoreX[OnlineLagoon[LagoonWidth[OnlineLagoon]<=0][0]]
    
    # Assemble the complete channel
    if Closed:
        ChanDx = np.full(RiverElev.size + OnlineLagoon.size - 1, Dx)
        ChanFlag = np.concatenate([np.full(RiverElev.size, 0), 
                                   np.full(OnlineLagoon.size, 1)])
        ChanElev = np.concatenate([RiverElev, ShoreZ[OnlineLagoon,3]])
        ChanWidth = np.concatenate([np.full(RiverElev.size, PhysicalPars['RiverWidth']), 
                                    LagoonWidth[OnlineLagoon]])
    else: # open
        ChanDx = np.full(RiverElev.size + OnlineLagoon.size + OutletChanIx.size + 1, Dx)
        if OutletEndX[0] < OutletEndX[1]:
            # Outlet angles from L to R
            ChanDx[-2] += OutletEndX[1] % Dx
        else:
            # Outlet from R to L
            ChanDx[-2] += Dx - (OutletEndX[1] % Dx)
        ChanFlag = np.concatenate([np.full(RiverElev.size, 0), 
                                   np.full(OnlineLagoon.size, 1), 
                                   np.full(OutletChanIx.size, 2), [3,4]])
        ChanElev = np.concatenate([RiverElev, ShoreZ[OnlineLagoon,3], 
                                   ShoreZ[OutletChanIx,1], 
                                   [OutletEndElev, min(PhysicalPars['MaxOutletElev'], OutletEndElev)]])
        ChanWidth = np.concatenate([np.full(RiverElev.size, PhysicalPars['RiverWidth']), 
                                    LagoonWidth[OnlineLagoon], OutletWidth, 
                                    np.full(2, OutletEndWidth)])
    LagArea = np.zeros(ChanElev.size)
    LagArea[RiverElev.size] = StartArea
    LagArea[RiverElev.size + OnlineLagoon.size - 1] = EndArea
    LagLen = (ChanFlag==1) * Dx
    LagLen[RiverElev.size] += StartLen
    LagLen[RiverElev.size + OnlineLagoon.size - 1] += EndLen
    
    # Assemble the hydraulic initial conditions
    if Closed:
        ChanDep = np.concatenate([RivDep,
                                  LagoonWL[OnlineLagoon]-ShoreZ[OnlineLagoon,3]])
        ChanVel = np.concatenate([RivVel,
                                  LagoonVel[OnlineLagoon]])
        ChanVel[-1] = 0.0
    else:
        ChanDep = np.concatenate([RivDep,
                                  LagoonWL[OnlineLagoon]-ShoreZ[OnlineLagoon,3],
                                  OutletDep[OutletChanIx], 
                                  OutletEndDep])
        ChanVel = np.concatenate([RivVel,
                                  LagoonVel[OnlineLagoon], 
                                  OutletVel[OutletChanIx], 
                                  OutletEndVel])
    # If depth is missing then interpolate it
    DepNan = np.isnan(ChanDep)
    if np.any(DepNan):
        logging.debug('Interpolating %i missing channel depths' % np.sum(DepNan))
        ChanDep[DepNan] = np.interp(np.where(DepNan)[0], np.where(~DepNan)[0], ChanDep[~DepNan])
    
    
    # If vel is missing then interpolate it (based on flow rather than vel)
    VelNan = np.isnan(ChanVel)
    if np.any(VelNan):
        logging.debug('Interpolating %i missing channel velocities' % np.sum(VelNan))
        ChanQ = ChanDep*ChanVel
        ChanVel[VelNan] = (np.interp(np.where(VelNan)[0], np.where(~VelNan)[0], ChanQ[~VelNan])
                           / ChanDep[VelNan])
    
    # Check there are no nan values in the channel!
    assert not np.any(ChanWidth<=0), 'Outlet width <= 0 in one or more cross-sections after assembleChannel'
    assert not np.any(np.isnan(ChanWidth)), 'NaN values in ChanWidth after assembleChannel'
    assert not np.any(np.isnan(ChanElev)), 'NaN values in ChanElev after assembleChannel'
    assert not np.any(np.isnan(ChanVel)), 'NaN values in ChanVel after assembleChannel'
    assert not np.any(np.isnan(ChanDep)), 'NaN values in ChanDep after assembleChannel'
    
    # Check consistency of key outputs
    assert OnlineLagoon.size == sum(ChanFlag==1), 'Missmatch between length of online lagoon in "OnlineLagoon" (%i) and "ChanFlag" (%i)' % (OnlineLagoon.size, sum(ChanFlag[:-1]==1))
    
    return (ChanDx, ChanElev, ChanWidth, LagArea, LagLen, ChanDep, ChanVel, 
            OnlineLagoon, OutletChanIx, ChanFlag, Closed)

def storeHydraulics(ChanDep, ChanVel, OnlineLagoon, OutletChanIx, ChanFlag, 
                    LagoonElev, Closed):
    """ Extract lagoon/outlet hydraulic conditions.
        
        (LagoonWL, LagoonVel, OutletDep, OutletVel, 
         OutletEndDep, OutletEndVel) = \
            storeHydraulics(ChanDep, ChanVel, OnlineLagoon, OutletChanIx, 
                            ChanFlag, LagoonElev)
        
        All the hydraulics calculations are carried out on a merged channel
        which includes the river upstream of the lagoon, the online part of
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
    
    # Outlet depth and velocity
    OutletDep = np.full(LagoonElev.size, np.nan)
    OutletVel = np.full(LagoonElev.size, np.nan)
    if Closed:
        OutletEndDep = np.full(2, np.nan)
        OutletEndVel = np.full(2, np.nan)
    else:        
        OutletDep[OutletChanIx] = ChanDep[ChanFlag==2]
        OutletVel[OutletChanIx] = ChanVel[ChanFlag==2]
        EndNodes = np.logical_or(ChanFlag==3, ChanFlag==4)
        OutletEndDep = ChanDep[EndNodes]
        OutletEndVel = ChanVel[EndNodes]
    
    return (LagoonWL, LagoonVel, OutletDep, OutletVel, 
            OutletEndDep, OutletEndVel)
    
def calcBedload(z, B, h, V, dx, PhysicalPars, Psi):
    """ Calculate bedload transport using Meyer-Peter-Muller approach
    
        Note that output is reach, rather than cross-section bedload transport.
        The conversion from cross-section to reach uses a partial upwinding 
        approach controlled by the spatial weighting factor Psi.
        
        Parameters:
            z = bed level at each cross-section [m]
            B = cross-section width at each cross-section [m]
            h = water depth at each cross-section (m)
            V = velocity (average) at each cross-section [m/s]
            dx = distance-to-next at each cross section (note, the array dx is
                 one smaller than z, B, etc) [m]
            PhysicalPars = dict of model input parameters
            Psi = spatial weighting factor for bedload transport
        
        Returns:
            Qs = numpy array of bedload transport in each reach 
                 [m3(bulk including voids)/s]
        
        Bagnold R.A. (1980) An empirical correlation of bedload transport rates
        in flumes and natural rivers. Proc R Soc Lond A Math Phys Sci 
        372(October):453–473. http://www.jstor.org/stable/2397042
    """
    
    Rho = PhysicalPars['RhoRiv']
    RhoS = PhysicalPars['RhoSed']
    g = PhysicalPars['Gravity']
    D = PhysicalPars['GrainSize']
    
    # Use friction slope rather than actual energy or water surface slope as it is more reliable
    S = V**2 * PhysicalPars['GrainRoughness']**2 / h**(4/3)
    
    # Total shear stress (N/m2) and dimensionless shear stress
    Tau = Rho * g * h * S
    Theta = Tau / ((RhoS - Rho) * g * D)
    
    # bedload transport rate at each node (bulk volume accounting for voids) [m^3/s]
    qs_star = PhysicalPars['MPM_coef'] * np.maximum(Theta - PhysicalPars['CritShieldsStress'], 0.0)**PhysicalPars['MPM_exp']
    Qs_node_bulk = qs_star * B * ((RhoS/Rho - 1) * g * D**3)**0.5 / (1 - PhysicalPars['VoidRatio'])
    Qs_node_bulk[V<0] *= -1
    
    # bedload transport in each reach
    Qs_reach_bulk = (1-Psi)*Qs_node_bulk[:-1] + Psi*Qs_node_bulk[1:]
    
    return(Qs_reach_bulk)

def storeBedload(Bedload, NTransects, OnlineLagoon, OutletChanIx, 
                 ChanFlag, Closed):
    """ Extract lagoon/outlet hydraulic conditions.
        
        (LagoonBedload, OutletBedload, OutletEndBedload) = \
            storeBedload(Bedload, NTransects, OnlineLagoon, OutletChanIx, 
                         ChanFlag, Closed)
        
        The river bedload calculations are carried out on a merged channel
        which includes the river upstream of the lagoon, the online part of
        lagoon itself, and the outlet channel. This function extracts the 
        bedload in the lagoon and outlet channel and stores them in the
        shore parallel model schematisation. This is necessary for saving the 
        bedload transport rate into the output netCDF file.
    """
    
    OutletBedload = np.zeros(NTransects)
    LagoonBedload = np.zeros(NTransects)
    if Closed:
        OutletEndBedload = 0.0
        LagoonBedload[OnlineLagoon[:-1]] = Bedload[ChanFlag[:-1]==1]
    else:        
        OutletBedload[OutletChanIx] = Bedload[ChanFlag[:-1]==2]
        OutletEndBedload = Bedload[ChanFlag[:-1]==3]
        LagoonBedload[OnlineLagoon] = Bedload[ChanFlag[:-1]==1]
    
    return (LagoonBedload, OutletBedload, OutletEndBedload)