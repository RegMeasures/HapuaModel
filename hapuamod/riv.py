# -*- coding: utf-8 -*-
""" Fluvial hydraulics calculations """

# import standard packages
import numpy as np
from scipy import linalg
import logging

def solveSteady(ChanDx, ChanElev, ChanWidth, Roughness, Beta, Qin, DsWL):
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
            assert CheckCount < MaxIter, 'Maximum iterations exceeded solving steady state water level'
        
        # Check for supercritical
        if ChanDep[XS] < CritDep[XS]:
            ChanDep[XS] = CritDep[XS]
            logging.warning('Steady state solution results in critical depth at XS%i' % XS)
        ChanVel[XS] = Qin / (ChanWidth[XS] * ChanDep[XS])
        Energy[XS] = ChanElev[XS] + ChanDep[XS] + Beta * ChanVel[XS]**2 / (2*Grav)
        S_f[XS] = ChanVel[XS]**2 * Roughness**2 / ChanDep[XS]**(4/3)
    
    return ChanDep, ChanVel

def solveFullPreissmann(z, B, LagArea, h, V, dx, dt, n, Q_Ts, DsWl_Ts, 
                        NumericalPars):
    """ Solve full S-V eqns for a rectangular channel using preissmann scheme.
        Uses a newton raphson solution to the preissmann discretisation of the 
        Saint-Venant equations.
        
        solveFullPreissmann(z, B, LagArea, h, V, dx, dt, n, Q_Ts, DsWl_Ts, 
                            NumericalPars)
        
        Parameters:
            z (np.ndarray(float64)): Bed elevation at each node (m)
            B (np.ndarray(float64)): Channel width at each node (m)
            LagArea (np.ndarray(float64)): Offline lagoon area connected to 
                each node (m^2)
            h (np.ndarray(float64)): Depth at each node at the last timestep (m)
            V (np.ndarray(float64)): Mean velocity at each node at the last timestep (m)
            dx (np.ndarray(float64)): Length of each reach between nodes (m)
                note the dx array should be one shorter than for z, B, etc
            dt (pd.timedelta): timestep
            n (float): mannings 'n' roughness
            Q_Ts (np.ndarray(float64)): List (as np.array) of inflows at 
                upstream end of channel corresponding to each timestep. 
                Function will loop over each inflow in turn (m^3/s)
            DsWl_Ts (np.ndarray(float64)): List (as np.array) of downstream 
                boundary water levels at upstream end of channel corresponding 
                to each timestep. Function will loop over each inflow in turn 
                (m^3/s)
            NumericalPars (dict): Numerical parameters including:
                Theta
                Beta
                ErrTol
                MaxIt
                FrRelax1
                FrRelax2
        
        Modifies in place:
            h
            V
        
    """
    g = 9.81                    # gravity [m/s^2]
    sqrt_g = np.sqrt(g)
    
    Theta = NumericalPars['Theta']
    Beta = NumericalPars['Beta']
    Tol = NumericalPars['ErrTol']
    MaxIt = NumericalPars['MaxIt']
    FrMax = NumericalPars['FrRelax2']
    FrMin = NumericalPars['FrRelax1']
    FrRng = FrMax - FrMin
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
    Fr = V / (np.sqrt(g*h)) # Froude No. at each node
    R = (FrMax-Fr)/(FrRng) #np.ones(Fr.size)
    
    i0 = np.arange(0,N-1)
    i1 = np.arange(1,N)
    
    # Main timestepping loop
    for StepNo in range(Q_Ts.shape[0]):
        Q = Q_Ts[StepNo]
        DsWl = DsWl_Ts[StepNo]
        
        V_old = V.copy()
        h_old = h.copy()
        Sf_old = Sf.copy()
        A_old = A.copy()
        R_old = R.copy()
        Fr_old = Fr.copy()
        
        # Constant parts of the S-V Equations which can be computed outside the loop
        # For continuity equation
        C1 = (h_old[:-1]*Be[:-1] + h_old[1:]*Be[1:] 
              - 2*(dt/dx)*(1-Theta) * (V_old[1:]*A_old[1:] 
                                       - V_old[:-1]*A_old[:-1]))
        
        # For momentum equation
        M1 = (V_old[1:]*A_old[1:] + V_old[:-1]*A_old[:-1] 
              - Beta*(1-Theta)*(dt/dx)*((R_old[1:]*A_old[1:]*V_old[1:] + R_old[:-1]*A_old[:-1]*V_old[:-1])
                                            * (V_old[1:] - V_old[:-1])
                                        + (V_old[1:] + V_old[:-1])
                                            * (A_old[1:]*V_old[1:]-A_old[:-1]*V_old[:-1]))
              - g*(1-Theta)*(dt/dx)*(A_old[1:]+A_old[:-1])*(h_old[1:]-h_old[:-1])
              - g*(1-Theta)*dt*(A_old[1:]*(Sf_old[1:]-S_0)
                                 +A_old[:-1]*(Sf_old[:-1]-S_0)))
        
        # Iterative solution
        ItCount = 0
        Err = np.zeros(2*N)
        ConIx = np.arange(1,2*N-1,2)
        MomIx = np.arange(2,2*N-1,2)
        while ItCount < MaxIt:
            
            # Error in Us Bdy
            Err[0] = A[0]*V[0]-Q
            
            # Error in continuity equation
            Err[ConIx] = (h[:-1]*Be[:-1] + h[1:]*Be[1:]
                                         + 2*(dt/dx)*Theta * (V[1:]*A[1:] - V[:-1]*A[:-1])) - C1
            
            # Error in momentum equation
            Err[MomIx] = (A[1:]*V[1:] + A[:-1]*V[:-1]
                          + Beta*Theta*(dt/dx)*((R[1:]*A[1:]*V[1:] + R[:-1]*A[:-1]*V[:-1]) * (V[1:] - V[:-1])
                                                + (V[1:] + V[:-1]) * (A[1:]*V[1:]-A[:-1]*V[:-1]))
                          + g*Theta*(dt/dx)*(A[1:]+A[:-1])*(h[1:]-h[:-1])
                          + g*Theta*dt*(A[1:]*(Sf[1:]-S_0)
                                        +A[:-1]*(Sf[:-1]-S_0))) - M1
            
            # Error in Ds Bdy
            Err[2*N-1] = z[-1] + h[-1] - DsWl
            
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
            a_banded[3,ConIx-1] = Be[:-1] - 2*dt/dx*Theta*V[:-1]*B[:-1]
            # d/dV[0]
            a_banded[2,ConIx] = -2*dt/dx*Theta*A[:-1]
            # d/dh[1]
            a_banded[1,ConIx+1] = Be[1:] + 2*dt/dx*Theta*V[1:]*B[1:]
            # d/dV[1]
            a_banded[0,ConIx+2] = 2*dt/dx*Theta*A[1:]
            
            # Momentum equation derivatives - dependant on Fr
            Fr2 = (Theta*(Fr[1:]+Fr[-1:]) + (1-Theta)*(Fr_old[1:]+Fr_old[:-1]))/2
            
            # Base part retained at all Fr
            # d/dh[0]
            a_banded[4,MomIx-2] = (B[:-1]*V[:-1]
                                   - Beta*Theta*(dt/dx) * B[:-1]*V[:-1]*(V[1:]+V[:-1])
                                   + g*Theta*(dt/dx)*(-2*A[:-1] + B[:-1]*h[1:] - A[1:])
                                   - g*Theta*dt*B[:-1]*(Sf[:-1]/3 + S_0))
            # d/dV[0]
            a_banded[3,MomIx-1] = (A[:-1]
                                   + Beta*Theta*(dt/dx) * (V[1:]*(A[1:]-A[:-1]) - 2*A[:-1]*V[:-1])
                                   + 2*g*Theta*dt*A[:-1]*Sf[:-1]/np.abs(V[-1]))
            # d/dh[1]
            a_banded[2,MomIx] = (B[1:]*V[1:]
                                 + Beta*Theta*(dt/dx) * B[1:]*V[1:]*(V[1:]+V[:-1])
                                 + g*Theta*(dt/dx)*(2*A[1:] - B[1:]*h[:-1] + A[:-1])
                                 - g*Theta*dt*B[1:]*(Sf[1:]/3 + S_0))
            # d/dV[1]
            a_banded[1,MomIx+1] = (A[1:]
                                   + Beta*(dt/dx)*Theta * (V[:-1]*(A[1:]-A[:-1]) + 2*A[1:]*V[1:])
                                   + 2*g*Theta*dt*A[1:]*Sf[1:]/np.abs(V[1:]))
            
            # Fr <= FrMin --> Full St Venant, no relaxation
            Sub = Fr2 <= FrMin
            
            # d/dh[0]
            a_banded[4,MomIx[Sub]-2] += Beta*(dt/dx[Sub])*Theta * B[i0[Sub]]*V[i0[Sub]]*(V[i1[Sub]]-V[i0[Sub]])
            # d/dV[0]
            a_banded[3,MomIx[Sub]-1] += Beta*(dt/dx[Sub])*Theta * (V[i1[Sub]]*(A[i0[Sub]]-A[i1[Sub]]) - 2*A[i0[Sub]]*V[i0[Sub]])
            # d/dh[1]
            a_banded[2,MomIx[Sub]] += Beta*(dt/dx[Sub])*Theta * B[i1[Sub]]*V[i1[Sub]]*(V[i1[Sub]]-V[i0[Sub]])
            # d/dV[1]
            a_banded[1,MomIx[Sub]+1] += Beta*(dt/dx[Sub])*Theta * (V[i0[Sub]]*(A[i0[Sub]]-A[i1[Sub]]) + 2*A[i1[Sub]]*V[i1[Sub]])
            
            # FrMin < Fr < FrMax --> Partial relaxation
            Tra = np.logical_and(~Sub, Fr2<FrMax)
            
            # d/dh[0]
            a_banded[4,MomIx[Tra]-2] += Beta*(dt/dx[Tra])*Theta * (A[i0[Tra]]*(V[i0[Tra]]**2)/(2*sqrt_g*FrRng*(h[i0[Tra]]**1.5))) * (V[i1[Tra]]-V[i0[Tra]])
            # d/dV[0]
            a_banded[3,MomIx[Tra]-1] += Beta*(dt/dx[Tra])*Theta * ((A[i1[Tra]]*V[i1[Tra]]*(V[i1[Tra]]*(h[i1[Tra]]**-0.5) - sqrt_g*FrMax)
                                                                    + A[i0[Tra]]*V[i1[Tra]]*(sqrt_g*FrMax - 2*V[i0[Tra]]*(h[i0[Tra]]**-0.5))
                                                                    + A[i0[Tra]]*V[i0[Tra]]*(3*V[i0[Tra]]*(h[i0[Tra]]**-0.5) - 2*sqrt_g*FrMax)) 
                                                                   / (sqrt_g * FrRng))
            # d/dh[1]
            a_banded[2,MomIx[Tra]] += Beta*(dt/dx[Tra])*Theta * (A[i1[Tra]]*V[i1[Tra]]**2/(2*sqrt_g*FrRng*(h[i1[Tra]]**1.5))) * (V[i1[Tra]]-V[i0[Tra]])
            # d/dV[1]
            a_banded[1,MomIx[Tra]+1] += Beta*(dt/dx[Tra])*Theta * ((A[i1[Tra]]*V[i1[Tra]]*(2*sqrt_g*FrMax - 3*V[i1[Tra]]*(h[i1[Tra]]**-0.5))
                                                                    + A[i1[Tra]]*V[i0[Tra]]*(2*V[i1[Tra]]*(h[i1[Tra]]**-0.5) - sqrt_g*FrMax)
                                                                    + A[i0[Tra]]*V[i0[Tra]]*(sqrt_g*FrMax - V[i0[Tra]]*(h[i0[Tra]]**-0.5))) 
                                                                   / (sqrt_g * FrRng))
            
            # Ds Bdy condition derivatives
            a_banded[2,2*N-1] = 0
            #a_banded[2,2*N-1] = V[N-1]/g
            a_banded[3,2*N-2] = 1
            
            # Solve the banded matrix
            Delta = linalg.solve_banded([2,2],a_banded,Err)
            
            # Update h & V
            h -= Delta[np.arange(0,2*N,2)]
            V -= Delta[np.arange(1,2*N,2)]
            
            # Check if solution is within tolerance
            if np.all(np.abs(Delta) < Tol):
                break
            
            # Update Sf, A, Fr and R
            Sf = V*np.abs(V) * n**2 / h**(4/3)
            A = h*B
            Fr = V / (np.sqrt(g*h)) # Froude No. at each node
            R = (FrMax-Fr)/(FrRng)
            
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
    
def calcBedload(z, B, h, V, dx, PhysicalPars):
    """ Calculate bedload transport using Bagnold 1980 streampower approach
        
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
    