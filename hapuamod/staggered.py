
import numpy as np


def solveFullExplicitStaggered(h, V_mid, z, B, LagArea, dx_mid, dt, DsWl_Ts, Q_Ts, PhysicalPars):
    
    """
    
    Based on Mungkasi et al. 2018.
    
    z = bed level (at nodes)
    B = width (at nodes)
    
    
    Mungkasi S., Magdalena I., Pudjaprasetya S.R., Wiryanto L.H., Roberts S.G. 
        (2018) A staggered method for the shallow water equations involving 
        varying channel width and topography. 
        Int J Multiscale Comput Eng 16(3):231–244. 
        https://doi.org/10.1615/IntJMultCompEng.2018027042
    """
    
    g = PhysicalPars['Gravity']
    n = PhysicalPars['RoughnessManning']
    S_us = PhysicalPars['RiverSlope']
    
    #%% First calculate some properties which do not change over time
    
    # I assume I can use a central approach to calculate B at the mid points 
    # NB: B is padded into dummy reaches beyond boundaries
    B_mid = np.concatenate(([B[0]], (B[:-1] + B[1:]) / 2, [B[-1]]))
    
    # I assume I can also use a central approach to calculate dx at the nodes
    # NB: dx is padded so end nodes have a dx assigned to them
    dx = np.concatenate(([dx_mid[0]], (dx_mid[:-1] + dx_mid[1:]) / 2, [dx_mid[-1]]))
    
    # Calculate R, the ratio of the effective width which stores water, to the actual flowing width B
    R = 1 + (LagArea/dx) / B
    R_mid = (R[:-1] + R[1:]) / 2
    
    #%% Main timestepping loop
    for StepNo in range(Q_Ts.shape[0]):
        
        # Add boundary conditions as dummy cells
        
        # Use a transmissive boundary condition to start with
        h_wBdys = np.pad(h.copy(), pad_width=1, mode='edge')
        V_mid_wBdys = np.pad(V_mid.copy(), pad_width=1, mode='edge')
        
        # Add a d/s WL bdy
        DsWL = DsWl_Ts[StepNo]
        h_wBdys[-1] = DsWL - z[-1]
        # V_mid_wBdys[-1] = h[-1]*V_mid[-1]) / h_wBdys[-1]
        V_mid_wBdys[-1] = (((h[-1] - h_wBdys[-1]) * dx_mid[-1]/dt) + h[-1]*V_mid[-1]) / h_wBdys[-1]
        
        # Add an upstream flow boundary
        Q = Q_Ts[StepNo]
        h_wBdys[0] = (Q * n / (B[0] * S_us**0.5))**(3/5)
        V_mid_wBdys[0] = V_mid[0] + 2 * ((Q / (B[0] * h_wBdys[0])) - V_mid[0])
            
        # eq 5
        h_mid = np.where(V_mid_wBdys >= 0, h_wBdys[:-1], h_wBdys[1:])
        
        # Apply friction based on a euler explicit discretisation
        S_mid = np.abs(V_mid_wBdys)*V_mid_wBdys * n**2 / h_mid**(4/3)
        # V_mid_wBdys = V_mid_wBdys - dt*g*S_mid
        
        # eq 7
        q_mid = h_mid * V_mid_wBdys
        h_bar_mid = (h[:-1] + h[1:]) / 2
        q_bar = (q_mid[:-1] + q_mid[:1]) / 2
        
        # eq 6
        V = np.where(q_bar >= 0, V_mid_wBdys[:-1], V_mid_wBdys[1:])
        
        # CFL = np.max((np.abs(V) + np.sqrt(g*h)) * dt/dx)
        # print(CFL)
        
        # eq 12
        # h = h - (dt/dx) * ((h_mid[1:] * V_mid_wBdys[1:] - h_mid[:-1] * V_mid_wBdys[:-1]) + 
        #                       h * V * ((B_mid[1:] - B_mid[:-1]) / B))
        h = h - (dt/(dx * R)) * ((h_mid[1:] * V_mid_wBdys[1:] - h_mid[:-1] * V_mid_wBdys[:-1]) + 
                              h * V * ((B_mid[1:] - B_mid[:-1]) / B))
        
        # eq 13
        # V_mid = (V_mid 
        #          - (dt / (h_bar_mid * dx_mid)) * ((q_bar[1:] * V[1:] - q_bar[:-1] * V[:-1])
        #                                          - V_mid * (q_bar[1:] - q_bar[:-1]))
        #           - g * (dt/dx_mid) * ((h[1:] - h[:-1]) + (z[1:] - z[:-1]))
        #           - g * dt * S_mid[1:-1])
        V_mid = (V_mid 
                  - (dt / (h_bar_mid * dx_mid)) * ((q_bar[1:] * V[1:] - q_bar[:-1] * V[:-1])
                                                  - V_mid * (q_bar[1:] - q_bar[:-1])
                                                  + (1 - 1/R_mid) * q_mid[1:-1] * V_mid * (B[1:] - B[:-1]) / B_mid[1:-1])
                  - g * (dt/dx_mid) * ((h[1:] - h[:-1]) + (z[1:] - z[:-1]))
                  - g * dt * S_mid[1:-1])
        
    return(h, V_mid)
    