
import numpy as np


def solveFullExplicitStaggered(h, V_mid, z, B, dx_mid, dt, g):
    
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
    
    # Add boundary conditions as dummy cells
    # Use a transmissive boundary condition to start with
    h_wBdys = np.pad(h, pad_width=1, mode='edge')
    V_mid_wBdys = np.pad(V_mid, pad_width=1, mode='edge')
    
    # I assume I should use a central approach to calculate B at the mid points 
    # as it is not specified in the paper...
    # NB: B is padded into dummy reaches beyond boundaries
    B_mid = np.concatenate(([B[0]], (B[:-1] + B[1:]) / 2, [B[-1]]))
    
    # I assume I should also use a central approach to calculate dx at the nodes
    # NB: dx is padded so end nodes have a dx assigned to them
    dx = np.concatenate(([dx_mid[0]], (dx_mid[:-1] + dx_mid[1:]) / 2, [dx_mid[-1]]))
    
    # eq 5
    h_mid = np.where(V_mid_wBdys >= 0, h_wBdys[:-1], h_wBdys[1:])
    
    # eq 7
    q_mid = h_mid * V_mid_wBdys
    h_bar_mid = (h[:-1] + h[1:]) / 2
    q_bar = (q_mid[:-1] + q_mid[:1]) / 2
    
    # eq 6
    V = np.where(q_bar >= 0, V_mid_wBdys[:-1], V_mid_wBdys[1:])
    
    CFL = np.max((np.abs(V) + np.sqrt(g*h)) * dt/dx)
    print(CFL)
    
    # eq 12
    h = h - dt/dx * ((h_mid[1:] * V_mid_wBdys[1:] - h_mid[:-1] * V_mid_wBdys[:-1]) + 
                     h * V * ((B_mid[1:] - B_mid[:-1]) / B))
    
    # eq 13
    V_mid = (V_mid
             - (dt / (h_bar_mid * dx_mid)) * ((q_bar[1:] * V[1:] - q_bar[:-1] * V[:-1])
                                          - V_mid * (q_bar[1:] - q_bar[:-1]))
             - g * dt/dx_mid * ((h[1:] - h[:-1]) + (z[1:] - z[:-1])))
    
    
    return(h, V_mid)
    