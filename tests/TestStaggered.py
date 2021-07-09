import numpy as np
import hapuamod.visualise as vis
import hapuamod.riv as riv

# Lake test with variable bed elevation, width and dx
g = 9.81
dt = 0.1
dx_mid = np.full(19, 20.0)  - np.random.rand(19) * 10
z = np.random.rand(20)
B = np.full(20, 10.0) + np.random.rand(20) * 5
h = np.full(20, 2.0) - z
V_mid = np.full(19, 0)
n = 0

Vplot = np.pad((V_mid[:-1]+V_mid[1:])/2, pad_width=1, mode='edge')
LongSecFig = vis.longSection(dx_mid, z, B, h, Vplot)

for i in range(100):
    (h, V_mid) = solveFullExplicitStaggered(h, V_mid, z, B, dx_mid, dt, g, n)
    Vplot = np.pad((V_mid[:-1]+V_mid[1:])/2, pad_width=1, mode='edge')
    vis.updateLongSection(LongSecFig, dx_mid, z, B, h, Vplot)


# Dam break test in uniform channel
# Section 4.1 in paper
g = 9.81
dt = 0.01 #0.026
dx_mid = np.full(399, 0.5)
dx = np.concatenate(([dx_mid[0]], (dx_mid[:-1] + dx_mid[1:]) / 2, [dx_mid[-1]]))
z = np.full(400, 0.0)
B = np.full(400, 1.0)
h = np.where(np.arange(400)<200, 10, 1)
V_mid = np.full(399, 0)
n = 0

Vplot = np.pad((V_mid[:-1]+V_mid[1:])/2, pad_width=1, mode='edge')
LongSecFig = vis.longSection(dx_mid, z, B, h, Vplot)

for i in range(200):
    (h, V_mid) = solveFullExplicitStaggered(h, V_mid, z, B, dx_mid, dt, g, n)
    Vplot = np.pad((V_mid[:-1]+V_mid[1:])/2, pad_width=1, mode='edge')
    vis.updateLongSection(LongSecFig, dx_mid, z, B, h, Vplot)


# Flow over a bump
g = 9.81
dt = 0.02
dx_mid = np.full(99, 0.1)
z = np.full(100, 0.0)
z[45:55] = 0.2
B = np.full(100, 1.0)
h = 1.0 - z
V = 1/h
V_mid = (V[:-1]+V[1:])/2
n = 0

h_wBdys = np.pad(h, pad_width=1, mode='edge')
V_mid_wBdys = np.pad(V_mid, pad_width=1, mode='edge')
h_mid = np.where(V_mid_wBdys >= 0, h_wBdys[:-1], h_wBdys[1:])
q_mid = h_mid * V_mid_wBdys
q_bar = (q_mid[:-1] + q_mid[:1]) / 2
Vplot = np.where(q_bar >= 0, V_mid_wBdys[:-1], V_mid_wBdys[1:])
LongSecFig = vis.longSection(dx_mid, z, B, h, Vplot)

for i in range(100):
    (h, V_mid) = solveFullExplicitStaggered(h, V_mid, z, B, dx_mid, dt, g, n)
    h_wBdys = np.pad(h, pad_width=1, mode='edge')
    V_mid_wBdys = np.pad(V_mid, pad_width=1, mode='edge')
    h_mid = np.where(V_mid_wBdys >= 0, h_wBdys[:-1], h_wBdys[1:])
    q_mid = h_mid * V_mid_wBdys
    q_bar = (q_mid[:-1] + q_mid[:1]) / 2
    Vplot = np.where(q_bar >= 0, V_mid_wBdys[:-1], V_mid_wBdys[1:])
    vis.updateLongSection(LongSecFig, dx_mid, z, B, h, Vplot)

# Steady flow
g = 9.81
dt = 2
dx_mid = np.full(50, 20.0)
z = np.linspace(0.003*np.sum(dx_mid) - 2, -2, dx_mid.size+1)
B = np.full(z.size, 70.0)
LagArea = np.full(z.size ,0.0)
Q_in = np.full(1, 70.0)
DsWL = np.full(1, 0.0)
n = 0.03
S_us = 0.003
NumericalPars = {'Beta':1.0,
                 'MaxIt':20,
                 'ErrTol':0.0001}
PhysicalPars = {'RiverSlope': S_us,
                'Gravity': 9.81,
                'RoughnessManning': n}

(h_steady, V_steady) = riv.solveSteady(dx_mid, z, B, n, Q_in, DsWL, NumericalPars)

V_mid = (V_steady[:-1] + V_steady[1:]) / 2
h = h_steady.copy()

Vplot = np.pad((V_mid[:-1]+V_mid[1:])/2, pad_width=1, mode='edge')
LongSecFig = vis.longSection(dx_mid, z, B, h, Vplot)

for i in range(100):
    (h, V_mid) = solveFullExplicitStaggered(h, V_mid, z, B, LagArea, dx_mid, dt, DsWL, Q_in, PhysicalPars)
    Vplot = np.pad((V_mid[:-1]+V_mid[1:])/2, pad_width=1, mode='edge')
    vis.updateLongSection(LongSecFig, dx_mid, z, B, h, Vplot)

Dist = np.insert(np.cumsum(dx_mid),0,0)
LongSecFig['RivFig'].suptitle('Test 1: Steady flow')
LongSecFig['ElevAx'].plot(Dist, z + h_steady, 'm--', label='Steady WL')
LongSecFig['VelAx'].plot(Dist, V_steady, 'm--', label='Steady vel')
