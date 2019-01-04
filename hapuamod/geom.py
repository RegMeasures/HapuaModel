# -*- coding: utf-8 -*-

import numpy as np

# functions for converting between model and real world coordinate systems
def mod2real(Xmod, Ymod, Origin, ShoreNormalDir):
    Xreal = (Origin[0] + 
             Xmod * np.sin(ShoreNormalDir+np.pi/2) - 
             Ymod * np.cos(ShoreNormalDir+np.pi/2))
    Yreal = (Origin[1] + 
             Xmod * np.cos(ShoreNormalDir+np.pi/2) + 
             Ymod * np.sin(ShoreNormalDir+np.pi/2))
    return (Xreal, Yreal)

def real2mod(Xreal, Yreal, Origin, ShoreNormalDir):
    Xrelative = Xreal - Origin[0]
    Yrelative = Yreal - Origin[1]
    Dist = np.sqrt(Xrelative**2 + Yrelative**2)
    Dir = np.arctan2(Xrelative, Yrelative)
    Xmod = Dist * np.cos(ShoreNormalDir - Dir + np.pi/2)
    Ymod = Dist * np.sin(ShoreNormalDir - Dir + np.pi/2)
    return (Xmod, Ymod)