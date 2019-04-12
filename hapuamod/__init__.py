"""hapuamode is a model for simulating hapua morphology.

hapuamod contains functions grouped into several sub-modules:
    load      Reading model inputs and model pre-processing
    geom      Geometry transformations between model and real world 
              coordinate systems
    visualise Plotting hapuamod variables
    coast     1-line shoreline model
    riv       1D river model
    core      Core model simulation functions
    
hapuamod requires: numpy, pandas, matplotlib, configobj, os, shapefile
"""

import hapuamod.load
import hapuamod.geom
import hapuamod.visualise
import hapuamod.coast
import hapuamod.riv
import hapuamod.core