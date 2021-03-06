"""hapuamode is a model for simulating hapua morphology.

hapuamod can be run from the commandline using the command

> python -m hapuamod MODEL_CONFIG_FILE.cnf

hapuamod contains functions grouped into several sub-modules:
    load      Reading model inputs and model pre-processing
    geom      Geometry transformations between model and real world 
              coordinate systems
    visualise Plotting hapuamod variables
    coast     1-line shoreline model
    riv       1D river model
    core      Core model simulation functions
    mor       Morphology updating
    out       Writing outputs to netCDF
    animate   Producing animations of model outputs
    synth     Generate synthetic timeseries for input boundary conditions
    
hapuamod requires: numpy, pandas, matplotlib, configobj, os, shapefile
"""