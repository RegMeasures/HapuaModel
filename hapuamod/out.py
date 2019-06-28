# -*- coding: utf-8 -*-

# Import standard packages
import netCDF4
import numpy as np
import pandas as pd
import os
import logging

def newOutFile(FileName, ModelName, StartTime,
               ShoreX, Dx, RiverElev, Overwrite=False):
    """ Create new netCDF output file for hapuamod results
    
        newOutFile(FileName, ModelName, ShoreX, Dx, RiverElev, 
                   StartTime, Overwrite=False)
    
        Parameters:
            FileName (string):
            ModelName (string):
            ShoreX
            Dx
            RiverElev
            StartTime
            Overwrite(boolean): Automatically overwrite netCDF file without 
                prompting (optional, default=False).
    """
    
    #%% setup
    # check if file already exists
    if os.path.isfile(FileName):
        if not Overwrite:
            Confirm = input('Confirm ok to overwrite "%s" (y/n):' % FileName)
            Overwrite = Confirm in ['Y', 'y', 'Yes', 'yes', 'YES']
        
        if Overwrite:
            os.remove(FileName)
            logging.info('Overwriting output file "%s"' % FileName)
        else:
            raise Exception('Conflict with existing output file "%s"' % FileName)
    else:
        logging.info('Creating new output file "%s"' % FileName)
    
    # create new empty netcdf file and open it for writing
    NcFile = netCDF4.Dataset(FileName, mode='w', format='NETCDF4_CLASSIC') 
    
    # create dimensions
    XDim = NcFile.createDimension('transect_x', ShoreX.size)
    XSegDim = NcFile.createDimension('shore_segment_x', ShoreX.size-1)
    RivDim = NcFile.createDimension('river_dist', RiverElev.size)
    EndsDim = NcFile.createDimension('outlet_ends', 2)
    TimeDim = NcFile.createDimension('time', None)
    
    # create attributes
    NcFile.ModelName = ModelName
    NcFile.ModelStartTime = pd.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # create coordinate variables
    XVar = NcFile.createVariable(XDim.name, np.float32, (XDim.name,))
    XVar.units = 'm'
    XVar.long_name = 'X co-ordinates of model shore transects'
    XVar[:] = ShoreX
    
    XSegVar = NcFile.createVariable(XSegDim.name, np.float32, (XSegDim.name,))
    XSegVar.units = 'm'
    XSegVar.long_name = 'X co-ordinates of midpoints of shoreline segments between transects'
    XSegVar[:] = (ShoreX[:-1]+ShoreX[1:])/2
    
    RivVar = NcFile.createVariable(RivDim.name, np.float32, (RivDim.name,))
    RivVar.units = 'm'
    RivVar.long_name = 'Distance along river to each cross-section upstream of lagoon'
    RivVar[:] = np.arange(RiverElev.size) * Dx
    
    TimeVar = NcFile.createVariable(TimeDim.name, np.float64, (TimeDim.name,))
    TimeVar.units = 'seconds since %s' % StartTime.strftime('%Y-%m-%d')
    TimeVar.calendar = 'standard'
    TimeVar.long_name = 'Model output times'
    
    #%% create shore-transect data variables
    
    # Shoreline Y coordinates
    ShoreYVar = NcFile.createVariable('shoreline_y', np.float32, 
                                      (TimeDim.name, XDim.name))
    ShoreYVar.units = 'm'
    ShoreYVar.long_name = 'Y coordinate of shoreline at each transect'
    
    # Outlet bank Y coordinates (x2 seaward and lagoonward)
    Outlet1YVar = NcFile.createVariable('outlet_bank1_y', np.float32, 
                                        (TimeDim.name, XDim.name))
    Outlet1YVar.units = 'm'
    Outlet1YVar.long_name = 'Y coordinate of seaward bank of outlet channel at each transect'
    Outlet2YVar = NcFile.createVariable('outlet_bank2_y', np.float32, 
                                        (TimeDim.name, XDim.name))
    Outlet2YVar.units = 'm'
    Outlet2YVar.long_name = 'Y coordinate of lagoonward bank of outlet channel at each transect'
    
    # Lagoon bank y coorindates
    LagoonYVar = NcFile.createVariable('lagoon_y', np.float32, 
                                       (TimeDim.name, XDim.name))
    LagoonYVar.units = 'm'
    LagoonYVar.long_name = 'Y coordinate of seaward edge of lagoon at each transect'
    
    # Cliff toe y coordinates
    CliffYVar = NcFile.createVariable('cliff_y', np.float32, 
                                      (TimeDim.name, XDim.name))
    CliffYVar.units = 'm'
    CliffYVar.long_name = 'Y coordinate of cliff toe at each transect'
    
    # Lagoon bed level
    LagoonElevVar = NcFile.createVariable('lagoon_bed_z', np.float32, 
                                          (TimeDim.name, XDim.name))
    LagoonElevVar.units = 'm'
    LagoonElevVar.long_name = 'Lagoon bed elevation at each transect'
    
    # Outlet bed level
    OutletElevVar = NcFile.createVariable('outlet_bed_z', np.float32, 
                                          (TimeDim.name, XDim.name))
    OutletElevVar.units = 'm'
    OutletElevVar.long_name = 'Outlet bed elevation at each transect'
    
    # Lagoon water level
    LagoonWlVar = NcFile.createVariable('lagoon_wl', np.float32, 
                                        (TimeDim.name, XDim.name))
    LagoonWlVar.units = 'm'
    LagoonWlVar.long_name = 'Lagoon water level at each transect'
    
    # Lagoon velocity
    LagoonVelVar = NcFile.createVariable('lagoon_vel', np.float32, 
                                          (TimeDim.name, XDim.name))
    LagoonVelVar.units = 'm/s'
    LagoonVelVar.long_name = 'Mean lagoon velocity at each transect (+ve = outflowing i.e. toward outlet channel)'
        
    # Longshore transport
    LstVar = NcFile.createVariable('lst', np.float32, (TimeDim.name, XSegDim.name))
    LstVar.units = 'm3/day'
    LstVar.long_name = 'longshore transport rate at each transect'
    
    #%% Create river variables
    
    # River bed elevation
    RiverElevVar = NcFile.createVariable('river_bed_z', np.float32, 
                                         (TimeDim.name, RivDim.name))
    RiverElevVar.units = 'm'
    RiverElevVar.long_name = 'River cross-section bed elevation'
    
    # River bed water level
    RiverWlVar = NcFile.createVariable('river_wl', np.float32, 
                                       (TimeDim.name, RivDim.name))
    RiverWlVar.units = 'm'
    RiverWlVar.long_name = 'River cross-section water level'
    
    # River bed velocity
    RiverVelVar = NcFile.createVariable('river_vel', np.float32, 
                                        (TimeDim.name, RivDim.name))
    RiverVelVar.units = 'm/s'
    RiverVelVar.long_name = 'River cross-section mean velocity'
    
    #%% Create outlet end variables
    
    # Outlet end position
    OutletEndXVar = NcFile.createVariable('outlet_end_x', np.float32, 
                                          (TimeDim.name, EndsDim.name))
    OutletEndXVar.units = 'm'
    OutletEndXVar.long_name = 'Outlet end position'
    
    # Outlet end width
    OutletEndBVar = NcFile.createVariable('outlet_end_width', np.float32, 
                                          (TimeDim.name, EndsDim.name))
    OutletEndBVar.units = 'm'
    OutletEndBVar.long_name = 'Outlet end width'
    
    # Outlet end elevation
    OutletEndElevVar = NcFile.createVariable('outlet_end_z', np.float32, 
                                             (TimeDim.name, EndsDim.name))
    OutletEndElevVar.units = 'm'
    OutletEndElevVar.long_name = 'Outlet end bed elevation'
    
    # Outlet end water level
    OutletEndWlVar = NcFile.createVariable('outlet_end_wl', np.float32, 
                                           (TimeDim.name, EndsDim.name))
    OutletEndWlVar.units = 'm'
    OutletEndWlVar.long_name = 'Outlet end water level'
    
    # Outlet end velocity
    OutletEndVelVar = NcFile.createVariable('outlet_end_vel', np.float32, 
                                          (TimeDim.name, EndsDim.name))
    OutletEndVelVar.units = 'm/s'
    OutletEndVelVar.long_name = 'Outlet end velocity'
    
    # Close netCDF file
    NcFile.close()

def writeCurrent(FileName, CurrentTime,
                 ShoreY, LagoonElev, OutletElev, LagoonWL, LagoonVel, LST, 
                 RiverElev, RiverDep, RiverVel,
                 OutletEndX, OutletEndElev, OutletEndWidth, 
                 OutletEndDep, OutletEndVel):
    """ Write hapuamod outputs for current timestep to netCDF 
    
    """
    
    # Open netCDF file for appending
    NcFile = netCDF4.Dataset(FileName, mode='a') 
    
    # Get index of new time row to add and add current time
    TimeVar = NcFile.variables['time']
    TimeIx = TimeVar.size
    TimeVar[TimeIx] = netCDF4.date2num(CurrentTime, TimeVar.units)
    
    # Append new data to shore transect variables
    NcFile.variables['shoreline_y'][TimeIx,:] = ShoreY[:,0]
    NcFile.variables['outlet_bank1_y'][TimeIx,:] = ShoreY[:,1]
    NcFile.variables['outlet_bank2_y'][TimeIx,:] = ShoreY[:,2]
    NcFile.variables['lagoon_y'][TimeIx,:] = ShoreY[:,3]
    NcFile.variables['cliff_y'][TimeIx,:] = ShoreY[:,4]
    
    NcFile.variables['lagoon_bed_z'][TimeIx,:] = LagoonElev
    NcFile.variables['outlet_bed_z'][TimeIx,:] = OutletElev
    
    NcFile.variables['lagoon_wl'][TimeIx,:] = LagoonWL
    NcFile.variables['lagoon_vel'][TimeIx,:] = LagoonVel
    
    NcFile.variables['lst'][TimeIx,:] = LST*86400
    
    # Append new data to river variables
    NcFile.variables['river_bed_z'][TimeIx,:] = RiverElev
    NcFile.variables['river_wl'][TimeIx,:] = RiverElev + RiverDep
    NcFile.variables['river_vel'][TimeIx,:] = RiverVel
    
    # Append new data to outlet end variables
    NcFile.variables['outlet_end_x'][TimeIx,:] = OutletEndX
    NcFile.variables['outlet_end_width'][TimeIx,:] = OutletEndWidth
    NcFile.variables['outlet_end_z'][TimeIx,:] = OutletEndElev
    NcFile.variables['outlet_end_wl'][TimeIx,:] = OutletEndElev + OutletEndDep[0:2]
    NcFile.variables['outlet_end_vel'][TimeIx,:] = OutletEndVel[0:2]
    
    NcFile.close()
