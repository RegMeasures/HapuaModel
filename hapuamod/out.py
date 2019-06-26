# -*- coding: utf-8 -*-

# Import standard packages
import netCDF4
import numpy as np
import pandas as pd
import os

def newOutFile(FileName, ModelName, ShoreX, StartTime, Overwrite=False):
    """
    
    Parameters:
        FileName(string):
        ModelName
        ShoreX
        StartTime
        Overwrite(boolean): Automatically overwrite netCDF file without 
            prompting (optional, default=False).
    """
    
    # check if file already exists
    if os.path.isfile(FileName):
        if not Overwrite:
            Confirm = input('Confirm ok to overwrite "%s" (y/n):' % FileName)
            Overwrite = Confirm in ['Y', 'y', 'Yes', 'yes', 'YES']
        
        if Overwrite:
            os.remove(FileName)
        else:
            raise Exception('Conflict with existing output file "%s"' % FileName)
    
    # create new empty netcdf file and open it for writing
    NcFile = netCDF4.Dataset(FileName, mode='w', format='NETCDF4_CLASSIC') 
    
    # create dimensions
    XDim = NcFile.createDimension('transect_x', ShoreX.size)
    XSegDim = NcFile.createDimension('shore_segment_x', ShoreX.size-1)
    TimeDim = NcFile.createDimension('time', None)
    
    # create attributes
    NcFile.ModelName = ModelName
    NcFile.ModelStartTime = pd.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # create variables
    XVar = NcFile.createVariable(XDim.name, np.float32, (XDim.name,))
    XVar.units = 'm'
    XVar.long_name = 'X co-ordinates of model shore transects'
    
    XSegVar = NcFile.createVariable(XSegDim.name, np.float32, (XSegDim.name,))
    XSegVar.units = 'm'
    XSegVar.long_name = 'X co-ordinates of midpoints of shoreline segments between transects'
    
    TimeVar = NcFile.createVariable(TimeDim.name, np.float64, (TimeDim.name,))
    TimeVar.units = 'seconds since %s' % StartTime.strftime('%Y-%m-%d')
    TimeVar.calendar = 'standard'
    TimeVar.long_name = 'Model output times'
    
    ShoreYVar = NcFile.createVariable('shoreline_y', np.float32, 
                                      (TimeDim.name, XDim.name))
    ShoreYVar.units = 'm'
    ShoreYVar.long_name = 'Y coordinate of shoreline at each transect'
    
    Outlet1YVar = NcFile.createVariable('outlet_bank1_y', np.float32, 
                                        (TimeDim.name, XDim.name))
    Outlet1YVar.units = 'm'
    Outlet1YVar.long_name = 'Y coordinate of seaward bank of outlet channel at each transect'
    
    Outlet2YVar = NcFile.createVariable('outlet_bank2_y', np.float32, 
                                        (TimeDim.name, XDim.name))
    Outlet2YVar.units = 'm'
    Outlet2YVar.long_name = 'Y coordinate of lagoonward bank of outlet channel at each transect'
    
    LagoonYVar = NcFile.createVariable('lagoon_y', np.float32, 
                                       (TimeDim.name, XDim.name))
    LagoonYVar.units = 'm'
    LagoonYVar.long_name = 'Y coordinate of seaward edge of lagoon at each transect'
    
    CliffYVar = NcFile.createVariable('cliff_y', np.float32, 
                                      (TimeDim.name, XDim.name))
    CliffYVar.units = 'm'
    CliffYVar.long_name = 'Y coordinate of cliff toe at each transect'
    
    LstVar = NcFile.createVariable('lst', np.float32, (TimeDim.name, XSegDim.name))
    LstVar.units = 'm3/day'
    LstVar.description = 'longshore transport rate (averaged over last output interval)'
    
    # write data to fixed variables
    XVar[:] = ShoreX
    XSegVar[:] = (ShoreX[:-1]+ShoreX[1:])/2
    
    NcFile.close()

def writeCurrent(FileName, ShoreY, LST, CurrentTime):
    
    # Open netCDF file for appending
    NcFile = netCDF4.Dataset(FileName, mode='a') 
    
    # Get index of new time row to add and add current time
    TimeVar = NcFile.variables['time']
    TimeIx = TimeVar.size
    TimeVar[TimeIx] = netCDF4.date2num(CurrentTime, TimeVar.units)
    
    # Append new data
    NcFile.variables['shoreline_y'][TimeIx,:] = ShoreY[:,0]
    NcFile.variables['outlet_bank1_y'][TimeIx,:] = ShoreY[:,1]
    NcFile.variables['outlet_bank2_y'][TimeIx,:] = ShoreY[:,2]
    NcFile.variables['lagoon_y'][TimeIx,:] = ShoreY[:,3]
    NcFile.variables['cliff_y'][TimeIx,:] = ShoreY[:,4]
    
    NcFile.variables['lst'][TimeIx,:] = LST*86400
    
    NcFile.close()
