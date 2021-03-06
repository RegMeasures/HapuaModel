# -*- coding: utf-8 -*-
""" Sub-module of hapuamod containing functions to read/write model outputs
    from/to netCDF file.
"""

# Import standard packages
import netCDF4
import numpy as np
import pandas as pd
import datetime
import os
import logging

def newOutFile(FileName, ModelName, StartTime,
               ShoreX, Dx, RiverElev, 
               Origin, ShoreNormDir, PhysicalPars,
               Overwrite=False):
    """ Create new netCDF output file for hapuamod results
    
        newOutFile(FileName, ModelName, StartTime, 
                   ShoreX, Dx, RiverElev, 
                   Origin, ShoreNormDir, PhysicalPars,
                   Overwrite=False)
    
        Parameters:
            FileName (string):
            ModelName (string):
            StartTime
            ShoreX
            Dx
            RiverElev
            Origin
            ShoreNormDir
            RiverWidth
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
    NcFile.ModelStartTime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    NcFile.ModelOriginX = Origin[0]
    NcFile.ModelOriginY = Origin[1]
    NcFile.ModelOrientation = np.rad2deg(ShoreNormDir)
    
    NcFile.Kcoef = PhysicalPars['Kcoef']
    NcFile.RhoSed = PhysicalPars['RhoSed']
    NcFile.RhoSea = PhysicalPars['RhoSea']
    NcFile.RhoRiv = PhysicalPars['RhoRiv']
    NcFile.VoidRatio = PhysicalPars['VoidRatio']
    NcFile.Gravity = PhysicalPars['Gravity']
    NcFile.GammaRatio = PhysicalPars['GammaRatio']
    NcFile.WaveDataDepth = PhysicalPars['WaveDataDepth']
    NcFile.ClosureDepth = PhysicalPars['ClosureDepth']
    NcFile.BeachSlope = PhysicalPars['BeachSlope']
    NcFile.RiverSlope = PhysicalPars['RiverSlope']
    NcFile.GrainSize = PhysicalPars['GrainSize']
    NcFile.CritShieldsStress = PhysicalPars['CritShieldsStress']
    NcFile.UpstreamLength = PhysicalPars['UpstreamLength']
    NcFile.RiverWidth = PhysicalPars['RiverWidth']
    NcFile.RoughnessManning = PhysicalPars['RoughnessManning']
    NcFile.WidthDepthRatio = PhysicalPars['WidthDepthRatio']
    NcFile.BackshoreElev = PhysicalPars['BackshoreElev']
    NcFile.MaxOutletElev = PhysicalPars['MaxOutletElev']
    NcFile.OT_coef = PhysicalPars['OT_coef']
    NcFile.OT_exp = PhysicalPars['OT_exp']
    NcFile.BeachTopElev = PhysicalPars['BeachTopElev']
    NcFile.SpitWidth = PhysicalPars['SpitWidth']
    NcFile.TargetBarHeight = PhysicalPars['TargetBarHeight']
    NcFile.TargetBarWidth = PhysicalPars['TargetBarWidth']
    NcFile.MinOutletWidth = PhysicalPars['MinOutletWidth']
    NcFile.OutletSedSpreadDist = PhysicalPars['OutletSedSpreadDist']
    NcFile.OutletBankEroFac = PhysicalPars['OutletBankEroFac']
    NcFile.BarrierPermeability = PhysicalPars['BarrierPermeability']
    NcFile.ShorelineErosionRate = PhysicalPars['ShorelineErosionRate']
    
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
    
    #%% create boundary condition variables
    SeaLevelVar = NcFile.createVariable('sea_level', np.float32, 
                                        (TimeDim.name,))
    SeaLevelVar.units = 'm'
    SeaLevelVar.long_name = 'Sea level'
    
    RiverFlowVar = NcFile.createVariable('river_flow', np.float32, 
                                         (TimeDim.name,))
    RiverFlowVar.units = 'm3/s'
    RiverFlowVar.long_name = 'River flow upstream of hapua'
    
    WaveHeightVar = NcFile.createVariable('hs_offshore', np.float32, 
                                          (TimeDim.name,))
    WaveHeightVar.units = 'm'
    WaveHeightVar.long_name = 'Offshore significant wave height'
    
    WaveDirVar = NcFile.createVariable('wave_dir', np.float32, 
                                       (TimeDim.name,))
    WaveDirVar.units = 'degrees'
    WaveDirVar.long_name = 'Direction of wave energy at model input point (depth=WaveDataDepth)'
    
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
    
    # Barrier crest level
    BarrierElevVar = NcFile.createVariable('barrier_crest_z', np.float32, 
                                           (TimeDim.name, XDim.name))
    BarrierElevVar.units = 'm'
    BarrierElevVar.long_name = 'Barrier crest elevation at each transect'
    
    # Inner barrier crest level
    InnerBarrierElevVar = NcFile.createVariable('inner_barrier_crest_z', np.float32, 
                                                (TimeDim.name, XDim.name))
    InnerBarrierElevVar.units = 'm'
    InnerBarrierElevVar.long_name = 'Inner barrier crest elevation at each transect'
    
    # Lagoon water level
    LagoonWlVar = NcFile.createVariable('lagoon_wl', np.float32, 
                                        (TimeDim.name, XDim.name))
    LagoonWlVar.units = 'm'
    LagoonWlVar.long_name = 'Lagoon water level at each transect'
    
    # Lagoon velocity
    LagoonVelVar = NcFile.createVariable('lagoon_vel', np.float32, 
                                          (TimeDim.name, XDim.name))
    LagoonVelVar.units = 'm/s'
    LagoonVelVar.long_name = 'Mean lagoon velocity at each transect (+ve = outflowing i.e. towards outlet channel)'
    
    # Lagoon bedload transport
    LagoonBedloadVar = NcFile.createVariable('lagoon_bedload', np.float32, (TimeDim.name, XDim.name))
    LagoonBedloadVar.units = 'm3(bulk including voids)/s'
    LagoonBedloadVar.long_name = 'bedlaod transport in lagoon at each transect (+ve = outflowing i.e. towards outlet channel)'
    
    # Outlet water level
    OutletWlVar = NcFile.createVariable('outlet_wl', np.float32, 
                                          (TimeDim.name, XDim.name))
    OutletWlVar.units = 'm'
    OutletWlVar.long_name = 'Outlet water level at each transect'
    
    # Outlet velocity
    OutletVelVar = NcFile.createVariable('outlet_vel', np.float32, 
                                          (TimeDim.name, XDim.name))
    OutletVelVar.units = 'm/s'
    OutletVelVar.long_name = 'Mean outlet channel velocity at each transect (+ve = outflowing i.e. towards sea)'
    
    # Outlet bedload transport
    OutletBedloadVar = NcFile.createVariable('outlet_bedload', np.float32, (TimeDim.name, XDim.name))
    OutletBedloadVar.units = 'm3(bulk including voids)/s'
    OutletBedloadVar.long_name = 'bedlaod transport in outlet channel at each transect (+ve = outflowing i.e. towards sea)'
        
    # Longshore transport
    LstVar = NcFile.createVariable('lst', np.float32, (TimeDim.name, XSegDim.name))
    LstVar.units = 'm3/hour'
    LstVar.long_name = 'longshore transport rate at each transect'
    
    # Crossshore transport
    CstVar = NcFile.createVariable('cst', np.float32, (TimeDim.name, XDim.name))
    CstVar.units = 'm3/hour'
    CstVar.long_name = 'cross-shore transport rate from the shore face to the barrier at each transect'
    
    # Overwash proportion
    OWVar = NcFile.createVariable('overwash_proportion', np.float32, (TimeDim.name, XDim.name))
    OWVar.units = '0 to 1'
    OWVar.long_name = 'proportion of cross-shore transport going over barrier to backshore (as opposed to onto barrier crest)'
    
    
    
    #%% Create river variables
    
    # River bed elevation
    RiverElevVar = NcFile.createVariable('river_bed_z', np.float32, 
                                         (TimeDim.name, RivDim.name))
    RiverElevVar.units = 'm'
    RiverElevVar.long_name = 'River cross-section bed elevation'
    
    # River water level
    RiverWlVar = NcFile.createVariable('river_wl', np.float32, 
                                       (TimeDim.name, RivDim.name))
    RiverWlVar.units = 'm'
    RiverWlVar.long_name = 'River cross-section water level'
    
    # River velocity
    RiverVelVar = NcFile.createVariable('river_vel', np.float32, 
                                        (TimeDim.name, RivDim.name))
    RiverVelVar.units = 'm/s'
    RiverVelVar.long_name = 'River cross-section mean velocity'
    
    # Bedload transport
    RiverBedloadVar = NcFile.createVariable('river_bedload', np.float32, 
                                            (TimeDim.name, RivDim.name))
    RiverBedloadVar.units = 'm3(bulk including voids)/s'
    RiverBedloadVar.long_name = 'River cross-section bedload transport'
    
    #%% Create outlet end variables
    
    # Outlet end position
    OutletEndXVar = NcFile.createVariable('outlet_end_x', np.float32, 
                                          (TimeDim.name, EndsDim.name))
    OutletEndXVar.units = 'm'
    OutletEndXVar.long_name = 'Outlet end position'
    
    # Outlet end width
    OutletEndBVar = NcFile.createVariable('outlet_end_width', np.float32, 
                                          TimeDim.name)
    OutletEndBVar.units = 'm'
    OutletEndBVar.long_name = 'Outlet end width'
    
    # Outlet end elevation
    OutletEndElevVar = NcFile.createVariable('outlet_end_z', np.float32, 
                                             TimeDim.name)
    OutletEndElevVar.units = 'm'
    OutletEndElevVar.long_name = 'Outlet end bed elevation'
    
    # Outlet end water level
    OutletEndWlVar = NcFile.createVariable('outlet_end_wl', np.float32, 
                                           TimeDim.name)
    OutletEndWlVar.units = 'm'
    OutletEndWlVar.long_name = 'Outlet end water level'
    
    # Outlet end velocity
    OutletEndVelVar = NcFile.createVariable('outlet_end_vel', np.float32, 
                                            TimeDim.name)
    OutletEndVelVar.units = 'm/s'
    OutletEndVelVar.long_name = 'Outlet end velocity (+ve = outflowing i.e. towards sea)'
    
    # Outlet end bedload transport
    OutletEndBedloadVar = NcFile.createVariable('outlet_end_bedload', np.float32, 
                                                TimeDim.name)
    OutletEndBedloadVar.units = 'm3(bulk including voids)/s'
    OutletEndBedloadVar.long_name = 'bedlaod transport at outlet channel ends (+ve = outflowing i.e. towards sea)'
    
    # Dummy XS water level
    DummyXsWlVar = NcFile.createVariable('dummy_xs_dep', np.float32, 
                                         TimeDim.name)
    DummyXsWlVar.units = 'm'
    DummyXsWlVar.long_name = 'Dummy XS water depth (downstream of outlet i.e. in sea)'
    
    # Dummy XS velocity
    DummyXsVelVar = NcFile.createVariable('dummy_xs_vel', np.float32, 
                                            TimeDim.name)
    DummyXsVelVar.units = 'm/s'
    DummyXsVelVar.long_name = 'Dummy XS velocity (downstream of outlet i.e. in sea, +ve = outflowing i.e. towards sea)'
    
    # Closed/open status
    OutletClosedVar = NcFile.createVariable('outlet_closed', 'i1', 
                                            TimeDim.name)
    OutletClosedVar.units = ''
    OutletClosedVar.long_name = 'Outlet status: true=closed, false=open'
    
    # Close netCDF file
    NcFile.close()

def writeCurrent(FileName, CurrentTime, SeaLevel, RivFlow, Hs_offshore, EDir_h,
                 ShoreY, ShoreZ, LagoonWL, LagoonVel, LagoonBedload, 
                 OutletDep, OutletVel, OutletBedload, 
                 LST, CST, OverwashProp,
                 RiverElev, RiverDep, RiverVel, RiverBedload,
                 OutletEndX, OutletEndElev, OutletEndWidth, 
                 OutletEndDep, OutletEndVel, OutletEndBedload, Closed):
    """ Write hapuamod outputs for current timestep to netCDF 
        
        writeCurrent(FileName, CurrentTime, SeaLevel, RivFlow,
                     Hs_offshore, EDir_h,
                     ShoreY, ShoreZ, LagoonWL, LagoonVel, LagoonBedload, 
                     OutletDep, OutletVel, OutletBedload, 
                     LST, CST, OverwashProp,
                     RiverElev, RiverDep, RiverVel, RiverBedload,
                     OutletEndX, OutletEndElev, OutletEndWidth, 
                     OutletEndDep, OutletEndVel, OutletEndBedload, Closed)
        
        Note: For this function boundary condition data should be supplied as 
              single floats rather than arrays/dataframes as used elsewhere.
    """
    
    # Open netCDF file for appending
    NcFile = netCDF4.Dataset(FileName, mode='a') 
    
    # Get index of new time row to add and add current time
    TimeVar = NcFile.variables['time']
    TimeIx = TimeVar.size
    TimeVar[TimeIx] = netCDF4.date2num(CurrentTime, TimeVar.units)
    
    # Append data to boundary condition variables
    NcFile.variables['sea_level'][TimeIx] = SeaLevel
    NcFile.variables['river_flow'][TimeIx] = RivFlow
    NcFile.variables['hs_offshore'][TimeIx] = Hs_offshore
    NcFile.variables['wave_dir'][TimeIx] = EDir_h
    
    # Append new data to shore transect variables
    NcFile.variables['shoreline_y'][TimeIx,:] = ShoreY[:,0]
    NcFile.variables['outlet_bank1_y'][TimeIx,:] = ShoreY[:,1]
    NcFile.variables['outlet_bank2_y'][TimeIx,:] = ShoreY[:,2]
    NcFile.variables['lagoon_y'][TimeIx,:] = ShoreY[:,3]
    NcFile.variables['cliff_y'][TimeIx,:] = ShoreY[:,4]
    
    NcFile.variables['barrier_crest_z'][TimeIx,:] = ShoreZ[:,0]
    NcFile.variables['outlet_bed_z'][TimeIx,:] = ShoreZ[:,1]
    NcFile.variables['inner_barrier_crest_z'][TimeIx,:] = ShoreZ[:,2]
    NcFile.variables['lagoon_bed_z'][TimeIx,:] = ShoreZ[:,3]
    
    NcFile.variables['lagoon_wl'][TimeIx,:] = LagoonWL
    NcFile.variables['lagoon_vel'][TimeIx,:] = LagoonVel
    NcFile.variables['lagoon_bedload'][TimeIx,:] = LagoonBedload
    NcFile.variables['outlet_wl'][TimeIx,:] = OutletDep + ShoreZ[:,1]
    NcFile.variables['outlet_vel'][TimeIx,:] = OutletVel
    NcFile.variables['outlet_bedload'][TimeIx,:] = OutletBedload
    
    NcFile.variables['cst'][TimeIx,:] = CST*3600
    NcFile.variables['overwash_proportion'][TimeIx,:] = OverwashProp
    
    NcFile.variables['lst'][TimeIx,:] = LST*3600
    
    # Append new data to river variables
    NcFile.variables['river_bed_z'][TimeIx,:] = RiverElev
    NcFile.variables['river_wl'][TimeIx,:] = RiverElev + RiverDep
    NcFile.variables['river_vel'][TimeIx,:] = RiverVel
    NcFile.variables['river_bedload'][TimeIx,:] = RiverBedload
    
    # Append new data to outlet end variables
    NcFile.variables['outlet_end_x'][TimeIx,:] = OutletEndX
    NcFile.variables['outlet_end_width'][TimeIx] = OutletEndWidth
    NcFile.variables['outlet_end_z'][TimeIx] = OutletEndElev
    NcFile.variables['outlet_end_wl'][TimeIx] = OutletEndElev + OutletEndDep[0]
    NcFile.variables['outlet_end_vel'][TimeIx] = OutletEndVel[0]
    NcFile.variables['dummy_xs_dep'][TimeIx] = OutletEndDep[1]
    NcFile.variables['dummy_xs_vel'][TimeIx] = OutletEndVel[1]
    NcFile.variables['outlet_closed'][TimeIx] = Closed
    
    NcFile.close()

def readTimestep(NcFile, TimeIx):
    """ Extract model results for a specific timestep from netcdf file
        
        (SeaLevel, RivFlow, Hs_offshore, EDir_h, 
         ShoreX, ShoreY, ShoreZ, LagoonWL, LagoonVel, OutletWL, 
         OutletVel, OutletEndX, OutletEndWidth, OutletEndElev, OutletEndVel, 
         OutletEndWL, OutletChanIx, WavePower, EDir_h, LST, CST, Closed, 
         RiverElev, RiverWL, RiverVel, ModelTime) = readTimestep(NcFile, TimeIx)
        
        Example
        -------
        > NcFile = netCDF4.Dataset(FileName, mode='r', format='NETCDF4_CLASSIC') 
        
        > (SeaLevel, RivFlow, Hs_offshore, EDir_h, 
           ShoreX, ShoreY, ShoreZ, LagoonWL, LagoonVel, OutletWL, 
        >  OutletVel, OutletEndX, OutletEndWidth, OutletEndElev, OutletEndVel, 
        >  OutletEndWL, OutletChanIx, WavePower, EDir_h, LST, CST, Closed, 
        >  RiverElev, RiverWL, RiverVel, ModelTime) = readTimestep(NcFile, TimeIx)
        
        > NcFile.close()
    """
    
    NTransects = NcFile.dimensions['transect_x'].size
    
    SeaLevel = NcFile.variables['sea_level'][TimeIx]
    RivFlow = NcFile.variables['river_flow'][TimeIx]
    Hs_offshore = NcFile.variables['hs_offshore'][TimeIx]
    EDir_h = NcFile.variables['wave_dir'][TimeIx]
    
    ShoreX = NcFile.variables['transect_x'][:]
    
    ModelTime = netCDF4.num2date(NcFile.variables['time'][TimeIx],
                                 NcFile.variables['time'].units)
    
    # ShoreY
    ShoreY = np.empty((NTransects, 5))
    ShoreY[:,0] = NcFile.variables['shoreline_y'][TimeIx,:]
    ShoreY[:,1] = NcFile.variables['outlet_bank1_y'][TimeIx,:]
    ShoreY[:,2] = NcFile.variables['outlet_bank2_y'][TimeIx,:]
    ShoreY[:,3] = NcFile.variables['lagoon_y'][TimeIx,:]
    ShoreY[:,4] = NcFile.variables['cliff_y'][TimeIx,:]
    
    # ShoreZ
    ShoreZ=np.empty((NTransects, 4))
    ShoreZ[:,0] = NcFile.variables['barrier_crest_z'][TimeIx,:]
    ShoreZ[:,1] = NcFile.variables['outlet_bed_z'][TimeIx,:]
    ShoreZ[:,2] = NcFile.variables['inner_barrier_crest_z'][TimeIx,:]
    ShoreZ[:,3] = NcFile.variables['lagoon_bed_z'][TimeIx,:]
    
    LagoonWL = NcFile.variables['lagoon_wl'][TimeIx,:].squeeze()
    LagoonVel = NcFile.variables['lagoon_vel'][TimeIx,:].squeeze()
    OutletWL = NcFile.variables['outlet_wl'][TimeIx,:].squeeze()
    OutletVel = NcFile.variables['outlet_vel'][TimeIx,:].squeeze()
    
    OutletEndX = NcFile.variables['outlet_end_x'][TimeIx,:].squeeze()
    OutletEndWidth = np.asarray(NcFile.variables['outlet_end_width'][TimeIx].squeeze())
    OutletEndElev = np.asarray(NcFile.variables['outlet_end_z'][TimeIx].squeeze())
    OutletEndVel = np.asarray(NcFile.variables['outlet_end_vel'][TimeIx].squeeze())
    OutletEndWL = np.asarray(NcFile.variables['outlet_end_wl'][TimeIx].squeeze())
    DummyXsDep = np.asarray(NcFile.variables['dummy_xs_dep'][TimeIx].squeeze())
    DummyXsVel = np.asarray(NcFile.variables['dummy_xs_vel'][TimeIx].squeeze())
    Closed = bool(NcFile.variables['outlet_closed'][TimeIx])
    
    WavePower=None
    EDir_h=0
    LST = NcFile.variables['lst'][TimeIx,:] / 3600
    CST = NcFile.variables['cst'][TimeIx,:] / 3600
    
    RiverElev = NcFile.variables['river_bed_z'][TimeIx,:].squeeze()
    RiverWL  = NcFile.variables['river_wl'][TimeIx,:].squeeze()
    RiverVel  = NcFile.variables['river_vel'][TimeIx,:].squeeze()
    
    # Calculate OutletChanIx
    if Closed:
        # Outlet closed
        OutletChanIx = np.empty(0)
    elif OutletEndX[0] < OutletEndX[1]:
        # Outlet angles from L to R
        OutletChanIx = np.where(np.logical_and(OutletEndX[0] <= ShoreX, 
                                               ShoreX <= OutletEndX[1]))[0]
    else:
        # Outlet from R to L
        OutletChanIx = np.flipud(np.where(np.logical_and(OutletEndX[1] <= ShoreX,
                                                         ShoreX <= OutletEndX[0]))[0])
    
    return(SeaLevel, RivFlow, Hs_offshore, EDir_h,
           ShoreX, ShoreY, ShoreZ, LagoonWL, LagoonVel, OutletWL, 
           OutletVel, OutletEndX, OutletEndWidth, OutletEndElev, OutletEndVel, 
           OutletEndWL, DummyXsDep, DummyXsVel, OutletChanIx, WavePower, EDir_h, LST, CST, Closed, 
           RiverElev, RiverWL, RiverVel, ModelTime)
    
def closestTimeIx(NcFile, DatetimeOfInterest):
    """ Find result file timestep index closest to desired datetime value
        
        TimeIx = closestTimeIx(NcFile, DatetimeOfInterest)
    """
    NetCdfTimes = NcFile.variables['time'][:]
    OutputTimes = pd.to_datetime(netCDF4.num2date(NetCdfTimes, NcFile.variables['time'].units))
    
    TimeDiffs = abs(OutputTimes - DatetimeOfInterest)
    TimeIx = np.where(TimeDiffs==min(TimeDiffs))[0]
    ClosestTime = OutputTimes[TimeIx].to_pydatetime()[0]
    
    return (TimeIx, ClosestTime)

def newTsOutFile(FileName, ModelName, StartTime, Overwrite=False):
    """ Crete new output file for high frequency timeseries 
        
        newTsOutFile(FileName, ModelName, StartTime, Overwrite=False)
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
    TimeDim = NcFile.createDimension('time', None)
    EndsDim = NcFile.createDimension('outlet_ends', 2)
    
    # create attributes
    NcFile.ModelName = ModelName
    NcFile.ModelStartTime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # create coordinate variables
    TimeVar = NcFile.createVariable(TimeDim.name, np.float64, (TimeDim.name,))
    TimeVar.units = 'seconds since %s' % StartTime.strftime('%Y-%m-%d')
    TimeVar.calendar = 'standard'
    TimeVar.long_name = 'Model output times'
    
    #%% create boundary condition variables
    
    # sea level boundary condition
    SeaLevelVar = NcFile.createVariable('sea_level', np.float32, 
                                        (TimeDim.name,))
    SeaLevelVar.units = 'm'
    SeaLevelVar.long_name = 'Sea level'
    
    # river inflow boundary condition
    RiverFlowVar = NcFile.createVariable('river_flow', np.float32, 
                                         (TimeDim.name,))
    RiverFlowVar.units = 'm3/s'
    RiverFlowVar.long_name = 'River flow upstream of hapua'
    
    # offshore significant wave height boundary condition
    WaveHeightVar = NcFile.createVariable('hs_offshore', np.float32, 
                                          (TimeDim.name,))
    WaveHeightVar.units = 'm'
    WaveHeightVar.long_name = 'Offshore significant wave height'
    
    # wave direction boundary condition
    WaveDirVar = NcFile.createVariable('wave_dir', np.float32, 
                                       (TimeDim.name,))
    WaveDirVar.units = 'degrees'
    WaveDirVar.long_name = 'Direction of wave energy at model input point (depth=WaveDataDepth)'
    
    #%% create other timeseries variables
    
    # mean lagoon level
    LagoonLevelVar = NcFile.createVariable('lagoon_level', np.float32, 
                                           (TimeDim.name,))
    LagoonLevelVar.units = 'm'
    LagoonLevelVar.long_name = 'Mean lagoon water level'
    
    # lagoon outflow rate
    LagoonOutflowVar = NcFile.createVariable('lagoon_outflow', np.float32, 
                                             (TimeDim.name,))
    LagoonOutflowVar.units = 'm3/s'
    LagoonOutflowVar.long_name = 'Lagoon outflow to ocean (negative value = inflow from ocean to lagoon)'
    
    # sed inflow rate
    SedInflowVar = NcFile.createVariable('sed_inflow', np.float32, 
                                         (TimeDim.name,))
    SedInflowVar.units = 'm3/s'
    SedInflowVar.long_name = 'Bedload transport into upstream end of modelled river reach'
    
    # Outlet end positions
    OutletEndXVar = NcFile.createVariable('outlet_end_x', np.float32, 
                                          (TimeDim.name, EndsDim.name))
    OutletEndXVar.units = 'm'
    OutletEndXVar.long_name = 'Outlet end position'
    
    # Closed/open status
    OutletClosedVar = NcFile.createVariable('outlet_closed', 'i1', 
                                            TimeDim.name)
    OutletClosedVar.units = ''
    OutletClosedVar.long_name = 'Outlet status: true=closed, false=open'
    
    # Morphological timestep
    MorDtVar = NcFile.createVariable('mordt', np.float32, 
                                     (TimeDim.name,))
    MorDtVar.units = 's'
    MorDtVar.long_name = 'Morphological timestep'
    
    #%% Close netCDF file
    NcFile.close()

def writeTsOut(FileName, CurrentTime, SeaLevel, RivFlow, Hs_offshore, EDir_h,
               MeanLagoonWl, LagOutflow, SedInflow, OutletEndX, Closed, MorDt):
    """ Write hapuamod outputs to timeseries output netCDF file
        
        writeTsOut(FileName, CurrentTime, SeaLevel, RivFlow, 
                   Hs_offshore, EDir_h, MeanLagoonWl, LagOutflow, 
                   SedInflow, OutletEndX, Closed, MorDt)
    """
    
    # Open netCDF file for appending
    NcFile = netCDF4.Dataset(FileName, mode='a') 
    
    # Get index of new time row to add and add current time
    TimeVar = NcFile.variables['time']
    TimeIx = TimeVar.size
    TimeVar[TimeIx] = netCDF4.date2num(CurrentTime, TimeVar.units)
    
    # Append data to boundary condition variables
    NcFile.variables['sea_level'][TimeIx] = SeaLevel
    NcFile.variables['river_flow'][TimeIx] = RivFlow
    NcFile.variables['hs_offshore'][TimeIx] = Hs_offshore
    NcFile.variables['wave_dir'][TimeIx] = EDir_h
    
    # Append new data to other high frequency outputs
    NcFile.variables['lagoon_level'][TimeIx] = MeanLagoonWl
    NcFile.variables['lagoon_outflow'][TimeIx] = LagOutflow             
    NcFile.variables['sed_inflow'][TimeIx] = SedInflow
    NcFile.variables['outlet_end_x'][TimeIx,:] = OutletEndX
    NcFile.variables['outlet_closed'][TimeIx] = Closed
    NcFile.variables['mordt'][TimeIx] = MorDt.seconds
    
    NcFile.close()