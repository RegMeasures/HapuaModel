""" Functions to analyse model results
"""

import netCDF4
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import logging
import argparse

def main(ResultsFileName):
    """
    """
    
    #%% Parse the inputs
    if ResultsFileName[-13:] == '_TSoutputs.nc':
        TsFileName = ResultsFileName
        MapFileName = ResultsFileName.replace('_TSoutputs.nc', '_outputs.nc')
    elif ResultsFileName[-11:] == '_outputs.nc':
        TsFileName = ResultsFileName.replace('_outputs.nc', '_TSoutputs.nc')
        MapFileName = ResultsFileName
    else:
        logging.error('Invalid results file name (%s). Expected *_TSoutputs.nc or *_outputs.nc' % ResultsFileName)
    
    BaseFileName = TsFileName.replace('_TSoutputs.nc', '')
    AnalysisReportFile = TsFileName.replace('_TSoutputs.nc', '_analysis.html')
    logging.info('Analysis report will be saved at %s' % AnalysisReportFile)
    
        
    #%% Setup to read data from NetCDF files
    # Open the NetCDF files for reading
    TsFile = netCDF4.Dataset(TsFileName, mode='r', format='NETCDF4_CLASSIC') 
    MapFile = netCDF4.Dataset(MapFileName, mode='r', format='NETCDF4_CLASSIC') 
    
    # Read time data
    TsTimes = pd.to_datetime(netCDF4.num2date(TsFile.variables['time'][:], 
                                              TsFile.variables['time'].units))
    MapTimes = pd.to_datetime(netCDF4.num2date(MapFile.variables['time'][:], 
                                               MapFile.variables['time'].units))
    
    # Derive some useful data from the times
    TsPeriods = np.empty(TsTimes.shape, dtype='timedelta64[s]')
    TsPeriods[1:-1] = (TsTimes[2:] - TsTimes[:-2]) / 2
    TsPeriods[0] = (TsTimes[1] - TsTimes[0])
    TsPeriods[-1] = (TsTimes[-1] - TsTimes[-2])
    
    TsYear = TsTimes.year.values
    
    #%% Lagoon water level distribution
    
    # Read WL data
    LagoonWL = pd.Series(TsFile.variables['lagoon_level'][:],
                         index = TsTimes)
    
    # Plot histogram
    LagWlFig, ax1 = plt.subplots()
    ax1.hist(LagoonWL, density=True, bins=np.arange(-1,4.1,0.1))
    ax1.set_xlabel('Lagoon water level (m)')
    ax1.set_ylabel('Frequency')
    ax1.set_yticks([])
    
    # Level duration curve
    ax2=ax1.twinx()
    Percentiles = np.arange(100)
    PercLevels = np.percentile(LagoonWL, Percentiles)
    ax2.plot(PercLevels, 100-Percentiles, c='blue')
    ax2.set_xlabel('Lagoon water level (m)')
    ax2.set_ylabel('Proportion of time level exceeded')
    ax2.set_xlim([-1,4])
    
    # Save plot
    LagWlFig.savefig(BaseFileName + '_LagoonWL.png')
    
    # add to html
    
    #%% Annual river sediment load
    
    # Read bedload supply rate [m3(bulk including voids)/s]
    SedFeed = pd.Series(TsFile.variables['sed_inflow'][:],
                        index = TsTimes)
    
    # Read void ratio to allow conversion to m3 solid rock
    VoidRatio = OutFile.VoidRatio
    
    # Calculate annual average load [m3/yr solid rock]
    SedLoad = SedFeed*TsPeriods.astype(int) / (1-VoidRatio)
    AnnualFeed = np.sum(SedLoad) / (((TsTimes[-1] - TsTimes[0]).days)/365.25)
    
    
    # look at annual variability
    YearlyLoad = SedLoad.resample('Y').sum()
    YearlySampleCount = SedLoad.resample('Y').count()
    WholeYears = YearlySampleCount > np.max(YearlySampleCount) * 0.95
    YearlyLoad = YearlyLoad[WholeYears]
    
    # Plot output
    LagWlFig, ax1 = plt.subplots()
    YearlyLoad.plot.hist(bins = np.arange(0,np.max(YearlyLoad)+20000,20000))
    YLim = plt.ylim()
    plt.plot([AnnualFeed, AnnualFeed], YLim, 'k--')
    plt.ylim(YLim)
    plt.xlabel('River sediment input ($m^3/yr$)')
    plt.ylabel('Number of years (%i to %i)' % 
               (YearlyLoad.index[0].year, YearlyLoad.index[-1].year))
    ' Mean annual bedload \n = %.1f thousand m³ per year' % (AnnualFeed/1000)
    
    #%% Coastal advance/retreat
    
    
    #%% Closure stats
    
    # Read closure data
    
    
    # Frequency of short closures (<12 hours)
    
    
    # Frequency of long closures (>12 hours)
    
    
    #%% Outlet position analysis
    
    # Read outlet position data
    
    # Calculate distribution of outlet length
    
    
    # Calculate distribution of outlet end position
    
    #%% Close read access to the netCDF files
    TsFile.close()
    MapFile.close()

if __name__ == "__main__":
    # Set up logging
    RootLogger = logging.getLogger()
    RootLogger.setLevel(logging.INFO)
    
    ConsoleHandler = logging.StreamHandler()
    ConsoleHandler.setLevel(logging.INFO)
    RootLogger.addHandler(ConsoleHandler)
    
    # Parse arguments from commandline
    Parser = argparse.ArgumentParser(description='Generate summary analysis of outputs from HapuaMod')
    Parser.add_argument('ResultsFile', help='HapuaMod netCDF formatted results file', type=str)
    # Parser.add_argument('-s', '--StartTimestep', type=int, default=0,
    #                     help='Optional timestep number to start analysis from ' +
    #                          '(integer, default=0)')
    # Parser.add_argument('-e', '--EndTimestep', type=int, default=None,
    #                     help='Optional timestep number to end analysis at ' +
    #                          '(integer, default=last available timestep)')
    ArgsIn = Parser.parse_args()
    
    # Run main animation function with specified arguments
    main(ArgsIn.ResultsFile)