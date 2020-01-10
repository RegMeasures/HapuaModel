# -*- coding: utf-8 -*-
"""
Synthetic timeseries generation module
"""
import pandas as pd
import numpy as np
import math
import random
import logging

def dailyShotNoise(StartDate, EndDate, OutputInt,
                   MeanTimeBetweenEvents, MeanEventIncrease, DecayRate1, 
                   PropRate1 = 1.0, DecayRate2 = None, RisingLimbTime = pd.Timedelta(minutes=1), 
                   CalcTimeInt = None, RandomSeed = None):
    """ Creates a stochastic shot noise timeseries (optionally second order)
        
    Following the principles described by Weiss (1977) except for the 
    addition of a parameter allowing for a simple linear rising limb 
    (original model has instantaneously increases associated wiht each 
    event).
    
    (OutputTs, EventList) = dailyShotNoise(StartDate, EndDate, OutputInt,
                                           MeanTimeBetweenEvents, 
                                           MeanEventIncrease, DecayRate1, 
                                           PropRate1, DecayRate2, 
                                           RisingLimbTime, CalcTimeInt, 
                                           RandomSeed)
    
    Parameters
    ----------
    StartDate : pd.datetime
        Datetime for start of timeseries.
    EndDate : pd.datetime
        Datetime for end of timeseries.
    OutputInt : pd.Timedelta
        Time interval for output timeseries.
    MeanTimeBetweenEvents : pd.Timedelta
        Input parameter controlling frequency of events on the shot noise 
        model.
    MeanEventIncrease : float
        Input parameter controlling the magnitude of events in the shot noise 
        model.
    DecayRate1 : float
        Input parameter controlling the exponential decay rate in the 
        shot-noise model (units = per day).
    PropRate1 : float, optional
        Proportion of event magnitude which decays at DecayRate1 as opposed to 
        DecayRate2. (0 < PropRate1 <= 1, default = 1 i.e. first order shot 
        noise) .
    DecayRate2 : float, optional
        As for DecayRate1 (but only required if PropRate1 < 1)
    RisingLimbTime : pd.Timedelta, optional
        Time over which the output timeseries increases for each event 
        (default = 1 minute).
    CalcTimeInt : pd.Timedelta
        Max time interval for calculation of timeseries prior to resampling at 
        desired OutputTimeInt.
    RandomSeed : int, optional
        Seed for initialising the random number generator. Allows identical 
        'random' timeseries to be regenerated. See random.seed for more 
                 information (default = None, whilst non-integer values are 
                 valid an integer value is recommended to ensure 
                 cross-platform consistency).
    
    Returns
    -------
    OutputTs : pd.Series
        stochastic shot noise timeseries with regularly spaced datapoints.
    EventList  : pd.Series
        Series with the exact time and magnitude of each event in the generated
        output timeseries.
            
    References
    ----------
    Weiss G. (1977) Shot noise models for the generation of synthetic 
    streamflow data. Water Resources Research 13(1):101–108.
    
    Examples
    --------
    >>> StartDate = pd.datetime(2020,1,1)
    >>> EndDate = pd.datetime(2021,1,1)
    >>> OutputInt = pd.Timedelta(hours=1)
    >>> (OutputTs, EventList) = dailyShotNoise(StartDate, EndDate, OutputInt,
    ...                                        pd.Timedelta(days=15), 
                                               100, 0.3, 0.9, 0.015)
    >>> OutputTS.plot()
    """
    # Validate inputs
    assert DecayRate1 > 0, 'DecayRate must be greater than 0'
    assert 0.0 <= PropRate1 <= 1.0, 'PropRate1 must be in the range 0 to 1'
    if PropRate1 < 1:
        assert not DecayRate2 is None, 'If PropRate1 is less than 1 then DecayRate2 must be specified'
        assert DecayRate2 > 0, 'DecayRate must be greater than 0'
    assert RisingLimbTime > pd.Timedelta(days=0), 'Rising limb time must be greater than 0'
    if CalcTimeInt is None:
        CalcTimeInt = OutputInt/4
    
    # Initialise a random number generator from a specific seed value (to allow repeatability)
    r = random.Random(RandomSeed)
    
    # Calculation additional (derived) input parameters
    EventRate = 1/MeanTimeBetweenEvents.days
    PropRate2 = 1-PropRate1
    Rate1Vals = [MeanEventIncrease * (PropRate1 / DecayRate1) * EventRate]
    Rate2Vals = [MeanEventIncrease * (PropRate2 / DecayRate2) * EventRate]
    MeanVal = Rate1Vals[0] + Rate2Vals[0]
    
    logging.info('Generating synthetic shot-noise timeseries for period %s to %s' % (StartDate, EndDate))
    logging.info('Estimated mean value of generated timeseries = %.3f' % MeanVal)
        
    EventList = pd.Series([MeanVal], index=pd.DatetimeIndex([StartDate]))
    OutputTs = EventList.copy()
    
    LastEventTime = StartDate
    while LastEventTime < EndDate:
        # Get the (random) parameters of the next event
        GapLength = pd.Timedelta(days = -math.log(r.random()) / EventRate)
        EventSize = -MeanEventIncrease * math.log(1 - r.random())
        
        # Extend the output timeseries up to the next event
        if GapLength < RisingLimbTime:
            # Deal with situation where the gap between events is shorter than the rising limb time...
          
            # Insert new (interpolated) time into output TS at start of new events rising limb
            OutputTs = OutputTs.append(pd.Series(np.interp(pd.DatetimeIndex([LastEventTime + GapLength - RisingLimbTime]).values.astype(float), 
                                                           OutputTs.index[-2:].values.astype(float), 
                                                           OutputTs[-2:]),
                                                 index = pd.DatetimeIndex([LastEventTime + GapLength - RisingLimbTime])))
            OutputTs.sort_index(inplace=True)
            
            # Increase the flow at the last time in the output TS (as its part way through the rising limb of the next event)
            OutputTs[-1] += np.interp(OutputTs.index[-1:].values.astype(float), 
                                      pd.DatetimeIndex([LastEventTime + GapLength - RisingLimbTime, LastEventTime + GapLength]).values.astype(float), 
                                      [0, EventSize])
            
            # Add a new data point at the peak of the new event
            TimeList = pd.DatetimeIndex([LastEventTime + GapLength])

        else:
            # Normal case where gap between events is longer than the rising limb time
            
            # Only need to add new data points - no other messing arund required...
            TimeList = pd.date_range(start = LastEventTime, freq = CalcTimeInt,
                                     end = LastEventTime + GapLength - RisingLimbTime,
                                     closed = 'right')
            TimeList = TimeList.append(pd.DatetimeIndex([LastEventTime + GapLength - RisingLimbTime, LastEventTime + GapLength]))
        
        DaysSinceLastEvent = (TimeList - LastEventTime).total_seconds().values / 86400
        Rate1Vals = Rate1Vals[-1] * np.exp(-DecayRate1 * DaysSinceLastEvent)
        Rate2Vals = Rate2Vals[-1] * np.exp(-DecayRate2 * DaysSinceLastEvent)
        Rate1Vals[-1] += EventSize * PropRate1
        Rate2Vals[-1] += EventSize * PropRate2
        TotVals = Rate1Vals + Rate2Vals
        OutputTs = OutputTs.append(pd.Series(TotVals, index = TimeList))
        
        
        # Also create a timeseries of 'Events'
        LastEventTime += GapLength
        EventList = EventList.append(pd.Series([EventSize], index = pd.DatetimeIndex([LastEventTime])))
    
    # Resample the output timeseries and clip
    OutputTs = OutputTs.resample(OutputInt).mean()
    OutputTs = OutputTs.loc[:EndDate]    
    
    return(OutputTs, EventList)