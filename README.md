# HapuaModel
Physically based model of hapua morphology.

## Processes
* Longshore transport & shoreline evolution
* Wave overtopping and overwashing effect on barrier height and width (simplified!)
* Bedload transport (simplified)
* Bank erosion/channel width adjustment
* Water storage in lagoon (mass balance)
* Quasi steady hydraulics in outlet channel
* Sediment balance
* Barrier breach

## Inputs
* Timeseries
  * River flow
  * Wave parameters (Deep water significant wave height & period for runup, 
    Wave power & Deep water direction of wave power for LST)
  * Tide level including storm surge (could be harmonic or constituent based 
    rather than timeseries?)
* Initial conditions
  * Shoreline position
  * Lagoon foreshore and backshore position
  * Outlet channel position
* Other location descriptors?
  * River slope
  * Location river enters lagoon
  * Sed size
  * SLR rate
  * Backshore elevation
  * Barrier height
  * Coastal erosion rate (or perhaps sediment supply deficit?)
  * Beach slope
* Model parameters/coefficients
  * Mouth bypassing?
  * Overtopping/overwashing?
  * Bedload transport
  * Longshore transport
