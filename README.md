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
  * Wave parameters (Hs, period, direction)
  * Tide level (could be harmonic or constituent based rather than timeseries?)
* Initial conditions
* Other location descriptors?
  * River slope
  * Sed size
  * SLR rate
  * Backshore elevation
  * Barrier height
  * Coastal erosion rate (or perhaps sediment supply deficit?)
* Model parameters/coefficients
  * Mouth bypassing?
  * Overtopping/overwashing?
  * Bedload transport
  * Longshore transport
