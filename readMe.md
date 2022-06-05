# ALENALYSIS

[![Language grade: Python](https://img.shields.io/lgtm/grade/python/g/saadjansari/alenalysis.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/saadjansari/alenalysis/context:python)
___
A python-based analysis pipeline for simulations performed via the aLENS software.
___
## Table of Contents:
* [Requirements](#requirements)
* [Installation](#installation)
* [Initialization](#initialization)
* [Usage](#usage)

## Requirements:

1. Simulation data from aLENS.
2. Anaconda

## Installation:
You can download alenalysis using git.

```
git clone https://github.com/saadjansari/alenalysis.git
```

## Initialization:

Begin by creating, and activating the conda environment:
```
conda env create -f environment.yml
conda activate pynalysis
```

There are 2 essential files that need to be configured for running alenalysis.
Start by duplicating the sample files:
```
cp sims_list.yaml.sample sims_list.yaml
cp config.yaml.sample config.yaml
```
#### ```sims_list.yaml```

* Set the correct simulation path:
``` 
path: <path/to/parent>
```

* Set the correct simulation names:
``` 
sims: 
    - <simulation1> 
    - <simulation2>
    - ...
```

#### ```config.yaml```

Turn on/off specific configuration parameters (boolean type):

_Computation Modes_

CorrelationMode
: Compute mode for intensive pair-pair correlations (default False).

ComputeFilamentCondensates
: Compute filament condensates. Splits the population into a dense and a vapor state.

ComputeFilamentClusters
: Compute high-density filament clusters.

ComputeCrosslinkerClusters
: Compute high-density crosslinker clusters.

_Graphics_

PlotTrajectories
: Filament trajectories of some N random filaments.

PlotCrosslinkerStats
: Crosslinker population statistics (number, state distributions)

PlotCrosslinkerLength
: Crosslinker stretch lengths/energies.

PlotGlobalOrderParameters
: Global order of filaments (nematic/polar order).

PlotExtensileMotors
: Population of motors causing causing extensile stresses (that bind anti-parallel filaments).

PlotEffectiveDiameter
: Effective diameter of filaments. This is a good measure of the probability of filament overlaps.

PlotCorrelations
: Correlations (Correlations data must be saved)

PlotDynamics
: Filament MSD/dynamics.

PlotRDF
: Radial density functions

PlotLocalMaps
: Local density/flux maps/movies

PlotPairDistribution
: Pair Distribution functions

_Write to VTK files (for visualization in paraview)_

WriteCorrelationsToVTK
: Correlation information

WriteAngleToVTK
: Polar and azimuthal angles.

WriteFilamentCondensatesToVTK
: Boolean flag determining whether a filament belongs to a condensate (True) or vapor (False)

### Usage:
To launch analysis, run:
``` python LAUNCH_LOCAL.py```
This calls ```main.py``` sequentially on the list of simulations.

Alternatively, to launch analysis in parallel on the CU Boulder supercomputing cluster, run:
``` python LAUNCH_SUMMIT.py```
**Note: you may need to edit the ```LAUNCH_SUMMIT.py``` file to meet your needs!**

