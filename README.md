# OptimESM Tools
[![Coverage Status](https://coveralls.io/repos/github/JoranAngevaare/optim_esm_tools/badge.svg)](https://coveralls.io/github/JoranAngevaare/optim_esm_tools)
[![PyPI version shields.io](https://img.shields.io/pypi/v/optim-esm-tools.svg)](https://pypi.python.org/pypi/optim-esm-tools/)
[![Python Versions](https://img.shields.io/pypi/pyversions/optim-esm-tools.svg)](https://pypi.python.org/pypi/optim-esm-tools)
[![PyPI downloads](https://img.shields.io/pypi/dm/optim-esm-tools.svg)](https://pypistats.org/packages/optim-esm-tools)
[![CodeFactor](https://www.codefactor.io/repository/github/joranangevaare/optim_esm_tools/badge)](https://www.codefactor.io/repository/github/joranangevaare/optim_esm_tools)


J.R. Angevaare (KNMI)

## Software
This software is used in the scope of the [OptimESM](https://cordis.europa.eu/project/id/101081193) project.
The scientific aim is to isolate regions of three dimensional earth science data (time, latitude and longitude) from CMIP6 and identify regions in latitude-longitude that show dramatic changes as function of time.

## Setup
This software requires [`cdo`](https://code.mpimet.mpg.de/projects/cdo) and [`cartopy`](https://github.com/SciTools/cartopy), and preferably also `latex` and `R`. 
For downloading CMIP6 data, [`synda`](https://espri-mod.github.io/synda/index.html#) is a useful tool, and few routines work best with the associated the [`ESGF`](https://pcmdi.llnl.gov/)-file structure.
Since `synda` list is only supported in python 3.8, we created a separate repository [`optim_esm_base`](https://github.com/JoranAngevaare/optim_esm_base) that has a working set of  software versions that are compatible with these requirements.

After these base requirements are fulfilled, one can install this software via pip
```
pip install optim_esm_tools
```

Alternatively, setting up a miniforge/conda environment is documented in [`optim_esm_base`](https://github.com/JoranAngevaare/optim_esm_base).

## Example
In the [notebooks folder](https://github.com/JoranAngevaare/optim_esm_tools/tree/master/notebooks), we have an [example notebook](https://github.com/JoranAngevaare/optim_esm_tools/blob/master/notebooks/example.ipynb) to help you get started. 
More advanced tutorials are also available in the notebooks folder.
