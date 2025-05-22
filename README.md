# OptimESM Tools
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15275184.svg)](https://doi.org/10.5281/zenodo.15275184)
[![Coverage Status](https://coveralls.io/repos/github/JoranAngevaare/optim_esm_tools/badge.svg)](https://coveralls.io/github/JoranAngevaare/optim_esm_tools)
[![PyPI version shields.io](https://img.shields.io/pypi/v/optim-esm-tools.svg)](https://pypi.python.org/pypi/optim-esm-tools/)
[![Python Versions](https://img.shields.io/pypi/pyversions/optim-esm-tools.svg)](https://pypi.python.org/pypi/optim-esm-tools)
[![PyPI downloads](https://img.shields.io/pypi/dm/optim-esm-tools.svg)](https://pypistats.org/packages/optim-esm-tools)
[![CodeFactor](https://www.codefactor.io/repository/github/joranangevaare/optim_esm_tools/badge)](https://www.codefactor.io/repository/github/joranangevaare/optim_esm_tools)


J.R. Angevaare (KNMI)

## Software
This software is used in the scope of the [OptimESM](https://cordis.europa.eu/project/id/101081193) project.
The scientific aim is to isolate regions of three dimensional earth science data (time, latitude and longitude) from CMIP6 and identify regions in latitude-longitude that show dramatic changes as function of time.

## Setup and installation
This software requires [`cdo`](https://code.mpimet.mpg.de/projects/cdo) and [`cartopy`](https://github.com/SciTools/cartopy), and preferably also `latex`.

To install `cdo` and `py-cdo`, we use `miniforge`, but alternative methods exits. To use `miniforge` (on linux):
```
wget https://github.com/conda-forge/miniforge/releases/download/25.3.0-3/Miniforge3-Linux-x86_64.sh
bash Miniforge3-Linux-x86_64.sh
```
Create an environment, e.g.:
```
conda create -n py310 python=3.10.13 cdo
# and activate
conda ativate py310
```
```
# install the software
pip install optim_esm_tools
```

You should be ready to go!

### Testing installation (optional)
After installation, you could test that everything is properly installed and all the dependencies are working.
```
# Clone https://github.com/JoranAngevaare/optim_esm_tools
git clone https://github.com/JoranAngevaare/optim_esm_tools.git
cd optim_esm_tools
pytest -v --durations 0
```
You could even test that all the notebooks work out of the box with the jupyter setup in your environment:
```
# Clone https://github.com/JoranAngevaare/optim_esm_tools
git clone https://github.com/JoranAngevaare/optim_esm_tools.git
cd optim_esm_tools
pip install -r requirements_tests.txt
pytest optim_esm_tools -v --nbmake -n3 notebooks/*.ipynb --durations 0
```

## Extended setup (optional)
For downloading CMIP6 data, [`synda`](https://espri-mod.github.io/synda/index.html#) is a useful tool, and few routines work best with the associated the [`ESGF`](https://pcmdi.llnl.gov/)-file structure.
Since `synda` list is only supported in python 3.8, we created a separate repository [`optim_esm_base`](https://github.com/JoranAngevaare/optim_esm_base) that has a working set of  software versions that are compatible with these requirements.


Alternatively, setting up a miniforge/conda environment is documented in [`optim_esm_base`](https://github.com/JoranAngevaare/optim_esm_base).

## Example
In the [notebooks folder](https://github.com/JoranAngevaare/optim_esm_tools/tree/master/notebooks), we have an [example notebook](https://github.com/JoranAngevaare/optim_esm_tools/blob/master/notebooks/example.ipynb) to help you get started.
More advanced tutorials are also available in the notebooks folder.
