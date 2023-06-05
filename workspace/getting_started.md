# Getting started is easy:

```
env_name='py310'
yes | conda create -n $env_name python=3.10.11 numpy numba scipy matplotlib jupyter ipython esgpull
conda install netcdf4 bottleneck
conda install -c conda-forge cdo
conda install -c IPSL synda
conda env config vars set ST_HOME=/nobackup/users/angevaar/synda


# sudo apt-get install myproxy
conda activate $env_name

conda install -c conda-forge cartopy

# https://github.com/jupyter-server/jupyter-resource-usage
pip install jupyter-resource-usage
jupyter serverextension enable --py jupyter_resource_usage --sys-prefix
jupyter nbextension install --py jupyter_resource_usage --sys-prefix
jupyter nbextension enable --py jupyter_resource_usage --sys-prefix

pip install jupyter_contrib_nbextensions
jupyter contrib nbextension install --sys-prefix
jupyter nbextension enable scratchpad/main --sys-prefix
jupyter nbextension enable execute_time/main --sys-prefix

pip install netcdf4 xarray pandas h5netcdf
```
