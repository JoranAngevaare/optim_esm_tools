import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

import optim_esm_tools as oet


def test_map_basic():
    oet.plotting.plot.setup_map()


def test_plot_da():
    oet.plotting.plot.plot_da(
        xr.DataArray(np.ones(12).reshape(4, 3), dims=('lat', 'lon')),
    )


def plot_something_else(**alt_option):
    plt.plot([-90, 90], [0, 360])
    oet.plotting.plot.setup_map(**alt_option)
    plt.close()
    plt.clf()


def test_plate_carree():
    plot_something_else(projection='PlateCarree')


def test_add_feature():
    plot_something_else(
        add_features='LAND OCEAN COASTLINE BORDERS LAKES RIVERS'.split(),
    )
