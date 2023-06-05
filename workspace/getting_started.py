"""
Example of how to use xarray, cartopy and numpy to create annual changing maps

"""
from IPython.core.getipython import get_ipython

get_ipython().system('which python')

import os
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import cartopy.crs as ccrs
from tqdm.notebook import tqdm

import cartopy.feature as cfeature

tas_data = xr.open_dataset("tas_Amon_EC-Earth3_historical_r2i1p1f1_gr_185001-201412.nc")
air2d = tas_data.isel(time=500, bnds=0)
d_avg = tas_data.isel(bnds=0).mean('time')


def moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1 :] / n


def plot_time_average():
    plt.plot(
        t := tas_data.isel(lon=45, bnds=1, lat=45)['time'],
        y := tas_data.isel(lon=45, bnds=1, lat=45)['tas'],
    )
    kw = dict(n=100, label='ma with window of 3')
    plt.plot(
        np.array(
            moving_average(t.values.astype(float) / 1e9, **kw), dtype='datetime64[s]'
        ),
        moving_average(y.values, **kw),
    )
    plt.show()


def example_cartopy_plot():
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    ax.coastlines()
    d = d_avg
    x = ax.contourf(
        d['lon'],
        d['lat'],
        d['tas'],
        transform=ccrs.PlateCarree(),
        cmap='RdBu_r',  # vmin=-10, vmax=10
        #                        levels=np.linspace(-5,35, 100)
    )
    plt.gcf().colorbar(x, label='Temperature [K]', orientation='horizontal')
    plt.show()


def basic_cartopy_map():
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND)
    ax.add_feature(cfeature.COASTLINE)


def changes_with_respect_to_t0():
    vmin = (tas_data['tas'] - d_avg['tas']).values.min()
    vmax = (tas_data['tas'] - d_avg['tas']).values.max()

    def plot_tas(data_at_t0, data):
        plt.gca().coastlines()
        x = plt.gca().contourf(
            data['lon'],
            data['lat'],
            data['tas'].values - data_at_t0['tas'].values,
            transform=ccrs.PlateCarree(),
            cmap='RdBu_r',
            levels=np.arange(np.floor(vmin), np.ceil(vmax)),
        )
        plt.gcf().colorbar(x, label='Temperature [K]', orientation='horizontal')

    save_in = 'figs_lowres'
    os.mkdirs(save_in, exist_ok=True)
    for i, t in enumerate(tqdm(tas_data['time'].values)):
        n = f'{save_in}/{i:05}.png'

        fig = plt.figure()
        plt.title(t)
        plt.axis('off')

        for b in range(1):
            d = tas_data.isel(time=i, bnds=b)
            ax = fig.add_subplot(1, 1, 1 + b, projection=ccrs.PlateCarree())
            plt.sca(ax)
            plot_tas(d_avg, d)
            del d

        plt.savefig(n, dpi=100)
        plt.clf()
        plt.close()


def to_annual(data, time_index=0, rebin_by=12):  # months
    if not time_index == 0:
        raise ValueError
    shape = list(data.shape)

    new_shape = [shape[time_index] // rebin_by, rebin_by]
    shape.pop(time_index)
    new_shape = [*new_shape, *shape]
    print(new_shape)
    return data.reshape(*new_shape).mean(axis=1)


def annual_changes_with_respect_to_t0(work_in='annual'):
    array = tas_data.isel()['tas'].values
    re_array = to_annual(array)
    data_t0 = re_array[0].copy()
    vmin = (re_array - data_t0).min()
    vmax = (re_array - data_t0).max()
    min_max = np.max(np.abs([np.floor(vmin), np.ceil(vmax)]))

    def plot_tas(d0, d):
        plt.gca().coastlines()
        x = plt.gca().contourf(
            tas_data['lon'].values,
            tas_data['lat'].values,
            d - d0,
            transform=ccrs.PlateCarree(),
            cmap='RdBu_r',  # vmin=vmin, vmax=vmax
            levels=np.arange(-min_max, min_max),
        )
        plt.gcf().colorbar(x, label='Temperature [K]', orientation='horizontal')

    for i, (d, t) in enumerate(zip(tqdm(re_array), times)):
        n = f'{work_in}/{i:05}.png'

        fig = plt.figure()
        plt.title(t)
        plt.axis('off')

        ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
        plt.sca(ax)
        ax.add_feature(cfeature.BORDERS, linestyle='-', alpha=0.5, lw=1)
        plot_tas(data_t0, d)
        del d
        plt.savefig(n, dpi=100)
        plt.close()


if __name__ == '__main__':
    plot_time_average()
    example_cartopy_plot()
    changes_with_respect_to_t0()
