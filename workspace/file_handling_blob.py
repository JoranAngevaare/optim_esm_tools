import typing
from collections import defaultdict
import os
import glob
import xarray as xr
from tqdm.notebook import tqdm
import numpy as np
from xmip.preprocessing import promote_empty_dims, replace_x_y_nominal_lat_lon, rename_cmip6
from xmip.preprocessing import correct_coordinates, parse_lon_lat_bounds, maybe_convert_bounds_to_vertex, \
    maybe_convert_vertex_to_bounds, broadcast_lonlat
from abc import ABC

#
# from cdo import Cdo
# from netCDF4 import Dataset as ds
# import bottleneck as bn
# import matplotlib.pyplot as plt
# import cartopy
# import pandas as pd
# import matplotlib.pyplot as plt
# import cartopy.crs as ccrs
# from xmip.postprocessing import interpolate_grid_label


def load_glob(
        pattern,
) -> xr.Dataset:
    return xr.open_mfdataset(
        pattern,
        concat_dim="time",
        combine="nested",
        data_vars='minimal',
        coords='minimal',
        compat='override',
    )


def recast(data_set):
    ds = rename_cmip6(data_set)
    ds = promote_empty_dims(ds)
    ds = broadcast_lonlat(ds)
    ds = replace_x_y_nominal_lat_lon(ds)
    return ds


class _ResultBase(ABC):
    path: str
    data: xr.Dataset


class ResultIO(_ResultBase):
    def __init__(self, path):
        self.path = path
        assert os.path.exists(path)
        self.data = self.read()

    def read(self) -> xr.Dataset:
        pattern = os.path.join(self.path, '*')
        files = sorted(glob.glob(pattern))
        print(f'Got {len(files)} files')
        files_merged = load_glob(pattern)
        print('Read done')
        return files_merged


class ResultOperations(_ResultBase):
    _warned = False

    def get_yearly_means(self, data: xr.Dataset) -> xr.Dataset:
        import warnings
        # https://stackoverflow.com/a/71963300
        if not self._warned:
            warnings.warn('Averaging monthly (not accounting for days/month)')
            self._warned = True
        return data.groupby('time.year').mean('time')  # .mean(dim='time')

    @staticmethod
    def set_temperature_std(merged) -> None:
        pass

    def set_yearly_means(self):
        self.data = self.get_yearly_means(self.data)


class ResultShower(_ResultBase):
    _tqdm_active = True

    def __repr__(self):
        return f'Data from {self.path}, {self.nbytes / 1e9:.1f} GB'

    @property
    def nbytes(self):
        return self.data.nbytes

    @property
    def log(self):
        raise NotImplementedError

    def tqdm(self, *args, **kwargs):
        kwargs.setdefault('disable', self._tqdm_active)
        return tqdm(*args, **kwargs)


class ExtractChange(_ResultBase):
    deg_res = 10
    y_bins = np.arange(-90, 90 + 1, deg_res)
    x_bins = np.arange(0, 360 + 1, deg_res)
    registry: typing.Dict[str, typing.Dict[str, xr.Dataset]] = defaultdict(dict)

    def set_slices(self) -> None:
        pbar = self.tqdm(total=(len(self.x_bins) - 1) * (len(self.y_bins) - 1)
                         )
        if self.registry:
            raise ValueError('sliced already')
        for x_i, x_l in enumerate(self.x_bins[:-1]):
            x_r = self.x_bins[x_i + 1]
            for y_i, y_l in enumerate(self.y_bins[:-1]):
                y_r = self.y_bins[y_i + 1]
                x_label = f'_x_{x_l:03}:{x_r:03}'
                y_label = f'_y_{y_l:03}:{y_r:03}'
                pbar.desc = x_label + y_label
                pbar.display()
                pbar.n += 1
                selection = self.data.sel(
                    y=slice(y_l, y_r),
                    x=slice(x_l, x_r)
                )
                check = [len(
                    selection[xy].values
                )
                    for xy in 'xy']
                if not all(check):
                    print(f'No result for {pbar.desc} {check}')
                    continue
                self.registry[y_label][x_label] = selection.mean(['x', 'y'])

                pbar.display()
        pbar.close()


class Result(ResultIO, ResultShower, ResultOperations, ExtractChange):
    pass


if __name__ == '__main__':
    changes = Result('/nobackup/users/angevaar/synda/data/CMIP6/ScenarioMIP'
                     '/MIROC/MIROC-ES2L/ssp585/r3i1p1f2/Omon/tos/gr1/v20201222/*.nc')
    changes.set_slices()
