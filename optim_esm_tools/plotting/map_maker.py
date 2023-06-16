# -*- coding: utf-8 -*-
import optim_esm_tools as oet
import xarray as xr
import numpy as np

import typing as ty
import collections
from warnings import warn

import matplotlib.pyplot as plt

from immutabledict import immutabledict


# import xrft

from optim_esm_tools.analyze.globals import _SECONDS_TO_YEAR
from optim_esm_tools.analyze.tipping_criteria import (
    StartEndDifference,
    StdDetrended,
    MaxJump,
    MaxDerivitive,
)


class MapMaker(object):
    data_set: xr.Dataset
    labels = set('i ii iii iv v vi vii viii ix x'.split())
    kw: ty.Mapping
    contitions: ty.Mapping

    def set_kw(self):
        import cartopy.crs as ccrs

        self.kw = immutabledict(
            fig=dict(dpi=200, figsize=(14, 10)),
            title=dict(fontsize=12),
            gridspec=dict(hspace=0.3),
            cbar=dict(orientation='horizontal', extend='both'),
            plot=dict(transform=ccrs.PlateCarree()),
        )

    def set_conditions(self, **condition_kwargs):
        conditions = [
            cls(**condition_kwargs)
            for cls in [StartEndDifference, StdDetrended, MaxJump, MaxDerivitive]
        ]

        self.conditions = {
            label: (condition.long_description, condition.calculate)
            for label, condition in zip(self.labels, conditions)
        }
        print(self.conditions)

    normalizations: ty.Optional[ty.Mapping] = None

    _cache: bool = False

    def __init__(
        self,
        data_set: xr.Dataset,
        normalizations: ty.Union[None, ty.Mapping, ty.Iterable] = None,
        cache: bool = False,
        **conditions,
    ):
        self.data_set = data_set
        self.set_kw()
        self.set_conditions(**conditions)
        if normalizations is None:
            self.normalizations = {i: [None, None] for i in self.conditions.keys()}
        elif isinstance(normalizations, collections.abc.Mapping):
            self.normalizations = normalizations
        elif isinstance(normalizations, collections.abc.Iterable):
            self.normalizations = {
                i: normalizations[j] for j, i in enumerate(self.conditions.keys())
            }

        def _incorrect_format():
            return (
                any(
                    not isinstance(v, collections.abc.Iterable)
                    for v in self.normalizations.values()
                )
                or any(len(v) != 2 for v in self.normalizations.values())
                or any(k not in self.normalizations for k in self.conditions)
            )

        if self.normalizations is None or _incorrect_format():
            raise TypeError(
                f'Normalizations should be mapping from'
                f'{self.conditions.keys()} to vmin, vmax, '
                f'got {self.normalizations} (from {normalizations})'
            )
        self._cache = cache

    def plot(self, *a, **kw):
        print('Depricated use plot_all')
        return self.plot_all(*a, **kw)

    @oet.utils.timed()
    def plot_all(
        self,
        nx=2,
        fig=None,
        **kw,
    ):
        import cartopy.crs as ccrs

        ny = np.ceil(len(self.conditions) / nx).astype(int)
        if fig is None:
            fig = plt.figure(**self.kw['fig'])

        from matplotlib.gridspec import GridSpec

        gs = GridSpec(nx, ny, **self.kw['gridspec'])
        plt_axes = []

        for i, label in enumerate(self.conditions.keys()):
            ax = fig.add_subplot(
                gs[i],
                projection=ccrs.PlateCarree(
                    central_longitude=0.0,
                ),
            )
            self.plot_i(label, ax=ax, **kw)
            plt_axes.append(ax)
        return plt_axes

    @oet.utils.timed()
    def plot_i(self, label, ax=None, coastlines=True, **kw):
        if ax is None:
            ax = plt.gca()
        if coastlines:
            ax.coastlines()

        prop = getattr(self, label)

        cmap = plt.get_cmap('viridis').copy()
        cmap.set_extremes(under='cyan', over='orange')

        c_kw = self.kw['cbar'].copy()
        c_range_kw = {
            vm: self.normalizations[label][j]
            for j, vm in enumerate('vmin vmax'.split())
        }

        for k, v in {
            **self.kw['plot'],
            **c_range_kw,
            **dict(
                cbar_kwargs=c_kw,
                cmap=cmap,
            ),
        }.items():
            kw.setdefault(k, v)

        plt_ax = prop.plot(**kw)

        plt.xlim(-180, 180)
        plt.ylim(-90, 90)
        description = self.conditions[label][0]
        ax.set_title(label.upper() + '\n' + description, **self.kw['title'])
        gl = ax.gridlines(draw_labels=True)
        gl.top_labels = False
        gl.right_labels = False
        return plt_ax

    def __getattr__(self, item):
        print(item)
        if item in self.conditions:
            _, function = self.conditions[item]
            key = f'_{item}'
            if self._cache:
                if not isinstance(self._cache, dict):
                    self._cache = dict()
                if key in self._cache:
                    data = self._cache.get(key)
                    return data

            data = function(self.data_set)
            if self._cache or isinstance(self._cache, dict):
                self._cache[key] = data.load()
            return data

        return self.__getattribute__(item)

    @staticmethod
    def _ts_single(time_val, mean, std, plot_kw, fill_kw):
        if fill_kw is None:
            fill_kw = dict(alpha=0.4, step='mid')
        l = mean.plot(**plot_kw)

        if std is not None:
            # TODO, make this more elegant!
            # import cftime
            # plt.fill_between(   [cftime.real_datetime(dd.year, dd.month, dd.day) for dd in time_val], mean - std, mean+std, **fill_kw)

            (mean - std).plot(color=l[0]._color, alpha=0.4)
            (mean + std).plot(color=l[0]._color, alpha=0.4)

    def _ts(
        self,
        variable,
        ds=None,
        time='time',
        other_dim=(),
        running_mean=10,
        fill_kw=None,
        labels=dict(),
        only_rm=False,
        **plot_kw,
    ):
        if ds is None:
            ds = self.data_set
        if not only_rm:
            mean, std = self._mean_and_std(ds, variable, other_dim)
            # return mean, std
            plot_kw['label'] = labels.get(variable, variable)
            self._ts_single(ds[time].values, mean, std, plot_kw, fill_kw)

        mean, std = self._mean_and_std(
            ds, f'{variable}_run_mean_{running_mean}', other_dim
        )
        plot_kw['label'] = labels.get(
            f'{variable}_run_mean_{running_mean}',
            f'{variable} running mean {running_mean}',
        )
        self._ts_single(ds[time].values, mean, std, plot_kw, fill_kw)

        plt.ylabel('T [K]')
        plt.legend()
        plt.title('')

    def _det_ts(
        self,
        variable,
        ds=None,
        time='time',
        other_dim=(),
        running_mean=10,
        fill_kw=None,
        labels=dict(),
        only_rm=False,
        **plot_kw,
    ):
        if ds is None:
            ds = self.data_set
        if not only_rm:
            mean, std = self._mean_and_std(ds, f'{variable}_detrend', other_dim)
            plot_kw['label'] = labels.get(
                f'{variable}_detrend', f'detrended {variable}'
            )
            self._ts_single(ds[time].values, mean, std, plot_kw, fill_kw)

        mean, std = self._mean_and_std(
            ds, f'{variable}_detrend_run_mean_{running_mean}', other_dim
        )
        plot_kw['label'] = labels.get(
            f'{variable}_detrend_run_mean_{running_mean}',
            f'detrended {variable} running mean {running_mean}',
        )
        self._ts_single(ds[time].values, mean, std, plot_kw, fill_kw)
        plt.ylabel('detrend(T) [K]')
        plt.legend()
        plt.title('')

    def _ddt_ts(
        self,
        variable,
        ds=None,
        time='time',
        other_dim=(),
        running_mean=10,
        fill_kw=None,
        labels=dict(),
        only_rm=False,
        **plot_kw,
    ):
        if ds is None:
            ds = self.data_set
        variable_rm = f'{variable}_run_mean_{running_mean}'

        da = ds[variable]
        da_rm = ds[variable_rm]

        if other_dim:
            da = da.mean(other_dim)
            da_rm = da_rm.mean(other_dim)
        if not only_rm:
            # Dropna should take care of any nones in the data-array
            dy_dt = da.dropna(time).differentiate(time)
            dy_dt *= _SECONDS_TO_YEAR
            # mean, std = self._mean_and_std(dy_dt, variable=None, other_dim=other_dim)
            # plot_kw['label'] = variable
            # self._ts_single(ds[time].values, mean, std, plot_kw, fill_kw)
            label = f'd/dt {labels.get(variable, variable)}'
            dy_dt.plot(label=label, **plot_kw)

        dy_dt_rm = da_rm.dropna(time).differentiate(time)
        dy_dt_rm *= _SECONDS_TO_YEAR
        label = (
            f"d/dt {labels.get(variable_rm, f'{variable} running mean {running_mean}')}"
        )
        dy_dt_rm.plot(label=label, **plot_kw)
        # mean, std = self._mean_and_std(dy_dt_rm, variable=None, other_dim=other_dim)
        # plot_kw['label'] = variable
        # self._ts_single(ds[time].values, mean, std, plot_kw, fill_kw)

        plt.ylim(dy_dt_rm.min() / 1.05, dy_dt_rm.max() * 1.05)
        plt.ylabel('$\partial T/\partial t$ [K/yr]')
        plt.legend()
        plt.title('')

    @staticmethod
    def _mean_and_std(ds, variable, other_dim):
        if variable is None:
            da = ds
        else:
            da = ds[variable]
        if other_dim is None:
            return da, None
        return da.mean(other_dim), da.std(other_dim)

    @oet.utils.timed()
    def time_series(
        self,
        variable='tas',
        time='time',
        other_dim=('x', 'y'),
        running_mean=10,
        interval=True,
        axes=None,
        **kw,
    ):
        if variable != 'tas':
            raise NotImplementedError('Currently only works for tas')

        ds = self.data_set
        if interval is False:
            ds = ds.copy().mean(other_dim)
            other_dim = None

        plot_kw = dict(**kw)

        if axes is None:
            _, axes = plt.subplots(3, 1, figsize=(12, 10), gridspec_kw=dict(hspace=0.3))

        plt.sca(axes[0])
        self._ts(
            variable, ds=ds, running_mean=running_mean, other_dim=other_dim, **plot_kw
        )

        plt.sca(axes[1])
        self._det_ts(
            variable, ds=ds, running_mean=running_mean, other_dim=other_dim, **plot_kw
        )

        plt.sca(axes[2])
        self._ddt_ts(
            variable,
            ds=ds,
            time=time,
            running_mean=running_mean,
            other_dim=other_dim,
            **plot_kw,
        )

        return axes

    @property
    def ds(self):
        warn('MapMaker.ds is depricated, use MapMaker.data_set')
        return self.data_set

    @property
    def title(self):
        return make_title(self.data_set)


def make_title(ds):
    return '{institution_id} {source_id} {experiment_id} {variant_label} {variable_id} {version}'.format(
        **ds.attrs
    )
