import os
import optim_esm_tools as oet
from optim_esm_tools.plotting.map_maker import MapMaker
import numpy as np
import matplotlib.pyplot as plt
import typing as ty
from optim_esm_tools.analyze import tipping_criteria
import logging
from optim_esm_tools.analyze.cmip_handler import transform_ds, read_ds
import typing as ty
import matplotlib.pyplot as plt
from functools import wraps

import inspect
from optim_esm_tools.analyze.clustering import build_cluster_mask
from optim_esm_tools.plotting.plot import setup_map
from immutabledict import immutabledict

def _show(show):
    if show:
        plt.show()
    else:
        plt.clf()
        plt.close()


def mask_xr_ds(ds_masked, da_mask):
    for k, v in ds_masked.data_vars.items():
        if all(xy in list(v.dims) for xy in 'xy'):
            ds_masked[k] = ds_masked[k].where(da_mask, drop=False)
    return ds_masked


def plt_show(*a):
    def somedec_outer(fn):
        @wraps(fn)
        def plt_func(*args, **kwargs):
            res = fn(*args, **kwargs)
            self = args[0]
            if getattr(self, 'show', False):
                plt.show()
            else:
                plt.clf()
                plt.close()
            return res

        return plt_func

    if a and isinstance(a[0], ty.Callable):
        # Decorator that isn't closed
        return somedec_outer(a[0])
    return somedec_outer


def apply_options(*a):
    def somedec_outer(fn):
        @wraps(fn)
        def timed_func(*args, **kwargs):
            self = args[0]
            takes = inspect.signature(fn).parameters
            kwargs.update({k: v for k, v in self.extra_opt.items()
                           if k in takes
                           })
            res = fn(*args, **kwargs)
            return res

        return timed_func

    if a and isinstance(a[0], ty.Callable):
        # Decorator that isn't closed
        return somedec_outer(a[0])
    return somedec_outer


class ResultDataSet:
    _logger: logging.Logger = None
    labels: tuple = tuple('ii iii'.split())
    show: bool = True

    show_basic = True
    criteria = (tipping_criteria.StdDetrended, tipping_criteria.MaxJump)
    extra_opt = None

    def __init__(self,
                 variable='tas',
                 path=None,
                 dataset=None,
                 transform=True,
                 save_kw=None,
                 extra_opt=None,
                 read_ds_kw=None) -> None:
        if path is None:
            if transform:
                self.log.warning(
                    f'Best is to start {self.__class__.__name__} from a synda path'
                )
                self.dataset = transform_ds(dataset)
            else:
                self.dataset = dataset
        else:
            read_ds_kw = dict() if read_ds_kw is None else read_ds_kw
            self.dataset = read_ds(path, **read_ds_kw)
        if save_kw is None:
            save_kw = dict(save_in='./',
                           file_types=('png', 'pdf',),
                           skip=False,
                           sub_dir=None)
        if extra_opt is None:
            extra_opt = dict()
        self.extra_opt = extra_opt
        self.save_kw = save_kw
        self.variable = variable

    @property
    def log(self):
        if self._logger is None:
            self._logger = oet.config.get_logger()
        return self._logger

    @apply_options
    def workflow(self, show_basic=True):
        if show_basic:
            self.plot_basic_map()
        masks = self.get_masks()
        self.plot_masks(masks)
        self.plot_mask_time_series(masks)

    @plt_show
    def plot_basic_map(self):
        self._plot_basic_map()
        self.save(f'{self.title_label}_global_map')

    def _plot_basic_map(self):
        mm = MapMaker(self.dataset)
        return mm.plot_all(2)

    def save(self, name):
        oet.utils.save_fig(name, **self.save_kw)

    @property
    def title(self):
        return MapMaker(self.dataset).title

    @property
    def title_label(self):
        return MapMaker(self.dataset).title.replace(' ', '_')


class MaxRegion(ResultDataSet):
    def get_masks(self) -> dict:
        """Get mask for max of ii and iii and a box arround that"""
        labels = [crit.short_description for crit in self.criteria]
        masks = {label: self.dataset[label].values == self.dataset[label].values.max()
                 for label in labels
                 }
        return masks

    @plt_show
    def plot_masks(self, masks, ax=None, legend=True):
        res = self._plot_masks(masks=masks, ax=ax, legend=legend,)
        self.save(f'{self.title_label}_map_maxes_{"-".join(self.labels)}')

    @apply_options
    def _plot_masks(self, masks, ax=None, legend=True):
        points = {}
        for key, mask_2d in masks.items():
            points[key] = self._mask_to_coord(mask_2d)
        if ax is None:
            oet.plotting.plot.setup_map()
            ax = plt.gca()
        for i, (label, xy) in enumerate(zip(
            self.labels,
            points.values())
        ):
            ax.scatter(*xy, marker='oxv^'[i], label=f'Maximum {label}')
        if legend:
            ax.legend(**oet.utils.legend_kw())
        plt.suptitle(self.title, y=0.95)
        plt.ylim(-90, 90)
        plt.xlim(-180, 180)

    def _mask_to_coord(self, mask_2d):
        arg_mask = np.argwhere(mask_2d)[0]
        x = self.dataset.x[arg_mask[1]]
        y = self.dataset.y[arg_mask[0]]
        return x, y

    def _plot_basic_map(self):
        mm = MapMaker(self.dataset)
        axes = mm.plot_all(2)
        masks = self.get_masks()
        for ax in axes:
            self._plot_masks(masks, ax=ax, legend=False)
        plt.suptitle(self.title, y=0.95)

    @plt_show
    def plot_mask_time_series(self, masks, time_series_joined=True):
        res = self._plot_mask_time_series(
            masks, time_series_joined=time_series_joined)
        if time_series_joined:
            self.save(
                f'{self.title_label}_time_series_maxes_{"-".join(self.labels)}')
        return res

    @apply_options
    def _plot_mask_time_series(self, masks, time_series_joined=True, only_rm=False, axes=None):
        legend_kw = oet.utils.legend_kw(
            loc='upper left', bbox_to_anchor=None, mode=None, ncol=4)
        for label, mask_2d in zip(self.labels, masks.values()):
            x, y = self._mask_to_coord(mask_2d)
            plot_labels = {f'{self.variable}': f'{label} at {x:.1f}:{y:.1f}',
                           f'{self.variable}_detrend': f'{label} at {x:.1f}:{y:.1f}',
                           f'{self.variable}_detrend_run_mean_10': f'$RM_{{10}}$ {label} at {x:.1f}:{y:.1f}',
                           f'{self.variable}_run_mean_10': f'$RM_{{10}}$ {label} at {x:.1f}:{y:.1f}'}
            argwhere = np.argwhere(mask_2d)[0]
            ds_sel = self.dataset.isel(x=argwhere[1], y=argwhere[0])
            mm_sel = MapMaker(ds_sel)
            axes = mm_sel.time_series(
                other_dim=(),
                interval=False,
                labels=plot_labels,
                axes=axes,
                only_rm=only_rm)
            if time_series_joined is False:
                axes = None
                plt.suptitle(f'Max. {label} {self.title}', y=0.95)
                self.save(f'{self.title_label}_time_series_max_{label}')
                _show(self.show)
        if not time_series_joined:
            return

        for ax in axes:
            ax.legend(**legend_kw)
        plt.suptitle(f'Max. {"-".join(self.labels)} {self.title}', y=0.95)


class Percentiles(ResultDataSet):
    @apply_options
    def get_masks(self, percentiles=99) -> dict:
        """Get mask for max of ii and iii and a box arround that"""
        labels = [crit.short_description for crit in self.criteria]
        masks = []
        vmin_vmax = []

        for lab in labels:
            arr = self.dataset[lab].values.T
            vmin_vmax.append([np.min(arr), np.max(arr)])
            thr = np.percentile(arr, percentiles)
            masks.append(arr >= thr)

        all_mask = np.ones_like(masks[0])
        for m in masks:
            all_mask &= m

        masks, clusters = build_cluster_mask(
            all_mask, self.dataset['x'].values, self.dataset['y'].values)
        return masks, clusters

    @plt_show
    def plot_masks(self, masks_and_clusters, ax=None, legend=True):
        if not len(masks_and_clusters[0]):
            return
        res = self._plot_masks(
            masks_and_clusters=masks_and_clusters, ax=ax, legend=legend,)
        self.save(f'{self.title_label}_map_clusters_{"-".join(self.labels)}')

    @apply_options
    def _plot_masks(self, masks_and_clusters,
                    scatter_medians=True,
                    ax=None, legend=True, mask_cbar_kw=None,cluster_kw=None):
        masks, clusters = masks_and_clusters
        all_masks = np.zeros(masks[0].shape, np.int16)

        for m, c in zip(masks, clusters):
            all_masks[m] = len(c)
        if ax is None:
            setup_map()
            ax = plt.gca()
        if mask_cbar_kw is None:
            mask_cbar_kw = dict(extend='neither', label='Number of gridcells')
        mask_cbar_kw.setdefault('orientation', 'horizontal')
        ds_dummy = self.dataset.copy()

        all_masks = all_masks.astype(np.float16)
        all_masks[all_masks == 0] = np.nan
        ds_dummy['n_grid_cells'] = (('y', 'x'), all_masks)

        ds_dummy['n_grid_cells'].plot(cbar_kwargs=mask_cbar_kw,

                                      vmin=0, extend='neither')
        plt.title('')
        if scatter_medians:
            if cluster_kw is None:
                cluster_kw = dict()
            for m_i, cluster in enumerate(clusters):
                ax.scatter(*np.median(cluster, axis=0), label=f'cluster {m_i}', **cluster_kw)
            if legend:
                plt.legend(**oet.utils.legend_kw())
        plt.suptitle(f'Clusters {self.title}', y=0.95)
        return ax

    def _plot_basic_map(self):
        mm = MapMaker(self.dataset)
        axes = mm.plot_all(2)
        plt.suptitle(self.title, y=0.95)
        return axes

        # Could add some masked selection on top
#         masks, _ = self.get_masks()

#         all_masks = masks[0]
#         for m in masks[1:]:
#             all_masks &= m
#         ds_masked = mask_xr_ds(self.dataset.copy(), all_masks)
#         mm_sel = MapMaker(ds_masked)
#         for label, ax in zip(mm.labels, axes):
#             plt.sca(ax)
#             mm_sel.plot_i(label, ax=ax, coastlines=False)

    @plt_show
    def plot_mask_time_series(self, masks_and_clusters, time_series_joined=True):
        res = self._plot_mask_time_series(
            masks_and_clusters, time_series_joined=time_series_joined)
        if time_series_joined:
            self.save(f'{self.title_label}_all_clusters')
        return res

    @apply_options
    def _plot_mask_time_series(self, masks_and_clusters, time_series_joined=True, only_rm=None, axes=None):
        if only_rm is None:
            only_rm = True if (len(masks_and_clusters[0]) > 1 and time_series_joined) else False
        masks, clusters = masks_and_clusters
        legend_kw = oet.utils.legend_kw(
            loc='upper left', bbox_to_anchor=None, mode=None, ncol=4)
        for m_i, (mask, cluster) in enumerate(zip(masks, clusters)):
            x, y = np.median(cluster, axis=0)
            plot_labels = {f'{self.variable}': f'Cluster {m_i} near ~{x:.1f}:{y:.1f}',
                           f'{self.variable}_detrend': f'Cluster {m_i} near ~{x:.1f}:{y:.1f}',
                           f'{self.variable}_detrend_run_mean_10': f'Cluster {m_i} $RM_{{10}}$ near ~{x:.1f}:{y:.1f}',
                           f'{self.variable}_run_mean_10': f'Cluster {m_i} $RM_{{10}}$ near ~{x:.1f}:{y:.1f}'}
            ds_sel = mask_xr_ds(self.dataset.copy(), mask)
            mm_sel = MapMaker(ds_sel)
            axes = mm_sel.time_series(
                other_dim=('x', 'y'),
                interval=True,
                labels=plot_labels,
                axes=axes,
                only_rm=only_rm)
            if time_series_joined == False:
                axes = None
                plt.suptitle(f'Cluster. {m_i} {self.title}', y=0.95)
                self.save(f'{self.title_label}_cluster_{m_i}')
                _show(self.show)
        if not time_series_joined:
            return

        if axes is not None:
            for ax in axes:
                ax.legend(**legend_kw)
        plt.suptitle(f'Clusters {self.title}', y=0.95)


class PercentilesHistory(Percentiles):
    @apply_options
    def get_masks(self, percentiles=99, read_ds_kw=None) -> dict:
        if read_ds_kw is None:
            read_ds_kw = dict()
        for k, v in dict(min_time=None, max_time=None).items():
            read_ds_kw.setdefault(k, v)
        historical_path = self.find_historical()[0]
        historical_ds = read_ds(historical_path, **read_ds_kw)

        labels = [crit.short_description for crit in self.criteria]
        masks = []

        for lab in labels:
            arr = self.dataset[lab].values.T
            arr_historical = historical_ds[lab].values.T
            thr = np.percentile(arr_historical, percentiles)
            masks.append(arr >= thr)

        all_mask = np.ones_like(masks[0])
        for m in masks:
            all_mask &= m

        masks, clusters = build_cluster_mask(
            all_mask, self.dataset['x'].values, self.dataset['y'].values)
        return masks, clusters

    @apply_options
    def find_historical(self, match_to='piControl'):
        from optim_esm_tools.config import config
        base = os.path.join(os.sep,
                            *self.dataset.attrs['path'].split(os.sep)[:-len(config['CMIP_files']['folder_fmt'].split())-1])

        search = oet.cmip_files.find_matches.folder_to_dict(
            self.dataset.attrs['path'])
        search['activity_id'] = 'CMIP'
        if search['experiment_id'] == match_to:
            raise NotImplementedError()
        search['experiment_id'] = match_to

        first_try = oet.cmip_files.find_matches.find_matches(base, **search)
        if first_try:
            return first_try
        self.log.warning(
            'No results at first try, retying with any variant_label')
        search.update(dict(variant_label='*', ))

        second_try = oet.cmip_files.find_matches.find_matches(base, **search)
        if second_try:
            return second_try
        self.log.warning('No results at second try, retying with any version')
        search.update(dict(version='*', ))
        third_try = oet.cmip_files.find_matches.find_matches(base, **search)
        if third_try:
            return third_try
        raise RuntimeError

    @property
    def log(self):
        if self._logger is None:
            self._logger = oet.config.get_logger()
        return self._logger
