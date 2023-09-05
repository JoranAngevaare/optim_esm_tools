import matplotlib.pyplot as plt
import numpy as np

import optim_esm_tools as oet
from ._base import _mask_cluster_type
from ._base import apply_options
from ._base import plt_show
from ._base import RegionExtractor
from optim_esm_tools.plotting.map_maker import MapMaker
from optim_esm_tools.plotting.plot import _show


class MaxRegion(RegionExtractor):
    def get_masks(self) -> _mask_cluster_type:
        """Get mask for max of ii and iii and a box around that."""

        def _val(label):
            return self.data_set[label].values

        def _max(label):
            return _val(label)[~np.isnan(_val(label))].max()

        masks = [_val(label) == _max(label) for label in self._labels]
        return masks, [np.array([]) for _ in range(len(masks))]

    @apply_options
    def filter_masks_and_clusters(self, masks_and_clusters, min_area_km_sq=0):
        """Wrap filter to work on dicts."""
        if min_area_km_sq:  # pragma: no cover
            message = f'Calling {self.__class__.__name__}.filter_masks_and_clusters is nonsensical as masks are single grid cells'
            self.log.warning(message)
        return masks_and_clusters

    @plt_show
    def plot_masks(self, masks, ax=None, legend=True):
        masks = masks[0]
        self._plot_masks(masks=masks, ax=ax, legend=legend)
        self.save(f'{self.title_label}_map_maxes_{"-".join(self.labels)}')

        self.make_mask_figures(masks)

    @apply_options
    def _plot_masks(self, masks, ax=None, legend=True):
        points = {
            key: self._mask_to_coord(mask_2d)
            for key, mask_2d in zip(self._labels, masks)
        }
        if ax is None:
            oet.plotting.plot.setup_map()
            ax = plt.gca()
        for i, (label, xy) in enumerate(zip(self.labels, points.values())):
            ax.scatter(
                *xy,
                marker='oxv^'[i],
                label=f'Maximum {label}',
                transform=oet.plotting.plot.get_cartopy_transform(),
            )
        if legend:
            ax.legend(**oet.utils.legend_kw())
        plt.suptitle(self.title, y=0.95)
        _xlim, _ylim = oet.plotting.plot.get_xy_lim_for_projection()
        plt.xlim(*_xlim)
        plt.ylim(*_ylim)

    def _mask_to_coord(self, mask_2d):
        arg_mask = np.argwhere(mask_2d)[0]
        x = self.data_set.lon[arg_mask[1]]
        y = self.data_set.lat[arg_mask[0]]
        return x, y

    def _plot_basic_map(self):
        mm = MapMaker(self.data_set)
        axes = mm.plot_selected(items=self.labels)
        masks = self.get_masks()
        for ax in axes:
            self._plot_masks(masks[0], ax=ax, legend=False)
        plt.suptitle(self.title, y=0.95)

    @plt_show
    @apply_options
    def plot_mask_time_series(self, masks, time_series_joined=True):
        res = self._plot_mask_time_series(masks, time_series_joined=time_series_joined)
        if time_series_joined:
            self.save(f'{self.title_label}_time_series_maxes_{"-".join(self.labels)}')
        return res

    @apply_options
    def _plot_mask_time_series(
        self,
        masks_and_clusters,
        time_series_joined=True,
        only_rm=False,
        axes=None,
        _ma_window=None,
    ):
        _ma_window = _ma_window or oet.config.config['analyze']['moving_average_years']
        masks = masks_and_clusters[0]
        legend_kw = oet.utils.legend_kw(
            loc='upper left',
            bbox_to_anchor=None,
            mode=None,
            ncol=2,
        )
        for label, mask_2d in zip(self._labels, masks):
            x, y = self._mask_to_coord(mask_2d)
            plot_labels = {
                f'{self.variable}': f'{label} at {x:.1f}:{y:.1f}',
                f'{self.variable}_detrend': f'{label} at {x:.1f}:{y:.1f}',
                f'{self.variable}_detrend_run_mean_{_ma_window}': f'$RM_{{{_ma_window}}}$ {label} at {x:.1f}:{y:.1f}',
                f'{self.variable}_run_mean_{_ma_window}': f'$RM_{{{_ma_window}}}$ {label} at {x:.1f}:{y:.1f}',
            }
            argwhere = np.argwhere(mask_2d)[0]
            ds_sel = self.data_set.isel(lat=argwhere[0], lon=argwhere[1])
            mm_sel = MapMaker(ds_sel)
            axes = mm_sel.time_series(
                variable=self.variable,
                other_dim=(),
                show_std=False,
                labels=plot_labels,
                axes=axes,
                only_rm=only_rm,
            )
            if time_series_joined is False:
                axes = None
                plt.suptitle(f'Max. {label} {self.title}', y=0.95)
                self.save(f'{self.title_label}_time_series_max_{label}')
                _show(self.show)
        if not time_series_joined:
            return
        assert axes is not None
        for ax in axes:
            ax.legend(**legend_kw)
        plt.suptitle(f'Max. {"-".join(self.labels)} {self.title}', y=0.95)

    @property
    def _labels(self):
        return [crit.short_description for crit in self.criteria]
