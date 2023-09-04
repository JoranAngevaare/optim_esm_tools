from abc import ABC

import numpy as np
import xarray as xr

import optim_esm_tools as oet
from ._base import apply_options
from .local_history import LocalHistory
from .percentiles import Percentiles
from .product_percentiles import ProductPercentiles
from optim_esm_tools.analyze.clustering import build_cluster_mask


class _ThresholdIterator(ABC):
    data_set: xr.Dataset

    def _get_masks_weighted(self, *a):
        raise NotImplementedError

    @apply_options
    def _get_masks_masked(
        self,
        percentile_range=(85, 100, 7),
        lon_lat_dim=('lon', 'lat'),
        _mask_method='all_pass_percentile',  # TODO or product_rank
        mask_min_area=1e12,
    ):
        already_seen = None
        masks, clusters = [], []
        pbar = oet.utils.tqdm(np.linspace(*percentile_range)[::-1])
        for percentile in pbar:
            pbar.desc = f'{percentile:.1f}%'
            pbar.display()
            all_mask = self._build_combined_mask(
                method=_mask_method,
                percentiles=percentile,
            )

            if already_seen is not None:
                all_mask[already_seen] = False

            these_masks, these_clusters = build_cluster_mask(
                all_mask,
                lon_coord=self.data_set[lon_lat_dim[0]].values,
                lat_coord=self.data_set[lon_lat_dim[1]].values,
            )
            for m, c in zip(these_masks, these_clusters):
                if self.mask_area(m).sum() >= mask_min_area:
                    masks.append(m)
                    clusters.append(c)
                    if already_seen is None:
                        already_seen = m
                    already_seen[m] = True
        pbar.close()
        return masks, clusters


class IterProductPercentiles(_ThresholdIterator, ProductPercentiles):
    pass


# class IterLocalHistory(_ThresholdIterator, LocalHistory):
#     pass


class IterPercentiles(_ThresholdIterator, Percentiles):
    pass
