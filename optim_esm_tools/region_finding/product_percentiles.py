from ._base import _two_sigma_percent, apply_options, _mask_cluster_type
import optim_esm_tools as oet
from optim_esm_tools.analyze import tipping_criteria
from optim_esm_tools.region_finding.percentiles import Percentiles
from optim_esm_tools.plotting.map_maker import HistoricalMapMaker
from optim_esm_tools.analyze.clustering import (
    build_cluster_mask,
    build_weighted_cluster,
)
from optim_esm_tools.utils import deprecated
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import abc
import typing as ty
import immutabledict
import os


class ProductPercentiles(Percentiles):
    labels = ('ii', 'iii', 'v')

    @oet.utils.check_accepts(
        accepts=immutabledict.immutabledict(cluster_method=('weighted', 'masked'))
    )
    @apply_options
    def get_masks(self, cluster_method='masked') -> _mask_cluster_type:
        """Get mask for max of ii and iii and a box around that"""
        if cluster_method == 'weighted':
            masks, clusters = self._get_masks_weighted()
        else:
            masks, clusters = self._get_masks_masked()
        if len(masks):
            self.check_shape(masks[0])
        return masks, clusters

    @apply_options
    def _get_masks_weighted(self, min_weight=0.95, lon_lat_dim=('lon', 'lat')):
        labels = [crit.short_description for crit in self.criteria]
        masks = []

        ds = self.data_set.copy()
        combined_score = np.ones_like(ds[labels[0]].values)

        for label in labels:
            combined_score *= tipping_criteria.rank2d(ds[label].values)
        self.check_shape(combined_score)
        masks, clusters = build_weighted_cluster(
            weights=combined_score,
            lon_coord=self.data_set[lon_lat_dim[0]].values,
            lat_coord=self.data_set[lon_lat_dim[1]].values,
            threshold=min_weight,
        )
        return masks, clusters

    @apply_options
    def _get_masks_masked(
        self, product_percentiles=_two_sigma_percent, lon_lat_dim=('lon', 'lat')
    ) -> _mask_cluster_type:
        """Get mask for max of ii and iii and a box around that"""
        labels = [crit.short_description for crit in self.criteria]
        masks = []

        ds = self.data_set.copy()
        combined_score = np.ones_like(ds[labels[0]].values)
        for label in labels:
            combined_score *= tipping_criteria.rank2d(ds[label].values)

        # Combined score is fraction, not percent!
        all_mask = combined_score > (product_percentiles / 100)

        self.check_shape(all_mask)

        masks, clusters = build_cluster_mask(
            all_mask,
            lon_coord=self.data_set[lon_lat_dim[0]].values,
            lat_coord=self.data_set[lon_lat_dim[1]].values,
        )
        return masks, clusters
