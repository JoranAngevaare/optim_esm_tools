import immutabledict
import numpy as np

import optim_esm_tools as oet
from ._base import _mask_cluster_type
from ._base import _two_sigma_percent
from ._base import apply_options
from optim_esm_tools.analyze import tipping_criteria
from optim_esm_tools.analyze.clustering import build_cluster_mask
from optim_esm_tools.analyze.clustering import build_weighted_cluster
from optim_esm_tools.region_finding.percentiles import Percentiles
from optim_esm_tools.utils import deprecated


class ProductPercentiles(Percentiles):
    labels = ('ii', 'iii', 'v')

    @apply_options
    def _get_masks_weighted(
        self,
        min_weight=0.95,
        lon_lat_dim=('lon', 'lat'),
        _mask_method='product_rank',
    ):
        return super()._get_masks_weighted(
            min_weight=min_weight,
            lon_lat_dim=lon_lat_dim,
            _mask_method=_mask_method,
        )

    @apply_options
    def _get_masks_masked(
        self,
        product_percentiles=_two_sigma_percent,
        lon_lat_dim=('lon', 'lat'),
        _mask_method='product_rank_past_threshold',
    ) -> _mask_cluster_type:
        """Get mask for max of ii and iii and a box around that."""
        all_mask = self._build_combined_mask(
            method=_mask_method,
            product_percentiles=product_percentiles,
        )
        masks, clusters = build_cluster_mask(
            all_mask,
            lon_coord=self.data_set[lon_lat_dim[0]].values,
            lat_coord=self.data_set[lon_lat_dim[1]].values,
        )
        return masks, clusters
