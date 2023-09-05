import numpy as np

from ._base import _mask_cluster_type
from ._base import _two_sigma_percent
from ._base import apply_options
from ._base import RegionExtractor
from optim_esm_tools.analyze.clustering import build_cluster_mask
from optim_esm_tools.region_finding.local_history import _HistroricalLookup
from optim_esm_tools.region_finding.percentiles import Percentiles
from optim_esm_tools.utils import deprecated


@deprecated
class PercentilesHistory(Percentiles, _HistroricalLookup):
    @apply_options
    def get_masks(
        self,
        percentiles_historical=_two_sigma_percent,
        read_ds_kw=None,
        lon_lat_dim=('lon', 'lat'),
    ) -> _mask_cluster_type:
        read_ds_kw = read_ds_kw or {}
        for k, v in dict(min_time=None, max_time=None).items():
            read_ds_kw.setdefault(k, v)

        historical_ds = self.get_historical_ds(read_ds_kw=read_ds_kw)
        labels = [crit.short_description for crit in self.criteria]
        masks = []

        for lab in labels:
            arr = self.data_set[lab].values
            arr_historical = historical_ds[lab].values
            thr = np.percentile(
                arr_historical[~np.isnan(arr_historical)],
                percentiles_historical,
            )
            masks.append(arr >= thr)

        all_mask = np.ones_like(masks[0])
        for m in masks:
            all_mask &= m

        self.check_shape(all_mask)
        masks, clusters = build_cluster_mask(
            all_mask,
            lon_coord=self.data_set[lon_lat_dim[0]].values,
            lat_coord=self.data_set[lon_lat_dim[1]].values,
        )
        return masks, clusters
