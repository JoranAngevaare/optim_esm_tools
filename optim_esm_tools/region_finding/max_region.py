import matplotlib.pyplot as plt
import numpy as np

import optim_esm_tools as oet
from ._base import _mask_cluster_type
from ._base import apply_options
from ._base import plt_show
from ._base import RegionExtractor

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

    @property
    def _labels(self):
        return [crit.short_description for crit in self.criteria]
