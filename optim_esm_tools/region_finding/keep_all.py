from ._base import RegionExtractor, _mask_cluster_type
import itertools
import numpy as np


class MaskAll(RegionExtractor):
    def get_masks(self) -> _mask_cluster_type:  # pragma: no cover
        mask_2d = ~self.data_set[self.variable].isnull().all(dim='time')
        mask_values = mask_2d.values
        lats = self.data_set['lat'].values
        lons = self.data_set['lon'].values
        masks = []
        coords = []

        for i, j in itertools.product(range(mask_2d.shape[0]), range(mask_2d.shape[1])):
            if mask_values[i][j]:
                coords.append(
                    np.array(
                        [
                            [
                                lats[i],
                                lons[j],
                            ],
                        ],
                    ),
                )
                this_mask = np.zeros_like(mask_values)
                this_mask[i][j] = True
                masks.append(this_mask)
        return masks, coords

    def filter_masks_and_clusters(
        self,
        masks_and_clusters: _mask_cluster_type,
    ) -> _mask_cluster_type:
        return masks_and_clusters
