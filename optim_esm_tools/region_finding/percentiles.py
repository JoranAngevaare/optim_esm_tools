import typing as ty

import immutabledict
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

import optim_esm_tools as oet
from ._base import _mask_cluster_type
from ._base import _two_sigma_percent
from ._base import apply_options
from ._base import plt_show
from ._base import RegionExtractor
from optim_esm_tools.analyze import tipping_criteria
from optim_esm_tools.analyze.clustering import build_cluster_mask
from optim_esm_tools.analyze.clustering import build_weighted_cluster
from optim_esm_tools.analyze.xarray_tools import mask_xr_ds
from optim_esm_tools.plotting.plot import _show
from optim_esm_tools.plotting.plot import setup_map
from optim_esm_tools.utils import check_accepts


class Percentiles(RegionExtractor):
    @oet.utils.check_accepts(
        accepts=immutabledict.immutabledict(cluster_method=('weighted', 'masked')),
    )
    @apply_options
    def get_masks(self, cluster_method='masked') -> _mask_cluster_type:
        """The function `get_masks` returns masks and clusters based on the
        specified cluster method.

        :param cluster_method: The `cluster_method` parameter is a string that determines the method used to
        generate masks. It can have two possible values:, defaults to masked (optional)
        :return: two values: masks and clusters.
        """
        if cluster_method == 'weighted':
            masks, clusters = self._get_masks_weighted()
        else:
            masks, clusters = self._get_masks_masked()
        if len(masks):
            self.check_shape(masks[0])
        return masks, clusters

    @apply_options
    def _get_masks_weighted(
        self,
        min_weight=0.95,
        lon_lat_dim=('lon', 'lat'),
        _mask_method='sum_rank',
    ):
        """The function `_get_masks_weighted` calculates weighted masks and
        clusters based on a minimum weight threshold.

        :param min_weight: The min_weight parameter is the minimum weight threshold for including a mask in
        the output. Masks with weights below this threshold will be excluded
        :param lon_lat_dim: The `lon_lat_dim` parameter is a tuple that specifies the names of the longitude
        and latitude dimensions in the dataset. These dimensions are used to extract the longitude and
        latitude coordinates from the dataset
        :param _mask_method: The `_mask_method` parameter is used to specify the method for building the
        combined mask. It is a string that can take one of the following values:, defaults to sum_rank
        (optional)
        :return: two variables: masks and clusters.
        """
        tot_sum = self._build_combined_mask(method=_mask_method)
        masks, clusters = build_weighted_cluster(
            weights=tot_sum,
            lon_coord=self.data_set[lon_lat_dim[0]].values,
            lat_coord=self.data_set[lon_lat_dim[1]].values,
            threshold=min_weight,
        )
        return masks, clusters

    @apply_options
    def _get_masks_masked(
        self,
        lon_lat_dim=('lon', 'lat'),
        percentiles=_two_sigma_percent,
        _mask_method='all_pass_percentile',
    ):
        """The function `_get_masks_masked` builds a combined mask using a
        specified method and percentiles, and then builds cluster masks based
        on the combined mask and lon/lat coordinates.

        :param lon_lat_dim: A tuple specifying the names of the longitude and latitude dimensions in the
        dataset
        :param percentiles: The `percentiles` parameter is a list of percentiles used for masking. It is set
        to `_two_sigma_percent` in the code, which suggests that it is a predefined variable containing a
        list of percentiles
        :param _mask_method: The `_mask_method` parameter is used to specify the method for building the
        combined mask. It determines how the individual masks are combined to create the final mask. The
        available options for `_mask_method` are:, defaults to all_pass_percentile (optional)
        :return: two values: masks and clusters.
        """
        all_mask = self._build_combined_mask(
            method=_mask_method,
            percentiles=percentiles,
        )
        masks, clusters = build_cluster_mask(
            all_mask,
            lon_coord=self.data_set[lon_lat_dim[0]].values,
            lat_coord=self.data_set[lon_lat_dim[1]].values,
        )
        return masks, clusters

    @check_accepts(
        accepts=dict(
            method=(
                'sum_rank',
                'all_pass_percentile',
                'product_rank',
                'product_rank_past_threshold',
                'all_pass_historical',
            ),
        ),
    )
    def _get_mask_function_and_kw(
        self,
        method: str,
        **kw,
    ) -> ty.Tuple[ty.Callable, dict]:
        functions = dict(
            all_pass_percentile=self._all_pass_percentile,
            all_pass_historical=self._all_pass_historical,
            product_rank=self._product_rank,
            product_rank_past_threshold=self._product_rank_past_threshold,
            sum_rank=self._sum_rank,
        )
        func = functions[method]
        assert isinstance(func, ty.Callable)
        filter_kw = oet.utils.filter_keyword_arguments(kw, func, allow_varkw=False)
        if removed := set(kw) - set(filter_kw):
            self.log.info(
                f'Removed {removed}. Started with {kw}, returning res {filter_kw}',
            )
        return func, filter_kw

    def _build_combined_mask(self, method: str, **kw) -> np.ndarray:
        """The `_build_combined_mask` function takes a method and keyword
        arguments, uses the method to select a function from a dictionary, and
        applies the selected function to a list of labels to generate a result.

        :param method: The "method" parameter is a string that specifies
            the method to be used for building the combined mask. The
            available methods are:
                - all_pass_percentile
                - all_pass_historical
                - product_rank
                - product_rank_past_threshold
                - sum_rank
        :type method: str
        :return: a numpy array.
        """
        labels = [crit.short_description for crit in self.criteria]
        func, filter_kw = self._get_mask_function_and_kw(method=method, **kw)
        result = func(labels, **filter_kw)
        self.check_shape(result)
        return result

    def _all_pass_percentile(
        self,
        labels: ty.List[str],
        percentiles: ty.Union[float, int],
    ) -> npt.NDArray[np.bool_]:
        """The `_all_pass_percentile` function calculates a mask that indicates
        whether each element in the data set is greater than or equal to the
        percentile threshold for each label.

        :param labels: A list of strings representing the labels of the data set. Each label corresponds to
        a column in the data set
        :type labels: ty.List[str]
        :param percentiles: The `percentiles` parameter is a list of values representing the percentiles at
        which you want to calculate the threshold. It can be a single value or a list of values. For
        example, if you pass `percentiles=90`, it will calculate the threshold at the 90th percentile
        :type percentiles: ty.Union[float, int]
        :return: a NumPy array of boolean values.
        """
        masks = []

        for lab in labels:
            arr = self.data_set[lab].values
            arr_no_nan = arr[~np.isnan(arr)]
            thr = np.percentile(arr_no_nan, percentiles)
            masks.append(arr >= thr)

        all_mask = np.ones_like(masks[0])
        for m in masks:
            all_mask &= m
        return all_mask

    def _all_pass_historical(self, *a, **kw):
        raise NotImplementedError(
            f'{self.__class__.__name__} has not method all_pass_historical',
        )  # pragma: no cover

    def _sum_rank(self, labels: ty.List[str]) -> npt.NDArray[np.float64]:
        """The `_sum_rank` function calculates the average rank of values in a
        dataset for a given list of labels.

        :param labels: The `labels` parameter is a list of strings representing the labels of the data set
        :type labels: ty.List[str]
        :return: a numpy array of type np.float64.
        """
        sums = []
        for lab in labels:
            vals = self.data_set[lab].values
            vals = tipping_criteria.rank2d(vals)
            vals[np.isnan(vals)] = 0
            sums.append(vals)

        tot_sum = np.zeros_like(sums[0], dtype=np.float64)
        for s in sums:
            tot_sum += s
        tot_sum /= len(sums)
        return tot_sum

    def _product_rank(self, labels: ty.List[str]) -> npt.NDArray[np.float64]:
        """The `_product_rank` function calculates the combined score of
        multiple labels using the `rank2d` function and returns the result as a
        numpy array.

        :param labels: A list of strings representing the labels of the
            columns in the dataset
        :type labels: ty.List[str]
        :return: a NumPy array of type np.float64.
        """
        ds = self.data_set.copy()
        combined_score = np.ones_like(ds[labels[0]].values, dtype=np.float64)

        for label in labels:
            try:
                combined_score *= tipping_criteria.rank2d(ds[label].values)
            except ValueError:
                raise ValueError(ds[label].values, label)
        return combined_score

    def _product_rank_past_threshold(
        self,
        labels: ty.List[str],
        product_percentiles: ty.Union[float, int],
    ) -> npt.NDArray[np.bool_]:
        """The function `_product_rank_past_threshold` calculates the combined
        score for a list of labels and returns a boolean array indicating
        whether each score is above a given percentile threshold.

        :param labels: A list of strings representing the labels of the products
        :type labels: ty.List[str]
        :param product_percentiles: The `product_percentiles` parameter is a value that represents the
        threshold for the combined score. It can be either a float or an integer. If it is a float, it
        represents a fraction (e.g., 0.5 represents 50%). If it is an integer, it represents a
        :type product_percentiles: ty.Union[float, int]
        :return: a NumPy array of boolean values.
        """
        combined_score = self._product_rank(labels)
        # Combined score is fraction, not percent!
        return combined_score > (product_percentiles / 100)
