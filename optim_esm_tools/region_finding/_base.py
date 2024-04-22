import inspect
import logging
import os
import typing as ty
from functools import wraps

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

import optim_esm_tools as oet
from optim_esm_tools.analyze import tipping_criteria
from optim_esm_tools.analyze.globals import _CMIP_HANDLER_VERSION
from optim_esm_tools.analyze.xarray_tools import mask_to_reduced_dataset
from optim_esm_tools.plotting.plot import _show

_mask_cluster_type = ty.Tuple[ty.List[np.ndarray], ty.List[np.ndarray]]

# >>> import scipy
# >>> scipy.stats.norm.cdf(3)
# 0.9986501019683699
# >> scipy.stats.norm.cdf(2)
# 0.9772498680518208
_two_sigma_percent = 97.72498680518208


def plt_show(*a):
    """Wrapper to disable class methods to follow up with show."""

    def somedec_outer(fn):
        @wraps(fn)
        def plt_func(*args, **kwargs):
            res = fn(*args, **kwargs)
            self = args[0]
            _show(getattr(self, 'show', False))
            return res

        return plt_func

    if a and isinstance(a[0], ty.Callable):
        # Decorator that isn't closed
        return somedec_outer(a[0])
    return somedec_outer


def apply_options(*a):
    """If a function takes any arguments in self.extra_opt, apply it to the
    method."""

    def somedec_outer(fn):
        @wraps(fn)
        def timed_func(*args, **kwargs):
            self = args[0]
            takes = inspect.signature(fn).parameters
            kwargs.update({k: v for k, v in self.extra_opt.items() if k in takes})
            res = fn(*args, **kwargs)
            return res

        return timed_func

    if a and isinstance(a[0], ty.Callable):
        # Decorator that isn't closed
        return somedec_outer(a[0])
    return somedec_outer


class RegionExtractor:
    _logger: ty.Optional[logging.Logger] = None
    labels: tuple = tuple('ii iii'.split())
    show: bool = True

    criteria: ty.Tuple = (tipping_criteria.StdDetrended, tipping_criteria.MaxJump)
    extra_opt: ty.Mapping
    save_kw: ty.Mapping
    save_statistics: bool = True
    data_set: xr.Dataset

    def __init__(
        self,
        data_set: xr.Dataset,
        variable: ty.Optional[str] = None,
        save_kw: ty.Optional[dict] = None,
        extra_opt: ty.Optional[dict] = None,
    ) -> None:
        """The function initializes an object with various parameters and
        assigns default values if necessary.

        :param variable: The `variable` parameter is used to specify the variable ID. If it is not
        provided, the code will try to retrieve the variable ID from the `data_set` attribute. If it is
        not found, it will default to the string 'NO_VAR_ID!'
        :param path: The path to the data file that will be read
        :param data_set: The `data_set` parameter is used to specify the dataset that will be used for
        further processing.
        :param save_kw: A dictionary containing the following keys and values:
        :param extra_opt: The `extra_opt` parameter is a dictionary that contains additional options for
        the object. It has the following keys:
        """
        self.data_set = data_set

        save_kw = save_kw or dict(
            save_in='./',
            file_types=(
                'png',
                'pdf',
            ),
            skip=False,
            sub_dir=None,
        )
        extra_opt = extra_opt or dict(show_basic=True)

        self.extra_opt = extra_opt
        self.save_kw = save_kw
        self.variable = variable or self.data_set.attrs.get('variable_id', 'NO_VAR_ID!')  # type: ignore

    @property
    def log(self) -> logging.Logger:
        """The function returns a logger object, creating one if it doesn't
        already exist.

        :return: The method is returning the `_logger` attribute.
        """
        if self._logger is None:
            self._logger = oet.config.get_logger(f'{self.__class__.__name__}')
        return self._logger

    def get_masks(self) -> _mask_cluster_type:  # pragma: no cover
        raise NotImplementedError(
            f'{self.__class__.__name__} has no get_masks',
        )

    def plot_masks(
        self,
        masks_and_clusters: _mask_cluster_type,
        **kw,
    ) -> _mask_cluster_type:  # pragma: no cover
        raise NotImplementedError(
            f'{self.__class__.__name__} has no plot_masks',
        )

    def plot_mask_time_series(
        self,
        masks_and_clusters: _mask_cluster_type,
        **kw,
    ) -> _mask_cluster_type:  # pragma: no cover
        raise NotImplementedError(
            f'{self.__class__.__name__} has no plot_mask_time_series',
        )

    def _plot_basic_map(self):  # pragma: no cover
        raise NotImplementedError(
            f'{self.__class__.__name__} has no _plot_basic_map',
        )

    def mask_area(self, mask: np.ndarray) -> np.ndarray:
        """The function `mask_area` returns the cell areas from a dataset based
        on a given mask.

        :param mask: The `mask` parameter is a numpy array that represents a mask. It is used to select
        specific elements from the `data_set['cell_area'].values` array. The mask should have the same shape
        as the `data_set['cell_area'].values` array, and its elements should be boolean
        :type mask: np.ndarray
        :return: an array containing the values of the 'cell_area' column from the 'data_set' attribute,
        filtered by the provided 'mask' array.
        """
        try:
            if mask is None or not np.sum(mask):
                return np.array([0])  # pragma: no cover
        except Exception as e:  # pragma: no cover
            raise ValueError(mask) from e
        self.check_shape(mask)
        return self.data_set['cell_area'].values[mask]

    def check_shape(
        self,
        data: ty.Union[np.ndarray, xr.DataArray],
        compare_with='cell_area',
    ) -> None:
        """The `check_shape` function compares the shape of a given data array
        with the shape of a reference array and raises a ValueError if they are
        not equal.

        :param data: The `data` parameter can be either a NumPy array (`np.ndarray`) or an xarray DataArray
        (`xr.DataArray`). It represents the data that needs to be checked for its shape
        :type data: ty.Union[np.ndarray, xr.DataArray]
        :param compare_with: The `compare_with` parameter is a string that specifies the variable name in
        the `self.data_set` object that you want to compare the shape of the `data` parameter with. It is
        used to determine the expected shape of the `data` parameter, defaults to cell_area (optional)
        :return: `None` if the shape of the input `data` matches the shape specified by `shape_should_be`.
        """
        shape_should_be = self.data_set[compare_with].shape
        if data.shape == shape_should_be:
            return
        error_message = f'Got {data.shape}, expected {shape_should_be}'
        if name := getattr(data, 'name', False):
            error_message = f'For {name}: {error_message}'
        if dims := getattr(data, 'dims', False):
            error_message = f'{error_message}. Dims are {dims}, expected'
        error_message += f'for {self.data_set[compare_with].dims}'
        raise ValueError(error_message)

    @apply_options
    def mask_is_large_enough(self, mask: np.ndarray, min_area_sq: float = 0.0) -> bool:
        """The function checks if the area of a given mask is larger than or
        equal to a specified minimum area.

        :param mask: The `mask` parameter is a numpy array representing a binary mask. It is typically used
        to represent a region of interest or a segmentation mask in an image. The mask should have the same
        shape as the image it corresponds to, with a value of 1 indicating the presence of the object and a
        :type mask: np.ndarray
        :param min_area_sq: The parameter `min_area_sq` represents the minimum area (in square units) that
        the `mask` should have in order for the function to return `True`
        :type min_area_sq: float
        :return: a boolean value, indicating whether the sum of the areas of the given mask is greater than
        or equal to the specified minimum area.
        """
        return self.mask_area(mask).sum() >= min_area_sq

    def mask_to_lon_lat(self, mask):
        ds = self.data_set
        if not isinstance(mask, np.ndarray):
            mask = mask.values
        lats, lons = np.meshgrid(ds.lat.values, ds.lon.values)
        lon_coords = lons.T[mask]
        lat_coords = lats.T[mask]
        return np.vstack([lon_coords, lat_coords]).T

    def filter_masks_and_clusters(
        self,
        masks_and_clusters: _mask_cluster_type,
    ) -> _mask_cluster_type:
        """The function filters a list of masks and clusters based on the size
        of the masks, and returns the filtered lists.

        :param masks_and_clusters: A tuple containing two lists. The first list contains masks, and the
        second list contains clusters
        :type masks_and_clusters: _mask_cluster_type
        :return: two lists: `ret_m` and `ret_c`.
        """
        if not len(masks_and_clusters[0]):
            return [], []
        ret_m = []
        ret_c = []
        for m, c in zip(*masks_and_clusters):
            if self.mask_is_large_enough(m):
                ret_m.append(m)
                ret_c.append(c)

        self.log.info(f'Keeping {len(ret_m)}/{len(masks_and_clusters[0])} of masks')
        return ret_m, ret_c
