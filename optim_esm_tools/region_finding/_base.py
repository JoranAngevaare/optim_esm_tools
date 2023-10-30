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
from optim_esm_tools.analyze.cmip_handler import read_ds
from optim_esm_tools.analyze.globals import _CMIP_HANDLER_VERSION
from optim_esm_tools.analyze.time_statistics import TimeStatistics
from optim_esm_tools.analyze.xarray_tools import mask_to_reduced_dataset
from optim_esm_tools.plotting.map_maker import MapMaker
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
        variable=None,
        path=None,
        data_set=None,
        save_kw=None,
        extra_opt=None,
        read_ds_kw=None,
    ) -> None:
        """The function initializes an object with various parameters and
        assigns default values if necessary.

        :param variable: The `variable` parameter is used to specify the variable ID. If it is not
        provided, the code will try to retrieve the variable ID from the `data_set` attribute. If it is
        not found, it will default to the string 'NO_VAR_ID!'
        :param path: The path to the data file that will be read
        :param data_set: The `data_set` parameter is used to specify the dataset that will be used for
        further processing. It can be either a path to a file or an already loaded dataset object. If
        `path` is provided, the dataset will be read from that path using the `read_ds` function. If
        :param save_kw: A dictionary containing the following keys and values:
        :param extra_opt: The `extra_opt` parameter is a dictionary that contains additional options for
        the object. It has the following keys:
        :param read_ds_kw: The `read_ds_kw` parameter is a dictionary that contains keyword arguments to
        be passed to the `read_ds` function. It is used to customize the behavior of the `read_ds`
        function when reading the dataset from a file
        """
        read_ds_kw = read_ds_kw or {}
        if path is None and data_set:
            self.data_set = data_set
        elif path:
            self.data_set = read_ds(path, **read_ds_kw)  # type: ignore
        else:
            raise ValueError('Both path and data_set are None?!')  # pragma: no cover

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
        extra_opt.update(dict(read_ds_kw=read_ds_kw))  # type: ignore
        self.extra_opt = extra_opt
        self.save_kw = save_kw
        self.variable = variable or self.data_set.attrs.get('variable_id', 'NO_VAR_ID!')  # type: ignore

    @property
    def read_ds_kw(self) -> ty.Mapping:
        """The function `read_ds_kw` returns the value of the 'read_ds_kw' key
        from the 'extra_opt' dictionary, or an empty dictionary if the key does
        not exist.

        :return: a mapping object, which is obtained from the
            'read_ds_kw' key in the 'extra_opt' dictionary. If the
            'read_ds_kw' key does not exist in the dictionary, an empty
            dictionary is returned.
        """
        return self.extra_opt.get('read_ds_kw', {})

    @property
    def log(self) -> logging.Logger:
        """The function returns a logger object, creating one if it doesn't
        already exist.

        :return: The method is returning the `_logger` attribute.
        """
        if self._logger is None:
            self._logger = oet.config.get_logger(f'{self.__class__.__name__}')
        return self._logger

    @apply_options()
    def workflow(self, show_basic=True) -> _mask_cluster_type:
        """The function "workflow" performs a series of tasks related to
        plotting maps and time series data.

        :param show_basic: The parameter "show_basic" is a boolean flag
            that determines whether to show the basic map or not. If set
            to True, the method will call the "plot_basic_map()"
            function to display the basic map. If set to False, the
            basic map will not be shown, defaults to True (optional)
        """
        if show_basic:
            self.plot_basic_map()
        masks_and_clusters = self.get_masks()
        masks_and_clusters = self.filter_masks_and_clusters(masks_and_clusters)

        self.plot_masks(masks_and_clusters)
        self.plot_mask_time_series(masks_and_clusters)
        return masks_and_clusters

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

    @plt_show()
    def plot_basic_map(self) -> None:
        """The function plots a basic map and saves it with a filename based on
        the title label."""
        self._plot_basic_map()
        self.save(f'{self.title_label}_global_map')

    def _plot_basic_map(self):  # pragma: no cover
        raise NotImplementedError(
            f'{self.__class__.__name__} has no _plot_basic_map',
        )

    def save(self, name) -> None:
        """The `save` function saves a figure with the given name and
        additional keyword arguments.

        :param name: The `name` parameter is a string that represents the name of the file to be saved
        """
        assert self.__class__.__name__ in name
        oet.utils.save_fig(name, **self.save_kw)

    @property
    def title(self) -> str:
        """The `title` function returns the title of a `MapMaker` object's
        `data_set`.

        :return: The `title` attribute of the `MapMaker` object created with the `data_set` attribute of the
        current object.
        """
        return MapMaker(self.data_set).title

    @property
    def title_label(self) -> str:
        """The function returns a string that combines the title of an object
        with the name of its class, with spaces replaced by underscores.

        :return: The method is returning a string that is the concatenation of the following elements:
        - The value of `self.title` with spaces replaced by underscores (`_`)
        - The string representation of the class name of `self` (obtained using
        `self.__class__.__name__`)
        """
        return self.title.replace(' ', '_') + f'_{self.__class__.__name__}'

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

    @plt_show
    def summarize_mask(self, mask: np.ndarray, m_i: int) -> None:
        """The function `summarize_mask` saves and stores a mask, and generates
        a plot with a title based on the cluster index.

        :param mask: The `mask` parameter is a NumPy array that represents a binary mask. It is typically
        used to identify specific regions or objects in an image
        :type mask: np.ndarray
        :param m_i: m_i is an integer representing the cluster index
        :type m_i: int
        """
        self._summarize_mask(mask)
        plt.suptitle(f'{self.title} cluster {m_i}')
        self.save(f'{self.title_label}_cluster-{m_i}')

        self.store_mask(mask, m_i)

    @apply_options
    def store_mask(self, mask: np.ndarray, m_i: int, store_masks: bool = True) -> None:
        """The function `store_mask` saves a masked dataset to a netCDF file,
        calculates statistics on the masked dataset, and adds an extended mask
        to the dataset.

        :param mask: The `mask` parameter is a numpy array that represents a mask to be applied to a
        dataset. It is used to select specific data points from the dataset based on certain conditions
        :type mask: np.ndarray
        :param m_i: The parameter `m_i` is an integer that represents the cluster index. It is used to
        create a unique filename for saving the masked dataset
        :type m_i: int
        :param store_masks: The `store_masks` parameter is a boolean flag that determines whether or not
        to store the masks. If it is set to `True`, the masks will be stored. If it is set to `False`,
        the function will return early and not store the masks, defaults to True
        :type store_masks: bool (optional)
        :return: If the `store_masks` parameter is `False`, then the function will return and nothing
        will be executed.
        """
        if not store_masks:
            return  # pragma: no cover
        save_in = self.save_kw['save_in']
        store_in_dir = os.path.join(save_in, 'masks')
        os.makedirs(store_in_dir, exist_ok=True)
        ds_masked = mask_to_reduced_dataset(self.data_set, mask)
        kw = {
            k: v
            for k, v in self.read_ds_kw.items()
            if k not in 'max_time min_time'.split()
        }
        self.log.debug('start start cal')

        # This line is very important, there is some non-optimal threading race condition going on
        # between dask and rpy2
        ds_masked = ds_masked.load()  # do not delete!

        statistics = TimeStatistics(
            ds_masked,
            calculation_kwargs=dict(max_jump=kw, n_std_global=kw),
        ).calculate_statistics()
        self.log.debug(f'done start cal {statistics}')
        ds_masked = ds_masked.assign_attrs(
            {k: (v if v is not None else float('nan')) for k, v in statistics.items()},
        )
        oet.analyze.xarray_tools.add_extended_mask(ds_masked)
        assert 'extended_mask' in ds_masked
        ds_masked.to_netcdf(
            os.path.join(
                store_in_dir,
                f'{self.title_label}_cluster-{m_i}_v{_CMIP_HANDLER_VERSION}.nc',
            ),
        )
        self.log.debug('done save')

    def _summarize_mask(
        self,
        mask: np.ndarray,
        plot: ty.Optional[str] = None,
    ) -> ty.List[plt.Axes]:
        """The function `_summarize_mask` calls another function
        `summarize_mask` from the `map_maker` module in the `oet.plotting`
        package to summarize a mask.

        :param mask: The `mask` parameter is a NumPy array that represents a binary mask. It is used to
        identify specific regions or elements in the data set
        :type mask: np.ndarray
        :param plot: The `plot` parameter is an optional argument that specifies the type of plot to be
        generated. It is a string that can take the following values:
        :type plot: ty.Optional[str]
        :return: a list of `plt.Axes` objects.
        """
        return oet.plotting.map_maker.summarize_mask(self.data_set, mask, plot=plot)

    def make_mask_figures(self, masks):
        """The function "make_mask_figures" iterates through a list of masks,
        skips masks with a sum of zero, and calls the "summarize_mask" function
        for each non-zero mask.

        :param masks: The `masks` parameter is a list of masks. Each mask is a binary array where each
        element represents a pixel in an image. A value of 1 indicates that the pixel belongs to the object
        of interest, while a value of 0 indicates that the pixel does not belong to the object
        """
        for m_i, mask in enumerate(masks):
            if np.sum(mask) == 0:
                continue  # pragma: no cover
            self.summarize_mask(mask, m_i)
