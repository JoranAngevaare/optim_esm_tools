# -*- coding: utf-8 -*-
import optim_esm_tools as oet

import os
import xarray as xr

import typing as ty
from warnings import warn

import xrft

from optim_esm_tools.utils import depricated

from .globals import _CMIP_HANDLER_VERSION, _FOLDER_FMT
from .xarray_tools import _native_date_fmt
from optim_esm_tools.plotting.map_maker import MapMaker, make_title
from optim_esm_tools.analyze import tipping_criteria
import logging


class ResultDataSet:
    _logger: logging.Logger = None
    labels: tuple = tuple('i ii iii iv'.split())

    def __init__(self, path=None, dataset=None) -> None:
        if path is None:
            self.log.warning(
                f'Best is to start {self.__class__.__name__} from a synda path'
            )
            self.dataset = transform_ds(dataset)
        else:
            self.dataset = read_ds(path)

    @property
    def log(self):
        if self._logger is None:
            self._logger = oet.config.get_logger()
        return self.logger


def transform_ds(
    ds: xr.Dataset,
    calculate_conditions: ty.Tuple[tipping_criteria._Condition] = None,
    condition_kwargs: ty.Mapping = None,
    variable_of_interest: ty.Tuple[str] = ('tas',),
    max_time: ty.Optional[ty.Tuple[int, int, int]] = (2100, 1, 1),
    min_time: ty.Optional[ty.Tuple[int, int, int]] = None,
    strict: bool = True,
    _time_var='time',
    _detrend_type='linear',
    _ma_window: int = 10,
):
    """Transform the dataset to get it ready for handling in optim_esm_tools

    Args:
        ds (xr.Dataset): input dataset
        variable_of_interest (ty.Tuple[str], optional): Variables to handle. Defaults to ('tas',).
        max_time (ty.Optional[ty.Tuple[int, int, int]], optional): Defines time range in which to load data. Defaults to (2100, 1, 1).
        min_time (ty.Optional[ty.Tuple[int, int, int]], optional): Defines time range in which to load data. Defaults to None.
        strict (bool, optional): raise errors on loading, if any. Defaults to True.
        _time_var (str, optional): Name of the time dimention. Defaults to 'time'.
        _detrend_type (str, optional): Type of detrending applied. Defaults to 'linear'.
        _ma_window (int, optional): Moving average window (assumed to be years). Defaults to 10.
    """
    if calculate_conditions is None:
        calculate_conditions = (
            tipping_criteria.StartEndDifference,
            tipping_criteria.StdDetrended,
            tipping_criteria.MaxJump,
            tipping_criteria.MaxDerivitive,
        )
    if condition_kwargs is None:
        condition_kwargs = dict()
    ds = _calculate_variables(
        oet.synda_files.format_synda.recast(ds),
        min_time,
        max_time,
        variable_of_interest,
        strict,
        _ma_window,
        _detrend_type,
        _time_var,
    )
    for cls in calculate_conditions:
        condition = cls(**condition_kwargs)
        condition_array = condition.calculate(ds)
        condition_array = condition_array.assign_attrs(
            dict(
                short_description=cls.short_description,
                long_description=condition.long_description,
                name=condition_array.name,
            )
        )
        ds[condition.short_description] = condition_array
    return ds


@oet.utils.timed()
def read_ds(
    base: str,
    variable_of_interest: ty.Tuple[str] = ('tas',),
    max_time: ty.Optional[ty.Tuple[int, int, int]] = (2100, 1, 1),
    min_time: ty.Optional[ty.Tuple[int, int, int]] = None,
    _ma_window: int = 10,
    _cache: bool = True,
    **kwargs,
) -> xr.Dataset:
    """Read a dataset from a folder called "base".

    Args:
        base (str): Folder to load the data from
        variable_of_interest (ty.Tuple[str], optional): Variables to handle. Defaults to ('tas',).
        max_time (ty.Optional[ty.Tuple[int, int, int]], optional): Defines time range in which to load data. Defaults to (2100, 1, 1).
        min_time (ty.Optional[ty.Tuple[int, int, int]], optional): Defines time range in which to load data. Defaults to None.
        _ma_window (int, optional): Moving average window (assumed to be years). Defaults to 10.
        _cache (bool, optional): cache the dataset with it's extra fields to alow faster (re)loading. Defaults to True.

    kwargs:
        any kwargs are passed onto transfor_ds.

    Returns:
        xr.Dataset: An xarray dataset with the appropriate variables
    """
    if kwargs:
        oet.config.get_logger().error(f'Not really advised yet to call with {kwargs}')
        _cache = False
    post_processed_file = _name_cache_file(
        base,
        variable_of_interest,
        min_time,
        max_time,
        _ma_window,
        _CMIP_HANDLER_VERSION,
    )

    if os.path.exists(post_processed_file) and _cache:
        return oet.synda_files.format_synda.load_glob(post_processed_file)

    data_path = os.path.join(base, 'merged.nc')
    if not os.path.exists(data_path):
        warn(f'No dataset at {data_path}')
        return None

    data_set = oet.synda_files.format_synda.load_glob(data_path)
    data_set = transform_ds(
        data_set,
        variable_of_interest=variable_of_interest,
        max_time=max_time,
        min_time=min_time,
        _ma_window=_ma_window,
        **kwargs,
    )

    folders = base.split(os.sep)

    # start with -1 (for i==0)
    metadata = {k: folders[-i - 1] for i, k in enumerate(_FOLDER_FMT[::-1])}
    metadata.update(
        dict(path=base, file=post_processed_file, running_mean_period=_ma_window)
    )

    data_set.attrs.update(metadata)

    if _cache:
        print(f'Write {post_processed_file}' + ' ' * 100, flush=True, end='\r')
        data_set.to_netcdf(post_processed_file)
    return data_set


def _name_cache_file(
    base,
    variable_of_interest,
    min_time,
    max_time,
    _ma_window,
    version,
):
    path = os.path.join(
        base,
        f'{variable_of_interest}'
        f'_{min_time if min_time else ""}'
        f'_{max_time if max_time else ""}'
        f'_ma{_ma_window}'
        f'_optimesm_v{version}.nc',
    )
    normalized_path = (
        path.replace('(', '')
        .replace(')', '')
        .replace(' ', '_')
        .replace(',', '')
        .replace('\'', '')
    )
    return normalized_path


@oet.utils.timed()
def _calculate_variables(
    data_set,
    min_time,
    max_time,
    variable_of_interest,
    strict,
    _ma_window,
    _detrend_type,
    _time_var,
):
    if min_time or max_time:
        time_slice = [
            _native_date_fmt(data_set[_time_var], d) if d is not None else None
            for d in [min_time, max_time]
        ]
        data_set = data_set.sel(**{_time_var: slice(*time_slice)})

    # Detrend and run_mean on the fly
    for variable in variable_of_interest:
        if (ds_len := len(data_set[variable])) < _ma_window:
            message = f'This data set is shorter {ds_len} than the moving average window ({_ma_window})'
            if strict:
                raise ValueError(message)
            oet.config.get_logger().warning(message)
        # NB these are DataArrays not Datasets!
        run_mean = data_set[variable].rolling(time=_ma_window, center=True).mean()
        detrended = xrft.detrend(
            data_set[variable], _time_var, detrend_type=_detrend_type
        )
        detrended_run_mean = xrft.detrend(
            run_mean.dropna(_time_var), _time_var, detrend_type=_detrend_type
        )
        detrended_run_mean.attrs['units'] = data_set[variable].attrs.get(
            'units', '{units}'
        )
        for det in detrended, detrended_run_mean:
            det.attrs.update(data_set[variable].attrs.copy())
        data_set[f'{variable}_run_mean_{_ma_window}'] = run_mean
        data_set[f'{variable}_detrend'] = detrended
        data_set[f'{variable}_detrend_run_mean_{_ma_window}'] = detrended_run_mean
    return data_set


class MapMaker(MapMaker):
    @depricated
    def __init__(self, *a, **kw):
        return super().__init__(*a, **kw)
