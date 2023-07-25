# -*- coding: utf-8 -*-
import os
import xarray as xr

import typing as ty
from warnings import warn

from .globals import _FOLDER_FMT, _DEFAULT_MAX_TIME
import optim_esm_tools as oet
from optim_esm_tools.analyze import tipping_criteria


def add_conditions_to_ds(
    ds: xr.Dataset,
    calculate_conditions: ty.Tuple[tipping_criteria._Condition] = None,
    condition_kwargs: ty.Mapping = None,
    variable_of_interest: ty.Tuple[str] = ('tas',),
    _ma_window: int = None,
) -> xr.Dataset:
    """Transform the dataset to get it ready for handling in optim_esm_tools

    Args:
        ds (xr.Dataset): input dataset
        calculate_conditions (ty.Tuple[tipping_criteria._Condition], optional): Calculate the
            results of these tipping conditions. Defaults to None.
        condition_kwargs (ty.Mapping, optional): kwargs for the tipping conditions. Defaults to
            None.
        variable_of_interest (ty.Tuple[str], optional): Variables to handle. Defaults to ('tas',).
        _ma_window (int, optional): Moving average window (assumed to be years). Defaults to 10.

    Raises:
        ValueError: If there are multiple tipping conditions with the same short_description

    Returns:
        xr.Dataset: The fully initialized dataset
    """
    _ma_window = _ma_window or oet.config.config['analyze']['moving_average_years']
    if calculate_conditions is None:
        calculate_conditions = (
            tipping_criteria.StartEndDifference,
            tipping_criteria.StdDetrended,
            tipping_criteria.MaxJump,
            tipping_criteria.MaxDerivitive,
            tipping_criteria.MaxJumpAndStd,
        )
    if len(set(desc := (c.short_description for c in calculate_conditions))) != len(
        calculate_conditions
    ):
        raise ValueError(f'One or more non unique descriptions {desc}')
    if condition_kwargs is None:
        condition_kwargs = dict()

    for variable in oet.utils.to_str_tuple(variable_of_interest):
        for cls in calculate_conditions:
            condition = cls(**condition_kwargs, variable=variable)
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


@oet.utils.add_load_kw
@oet.utils.timed()
def read_ds(
    base: str,
    variable_of_interest: ty.Tuple[str] = None,
    max_time: ty.Optional[ty.Tuple[int, int, int]] = _DEFAULT_MAX_TIME,
    min_time: ty.Optional[ty.Tuple[int, int, int]] = None,
    apply_transform: bool = True,
    pre_process: bool = True,
    strict: bool = True,
    load: bool = None,
    _ma_window: ty.Optional[int] = None,
    _cache: bool = True,
    _file_name: str = None,
    **kwargs,
) -> xr.Dataset:
    """Read a dataset from a folder called "base".

    Args:
        base (str): Folder to load the data from
        variable_of_interest (ty.Tuple[str], optional): Variables to handle. Defaults to ('tas',).
        max_time (ty.Optional[ty.Tuple[int, int, int]], optional): Defines time range in which to
            load data. Defaults to (2100, 12, 31).
        min_time (ty.Optional[ty.Tuple[int, int, int]], optional): Defines time range in which to
            load data. Defaults to None.
        apply_transform: (bool, optional): Apply analysis specific postprocessing algorithms.
            Defaults to True.
        pre_process (bool, optional): Should be true, this pre-processing of the data is required
            later on. Defaults to True.
        area_query_kwargs (ty.Mapping, optional): additionally keyword arguments for searching.
        strict (bool, optional): raise errors on loading, if any. Defaults to True.
        load (bool, optional): apply dataset.load to dataset directly. Defaults to False.
        _ma_window (int, optional): Moving average window (assumed to be years). Defaults to 10.
        _cache (bool, optional): cache the dataset with it's extra fields to alow faster
            (re)loading. Defaults to True.
        _file_name (str, optional): name to match. Defaults to configs settings.

    kwargs:
        any kwargs are passed onto transform_ds.

    Returns:
        xr.Dataset: An xarray dataset with the appropriate variables
    """
    log = oet.config.get_logger()
    _file_name = _file_name or oet.config.config['CMIP_files']['base_name']
    _ma_window = _ma_window or oet.config.config['analyze']['moving_average_years']
    data_path = os.path.join(base, _file_name)
    variable_of_interest = (
        variable_of_interest or oet.analyze.pre_process._read_variable_id(data_path)
    )

    if not isinstance(variable_of_interest, str):
        raise ValueError('Only single vars supported')
    if kwargs:
        log.error(f'Not really advised yet to call with {kwargs}')
        _cache = False
    if not apply_transform:
        # Don't cache the partial ds
        _cache = False

    log.debug(f'read_ds {variable_of_interest}')
    res_file = _name_cache_file(
        base,
        variable_of_interest,
        min_time,
        max_time,
        _ma_window,
    )

    if os.path.exists(res_file) and _cache:
        return oet.analyze.io.load_glob(res_file)

    if not os.path.exists(data_path):
        message = f'No dataset at {data_path}'
        if strict:
            raise FileNotFoundError(message)
        warn(message)
        return None

    if pre_process:
        data_set = oet.analyze.pre_process.get_preprocessed_ds(
            source=data_path,
            max_time=max_time,
            min_time=min_time,
            _ma_window=_ma_window,
            variable_id=variable_of_interest,
        )
    else:
        message = 'Not preprocessing file is dangerous, dimensions may differ wildly!'
        log.warning(message)
        data_set = oet.analyze.io.load_glob(data_path, load=load)

    if apply_transform:
        kwargs.update(
            dict(
                variable_of_interest=variable_of_interest,
                _ma_window=_ma_window,
            )
        )
        data_set = add_conditions_to_ds(data_set, **kwargs)

    folders = base.split(os.sep)

    # start with -1 (for i==0)
    metadata = {k: folders[-i - 1] for i, k in enumerate(_FOLDER_FMT[::-1])}
    metadata.update(dict(path=base, file=res_file, running_mean_period=_ma_window))

    data_set.attrs.update(metadata)

    if _cache:
        log.info(f'Write {res_file}')
        data_set.to_netcdf(res_file)

    return data_set


def _name_cache_file(
    base,
    variable_of_interest,
    min_time,
    max_time,
    _ma_window,
    version=None,
):
    """Get a file name that identifies the settings"""
    version = version or oet.config.config['versions']['cmip_handler']
    _ma_window = _ma_window or oet.config.config['analyze']['moving_average_years']
    path = os.path.join(
        base,
        f'{variable_of_interest}'
        f'_s{tuple(min_time) if min_time else ""}'
        f'_e{tuple(max_time) if max_time else ""}'
        f'_ma{_ma_window}'
        f'_optimesm_v{version}.nc',
    )
    normalized_path = (
        path.replace('(', '')
        .replace(')', '')
        .replace(']', '')
        .replace('[', '')
        .replace(' ', '_')
        .replace(',', '')
        .replace('\'', '')
    )
    oet.config.get_logger().debug(f'got {normalized_path}')
    return normalized_path
