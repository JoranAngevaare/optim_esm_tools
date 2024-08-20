import os
import typing as ty
from warnings import warn

import xarray as xr

import optim_esm_tools as oet
from .globals import _DEFAULT_MAX_TIME
from .globals import _FOLDER_FMT
from optim_esm_tools.analyze import tipping_criteria


def add_conditions_to_ds(
    ds: xr.Dataset,
    calculate_conditions: ty.Optional[
        ty.Tuple[tipping_criteria._Condition, ...]
    ] = None,
    condition_kwargs: ty.Optional[ty.Mapping] = None,
    variable_of_interest: ty.Tuple[str] = ('tas',),
    _ma_window: ty.Optional[ty.Union[int, str]] = None,
) -> xr.Dataset:
    """Transform the dataset to get it ready for handling in optim_esm_tools.

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
            tipping_criteria.StdDetrendedYearly,
            tipping_criteria.MaxJump,
            tipping_criteria.MaxJumpYearly,
            tipping_criteria.MaxDerivitive,
            tipping_criteria.MaxJumpAndStd,
            tipping_criteria.SNR,
        )  # type: ignore
    if len(set(desc := (c.short_description for c in calculate_conditions))) != len(  # type: ignore
        calculate_conditions,  # type: ignore
    ):
        raise ValueError(
            f'One or more non unique descriptions {desc}',
        )  # pragma: no cover
    if condition_kwargs is None:
        condition_kwargs = {}

    for variable in oet.utils.to_str_tuple(variable_of_interest):
        assert calculate_conditions is not None
        for cls in calculate_conditions:
            condition = cls(**condition_kwargs, variable=variable, running_mean=_ma_window)  # type: ignore
            oet.get_logger().debug(
                f'{condition} from {cls} set ma= {condition.running_mean} ma={_ma_window}',
            )
            condition_array = condition.calculate(ds)
            condition_array = condition_array.assign_attrs(
                dict(
                    short_description=cls.short_description,
                    long_description=condition.long_description,
                    name=condition_array.name,
                ),
            )
            ds[condition.short_description] = condition_array
    return ds


@oet.utils.add_load_kw
@oet.utils.timed(_stacklevel=3, _args_max=50)
def read_ds(
    base: str,
    variable_of_interest: ty.Optional[ty.Tuple[str]] = None,
    max_time: ty.Optional[ty.Tuple[int, ...]] = _DEFAULT_MAX_TIME,
    min_time: ty.Optional[ty.Tuple[int, ...]] = None,
    apply_transform: bool = True,
    pre_process: bool = True,
    strict: bool = True,
    load: ty.Optional[bool] = None,
    add_history: bool = False,
    _ma_window: ty.Optional[ty.Union[int, str]] = None,
    _cache: bool = True,
    _file_name: ty.Optional[str] = None,
    _skip_folder_info: bool = False,
    _historical_path: ty.Optional[str] = None,
    pre_proc_kw=None,
    **kwargs,
) -> ty.Optional[xr.Dataset]:
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
        add_history (bool, optional): start by merging historical dataset to the dataset.
        _ma_window (int, optional): Moving average window (assumed to be years). Defaults to 10.
        _cache (bool, optional): cache the dataset with it's extra fields to allow faster
            (re)loading. Defaults to True.
        _file_name (str, optional): name to match. Defaults to configs settings.
        _skip_folder_info (bool, optional): if set to True, do not infer the properties from the
            (synda) path of the file
        _historical_path (str, optional): If add_history is True, load from this (full) path

    kwargs:
        any kwargs are passed onto transform_ds.

    Returns:
        xr.Dataset: An xarray dataset with the appropriate variables
    """
    log = oet.config.get_logger()
    _file_name = _file_name or oet.config.config['CMIP_files']['base_name']
    _ma_window = _ma_window or int(oet.config.config['analyze']['moving_average_years'])
    data_path = os.path.join(base, _file_name)
    variable_of_interest = (
        variable_of_interest or oet.analyze.pre_process._read_variable_id(data_path)
    )
    _historical_path = _historical_file(add_history, base, _file_name, _historical_path)

    if not isinstance(variable_of_interest, str):
        raise ValueError('Only single vars supported')  # pragma: no cover
    if kwargs:
        log.error(f'Not really advised yet to call with {kwargs}')
        _cache = False
    if not apply_transform and strict:  # pragma: no cover
        # Don't cache the partial ds
        _cache = False

    log.debug(f'read_ds {variable_of_interest}')
    res_file = _name_cache_file(
        base,
        variable_of_interest,
        min_time,
        max_time,
        _ma_window,
        is_historical=_historical_path is not None,
    )

    if os.path.exists(res_file) and _cache:
        return oet.analyze.io.load_glob(res_file)

    if not os.path.exists(data_path):  # pragma: no cover
        message = f'No dataset at {data_path}'
        if strict:
            raise FileNotFoundError(message)
        log.warning(message)
        return None

    if pre_process:
        pre_proc_kw = pre_proc_kw or dict()
        data_set = oet.analyze.pre_process.get_preprocessed_ds(
            sources=data_path,
            historical_path=_historical_path,
            max_time=max_time,
            min_time=min_time,
            _ma_window=_ma_window,
            variable_id=variable_of_interest,
            **pre_proc_kw,
        )
    else:  # pragma: no cover
        message = 'Not preprocessing file is dangerous, dimensions may differ wildly!'
        if strict:
            raise ValueError(message)
        log.warning(message)
        data_set = oet.analyze.io.load_glob(data_path, load=load)

    if apply_transform:
        kwargs.update(
            dict(
                variable_of_interest=variable_of_interest,
                _ma_window=_ma_window,
            ),
        )
        data_set = add_conditions_to_ds(data_set, **kwargs)

    # start with -1 (for i==0)
    metadata = (
        {} if _skip_folder_info else oet.analyze.find_matches.folder_to_dict(base)
    )
    metadata.update(dict(path=base, file=res_file, running_mean_period=_ma_window))  # type: ignore
    if _historical_path:
        metadata.update(dict(historical_file=_historical_path))

    data_set.attrs.update(metadata)

    if _cache:
        log.info(f'Write {res_file}')
        if oet.config.config['CMIP_files']['compress'] == 'True':
            oet.analyze.pre_process.save_nc(data_set, res_file)
        else:
            data_set.to_netcdf(res_file)

    return data_set


def _historical_file(
    add_history: bool,
    base: str,
    _file_name: str,
    _historical_path: ty.Optional[str],
) -> ty.Optional[str]:
    if add_history:
        if _historical_path is not None:
            return _historical_path
        historical_heads = oet.analyze.find_matches.associate_historical(
            path=base,
            match_to='historical',
            strict=False,
        )
        if not historical_heads:
            raise FileNotFoundError(f'No historical matches for {base}')
        _historical_path = os.path.join(historical_heads[0], _file_name)
        if not os.path.exists(_historical_path):  # pragma: no cover
            raise ValueError(
                f'{_historical_path} not found, (check {historical_heads}?)',
            )
    elif _historical_path:
        raise ValueError(
            f'Wrong input _historical_path is {_historical_path} but add_history is False',
        )
    return _historical_path


def _name_cache_file(
    base,
    variable_of_interest: str,
    min_time: ty.Optional[ty.Tuple[int, ...]],
    max_time: ty.Optional[ty.Tuple[int, ...]],
    _ma_window: int,
    is_historical: bool,
    version: ty.Optional[str] = None,
) -> str:
    """Get a file name that identifies the settings."""
    version = version or oet.config.config['versions']['cmip_handler']
    _ma_window = _ma_window or int(oet.config.config['analyze']['moving_average_years'])
    path = os.path.join(
        base,
        f'{variable_of_interest}'
        f'_s{tuple(min_time) if min_time else ""}'
        f'_e{tuple(max_time) if max_time else ""}'
        f'_ma{_ma_window}'
        + ('_hist' if is_historical else '')
        + f'_optimesm_v{version}.nc',
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
