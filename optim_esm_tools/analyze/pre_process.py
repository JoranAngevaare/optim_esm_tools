import os
import shutil
import tempfile
import typing as ty

import numpy as np

from optim_esm_tools.analyze.globals import _DEFAULT_MAX_TIME
from optim_esm_tools.analyze.io import load_glob
from optim_esm_tools.analyze.xarray_tools import _native_date_fmt
from optim_esm_tools.config import config
from optim_esm_tools.config import get_logger
from optim_esm_tools.utils import timed, check_accepts, to_str_tuple
from pandas.util._decorators import deprecate_kwarg


@deprecate_kwarg('source', 'sources')
@check_accepts(dict(return_type=('path', 'data_set')))
def get_preprocessed_ds(
    sources: ty.Union[str, tuple, list],
    year_mean=False,
    _check_duplicate_years=True,
    return_type='data_set',
    skip_compression=False,
    temp_dir_location=None,
    **kw,
):
    """Create a temporary working directory for pre-process and delete all
    intermediate files."""
    if 'working_dir' in kw:  # pragma: no cover
        message = (
            f'Calling get_preprocessed_ds with working_dir={kw.get("working_dir")} is not '
            'intended, as this function is meant to open a temporary directory, load the '
            'dataset, and remove all local files.'
        )
        get_logger().warning(message)
    if return_type == 'path':
        assert 'save_as' in kw
        store_final = kw.pop('save_as')
    if temp_dir_location is None and os.path.exists('/data/volume_2/temp'):
        temp_dir_location = '/data/volume_2/temp'
    with tempfile.TemporaryDirectory(dir=temp_dir_location) as temp_dir:
        if year_mean:
            old_sources = to_str_tuple(sources)
            new_sources = [
                os.path.join(temp_dir, os.path.split(p)[1]) for p in old_sources
            ]
            for o, n in zip(old_sources, new_sources):
                _year_mon_mean(o, n)
                assert os.path.exists(n)
            sources = new_sources
        if isinstance(sources, (list, tuple)) and len(sources) > 1:
            source_tmp = os.path.join(temp_dir, 'sources_merged.nc')
            _merge_sources(list(sources), source_tmp)
            source = source_tmp
        elif len(sources) == 1:
            source = sources[0]
        else:
            assert isinstance(sources, str)
            source = sources
        defaults = dict(
            source=source,
            working_dir=temp_dir,
            clean_up=False,
            save_as=None,
        )
        for k, v in defaults.items():
            kw.setdefault(k, v)
        intermediate_file = pre_process(
            **kw,
            _check_duplicate_years=_check_duplicate_years,
        )
        ds = load_glob(intermediate_file)
        if return_type == 'data_set':
            # After with close this "with", we lose the file, so load it just to be sure we have all we need
            ds = ds.load()  # type: ignore
            ret = ds
        elif skip_compression:
            shutil.move(intermediate_file, store_final)
            ret = store_final
        else:
            save_nc(ds, store_final)
            ret = store_final

    if _check_duplicate_years:
        sanity_check(ds)
    return ret


def _merge_sources(source_files: ty.List[str], f_tmp: str) -> None:  # pragma: no cover
    import cdo

    cdo_int = cdo.Cdo()
    cdo_int.mergetime(input=source_files, output=f_tmp)


def _year_mon_mean(input_file, output_file):
    import cdo

    cdo_int = cdo.Cdo()
    cdo_int.yearmonmean(input=input_file, output=output_file)


def save_nc(ds, path):
    comp_kw = dict(
        format='NETCDF4',
        engine='netcdf4',
        encoding={k: {'zlib': True, 'complevel': 1} for k in ds.data_vars},
    )
    ds.to_netcdf(path, **comp_kw)


def sanity_check(ds):
    t_prev = None

    for i, t in enumerate(ds['time'].values):
        t_cur = getattr(t, 'year', None)
        if t_prev is not None and t_cur <= t_prev:  # pragma: no cover
            m = f'Got at least one overlapping year on index {i} {t_cur} {t_prev}'
            raise ValueError(m)
        t_prev = t_cur


def pre_process(
    source: str,
    historical_path: ty.Optional[str] = None,
    target_grid: ty.Union[None, str, bool] = None,
    max_time: ty.Optional[ty.Tuple[int, ...]] = _DEFAULT_MAX_TIME,
    min_time: ty.Optional[ty.Tuple[int, int, int]] = None,
    save_as: ty.Optional[str] = None,
    clean_up: bool = True,
    _ma_window: ty.Union[int, str, None] = None,
    variable_id: ty.Optional[str] = None,
    working_dir: ty.Optional[str] = None,
    _check_duplicate_years=True,
    do_detrend=True,
    do_running_mean=True,
) -> str:  # type: ignore
    """Apply several preprocessing steps to the file located at <source>:

      - Slice the data to desired time range
      - regrid to simple grid
      - calculate corresponding area
      - calculate running mean, detrended and not-detrended
      - merge all files into one

    Args:
        source (str): path of file to parse
        historical_path (str, None): if specified, first merge this file to the source.
        target_grid (str, optional): Grid specification (like n64, n90 etc.). Defaults to None and
            is taken from config.
        max_time (ty.Optional[ty.Tuple[int, int, int]], optional): Defines time range in which to
            load data. Defaults to (2100, 12, 30).
        min_time (ty.Optional[ty.Tuple[int, int, int]], optional): Defines time range in which to
            load data. Defaults to None.
        save_as (str, optional): path where to store the pre-processed folder. Defaults to None.
        clean_up (bool, optional): delete intermediate files. Defaults to True.
        _ma_window (int, optional): moving average window (assumed 10 years). Defaults to None.
        variable_id (str, optional): Name of the variable of interest. Defaults to None.

    Raises:
        ValueError: If source and dest are the same, we'll run into problems

    Returns:
        str: path of the dest file (same provided, if any)
    """

    import cdo

    _remove_bad_vars(source)
    if historical_path is not None:
        _remove_bad_vars(historical_path)
    variable_id = variable_id or _read_variable_id(source)
    use_max_time = max_time or (9999, 12, 30)  # unreasonably far away
    use_min_time = min_time or (0, 1, 1)  # unreasonably long ago

    do_regrid = target_grid != False
    target_grid = target_grid or config['analyze']['regrid_to']

    _ma_window = _ma_window or config['analyze']['moving_average_years']
    _check_time_range(source, use_max_time, use_min_time, _ma_window)

    cdo_int = cdo.Cdo()
    head, _ = os.path.split(source)
    working_dir = working_dir or head

    # Several intermediate_files
    f_time = os.path.join(working_dir, 'time_sel.nc')
    f_tmp = os.path.join(working_dir, 'tmp.nc')
    f_regrid = os.path.join(working_dir, 'regrid.nc')
    f_area = os.path.join(working_dir, 'area.nc')
    f_det = os.path.join(working_dir, 'detrend.nc')
    f_rm = os.path.join(working_dir, f'rm_{_ma_window}.nc')
    f_det_rm = os.path.join(working_dir, f'detrend_rm_{_ma_window}.nc')
    files = [f_time, f_det, f_det_rm, f_rm, f_tmp, f_regrid, f_area]

    save_as = save_as or os.path.join(working_dir, 'result.nc')

    # Several names:
    var = variable_id

    for p in files + [save_as]:
        if p == source:
            raise ValueError(f'source equals other path {p}')  # pragma: no cover
        if os.path.exists(p):  # pragma: no cover
            get_logger().warning(f'Removing {p}!')
            os.remove(p)
    if historical_path:
        _remap_and_merge(
            cdo_int,
            cdo,
            historical_path,
            source,
            target_grid,
            working_dir,
            f_tmp,
        )
        source = f_tmp
    if _check_duplicate_years:
        _remove_duplicate_time_stamps(source)

    next_source = source
    if min_time is not None or max_time is not None:
        time_range = f'{_fmt_date(use_min_time)},{_fmt_date(use_max_time)}'
        cdo_int.seldate(time_range, input=next_source, output=f_time)  # type: ignore
        next_source = f_time

    if do_regrid:
        cdo_int.remapbil(target_grid, input=next_source, output=f_regrid)  # type: ignore
        cdo_int.gridarea(input=f_regrid, output=f_area)  # type: ignore
        input_files = [f_regrid, f_area]
    else:
        input_files = [f_regrid]
        os.rename(next_source, f_regrid)
    next_source = f_regrid

    if do_detrend:
        var_det = f'{var}_detrend'
        cdo_int.detrend(input=next_source, output=f_tmp)  # type: ignore
        cdo_int.chname(f'{var},{var_det}', input=f_tmp, output=f_det)  # type: ignore
        os.remove(f_tmp)
        input_files += [f_det]

    if do_running_mean:
        var_rm = f'{var}_run_mean_{_ma_window}'
        cdo_int.runmean(_ma_window, input=f_regrid, output=f_tmp)  # type: ignore
        _run_mean_patch(
            f_start=f_regrid,
            f_rm=f_tmp,
            f_out=f_rm,
            ma_window=_ma_window,
            var_name=var,
            var_rm_name=var_rm,
        )
        os.remove(f_tmp)
        input_files += [f_rm]

    if do_running_mean and do_detrend:
        var_det_rm = f'{var_det}_run_mean_{_ma_window}'
        cdo_int.detrend(input=f_rm, output=f_tmp)  # type: ignore
        cdo_int.chname(f'{var_rm},{var_det_rm}', input=f_tmp, output=f_det_rm)  # type: ignore
        input_files += [f_det_rm]
    get_logger().info(f'Join {input_files} to {save_as}')
    cdo_int.merge(input=' '.join(input_files), output=save_as)  # type: ignore

    if clean_up:  # pragma: no cover
        for p in files:
            if os.path.exists(p):
                os.remove(p)
    return save_as


def _quick_drop_duplicates(ds, t_span, t_len, path):
    ds = ds.drop_duplicates('time')
    if (t_new_len := len(ds['time'])) > t_span + 1:
        raise ValueError(f'{t_new_len} too long! Started with {t_len} and {t_span}')
    get_logger().warning('Timestamp issue solved')
    with tempfile.TemporaryDirectory() as temp_dir:
        save_as = os.path.join(temp_dir, 'temp.nc')
        ds.to_netcdf(save_as)
        # move the old file
        os.rename(path, os.path.join(os.path.split(path)[0], 'faulty_merged.nc'))
        shutil.copy2(save_as, path)


def _drop_duplicates_carefully(ds, t_span, t_len, path):
    from tqdm import tqdm

    # As we only do this for huge datasets, it might be that /tmp doesn't allow storing sufficient data.
    work_dir = os.path.split(path)[0]
    with tempfile.TemporaryDirectory(dir=work_dir) as temp_dir:
        keep_years = np.argwhere(np.diff([t.year for t in ds['time'].values]) == 1)[
            :,
            0,
        ]
        keep_years = [0] + [int(i) + 1 for i in keep_years]
        saves = []
        for i in tqdm(keep_years):
            save_as = os.path.join(temp_dir, f'temp_{i}.nc')
            saves.append(save_as)
            ds.isel(time=slice(i, i + 1)).load().to_netcdf(save_as)
        # move the old file
        _tempf = os.path.join(temp_dir, 'temp_merge.nc')
        get_logger().warning(f'Merging {saves} -> {_tempf}')
        _merge_sources(saves, _tempf)
        ds = load_glob(_tempf)
        if (t_new_len := len(ds['time'])) > t_span + 1:
            raise ValueError(
                f'{t_new_len} too long! Started with {t_len} and {t_span}',
            )
        get_logger().warning('Timestamp issue solved')

        os.rename(path, os.path.join(os.path.split(path)[0], 'faulty_merged.nc'))
        os.rename(_tempf, path)
    get_logger().warning(f'_remove_duplicate_time_stamps - > Fixed!')


def _remove_duplicate_time_stamps(path):  # pragma: no cover
    ds = load_glob(path)
    t_len = len(ds['time'])
    if t_len <= 1:
        raise ValueError(f'No time length in {path}')
    if t_len > (t_span := (ds['time'].values[-1].year - ds['time'].values[0].year)) + 1:
        get_logger().warning(
            f'Finding {t_len} timestamps in {t_span} years - removing duplicates',
        )
        if ds.nbytes / 1e6 < 1_000:
            _quick_drop_duplicates(ds, t_span, t_len, path)
        else:
            _drop_duplicates_carefully(ds, t_span, t_len, path)


def _remap_and_merge(
    cdo_int,
    cdo,
    historical_path: str,
    source: str,
    target_grid: ty.Union[bool, str],
    working_dir: str,
    f_tmp: str,
) -> None:  # pragma: no cover
    """The function `_remap_and_merge` merges two input files,
    `historical_path` and `source`, and if an exception occurs, it regrids the
    files and tries again.

    :param cdo_int: The `cdo_int` parameter is an instance of the `cdo` module, which is used for
    executing CDO commands in Python
    :param cdo: The `cdo` parameter is an instance of the `cdo.Cdo` class. It is used to interact with
    the Climate Data Operators (CDO) library, which provides a collection of command-line tools for
    working with climate and weather data
    :param historical_path: The `historical_path` parameter is a string that represents the path to a
    file containing historical data
    :type historical_path: str
    :param source: The `source` parameter is a string that represents the path to the source file that
    needs to be merged with the `historical_path` file
    :type source: str
    :param target_grid: The `target_grid` parameter is a string that specifies the path to the target
    grid file. It is used in the `remapbil` function to regrid the input data to the target grid before
    merging
    :type target_grid: str
    :param working_dir: The `working_dir` parameter is a string that represents the directory where
    temporary files will be stored during the execution of the `_remap_and_merge` function
    :type working_dir: str
    :param f_tmp: The `f_tmp` parameter is a string that represents the path to the output file where
    the merged data will be saved
    :type f_tmp: str
    """
    try:
        cdo_int.mergetime(input=[historical_path, source], output=f_tmp)
    except cdo.CDOException as e:  # pragma: no cover
        get_logger().error(f"Ran into {e}, let's regrid first and retry")
        if target_grid == False:
            raise ValueError(
                f'Cannot merge {historical_path} and {source} since target grid is False and we ran into {e}',
            ) from e
        cdo_int.remapbil(
            target_grid,
            input=historical_path,
            output=(_a := os.path.join(working_dir, '_a.nc')),
        )
        cdo_int.remapbil(
            target_grid,
            input=source,
            output=(_b := os.path.join(working_dir, '_b.nc')),
        )
        cdo_int.mergetime(input=[_a, _b], output=f_tmp)


def _remove_bad_vars(path):
    """The function `_remove_bad_vars` removes specified variables from a
    dataset and replaces the original file with the modified dataset.

    :param path: The `path` parameter is a string that represents the file path to the dataset that
    needs to be processed
    """
    log = get_logger()
    to_delete = config['analyze']['remove_vars'].split()
    ds = load_glob(path)
    drop_any = False
    for var in to_delete:  # pragma: no cover
        if var in ds.data_vars:
            log.warning(f'{var} in dataset from {path}')
            drop_any = True
            ds = ds.load()
            ds = ds.drop_vars(var)
    if drop_any:  # pragma: no cover
        log.error(f'Replacing {path} after dropping at least one of {to_delete}')
        os.remove(path)
        ds.to_netcdf(path)
        if not os.path.exists(path):
            raise RuntimeError(f'Data loss, somehow {path} got removed!')


def _check_time_range(path, max_time, min_time, ma_window):
    """The function `_check_time_range` checks if the time stamps in a dataset
    fall within a specified time range and raises an error if the number of
    time stamps is less than a specified moving average window size.

    :param path: The `path` parameter is the file path or file pattern that specifies the location of
    the data files to be loaded
    :param max_time: The maximum time value to consider in the time range
    :param min_time: The `min_time` parameter represents the minimum time or date range for the data. It
    is a tuple with three elements representing the year, month, and day respectively
    :param ma_window: The `ma_window` parameter represents the moving average window size. It is used to
    determine the minimum number of time stamps required within the specified time range (`[min_time,
    max_time]`) in order to proceed with the calculation of the moving average. If the number of time
    stamps within the time
    """
    ds = load_glob(path)
    times = ds['time'].values
    time_mask = times < _native_date_fmt(times, max_time)
    if min_time != (0, 1, 1):
        # CF time does not always support year 0
        time_mask &= times > _native_date_fmt(times, min_time)
    if time_mask.sum() < float(ma_window):
        message = f'Data from {path} has {time_mask.sum()} time stamps in [{min_time}, {max_time}]'
        raise NoDataInTimeRangeError(message)


def _run_mean_patch(f_start, f_rm, f_out, ma_window, var_name, var_rm_name):
    """Patch running mean file, since cdo decreases the length of the file by
    the ma_window, merging two files of different durations results in bad
    data.

    As a solution, we take the original file (f_start) and use it's
    shape (like the number of timestamps) and fill those timestamps
    where the value of the running mean is defined. Everything else is
    set to zero.
    """
    ds_base = load_glob(f_start)
    ds_rm = load_glob(f_rm)
    ds_out = ds_base.copy()

    # Replace timestamps with the CDO computed running mean
    data = np.zeros(ds_base[var_name].shape, ds_base[var_name].dtype)
    data[:] = np.nan
    ma_window = int(ma_window)
    data[ma_window // 2 : 1 - ma_window // 2] = ds_rm[var_name].values
    ds_out[var_name].data = data

    # Patch variables and save
    ds_out = ds_out.rename({var_name: var_rm_name})
    ds_out.attrs = ds_rm.attrs
    ds_out.to_netcdf(f_out)


class NoDataInTimeRangeError(Exception):
    pass


def _fmt_date(date: tuple) -> str:
    """The function `_fmt_date` takes a tuple representing a date and returns a
    formatted string in the format 'YYYY-MM-DD'.

    :param date: The `date` parameter is a tuple containing three elements: year, month, and day
    :type date: tuple
    :return: a formatted string representing the date in the format "YYYY-MM-DD".
    """
    assert len(date) == 3
    y, m, d = date
    return f'{y:04}-{m:02}-{d:02}'


def _read_variable_id(path):
    """The function `_read_variable_id` reads the variable ID from a given path
    and raises an error if the information is not available.

    :param path: The `path` parameter is a string that represents the file path from which the variable
    ID needs to be read
    :return: the value of the 'variable_id' attribute from the file located at the given path.
    """
    try:
        return load_glob(path).attrs['variable_id']
    except KeyError as e:  # pragma: no cover
        message = f'When reading the variable_id from {path}, it appears no such information is available'
        raise KeyError(message) from e
