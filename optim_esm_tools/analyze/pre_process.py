from optim_esm_tools.utils import timed
from optim_esm_tools.config import config, get_logger
from optim_esm_tools.analyze.xarray_tools import _native_date_fmt
from optim_esm_tools.analyze.io import load_glob
from optim_esm_tools.analyze.globals import _DEFAULT_MAX_TIME
import numpy as np
import os
import typing as ty
import tempfile


def get_preprocessed_ds(source, **kw):
    """Create a temporary working directory for pre-process and delete all intermediate files"""
    if 'working_dir' in kw:
        message = (
            f'Calling get_preprocessed_ds with working_dir={kw.get("working_dir")} is not '
            'intended, as this function is meant to open a temporary directory, load the '
            'dataset, and remove all local files.'
        )
        get_logger().warning(message)
    with tempfile.TemporaryDirectory() as temp_dir:
        defaults = dict(
            source=source, working_dir=temp_dir, clean_up=False, save_as=None
        )
        for k, v in defaults.items():
            kw.setdefault(k, v)
        intermediate_file = pre_process(**kw)
        # After with close this "with", we lose the file, so load it just to be sure we have all we need
        ds = load_glob(intermediate_file).load()
    return ds


@timed
def pre_process(
    source: str,
    target_grid: str = None,
    max_time: ty.Optional[ty.Tuple[int, int, int]] = _DEFAULT_MAX_TIME,
    min_time: ty.Optional[ty.Tuple[int, int, int]] = None,
    save_as: str = None,
    clean_up: bool = True,
    _ma_window: int = None,
    variable_id: str = None,
    working_dir: str = None,
) -> str:
    """Apply several preprocessing steps to the file located at <source>:
      - Slice the data to desired time range
      - regrid to simple grid
      - calculate corresponding area
      - calculate running mean, detrended and not-detrended
      - merge all files into one

    Args:
        source (str): path of file to parse
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
    variable_id = variable_id or _read_variable_id(source)
    max_time = max_time or (9999, 12, 30)  # unreasonably far away
    min_time = min_time or (0, 1, 1)  # unreasonably long ago
    target_grid = target_grid or config['analyze']['regrid_to']
    _ma_window = _ma_window or config['analyze']['moving_average_years']
    _check_time_range(source, max_time, min_time, _ma_window)

    cdo_int = cdo.Cdo()
    head, _ = os.path.split(source)
    working_dir = working_dir or head

    # Several intermediate_files
    f_time = os.path.join(working_dir, 'time_sel.nc')
    f_det = os.path.join(working_dir, 'detrend.nc')
    f_det_rm = os.path.join(working_dir, f'detrend_rm_{_ma_window}.nc')
    f_rm = os.path.join(working_dir, f'rm_{_ma_window}.nc')
    f_tmp = os.path.join(working_dir, 'tmp.nc')
    f_regrid = os.path.join(working_dir, 'regrid.nc')
    f_area = os.path.join(working_dir, 'area.nc')
    files = [f_time, f_det, f_det_rm, f_rm, f_tmp, f_regrid, f_area]

    save_as = save_as or os.path.join(working_dir, 'result.nc')

    # Several names:
    var = variable_id
    var_det = f'{var}_detrend'
    var_rm = f'{var}_run_mean_{_ma_window}'
    var_det_rm = f'{var_det}_run_mean_{_ma_window}'

    for p in files + [save_as]:
        if p == source:
            raise ValueError(f'source equals other path {p}')
        if os.path.exists(p):
            get_logger().warning(f'Removing {p}!')
            os.remove(p)

    time_range = f'{_fmt_date(min_time)},{_fmt_date(max_time)}'
    cdo_int.seldate(time_range, input=source, output=f_time)

    cdo_int.remapbil(target_grid, input=f_time, output=f_regrid)
    cdo_int.gridarea(input=f_regrid, output=f_area)

    cdo_int.detrend(input=f_regrid, output=f_tmp)
    cdo_int.chname(f'{var},{var_det}', input=f_tmp, output=f_det)
    os.remove(f_tmp)

    cdo_int.runmean(_ma_window, input=f_regrid, output=f_tmp)
    _run_mean_patch(
        f_start=f_regrid,
        f_rm=f_tmp,
        f_out=f_rm,
        ma_window=_ma_window,
        var_name=var,
        var_rm_name=var_rm,
    )
    os.remove(f_tmp)

    cdo_int.detrend(input=f_rm, output=f_tmp)
    cdo_int.chname(f'{var_rm},{var_det_rm}', input=f_tmp, output=f_det_rm)
    # remove in cleanup

    input_files = ' '.join([f_regrid, f_det, f_det_rm, f_rm, f_area])
    cdo_int.merge(input=input_files, output=save_as)

    if clean_up:
        for p in files:
            os.remove(p)
    return save_as


def _remove_bad_vars(path):
    log = get_logger()
    to_delete = config['analyze']['remove_vars'].split()
    ds = load_glob(path)
    drop_any = False
    for var in to_delete:
        if var in ds.data_vars:
            log.warning(f'{var} in dataset from {path}')
            drop_any = True
            ds = ds.load()
            ds = ds.drop_vars(var)
    if drop_any:
        log.error(f'Replacing {path} after dropping at least one of {to_delete}')
        os.remove(path)
        ds.to_netcdf(path)
        if not os.path.exists(path):
            raise RuntimeError(f'Data loss, somehow {path} got removed!')


def _check_time_range(path, max_time, min_time, ma_window):
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
    """
    Patch running mean file, since cdo decreases the length of the file by the ma_window, merging
    two files of different durations results in bad data.

    As a solution, we take the original file (f_start) and use it's shape (like the number of
    timestamps) and fill those timestamps where the value of the running mean is defined.
    Everything else is set to zero.
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
    assert len(date) == 3
    y, m, d = date
    return f'{y:04}-{m:02}-{d:02}'


def _read_variable_id(path):
    try:
        return load_glob(path).attrs['variable_id']
    except KeyError as e:
        raise KeyError(
            f'When reading the variable_id from {path}, it appears no such information is available'
        ) from e
