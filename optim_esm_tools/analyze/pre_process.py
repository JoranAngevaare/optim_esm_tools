from optim_esm_tools.utils import timed
from optim_esm_tools.config import config, get_logger
import os
import typing as ty


@timed
def pre_process(
    source: str,
    target_grid: str = None,
    max_time: ty.Optional[ty.Tuple[int, int, int]] = (2100, 1, 1),
    min_time: ty.Optional[ty.Tuple[int, int, int]] = None,
    save_as: str = None,
    clean_up: bool = True,
    _ma_window: int = None,
    variable_id: str = None,
) -> str:
    """Apply several preprocessing steps to the file located at <source>:
      - Slice the data to desired time range
      - regrid to simple grid
      - calculate corresponding area
      - calculate running mean, detrended and not-detrended
      - merge all files into one

    Args:
        source (str): path of file to parse
        target_grid (str, optional): Grid specification (like n64, n90 etc.). Defaults to None and is taken from config.
        max_time (ty.Optional[ty.Tuple[int, int, int]], optional): Defines time range in which to load data. Defaults to (2100, 1, 1).
        min_time (ty.Optional[ty.Tuple[int, int, int]], optional): Defines time range in which to load data. Defaults to None.
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

    variable_id = variable_id or _read_variable_id(source)
    max_time = max_time or (9999, 1, 1)  # unreasonably far away
    min_time = min_time or (0, 1, 1)  # unreasonably long ago

    target_grid = target_grid or config['analyze']['regrid_to']
    _ma_window = _ma_window or config['analyze']['moving_average_years']

    cdo_int = cdo.Cdo()
    head, _ = os.path.split(source)

    # Several intermediate_files
    f_time = os.path.join(head, 'time_sel.nc')
    f_det = os.path.join(head, 'detrend.nc')
    f_det_rm = os.path.join(head, f'detrend_rm_{_ma_window}.nc')
    f_rm = os.path.join(head, f'rm_{_ma_window}.nc')
    f_tmp = os.path.join(head, 'tmp.nc')
    f_regrid = os.path.join(head, 'regrid.nc')
    f_area = os.path.join(head, 'area.nc')
    files = [f_time, f_det, f_det_rm, f_rm, f_tmp, f_regrid, f_area]

    save_as = save_as or os.path.join(head, 'result.nc')

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

    cdo_int.seldate(
        f'{_fmt_date(min_time)},{_fmt_date(max_time)}', input=source, output=f_time
    )

    cdo_int.remapbil(target_grid, input=f_time, output=f_regrid)
    cdo_int.gridarea(input=f_regrid, output=f_area)

    cdo_int.detrend(input=f_regrid, output=f_tmp)
    cdo_int.chname(f'{var},{var_det}', input=f_tmp, output=f_det)
    os.remove(f_tmp)

    cdo_int.runmean(_ma_window, input=f_regrid, output=f_tmp)

    cdo_int.chname(f'{var},{var_rm}', input=f_tmp, output=f_rm)
    os.remove(f_tmp)

    cdo_int.detrend(input=f_rm, output=f_tmp)
    cdo_int.chname(f'{var_rm},{var_det_rm}', input=f_tmp, output=f_det_rm)
    # remove in cleanup
    cdo_int.merge(
        input=' '.join([f_regrid, f_det, f_det_rm, f_rm, f_area]), output=save_as
    )

    if clean_up:
        for p in files:
            os.remove(p)
    return save_as


def _fmt_date(date: tuple) -> str:
    assert len(date) == 3
    y, m, d = date
    return f'{y:04}-{m:02}-{d:02}'


def _read_variable_id(path):
    import xarray as xr

    try:
        return xr.open_dataset(path).attrs['variable_id']
    except KeyError as e:
        raise KeyError(
            f'When reading the variable_id from {path}, it appears no such information is available'
        ) from e
