from .xarray_tools import apply_abs
from optim_esm_tools.utils import check_accepts
import xarray as xr
import numpy as np
import typing as ty

from .globals import _seconds_to_year


@apply_abs()
@check_accepts(accepts=dict(unit=('absolute', 'relative', 'std')))
def running_mean_diff(
    data_set: xr.Dataset,
    variable: str = 'tas',
    time_var: str = 'time',
    naming: str = '{variable}_run_mean_{running_mean}',
    running_mean: int = 10,
    rename_to: str = 'long_name',
    unit: str = 'absolute',
    apply_abs: bool = True,
    _t_0_date: ty.Optional[tuple] = None,
    _t_1_date: ty.Optional[tuple] = None,
) -> xr.Dataset:
    """Return difference in running mean of data set

    Args:
        data_set (xr.Dataset):
        variable (str, optional): . Defaults to 'tas'.
        time_var (str, optional): . Defaults to 'time'.
        naming (str, optional): . Defaults to '{variable}_run_mean_{running_mean}'.
        running_mean (int, optional): . Defaults to 10.
        rename_to (str, optional): . Defaults to 'long_name'.
        unit (str, optional): . Defaults to 'absolute'.
        apply_abs (bool, optional): . Defaults to True.
        _t_0_date (ty.Optional[tuple], optional): . Defaults to (2015, 1, 1).
        _t_1_date (ty.Optional[tuple], optional): . Defaults to (2100, 1, 1).

    Raises:
        ValueError: when no timestamps are not none?

    Returns:
        xr.Dataset:
    """
    var_name = naming.format(variable=variable, running_mean=running_mean)
    _time_values = data_set[time_var].dropna(time_var)

    if not len(_time_values):
        raise ValueError(f'No values for {time_var} in dataset?')

    data_var = _remove_any_none_times(data_set[var_name], time_var)

    if _t_0_date is not None:
        t_0 = _native_date_fmt(_time_values, _t_0_date)
        data_t_0 = data_var.sel(time=t_0, method='nearest')
    else:
        data_t_0 = data_var.isel(time=0)

    if _t_0_date is not None:
        t_1 = _native_date_fmt(_time_values, _t_1_date)
        data_t_1 = data_var.sel(time=t_1, method='nearest')
    else:
        data_t_1 = data_var.isel(time=-1)

    result = data_t_1 - data_t_0
    result = result.copy()
    var_unit = data_var.attrs.get('units', '{units}')
    name = data_var.attrs.get(rename_to, variable)

    if unit == 'absolute':
        result.name = f't[-1] - t[0] for {name} [{var_unit}]'
        return result

    if unit == 'relative':
        result = 100 * result / data_t_0
        result.name = f't[-1] - t[0] / t[0] for {name} $\%$'
        return result

    # Redundant if just for clarity
    if unit == 'std':
        result = result / result.std()
        result.name = f't[-1] - t[0] for {name} [$\sigma$]'
        return result


@apply_abs()
@check_accepts(accepts=dict(unit=('absolute', 'relative', 'std')))
def running_mean_std(
    data_set: xr.Dataset,
    variable: str = 'tas',
    time_var: str = 'time',
    naming: str = '{variable}_detrend_run_mean_{running_mean}',
    running_mean: int = 10,
    rename_to: str = 'long_name',
    apply_abs: bool = True,
    unit: str = 'absolute',
) -> xr.Dataset:
    data_var = naming.format(variable=variable, running_mean=running_mean)
    result = data_set[data_var].std(dim=time_var)
    result = result.copy()
    var_unit = data_set[data_var].attrs.get('units', '{units}')
    name = data_set[data_var].attrs.get(rename_to, variable)

    if unit == 'absolute':
        result.name = f'Std. {name} [{var_unit}]'
        return result

    if unit == 'relative':
        result = 100 * result / data_set[data_var].mean(dim=time_var)
        result.name = f'Relative Std. {name} [$\%$]'
        return result

    if unit == 'std':
        result = result / data_set[data_var].std()
        result.name = f'Std. {name} [$\sigma$]'
        return result


@apply_abs()
@check_accepts(accepts=dict(unit=('absolute', 'relative', 'std')))
def max_change_xyr(
    data_set: xr.Dataset,
    variable: str = 'tas',
    time_var: str = 'time',
    naming: str = '{variable}_run_mean_{running_mean}',
    x_yr: ty.Union[int, float] = 10,
    running_mean: int = 10,
    rename_to: str = 'long_name',
    apply_abs: bool = True,
    unit: str = 'absolute',
) -> xr.Dataset:
    data_var = naming.format(variable=variable, running_mean=running_mean)
    plus_x_yr = data_set.isel({time_var: slice(x_yr, None)})[data_var]
    to_min_x_yr = data_set.isel({time_var: slice(None, -x_yr)})[data_var]

    # Keep the metadata (and time stamps of the to_min_x_yr)
    result = to_min_x_yr.copy(data=plus_x_yr.values - to_min_x_yr.values)

    result = result.max(dim=time_var).copy()
    var_unit = data_set[data_var].attrs.get('units', '{units}')
    name = data_set[data_var].attrs.get(rename_to, variable)

    if unit == 'absolute':
        result.name = f'{x_yr} yr diff. {name} [{var_unit}]'
        return result

    if unit == 'relative':
        result = 100 * result / to_min_x_yr.mean(dim=time_var)
        result.name = f'{x_yr} yr diff. {name} [$\%$]'
        return result

    if unit == 'std':
        result = result / result.std()
        result.name = f'{x_yr} yr diff. {name} [$\sigma$]'
        return result


@apply_abs()
@check_accepts(accepts=dict(unit=('absolute', 'relative', 'std')))
def max_derivative(
    data_set: xr.Dataset,
    variable: str = 'tas',
    time_var: str = 'time',
    naming: str = '{variable}_run_mean_{running_mean}',
    running_mean: int = 10,
    rename_to: str = 'long_name',
    apply_abs: bool = True,
    unit: str = 'absolute',
) -> xr.Dataset:
    var_name = naming.format(variable=variable, running_mean=running_mean)

    data_array = _remove_any_none_times(data_set[var_name], time_var)
    result = data_array.differentiate(time_var).max(dim=time_var) * _seconds_to_year

    var_unit = data_array.attrs.get('units', '{units}')
    name = data_array.attrs.get(rename_to, variable)

    if unit == 'absolute':
        result.name = f'Max $\partial/\partial t$ {name} [{var_unit}/yr]'
        return result

    if unit == 'relative':
        result = 100 * result / data_array.mean(dim=time_var)
        result.name = f'Max $\partial/\partial t$ {name} [$\%$/yr]'
        return result

    if unit == 'std':
        # A local unit of sigma might be better X.std(dim=time_var)
        result = result / data_array.std()
        result.name = f'Max $\partial/\partial t$ {name} [$\sigma$/yr]'
        return result


def _remove_any_none_times(da, time_dim):
    data_var = da.copy()
    time_null = da.isnull().all(dim=set(da.dims) - {time_dim})
    if np.any(time_null):
        data_var = data_var.load().where(~time_null, drop=True)
    return data_var
