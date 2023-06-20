from .xarray_tools import apply_abs, _native_date_fmt
from optim_esm_tools.utils import check_accepts, timed
import xarray as xr
import numpy as np
import typing as ty
from .globals import _SECONDS_TO_YEAR
import abc
from immutabledict import immutabledict


class _Condition(abc.ABC):
    short_description: str
    defaults = immutabledict(
        rename_to='long_name',
        unit='absolute',
        apply_abs=True,
    )

    def __init__(self, variable='tas', running_mean=10, time_var='time', **kwargs):
        self.variable = variable
        self.running_mean = running_mean
        self.time_var = time_var
        if kwargs:
            for k, v in self.defaults.items():
                kwargs.setdefault(k, v)
            self.defaults = immutabledict(kwargs)

    def calculate(self, *arg, **kwarg):
        raise NotImplementedError

    @property
    def long_description(self):
        raise NotImplementedError


class StartEndDifference(_Condition):
    short_description: str = 'start end difference'

    @property
    def long_description(self):
        return f'Difference of running mean ({self.running_mean} yr) between start and end of time series. Not detrended'

    def calculate(self, dataset):
        return running_mean_diff(
            dataset,
            variable=self.variable,
            time_var=self.time_var,
            naming='{variable}_run_mean_{running_mean}',
            running_mean=self.running_mean,
            # TODO
            # Pass kw arguments on? I think not
            _t_0_date=None,
            _t_1_date=None,
            **self.defaults,
        )


class StdDetrended(_Condition):
    short_description: str = 'std detrended'

    @property
    def long_description(self):
        return f'Standard deviation of running mean ({self.running_mean} yr). Detrended'

    def calculate(self, dataset):
        return running_mean_std(
            dataset,
            variable=self.variable,
            time_var=self.time_var,
            naming='{variable}_detrend_run_mean_{running_mean}',
            running_mean=self.running_mean,
            **self.defaults,
        )


class MaxJump(_Condition):
    short_description: str = 'max jump'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.number_of_years = 10

    @property
    def long_description(self):
        return f'Max change in {self.number_of_years} yr in the running mean ({self.running_mean} yr). Not detrended'

    def calculate(self, dataset):
        return max_change_xyr(
            dataset,
            variable=self.variable,
            time_var=self.time_var,
            naming='{variable}_run_mean_{running_mean}',
            x_yr=self.number_of_years,
            running_mean=self.running_mean,
            **self.defaults,
        )


class MaxDerivitive(_Condition):
    short_description: str = 'max derivative'

    @property
    def long_description(self):
        return f'Max value of the first order derivative of the running mean ({self.running_mean} yr). Not deterended'

    def calculate(self, dataset):
        return max_derivative(
            dataset,
            variable=self.variable,
            time_var=self.time_var,
            naming='{variable}_run_mean_{running_mean}',
            running_mean=self.running_mean,
            **self.defaults,
        )


@timed
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


@timed
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


@timed
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


@timed
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
    result = data_array.differentiate(time_var).max(dim=time_var) * _SECONDS_TO_YEAR

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
    time_null = data_var.isnull().all(dim=set(data_var.dims) - {time_dim})
    if np.all(time_null):
        # If we take a running mean of 10 (the default), and the array is shorter than
        # 10 years we will run into issues here because a the window is longer than the
        # array. Perhaps we should raise higher up.
        raise ValueError(
            f'This array only has NaN values, perhaps array too short ({len(time_null)} < 10)?'
        )
    if np.any(time_null):
        try:
            # For some reason only alt_calc seems to work even if it should be equivalent to the data_var
            # I think there is some fishy indexing going on in pandas <-> dask
            # Maybe worth raising an issue?
            alt_calc = xr.where(~time_null, da, np.nan).dropna('time')
            data_var = data_var.load().where(~time_null, drop=True)
            assert np.all((alt_calc == data_var).values)
        except IndexError as e:
            print(e)
            return alt_calc
    return data_var
