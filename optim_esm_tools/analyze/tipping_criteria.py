import abc
import typing as ty

import numpy as np
import xarray as xr
from immutabledict import immutabledict

from .globals import _SECONDS_TO_YEAR
from .tools import rank2d
from .xarray_tools import _native_date_fmt
from .xarray_tools import _remove_any_none_times
from .xarray_tools import apply_abs
from optim_esm_tools.utils import check_accepts
from optim_esm_tools.utils import deprecated
from optim_esm_tools.utils import timed


class _Condition(abc.ABC):
    short_description: str
    defaults: immutabledict = immutabledict(
        rename_to='long_name',
        unit='absolute',
        apply_abs=True,
    )

    def __init__(
        self,
        variable: str,
        running_mean: int = 10,
        time_var: str = 'time',
        **kwargs,
    ):
        self.variable = variable
        self.running_mean = running_mean
        self.time_var = time_var
        if kwargs:
            for k, v in self.defaults.items():
                kwargs.setdefault(k, v)
            self.defaults = immutabledict(kwargs)

    def calculate(self, *arg, **kwarg):
        raise NotImplementedError  # pragma: no cover

    @property
    def long_description(self):
        raise NotImplementedError  # pragma: no cover


class StartEndDifference(_Condition):
    short_description: str = 'start end difference'

    @property
    def long_description(self) -> str:
        return f'Difference of running mean ({self.running_mean} yr) between start and end of time series. Not detrended'

    def calculate(self, data_set: xr.Dataset):
        return running_mean_diff(
            data_set,
            variable=self.variable,  # type: ignore
            time_var=self.time_var,  # type: ignore
            naming='{variable}_run_mean_{running_mean}',  # type: ignore
            running_mean=self.running_mean,  # type: ignore
            **self.defaults,
        )


class StdDetrended(_Condition):
    short_description: str = 'std detrended'

    @property
    def long_description(self) -> str:
        return f'Standard deviation of running mean ({self.running_mean} yr). Detrended'

    @property
    def use_variable(self) -> str:
        return '{variable}_detrend_run_mean_{running_mean}'

    def calculate(self, data_set: xr.Dataset):
        return running_mean_std(
            data_set,
            variable=self.variable,  # type: ignore
            time_var=self.time_var,  # type: ignore
            naming=self.use_variable,  # type: ignore
            running_mean=self.running_mean,  # type: ignore
            **self.defaults,
        )


class MaxJump(_Condition):
    short_description: str = 'max jump'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.number_of_years = 10

    @property
    def long_description(self) -> str:
        return f'Max change in {self.number_of_years} yr in the running mean ({self.running_mean} yr). Not detrended'

    @property
    def use_variable(self) -> str:
        return '{variable}_run_mean_{running_mean}'

    def calculate(self, data_set: xr.Dataset):
        return max_change_xyr(
            data_set,
            variable=self.variable,  # type: ignore
            time_var=self.time_var,  # type: ignore
            naming=self.use_variable,  # type: ignore
            x_yr=self.number_of_years,  # type: ignore
            running_mean=self.running_mean,  # type: ignore
            **self.defaults,
        )


class MaxJumpYearly(MaxJump):
    short_description: str = 'max jump yearly'

    def __init__(self, *args, **kwargs):
        kwargs['running_mean'] = 1
        super().__init__(*args, **kwargs)

    @property
    def use_variable(self) -> str:
        assert self.running_mean == 1
        return '{variable}'


class StdDetrendedYearly(StdDetrended):
    short_description: str = 'std detrended yearly'

    @property
    def long_description(self) -> str:
        return 'Standard deviation. Detrended'

    @property
    def use_variable(self) -> str:
        return '{variable}_detrend'


class MaxDerivitive(_Condition):
    short_description: str = 'max derivative'

    @property
    def long_description(self) -> str:
        return f'Max value of the first order derivative of the running mean ({self.running_mean} yr). Not deterended'

    def calculate(self, data_set: xr.Dataset):
        return max_derivative(
            data_set,
            variable=self.variable,  # type: ignore
            time_var=self.time_var,  # type: ignore
            naming='{variable}_run_mean_{running_mean}',  # type: ignore
            running_mean=self.running_mean,  # type: ignore
            **self.defaults,
        )


class MaxJumpAndStd(_Condition):
    short_description: str = 'percentile score std and max jump'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.number_of_years = 10

    @staticmethod
    def parents():
        return MaxJump, StdDetrended

    @property
    def long_description(self) -> str:
        p1, p2 = self.parents()
        return f'Product of {p1.short_description} and {p2.short_description}'

    def get_parents_init(self) -> ty.List[_Condition]:
        return [
            p(
                variable=self.variable,
                running_mean=self.running_mean,
                time_var=self.time_var,
                **self.defaults,
            )
            for p in self.parents()
        ]

    def get_parent_results(self, data_set: xr.Dataset) -> ty.Dict[str, float]:
        super_1, super_2 = self.get_parents_init()
        da_1 = super_1.calculate(data_set)
        da_2 = super_2.calculate(data_set)
        assert super_1.short_description != super_2.short_description, (
            super_1.short_description,
            super_2.short_description,
        )
        return {super_1: da_1, super_2: da_2}

    def calculate(self, data_set: xr.Dataset):
        da_1, da_2 = self.get_parent_results(data_set).values()
        combined_score = np.ones_like(da_1.values, dtype=np.float64)
        for da in [da_1, da_2]:
            combined_score *= rank2d(da.values)
        return xr.DataArray(
            combined_score,
            coords=da_1.coords,
            name=self.short_description,
        )


class SNR(MaxJumpAndStd):
    short_description: str = 'max jump div. std'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @staticmethod
    def parents():
        return MaxJump, StdDetrended

    @property
    def long_description(self) -> str:
        p1, p2 = self.parents()
        return f'Signal to noise ratio of {p1.short_description}/{p2.short_description}'

    def calculate(self, data_set: xr.Dataset):
        da_1, da_2 = self.get_parent_results(data_set).values()
        res = da_1 / da_2
        res.name = self.short_description
        return res


@timed
@apply_abs()
@check_accepts(accepts=dict(unit=('absolute', 'relative', 'std')))
def running_mean_diff(
    data_set: xr.Dataset,
    variable: str,
    time_var: str = 'time',
    naming: str = '{variable}_run_mean_{running_mean}',
    running_mean: int = 10,
    rename_to: str = 'long_name',
    unit: str = 'absolute',
    apply_abs: bool = True,
) -> xr.DataArray:  # type: ignore
    """Return difference in running mean of data set.

    Args:
        data_set (xr.Dataset):
        variable (str, optional): . Defaults to 'tas'.
        time_var (str, optional): . Defaults to 'time'.
        naming (str, optional): . Defaults to '{variable}_run_mean_{running_mean}'.
        running_mean (int, optional): . Defaults to 10.
        rename_to (str, optional): . Defaults to 'long_name'.
        unit (str, optional): . Defaults to 'absolute'.
        apply_abs (bool, optional): . Defaults to True.
    Raises:
        ValueError: when no timestamps are not none?

    Returns:
        xr.Dataset:
    """
    var_name = naming.format(variable=variable, running_mean=running_mean)
    _time_values = data_set[time_var].dropna(time_var)

    if not len(_time_values):
        raise ValueError(f'No values for {time_var} in data_set?')  # pragma: no cover

    data_var = _remove_any_none_times(data_set[var_name], time_var)

    data_t_0 = data_var.isel(time=0)
    data_t_1 = data_var.isel(time=-1)
    result = data_t_1 - data_t_0
    result = result.copy()
    var_unit = data_var.attrs.get('units', '{units}').replace('%', r'\%')
    name = data_var.attrs.get(rename_to, variable)

    if unit == 'absolute':
        result.name = f't[-1] - t[0] for {name} [{var_unit}]'
        return result

    if unit == 'relative':
        result = 100 * result / data_t_0
        result.name = fr't[-1] - t[0] / t[0] for {name} $\%$'
        return result

    # Redundant if just for clarity
    if unit == 'std':
        result = result / result.std()
        result.name = fr't[-1] - t[0] for {name} [$\sigma$]'
        return result
    raise ValueError(f'{unit} is invalid')


@timed
@apply_abs()
@check_accepts(accepts=dict(unit=('absolute', 'relative', 'std')))
def running_mean_std(
    data_set: xr.Dataset,
    variable: str,
    time_var: str = 'time',
    naming: str = '{variable}_detrend_run_mean_{running_mean}',
    running_mean: int = 10,
    rename_to: str = 'long_name',
    apply_abs: bool = True,
    unit: str = 'absolute',
) -> xr.DataArray:  # type: ignore
    data_var = naming.format(variable=variable, running_mean=running_mean)
    result = data_set[data_var].std(dim=time_var)
    result = result.copy()
    var_unit = data_set[data_var].attrs.get('units', '{units}').replace('%', r'\%')
    name = data_set[data_var].attrs.get(rename_to, variable)

    if unit == 'absolute':
        result.name = f'Std. {name} [{var_unit}]'
        return result

    if unit == 'relative':
        result = 100 * result / data_set[data_var].mean(dim=time_var)
        result.name = fr'Relative Std. {name} [$\%$]'
        return result

    if unit == 'std':
        result = result / data_set[data_var].std()
        result.name = fr'Std. {name} [$\sigma$]'
        return result


@timed
@apply_abs()
@check_accepts(accepts=dict(unit=('absolute', 'relative', 'std')))
def max_change_xyr(
    data_set: xr.Dataset,
    variable: str,
    time_var: str = 'time',
    naming: str = '{variable}_run_mean_{running_mean}',
    x_yr: ty.Union[int, float] = 10,
    running_mean: int = 10,
    rename_to: str = 'long_name',
    apply_abs: bool = True,
    unit: str = 'absolute',
) -> xr.DataArray:  # type: ignore
    data_var = naming.format(variable=variable, running_mean=running_mean)
    plus_x_yr = data_set.isel({time_var: slice(x_yr, None)})[data_var]
    to_min_x_yr = data_set.isel({time_var: slice(None, -x_yr)})[data_var]

    # Keep the metadata (and time stamps of the to_min_x_yr)
    result = to_min_x_yr.copy(data=plus_x_yr.values - to_min_x_yr.values)
    result.data = np.abs(result.values)
    result = result.max(dim=time_var)
    var_unit = data_set[data_var].attrs.get('units', '{units}').replace('%', r'\%')
    name = data_set[data_var].attrs.get(rename_to, variable)

    if unit == 'absolute':
        result.name = f'{x_yr} yr diff. {name} [{var_unit}]'  # type: ignore
        return result  # type: ignore

    if unit == 'relative':
        result = 100 * result / to_min_x_yr.mean(dim=time_var)
        result.name = fr'{x_yr} yr diff. {name} [$\%$]'  # type: ignore
        return result  # type: ignore

    if unit == 'std':
        result = result / result.std()
        result.name = fr'{x_yr} yr diff. {name} [$\sigma$]'  # type: ignore
        return result  # type: ignore


@timed
@apply_abs()
@check_accepts(accepts=dict(unit=('absolute', 'relative', 'std')))
def max_derivative(
    data_set: xr.Dataset,
    variable: str,
    time_var: str = 'time',
    naming: str = '{variable}_run_mean_{running_mean}',
    running_mean: int = 10,
    rename_to: str = 'long_name',
    apply_abs: bool = True,
    unit: str = 'absolute',
) -> xr.Dataset:  # type: ignore
    var_name = naming.format(variable=variable, running_mean=running_mean)

    data_array = _remove_any_none_times(data_set[var_name], time_var)
    result = (
        np.abs(data_array.differentiate(time_var)).max(dim=time_var) * _SECONDS_TO_YEAR
    )

    var_unit = data_array.attrs.get('units', '{units}').replace('%', r'\%')
    name = data_array.attrs.get(rename_to, variable)

    if unit == 'absolute':
        result.name = fr'Max $\partial/\partial t$ {name} [{var_unit}/yr]'
        return result

    if unit == 'relative':
        result = 100 * result / data_array.mean(dim=time_var)
        result.name = fr'Max $\partial/\partial t$ {name} [$\%$/yr]'
        return result

    if unit == 'std':
        # A local unit of sigma might be better X.std(dim=time_var)
        result = result / data_array.std()
        result.name = fr'Max $\partial/\partial t$ {name} [$\sigma$/yr]'
        return result


rank2d = deprecated(rank2d, 'Call rank2d from optim_esm_tools.analyze.tools')
