import typing as ty
from functools import wraps

import numpy as np
import xarray as xr
import numba
import cftime
from optim_esm_tools.config import config
from optim_esm_tools.utils import check_accepts


def _native_date_fmt(time_array: np.ndarray, date: ty.Tuple[int, int, int]):
    """Create date object using the date formatting from the time-array."""

    if isinstance(time_array, xr.DataArray):  # pragma: no cover
        return _native_date_fmt(time_array=time_array.values, date=date)

    if not len(time_array):  # pragma: no cover
        raise ValueError('No values in dataset?')

    # Support cftime.DatetimeJulian, cftime.DatetimeGregorian, cftime.DatetimeNoLeap and similar
    _time_class = time_array[0].__class__
    return _time_class(*date)


def apply_abs(apply=True, add_abs_to_name=True, _disable_kw='apply_abs') -> ty.Callable:
    """Apply np.max() to output of function (if apply=True) Disable in the
    function kwargs by using the _disable_kw argument.

    Example:
        ```
        @apply_abs(apply=True, add_abs_to_name=False)
        def bla(a=1, **kw):
            print(a, kw)
            return a
        assert bla(-1, apply_abs=True) == 1
        assert bla(-1, apply_abs=False) == -1
        assert bla(1) == 1
        assert bla(1, apply_abs=False) == 1
        ```
    Args:
        apply (bool, optional): apply np.abs. Defaults to True.
        _disable_kw (str, optional): disable with this kw in the function. Defaults to 'apply_abs'.
    """

    def somedec_outer(fn):
        @wraps(fn)
        def somedec_inner(*args, **kwargs):
            response = fn(*args, **kwargs)
            do_abs = kwargs.get(_disable_kw)
            if do_abs or (do_abs is None and apply):
                if add_abs_to_name and isinstance(getattr(response, 'name'), str):
                    response.name = f'Abs. {response.name}'
                return np.abs(response)
            return response  # pragma: no cover

        return somedec_inner

    return somedec_outer


def _remove_any_none_times(da: xr.DataArray, time_dim: bool, drop: bool = True) -> None:
    data_var = da.copy()
    time_null = data_var.isnull().all(dim=set(data_var.dims) - {time_dim})
    if np.all(time_null):
        # If we take a running mean of 10 (the default), and the array is shorter than
        # 10 years we will run into issues here because a the window is longer than the
        # array. Perhaps we should raise higher up.
        raise ValueError(
            f'This array only has NaN values, perhaps array too short ({len(time_null)} < 10)?',
        )  # pragma: no cover

    if np.any(time_null):
        try:
            # For some reason only alt_calc seems to work even if it should be equivalent to the data_var
            # I think there is some fishy indexing going on in pandas <-> dask
            # Maybe worth raising an issue?
            alt_calc = xr.where(~time_null, da, np.nan)
            if drop:
                alt_calc = alt_calc.dropna(time_dim)
            data_var = data_var.load().where(~time_null, drop=drop)
            assert np.all((alt_calc == data_var).values)
        except IndexError as e:  # pragma: no cover
            from optim_esm_tools.config import get_logger

            get_logger().error(e)
            if 'alt_calc' in locals():
                return alt_calc  # type: ignore
            raise e
    return data_var


@check_accepts(accepts=dict(drop_method=('xarray', 'numba')))
def mask_xr_ds(
    data_set: xr.Dataset,
    da_mask: xr.DataArray,
    masked_dims: ty.Optional[ty.Iterable[str]] = None,
    drop: bool = False,
    keep_keys: ty.Optional[ty.Iterable[str]] = None,
    drop_method: ty.Optional[str] = None,
):
    # Modify the ds in place - make a copy!
    data_set = data_set.copy()
    if masked_dims is None:
        masked_dims = config['analyze']['lon_lat_dim'].split(',')[::-1]

    ds_start = data_set.copy()
    drop_true_function: ty.Callable = (
        _drop_by_mask
        if (
            drop_method == 'xarray'
            or (drop_method is None and config['analyze']['use_drop_nb'] != 'True')
        )
        else _drop_by_mask_nb
    )

    func_by_drop = {
        True: drop_true_function,
        False: _mask_xr_ds,
    }[drop]
    data_set = func_by_drop(
        data_set,
        masked_dims,
        ds_start,
        da_mask,
        keep_keys=keep_keys,
    )
    data_set = data_set.assign_attrs(ds_start.attrs)
    return data_set


def reverse_name_mask_coords(
    da_mask: xr.DataArray,
    rename_dict: ty.Optional[dict] = None,
) -> xr.DataArray:
    rename_dict = rename_dict or {
        v: k for k, v in default_rename_mask_dims_dict().items()
    }
    return rename_mask_coords(da_mask, rename_dict=rename_dict)


def rename_mask_coords(
    da_mask: xr.DataArray,
    rename_dict: ty.Optional[ty.Mapping] = None,
) -> xr.DataArray:
    """Get a boolean DataArray with renamed dimensionality. For some
    applications, we want to prune a dataset of nan values along a given
    lon/lat mask. Removing data along a given mask can greatly reduce file-size
    and speed up data-set handling. This however makes it somewhat cumbersome
    to later re-apply said mask (to other data) since it's shape will be
    inconsistent with other (non-masked) data. To this end, we want to store
    the mask separately in a dataset. To avoid dimension clashes between masked
    data and the masked information, we rename the dimensions of the mask.

    Args:
        da_mask (xr.DataArray): Mask to be renamed.
        rename_dict (ty.Mapping, optional): Mapping from the dims in da_mask to renamed dims.

    Returns:
        xr.DataArray: da_mask with renamed dims.
    """
    rename_dict = rename_dict or default_rename_mask_dims_dict()
    if any(dim not in da_mask.dims for dim in rename_dict.keys()):
        raise KeyError(
            f'Trying to rename {rename_dict}, but this DataArray has {da_mask.dims}',
        )  # pragma: no cover
    mask = da_mask.copy().rename(rename_dict)
    message = (
        'Full global mask with full lat/lon dimensionality in order to be save the masked '
        'time series with all nan values dropped (to conserve disk space)'
    )
    mask.attrs.update(dict(info=message))
    return mask


def mask_to_reduced_dataset(
    data_set: xr.Dataset,
    mask: ty.Union[xr.DataArray, np.ndarray],
    add_global_mask: bool = True,
    _fall_back_field: str = 'cell_area',
    **kw,
) -> xr.Dataset:
    """Reduce data_set by dropping all data where mask is False. This greatly
    reduces the size (which is absolutely required for exporting time series
    from global data).

    Args:
        data_set (xr.Dataset): data set to mask by mask
        mask (ty.Union[xr.DataArray, np.ndarray]): boolean array to mask
        add_global_mask (bool, optional): Add global mask with full dimensionality (see
            rename_mask_coords for more info). Defaults to True.

    Raises:
        ValueError: If mask has a wrong shape

    Returns:
        xr.Dataset: Original dataset where mask is True
    """
    if isinstance(mask, np.ndarray):
        mask_da = data_set[_fall_back_field].astype(np.bool_).copy()
        mask_da.data = mask
        mask = mask_da
    if mask.shape != (expected := data_set[_fall_back_field].shape):
        raise ValueError(
            f'Inconsistent dimensionality, expected {expected}, got {mask.shape}',
        )  # pragma: no cover

    if all(m in list(mask.coords) for m in default_rename_mask_dims_dict().values()):
        from optim_esm_tools.config import get_logger

        get_logger().debug(
            f'Reversing coords {list(mask.coords)} != {list(data_set[_fall_back_field].coords)}',
        )
        mask = reverse_name_mask_coords(mask)
    ds_masked = mask_xr_ds(data_set.copy(), mask, drop=True, **kw)
    if add_global_mask:
        ds_masked = add_mask_renamed(ds_masked, mask)
    return ds_masked


def default_rename_mask_dims_dict() -> ty.Dict:
    return {k: f'{k}_mask' for k in config['analyze']['lon_lat_dim'].split(',')}


def add_mask_renamed(
    data_set: xr.Dataset,
    da_mask: xr.DataArray,
    mask_name: str = 'global_mask',
    **kw,
) -> xr.Dataset:
    data_set[mask_name] = rename_mask_coords(da_mask, **kw)
    return data_set


def _drop_by_mask(
    data_set: xr.Dataset,
    masked_dims: ty.Iterable[str],
    ds_start: xr.Dataset,
    da_mask: xr.DataArray,
    keep_keys: ty.Optional[ty.Iterable[str]] = None,
):
    """Drop values with masked_dims dimensions.

    Unfortunately, data_set.where(da_mask, drop=True) sometimes leads to
    bad results, for example for time_bnds (time, bnds) being dropped by
    (lon, lat). So we have to do some funny bookkeeping of which data
    vars we can drop with data_set.where.
    """
    if keep_keys is None:
        keep_keys = list(data_set.variables.keys())
    dropped = [
        k
        for k, data_array in data_set.data_vars.items()
        if any(dim not in list(data_array.dims) for dim in masked_dims)
        or k not in keep_keys
    ]
    data_set = data_set.drop_vars(dropped)

    try:
        data_set = data_set.where(da_mask.compute(), drop=True)
    except ValueError:
        from optim_esm_tools.config import get_logger

        get_logger().info(f'data_set {list(data_set.coords)}')
        get_logger().info(f'da_mask {list(da_mask.coords)}')
        raise

    # Restore ignored variables and attributes
    for k in dropped:  # pragma: no cover
        if k not in keep_keys:
            continue
        data_set[k] = ds_start[k]
    return data_set


def _mask_xr_ds(
    data_set: xr.Dataset,
    masked_dims: ty.Iterable[str],
    ds_start: xr.Dataset,
    da_mask: xr.DataArray,
    keep_keys: ty.Optional[ty.Iterable[str]] = None,
):
    """Rebuild data_set for each variable that has all masked_dims."""
    for k, data_array in data_set.data_vars.items():
        if keep_keys is not None and k not in keep_keys:
            continue
        if all(dim in list(data_array.dims) for dim in masked_dims):
            lat_lon = config['analyze']['lon_lat_dim'].split(',')[::-1]
            dim_incorrect = tuple(data_array.dims) not in [
                ('time', *lat_lon),
                (*lat_lon,),
            ]
            shape_incorrect = data_array.shape != data_array.T.shape
            if dim_incorrect and shape_incorrect:  # pragma: no cover
                message = f'Please make "{k}" {lat_lon}, now "{data_array.dims}"'
                raise ValueError(message)
            da = data_set[k].where(da_mask, drop=False)
            da = da.assign_attrs(ds_start[k].attrs)
            data_set[k] = da

    return data_set


def _prepare_dropped_dataset(
    data_set: xr.Dataset,
    fall_back_key: str,
    masked_dims: ty.Iterable[str],
    da_mask: xr.DataArray,
    keep_keys: ty.Optional[ty.Iterable[str]] = None,
) -> ty.Tuple[xr.Dataset, ty.List[str], ty.List[str]]:
    assert fall_back_key in data_set
    if keep_keys is None:
        keep_keys = list(data_set.variables.keys())
    else:
        if not all(isinstance(k, str) for k in keep_keys):
            raise TypeError(f'Got one or more non-string keys {keep_keys}')
    dropped = [
        str(k)
        for k, data_array in data_set.data_vars.items()
        if (
            any(dim not in list(data_array.dims) for dim in masked_dims)
            or k not in keep_keys
        )  # and k not in no_drop
    ]

    data_set = data_set.drop_vars(
        (set(dropped) | set(keep_keys)) - {*masked_dims, fall_back_key},
    )

    data_set = data_set.where(da_mask.compute(), drop=True)

    assert fall_back_key in data_set
    return data_set, keep_keys, dropped


def _drop_by_mask_nb(
    data_set: xr.Dataset,
    masked_dims: ty.Iterable[str],
    ds_start: xr.Dataset,
    da_mask: xr.DataArray,
    keep_keys: ty.Optional[ty.Iterable[str]] = None,
    fall_back_key='cell_area',
) -> xr.Dataset:
    """Drop values with masked_dims dimensions.

    Unfortunately, data_set.where(da_mask, drop=True) sometimes leads to
    bad results, for example for time_bnds (time, bnds) being dropped by
    (lon, lat). So we have to do some funny bookkeeping of which data
    vars we can drop with data_set.where.
    """
    data_set, keep_keys, dropped = _prepare_dropped_dataset(
        data_set=data_set,
        fall_back_key=fall_back_key,
        masked_dims=masked_dims,
        da_mask=da_mask,
        keep_keys=keep_keys,
    )
    x_map = map_array_to_index_array(ds_start.lat.values, data_set.lat.values)
    y_map = map_array_to_index_array(ds_start.lon.values, data_set.lon.values)
    mask_np = da_mask.values
    res_shape = data_set[fall_back_key].shape
    from optim_esm_tools.config import get_logger

    log = get_logger()

    for mask_key in keep_keys:
        if mask_key == fall_back_key:
            continue

        da = ds_start[mask_key]
        if da.dims == tuple(masked_dims):
            v = mapped_2d_mask(da.values, res_shape, mask_np, x_map, y_map)

            try:
                data_set[mask_key] = xr.DataArray(data=v, dims=da.dims, attrs=da.attrs)
            except ValueError:
                return dict(data_set=data_set, data=v, dims=da.dims, attrs=da.attrs)

        elif da.dims[1:] == tuple(masked_dims):
            v = mapped_3d_mask(da.values, res_shape, mask_np, x_map, y_map)
            data_set[mask_key] = xr.DataArray(data=v, dims=da.dims, attrs=da.attrs)
        elif mask_key not in masked_dims:
            log.debug(
                f'Skipping "{mask_key}" in masking as it\'s not following the {masked_dims} but has {da.dims}',
            )
            if any(m in da.dims for m in masked_dims):
                data_set[mask_key] = ds_start[mask_key].where(da_mask, drop=True)
            else:
                data_set[mask_key] = ds_start[mask_key]

    # Restore ignored variables and attributes
    for k in dropped:  # pragma: no cover
        if k not in keep_keys:
            continue
        data_set[k] = ds_start[k]
    return data_set


@numba.njit
def map_array_to_index_array(x0, x1, nan_int=-1234):
    """Map the indexes from x0 to x1, if there is no index in x1 that
    corresponds to a value of x0, fill it with the nan_int value.

    Example
    ``
    >>> arg_map_1d(np.array([1,2,3]), np.array([1,3]), nan_int=-1234)
    array([    0, -1234,     1])
    ``
    """
    res = np.ones(len(x0), dtype=np.int16) * nan_int
    for i0, x0_i in enumerate(x0):
        for i1, x1_i in enumerate(x1):
            if not np.isclose(x0_i, x1_i):
                continue
            res[i0] = i1
            break
    return res


@numba.njit
def mapped_2d_mask(
    source_values,
    result_shape,
    mask_2d,
    index_map_x,
    index_map_y,
    dtype=np.float64,
    nan_int=-1234,
):
    x, y = source_values.shape
    res = np.zeros(result_shape, dtype) * np.nan

    for i in range(x):
        x_fill = index_map_x[i]
        if x_fill == nan_int:
            continue
        for j in range(y):
            y_fill = index_map_y[j]
            if y_fill == nan_int:
                continue
            if not mask_2d[i][j]:
                continue
            res[x_fill][y_fill] = source_values[i][j]

    return res


@numba.njit
def mapped_3d_mask(
    source_values,
    result_shape,
    mask_2d,
    index_map_x,
    index_map_y,
    dtype=np.float64,
    nan_int=-1234,
):
    assert len(source_values.shape) == 3
    res = (
        np.zeros((len(source_values), result_shape[0], result_shape[1]), dtype) * np.nan
    )
    for ti in range(source_values.shape[0]):
        res[ti] = mapped_2d_mask(
            source_values[ti],
            result_shape=result_shape,
            mask_2d=mask_2d,
            index_map_x=index_map_x,
            index_map_y=index_map_y,
            dtype=dtype,
            nan_int=nan_int,
        )
    return res


def yearly_average(ds: xr.Dataset, time_dim='time') -> xr.Dataset:
    """Compute yearly averages for all variables in the dataset along the time
    dimension, handling both datetime and cftime objects."""

    def compute_weighted_mean(data, time):
        """Helper function to compute weighted mean for a given array of
        data."""
        if time_bounds is not None:
            dt = np.diff(ds[time_bounds].values, axis=1).squeeze()
        else:
            if isinstance(time[0], cftime.datetime):
                dt = np.array(
                    [(time[i + 1] - time[i]).days for i in range(len(time) - 1)]
                    + [(time[-1] - time[-2]).days],
                )
            else:
                # poor man solution, let's just assume that the last time-interval is as long as the second to last interval
                dt = np.diff(time)
                dt = np.concatenate([dt, [dt[-1]]])
        if len(time) != len(dt):
            raise ValueError(f'Inconsistent time lengths {len(time)} != {len(dt)}')
        years = [d.year for d in data[time_dim].values]
        if isinstance(time[0], cftime.datetime):
            keep_idx = np.array([t.year in years for t in time])
        elif isinstance(time, xr.DataArray) and isinstance(
            time.values[0],
            cftime.datetime,
        ):
            keep_idx = np.array([t.year in years for t in time.values])
        else:
            raise TypeError(type(time))

        dt = dt[keep_idx]
        dt_seconds = dt * 86400  # Convert days to seconds if cftime
        weights = dt_seconds / dt_seconds.sum()

        # Apply weighted mean over time axis
        weighted_mean = (data * weights[:, None, None]).sum(axis=0)
        return weighted_mean

    # Handle time bounds if present
    time_bounds = next(
        (k for k in [f'{time_dim}_bounds', f'{time_dim}_bnds'] if k in ds),
        None,
    )

    # Initialize a new dataset to hold the yearly averages
    ds_yearly = xr.Dataset()

    # Loop through the variables in the dataset
    for var in ds.data_vars:
        if time_dim in ds[var].dims:
            dtype = ds[var].dtype

            # Skip non-numeric data types
            if not np.issubdtype(dtype, np.number):
                print(f'Skipping {var} of dtype={dtype}')
                continue

            grouped = ds[var].groupby('time.year')
            yearly_mean = grouped.map(lambda x: compute_weighted_mean(x, ds[time_dim]))

            ds_yearly[var] = yearly_mean.astype(ds[var].dtype)

    return ds_yearly


def set_time_int(ds: xr.Dataset) -> xr.Dataset:
    if not isinstance(ds['time'].values[0], (int, np.int_)):
        years = [t.year for t in ds['time'].values]
        assert (
            np.unique(years, return_counts=True)[1].max() == 1
        ), 'Data has one or more non-unique years!'
        ds['time'] = years
    return ds
