# -*- coding: utf-8 -*-
import xarray as xr
import numpy as np
import os


def load_glob(
    pattern: str,
    **kw,
) -> xr.Dataset:
    """Load cmip dataset from provided pattern

    Args:
        pattern (str): Path where to load the data from

    Returns:
        xr.Dataset: loaded from pattern
    """
    if not os.path.exists(pattern):
        raise FileNotFoundError(f'{pattern} does not exists')
    for k, v in dict(
        use_cftime=True,
        concat_dim='time',
        combine='nested',
        data_vars='minimal',
        coords='minimal',
        compat='override',
        decode_times=True,
    ).items():
        kw.setdefault(k, v)
    return xr.open_mfdataset(pattern, **kw)


def _interp_nominal_lon_new(lon_1d):
    from optim_esm_tools.config import get_logger

    get_logger().debug('Using altered version _interp_nominal_lon_new')
    x = np.arange(len(lon_1d))
    idx = np.isnan(lon_1d)
    # TODO assume that longitudes are cyclic see https://github.com/jbusecke/xMIP/issues/299
    ret = np.interp(x, x[~idx], lon_1d[~idx], period=len(lon_1d))
    return ret


def recast(data_set):
    from xmip.preprocessing import (
        promote_empty_dims,
        replace_x_y_nominal_lat_lon,
        rename_cmip6,
        broadcast_lonlat,
    )

    ds = data_set.copy()
    # See https://github.com/jbusecke/xMIP/issues/299
    for k, v in {'longitude': 'lon', 'latitude': 'lat'}.items():
        if k in ds and v not in ds:
            ds = ds.rename({k: v})
    ds = rename_cmip6(ds)
    ds = promote_empty_dims(ds)
    ds = broadcast_lonlat(ds)
    import xmip.preprocessing

    xmip.preprocessing._interp_nominal_lon = _interp_nominal_lon_new
    ds = replace_x_y_nominal_lat_lon(ds)
    return ds
