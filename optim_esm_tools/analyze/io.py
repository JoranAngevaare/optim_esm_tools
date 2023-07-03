# -*- coding: utf-8 -*-
import xarray as xr
import numpy as np
import os
from optim_esm_tools.utils import add_load_kw
from optim_esm_tools.analyze.calculate_metric import add_grid_area_field


@add_load_kw
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


def recast(data_set):
    from xmip.preprocessing import (
        promote_empty_dims,
        rename_cmip6,
        broadcast_lonlat,
        _drop_coords,
        correct_coordinates,
        correct_lon,
        correct_units,
        parse_lon_lat_bounds,
        sort_vertex_order,
        maybe_convert_bounds_to_vertex,
        maybe_convert_vertex_to_bounds,
        fix_metadata,
    )

    ds = data_set.copy()
    # See https://github.com/jbusecke/xMIP/issues/299
    for k, v in {'longitude': 'lon', 'latitude': 'lat'}.items():
        if k in ds and v not in ds:
            ds = ds.rename({k: v})
    # fix naming
    ds = rename_cmip6(ds)
    # promote empty dims to actual coordinates
    ds = promote_empty_dims(ds)
    # demote coordinates from data_variables
    ds = correct_coordinates(ds)
    # broadcast lon/lat
    ds = broadcast_lonlat(ds)
    # shift all lons to consistent 0-360
    ds = correct_lon(ds)
    # fix the units
    ds = correct_units(ds)
    # rename the `bounds` according to their style (bound or vertex)
    ds = parse_lon_lat_bounds(ds)
    # sort verticies in a consistent manner
    ds = sort_vertex_order(ds)
    # convert vertex into bounds and vice versa, so both are available
    ds = maybe_convert_bounds_to_vertex(ds)
    ds = maybe_convert_vertex_to_bounds(ds)

    # Not really sure if we need this, it raises key errors since we aren't xmip
    # ds = fix_metadata(ds)
    ds = ds.drop_vars(_drop_coords, errors='ignore')

    # Add grid field (associate km2 to each x,y value)
    add_grid_area_field(ds)
    return ds


# Keeping this arround for sake of the old times
# def _interp_nominal_lon_new(lon_1d):
#     from optim_esm_tools.config import get_logger
#     get_logger().debug('Using altered version _interp_nominal_lon_new')
#     x = np.arange(len(lon_1d))
#     idx = np.isnan(lon_1d)
#     # TODO assume that longitudes are cyclic see https://github.com/jbusecke/xMIP/issues/299
#     ret = np.interp(x, x[~idx], lon_1d[~idx], period=len(lon_1d))
#     return ret

# def recast(data_set):
#     from xmip.preprocessing import (
#         promote_empty_dims,
#         replace_x_y_nominal_lat_lon,
#         rename_cmip6,
#         broadcast_lonlat,
#     )

#     ds = data_set.copy()
#     # See https://github.com/jbusecke/xMIP/issues/299
#     for k, v in {'longitude': 'lon', 'latitude': 'lat'}.items():
#         if k in ds and v not in ds:
#             ds = ds.rename({k: v})
#     ds = rename_cmip6(ds)
#     ds = promote_empty_dims(ds)
#     ds = broadcast_lonlat(ds)
#     import xmip.preprocessing

#     xmip.preprocessing._interp_nominal_lon = _interp_nominal_lon_new
#     ds = replace_x_y_nominal_lat_lon(ds)
#     return ds
