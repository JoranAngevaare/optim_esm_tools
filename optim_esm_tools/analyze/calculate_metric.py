"""
Basic module to calculate the size of grid cells. If a more robust method exists, I'd be happy to implement it here.
"""
import numba
import numpy as np
from .clustering import _distance_bf_coord
import xarray as xr
import typing as ty


def add_grid_area_field(data_set: xr.Dataset, **kw) -> None:
    """Add cell_area field to dataset"""
    data_set['cell_area'] = xr.DataArray(
        calucluate_grid(data_set, **kw).T,
        dims=('y', 'x'),
        name='Cell area km$^2$',
        attrs=dict(units='km$^2$'),
    )


def calucluate_grid(
    data_set: ty.Union[xr.Dataset, xr.DataArray],
    _do_numba: bool = True,
    x_label: str = 'x',
    y_label: str = 'y',
    _area_off_percent_threshold: float = 5,
) -> np.ndarray:
    """Calculate the area of each x,y coordinate in the dataset

    Args:
        data_set (ty.Union[xr.Dataset, xr.DataArray]): dataset to calculate grid metric of
        _do_numba (bool, optional): use fast numba calculation. Defaults to True.
        x_label (str, optional): label of x coord. Defaults to 'x'.
        y_label (str, optional): label of y coord. Defaults to 'y'.

    Raises:
        ValueError: If the total area differs siginificantly from the true area of the globe, raise an error

    Returns:
        np.ndarray: _description_
    """
    lon, lat = np.meshgrid(data_set[y_label], data_set[x_label])
    area = np.zeros_like(lon)
    if _do_numba:
        _n_calulate_mesh(lon, lat, area)
    else:
        _calulate_mesh(lon, lat, area)
    if np.abs(1 - area.sum() / 509600000) > _area_off_percent_threshold / 100:
        raise ValueError(
            f'This estimation leads to an area of {area.sum()} kmn2 which is off by at least {_area_off_percent_threshold}\% of the true value'
        )

    return area


@numba.njit
def clip(val, low, high):
    """Simple np.clip like numba function"""
    if high < low:
        raise ValueError
    return min(max(val, low), high)


def _calulate_mesh(lon: np.ndarray, lat: np.ndarray, area: np.ndarray) -> np.ndarray:
    """Calculate the area of each cell based on distance to adjacent cells.
    Averages the distance to left, right up and down:
    If we want to approximate the size of cell "x", we approximate


    . . U . . . .         . . - . . . .
    . L x R . . .  =>     . |-x-| . . .
    . . D . . . .         . . - . . . .
    area_x = (R-L) / 2 * (U-D) / 2

    For any boundary conditions we replace R, L, U or D with x and change
    the normalization appropriately

    Args:
        lon (np.ndarray): mesh of lon values
        lat (np.ndarray): mesh of lat values
        area (np.ndarray): area buffer to fill with values [in km2]!
    """
    len_y, len_x = lon.shape

    for i in range(len_y):
        for j in range(len_x):
            h_multiply = 1 if (i == 0 or i == len_y - 1) else 2
            w_multiply = 1 if (j == 0 or j == len_x - 1) else 2

            coord_up = (
                lon[i][clip(j + 1, 0, len_x - 1)],
                lat[i][clip(j + 1, 0, len_x - 1)],
            )
            coord_down = (
                lon[i][clip(j - 1, 0, len_x - 1)],
                lat[i][clip(j - 1, 0, len_x - 1)],
            )
            coord_left = (
                lon[clip(i + 1, 0, len_y - 1)][j],
                lat[clip(i + 1, 0, len_y - 1)][j],
            )
            coord_right = (
                lon[clip(i - 1, 0, len_y - 1)][j],
                lat[clip(i - 1, 0, len_y - 1)][j],
            )

            h_km = _distance_bf_coord(*coord_up, *coord_down)
            w_km = _distance_bf_coord(*coord_left, *coord_right)

            area[i][j] = h_km * w_km / (h_multiply * w_multiply)


_n_calulate_mesh = numba.njit(_calulate_mesh)
