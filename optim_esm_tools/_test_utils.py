# -*- coding: utf-8 -*-
import os

EXMPLE_DATA_SET = 'CMIP6/ScenarioMIP/CCCma/CanESM5/ssp585/r3i1p2f1/Amon/tas/gn/v20190429/tas_Amon_CanESM5_ssp585_r3i1p2f1_gn_201501-210012.nc'


def get_file_from_pangeo():
    from xmip.utils import google_cmip_col

    # import cftime
    col = google_cmip_col()
    search = col.search(
        source_id='CanESM5',
        variable_id='tas',
        table_id='Amon',
        experiment_id='ssp585',
        member_id=['r3i1p2f1'],
    )

    ddict = search.to_dataset_dict(
        xarray_open_kwargs={'use_cftime': True},
    )
    data = list(ddict.values())[0]
    # data = data.groupby('time.year').mean('time')
    # data = data.rename(year='time')
    data = data.mean(set(data.dims) - {'x', 'y', 'time'})
    # data['time'] = [cftime.DatetimeNoLeap(y,1,1) for y in data['time']]

    write_to = get_example_data_loc()
    dest_folder = os.path.split(write_to)[0]
    os.makedirs(dest_folder, exist_ok=True)
    if os.path.exists(write_to):
        print(f'already file at {write_to}')
        write_to = os.path.join(dest_folder, 'test.nc')
    data.to_netcdf(write_to)


def get_synda_loc():
    return os.path.join(
        os.environ.get('ST_HOME', os.path.join(os.path.abspath('.'), 'cmip')), 'data'
    )


def get_example_data_loc():
    return os.path.join(get_synda_loc(), EXMPLE_DATA_SET)


def synda_test_available():
    """Check if we can run a synda-dependent test"""
    return os.environ.get('ST_HOME') is not None and os.path.exists(get_example_data_loc())


def minimal_xr_ds():
    import numpy as np
    import xarray as xr

    lon = np.linspace(0, 360, 513)[:-1]
    lat = np.linspace(-90, 90, 181)[:-1]
    time = np.arange(10)
    # Totally arbitrary data
    data = (
        np.zeros(len(lat) * len(lon) * len(time)).reshape(len(time), len(lat), len(lon))
        * lon
    )

    # Add some NaN values just as an example
    data[:, :, len(lon) // 2 + 30 : len(lon) // 2 + 50] = np.nan

    ds_dummy = xr.Dataset(
        data_vars=dict(
            var=(
                ('time', 'x', 'y'),
                data,
            )
        ),
        coords=dict(
            time=time,
            lon=lon,
            lat=lat,
        ),
        attrs=dict(source_id='bla'),
    )
    return ds_dummy
