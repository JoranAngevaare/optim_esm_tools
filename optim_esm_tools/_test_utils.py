import os

EXMPLE_DATA_SET = 'CMIP6/ScenarioMIP/CCCma/CanESM5/ssp585/r3i1p2f1/Amon/tas/gn/v20190429/tas_Amon_CanESM5_ssp585_r3i1p2f1_gn_201501-210012.nc'


def cmip_store():
    import intake

    return intake.open_esm_datastore(
        'https://storage.googleapis.com/cmip6/pangeo-cmip6.json',
    )  # type: ignore


def get_file_from_pangeo(experiment_id='ssp585', refresh=True):
    # sourcery skip: dict-assign-update-to-union
    dest_folder = os.path.split(
        get_example_data_loc().replace('ssp585', experiment_id),
    )[0]
    if experiment_id in ['piControl', 'historical']:
        dest_folder = dest_folder.replace('ScenarioMIP', 'CMIP')
    write_to = os.path.join(dest_folder, 'test.nc')
    if os.path.exists(write_to) and not refresh:
        print(f'already file at {write_to}')
        return write_to

    col = cmip_store()
    query = dict(
        source_id='CanESM5',
        variable_id='tas',
        table_id='Amon',
        experiment_id=experiment_id,
    )
    if experiment_id in ['historical', 'ssp585']:
        query.update(dict(member_id=['r3i1p2f1']))  # type: ignore
    else:
        query.update(dict(member_id=['r1i1p1f1']))  # type: ignore
    search = col.search(**query)  # type: ignore

    ddict = search.to_dataset_dict(
        xarray_open_kwargs={'use_cftime': True},
    )
    data = list(ddict.values())[0]

    data = data.mean(set(data.dims) - {'x', 'y', 'lat', 'lon', 'time'})
    if query['variable_id'] != 'tas':  # pragma: no cover
        raise ValueError(
            'Only tas for now as only areacella is hardcoded (see line below)',
        )

    data.attrs.update(
        dict(
            external_variables='areacella',
            variable_id='tas',
        ),
    )
    os.makedirs(dest_folder, exist_ok=True)
    assert data.attrs.get('variable_id')
    data.to_netcdf(write_to)
    return write_to


def year_means(path, refresh=True):
    new_dir = os.path.split(path.replace('Amon', 'AYear'))[0]
    new_dest = os.path.join(new_dir, 'test_merged.nc')
    if os.path.exists(new_dest) and not refresh:
        print(f'File at {new_dest} already exists')
        return new_dest
    import cftime
    import optim_esm_tools as oet

    data = oet.analyze.io.load_glob(path)

    data = data.groupby('time.year').mean('time')
    data = data.rename(year='time')
    data['time'] = [cftime.DatetimeNoLeap(y, 1, 1) for y in data['time']]
    data.attrs.update(dict(external_variables='areacella', variable_id='tas'))
    os.makedirs(new_dir, exist_ok=True)
    assert os.path.exists(new_dir)
    assert data.attrs.get('variable_id')
    data.to_netcdf(new_dest)
    return new_dest


def get_synda_loc():
    return os.path.join(
        os.environ.get('ST_HOME', os.path.join(os.path.abspath('.'), 'cmip')),
        'data',
    )


def get_example_data_loc():
    return os.path.join(get_synda_loc(), EXMPLE_DATA_SET)


def synda_test_available():
    """Check if we can run a synda-dependent test."""
    return os.environ.get('ST_HOME') is not None and os.path.exists(
        get_example_data_loc(),
    )


def get_path_for_ds(data_name, refresh=True):
    path = get_file_from_pangeo(data_name, refresh=refresh)
    year_path = year_means(path, refresh=refresh)
    assert year_path
    assert os.path.exists(year_path)
    return year_path


def complete_ds(start_year=2000, **kw):
    import cftime
    import optim_esm_tools as oet

    ds = oet._test_utils.minimal_xr_ds(**kw)
    ds['time'] = [cftime.datetime(y + start_year, 1, 1) for y in range(len(ds['time']))]
    ds['lat'].attrs.update(
        {
            'standard_name': 'latitude',
            'long_name': 'Latitude',
            'units': 'degrees_north',
            'axis': 'Y',
        },
    )
    ds['lon'].attrs.update(
        {
            'standard_name': 'longitude',
            'long_name': 'Longitude',
            'units': 'degrees_east',
            'axis': 'X',
        },
    )
    return ds


def minimal_xr_ds(len_x=513, len_y=181, len_time=10, add_nans=True):
    import numpy as np
    import xarray as xr

    lon = np.linspace(0, 360, len_x + 1)[:-1]
    lat = np.linspace(-90, 90, len_y + 1)[:-1]
    time = np.arange(len_time)
    # Totally arbitrary data
    data = (
        np.zeros(len(lat) * len(lon) * len(time)).reshape(len(time), len(lat), len(lon))
        * lon
    )

    # Add some NaN values just as an example
    if add_nans:
        data[:, :, len(lon) // 2 + 30 : len(lon) // 2 + 50] = np.nan

    return xr.Dataset(
        data_vars=dict(
            var=(
                ('time', 'lat', 'lon'),
                data,
            ),
        ),
        coords=dict(
            time=time,
            lat=lat,
            lon=lon,
        ),
        attrs=dict(source_id='bla', variable_id='var'),
    )
