import optim_esm_tools as oet
import numpy as np
import xarray as xr


def test_calculate_norm(n_time=12):
    ds = oet._test_utils.complete_ds(
        len_x=180,
        len_y=90,
        len_time=n_time,
        add_nans=True,
    )

    data = np.random.random_sample(ds['var'].shape)
    # add some regions with temporal changes
    data[:, 10:50, 40:80] += np.linspace(0, 2, n_time).reshape(n_time, 1, 1)
    data[:, 80:85, 40:80] += np.linspace(3, 0, n_time).reshape(n_time, 1, 1)
    data[:, 80:85, 150:160] += np.linspace(0, 4, n_time).reshape(n_time, 1, 1)
    ds['var'].data += data

    oet.analyze.inferred_variable_field.inferred_fields_to_dataset(ds)
    ds['cell_area'] = xr.ones_like(ds['var'].isel(time=0))
    a_max = float(ds['cell_area'].sum())
    ds['var_run_mean_10'].load()
    for a_target in np.random.sample(10) * a_max:
        max_val = oet.analyze.region_calculation.find_max_in_equal_area(
            ds,
            a_target,
        )['max_in_sel']
        assert max_val > 0, (a_target, max_val)
    assert (
        oet.analyze.region_calculation.calculate_norm(
            ds,
            ds['var'].max('time') > 3,
            'var',
        )
        > 0
    )
