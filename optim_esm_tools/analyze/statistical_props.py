import optim_esm_tools as oet
import numpy as np
import scipy
import xarray as xr
import typing as ty


def calculate_statistics(ds: xr.Dataset) -> ty.Dict[ty.Optional[float]]:
    """
    For a given dataset calculate the statistical properties of the dataset based on three tests:
        1. The standard deviation w.r.t. the standard

    Args:
        ds (xr.Dataset): _description_

    Returns:
        ty.Dict[ty.Optional[float]]: _description_
    """
    functions = dict(
        std_his=calculate_historical_std,
        symmetry=calculate_skewtest,
        dip=calculate_dip_test,
    )
    return {k: f(ds) for k, f in functions.items()}


def make_mask(ds):
    oet.config.get_logger().error('Temporary patch, replace mask in region finding!')

    data = oet.analyze.io.load_glob('grid.nc').copy()
    data['mask'] = (
        data['cell_area'].dims,
        np.zeros(data['cell_area'].shape, dtype=np.bool_),
    )
    mask_y = np.in1d(data['lat'], ds['lat'])
    mask_x = np.in1d(data['lon'], ds['lon'])

    a, b = np.meshgrid(mask_x, mask_y)

    mask_2d = a & b
    mask_2d[mask_2d] = (ds['cell_area'] > 0).values.flatten()
    data['mask'].data = mask_2d
    return data


def get_historical_ds(ds):
    try:
        hist_path = oet.analyze.find_matches.associate_historical(path=ds.attrs['path'])
    except RuntimeError as e:
        print(e)
        return
    hist_ds = oet.read_ds(hist_path[0], max_time=None, min_time=None)
    return hist_ds


def get_values_from_data_set(ds, field, add='_detrend'):
    if field is None:
        field = ds.attrs['variable_id'] + add
    da = ds[field]
    da = da.mean(set(da.dims) - {'time'})
    return da.values


def calculate_dip_test(ds, field=None):
    import diptest

    values = get_values_from_data_set(ds, field)

    _, pval = diptest.diptest(values, boot_pval=False)
    return pval


def calculate_skewtest(ds, field=None):
    values = get_values_from_data_set(ds, field, add='')
    return scipy.stats.skewtest(values, nan_policy='omit').pvalue


def calculate_historical_std(ds, field='std detrended'):
    ds_hist = get_historical_ds(ds)
    if ds_hist is None:
        return None
    mask = make_mask(ds)
    ds_hist_masked = oet.analyze.region_finding.mask_xr_ds(
        ds_hist, mask['mask'], drop=True
    )
    assert (
        ds[field].shape == ds_hist_masked[field].shape
    ), f'{ds[field].shape} != {ds_hist_masked[field].shape}'
    cur = ds[field].values
    his = ds_hist_masked[field].values
    isnnan = np.isnan(cur) | np.isnan(his)
    cur = cur[~isnnan]
    his = his[~isnnan]

    return np.median(cur / his)
