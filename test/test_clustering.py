import xarray as xr
import numpy as np
import optim_esm_tools._test_utils
import optim_esm_tools.analyze.clustering as clustering


def test_clustering_empty():
    ds = optim_esm_tools._test_utils.minimal_xr_ds().copy()
    ds['var'] = (ds['var'].dims, np.zeros_like(ds['var']))
    ds = ds.isel(time=0)
    assert np.all(np.shape(ds['var']) > np.array([2, 2]))

    clusters, masks = clustering.build_cluster_mask(ds['var'] > 0, ds['x'], ds['y'])
    assert len(clusters) == len(masks) == 0


def test_clustering_double_blob(npoints=100, res_x=5, res_y=5):
    ds = optim_esm_tools._test_utils.minimal_xr_ds().copy()
    ds = ds.isel(time=0)

    arr = np.zeros_like(ds['var'])
    len_x, len_y = arr.shape[0:]
    x0, y0, x1, y1 = len_x // 4, len_y // 4, len_x // 2, len_y// 2

    for x, y in [x0, y0], [x1, y1]:
        for x_i, y_i in zip(
            np.clip(np.random.normal(x, res_x, npoints).astype(int), 0, len_x),
            np.clip(np.random.normal(y, res_y, npoints).astype(int), 0, len_y),
        ):
            arr[x_i][y_i] += 1

    assert np.sum(arr) == 2 * npoints
    ds['var'] = (ds['var'].dims, arr)
    assert np.all(np.shape(ds['var']) > np.array([2, 2]))

    clusters, masks = clustering.build_cluster_mask(
        ds['var'] > 1,
        ds['x'],
        ds['y'],
        max_distance_km=1000,
        cluster_opts=dict(min_samples=2),
    )
    assert len(clusters) == len(masks) == 2
