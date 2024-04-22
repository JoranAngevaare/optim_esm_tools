import unittest

import hypothesis
import numpy as np

import optim_esm_tools._test_utils
import optim_esm_tools.analyze.clustering as clustering
from optim_esm_tools.config import config
from optim_esm_tools.utils import timed


def test_clustering_empty():
    ds = optim_esm_tools._test_utils.minimal_xr_ds().copy()
    ds['var'] = (ds['var'].dims, np.zeros_like(ds['var']))
    ds = ds.isel(time=0)
    assert np.all(np.shape(ds['var']) > np.array([2, 2]))

    clusters, masks = clustering.build_cluster_mask(
        (ds['var'] > 0).values,
        ds['lat'].values,
        ds['lon'].values,
    )
    assert len(clusters) == len(masks) == 0


def test_clustering_mesh():
    return test_clustering_double_blob(use_mesh=True)


def test_clustering_double_blob(npoints=100, res_x=3, res_y=3, use_mesh=False):
    ds = optim_esm_tools._test_utils.minimal_xr_ds().copy()
    ds = ds.isel(time=0)

    arr = np.zeros_like(ds['var'])
    len_lat, len_lon = arr.shape
    y0, x0, y1, x1 = len_lat // 4, len_lon // 4, len_lat // 2, len_lon - len_lon // 4

    for x, y in [x0, y0], [x1, y1]:
        for x_i, y_i in zip(
            np.clip(np.random.normal(x, res_x, npoints).astype(int), 0, len_lat),
            np.clip(np.random.normal(y, res_y, npoints).astype(int), 0, len_lon),
        ):
            arr[y_i][x_i] += 1

    assert np.sum(arr) == 2 * npoints
    ds['var'] = (ds['var'].dims, arr)

    assert np.all(np.shape(ds['var']) > np.array([2, 2]))
    lat, lon = np.meshgrid(ds['lat'], ds['lon'])
    if use_mesh:
        lat, lon = np.meshgrid(ds['lat'], ds['lon'])
        clusters, masks = clustering.build_cluster_mask(
            ds['var'] > 1,
            lat.T,
            lon.T,
            max_distance_km=1000,
            min_samples=2,
        )
    else:
        clusters, masks = clustering.build_cluster_mask(
            ds['var'] > 1,
            ds['lat'].values,
            ds['lon'].values,
            max_distance_km=1000,
            min_samples=2,
        )
    assert len(clusters) == len(masks)
    assert len(clusters) == 2


def test_geopy_alternative():
    xs, ys = np.random.rand(1, 4).reshape(2, 2)
    xs *= 360
    ys = ys * 180 - 90
    # LAT:LON!
    coords = np.array([ys, xs]).T
    flat_coord = coords.flatten()
    print(coords, flat_coord)
    assert np.isclose(
        clustering._distance_bf_coord(*flat_coord),
        clustering._distance(coords),
        rtol=0.1,
    )
    assert np.isclose(
        clustering._distance_bf_coord(*flat_coord),
        clustering._distance(coords, force_math=True),
        rtol=0.1,
    )


def test_infer_step_size():
    ds = optim_esm_tools._test_utils.minimal_xr_ds().copy()
    res_0 = clustering.infer_max_step_size(ds.lat.values, ds.lon.values)
    assert np.isclose(res_0, 148.92)


class TestClustering(unittest.TestCase):
    _max_lat = 100
    _max_lon = 400

    @hypothesis.settings(max_examples=10, deadline=None)
    @hypothesis.given(
        hypothesis.strategies.integers(min_value=4, max_value=_max_lon),
        hypothesis.strategies.integers(min_value=4, max_value=_max_lat),
        hypothesis.strategies.floats(min_value=-90, max_value=90, exclude_max=True),
        hypothesis.strategies.floats(min_value=0, max_value=180),
        hypothesis.strategies.integers(min_value=0, max_value=_max_lon),
        hypothesis.strategies.integers(min_value=0, max_value=_max_lat),
        hypothesis.strategies.integers(
            min_value=1,
            max_value=int(config['analyze']['clustering_min_neighbors']),
        ),
    )
    @timed(seconds=1, _report='print', _args_max=300)
    def test_rand_cluster(
        self,
        len_lon,
        len_lat,
        coord_lat,
        coord_lon,
        lat_width,
        lon_width,
        min_samples_cluster,
    ):
        ds = optim_esm_tools._test_utils.minimal_xr_ds(
            len_x=len_lon,
            len_y=len_lat,
            len_time=2,
            add_nans=False,
        )
        mask = ds['var'].isel(time=0).values.copy()
        mask[:] = 0
        lat_idx = np.argmin(np.abs(ds['lat'].values - coord_lat))
        lon_idx = np.argmin(np.abs(ds['lon'].values - coord_lon))
        mask[slice(lat_idx - lat_width, lat_idx + lat_width + 1)] += 1
        lon_min = lon_idx - lon_width
        lon_max = lon_idx + lon_width + 1
        if lon_min < 0:
            mask[:, slice(360 + lon_min, 361)] += 1
            lon_min = 0
        if lon_max > len_lon:
            mask[:, slice(0, lon_max - len_lon)] += 1
            lon_max = len_lon
        mask[:, slice(lon_min, lon_max)] += 1

        candidates = np.sum(mask >= 2).sum()
        min_samples_cluster = np.clip(min_samples_cluster, 1, candidates - 1)
        clusters, masks = clustering.build_cluster_mask(
            (mask >= 2),
            ds.lat.values,
            ds.lon.values,
            min_samples=min_samples_cluster,
        )
        if lat_width <= 1 or lon_width <= 1:
            pass
        elif candidates > 1 and candidates > min_samples_cluster:
            assert len(masks)
        print('done')
