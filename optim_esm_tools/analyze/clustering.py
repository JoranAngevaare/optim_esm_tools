import contextlib
import typing as ty
from math import atan2
from math import cos
from math import radians
from math import sin
from math import sqrt

import numba
import numpy as np
import xarray as xr

from optim_esm_tools.config import config
from optim_esm_tools.config import get_logger
from optim_esm_tools.utils import timed, deprecated
from optim_esm_tools.utils import tqdm


@timed()
def build_clusters(
    coordinates_deg: np.ndarray,
    weights: ty.Optional[np.ndarray] = None,
    max_distance_km: ty.Union[float, int] = 750,
    only_core: bool = True,
    min_samples: int = int(config['analyze']['clustering_min_neighbors']),
    cluster_opts: ty.Optional[dict] = None,
    keep_masks: bool = False,
) -> ty.Union[ty.List[np.ndarray], ty.Tuple[ty.List[np.ndarray], ty.List[np.ndarray]]]:
    """Build clusters based on a list of coordinates, use halfsine metric for
    spherical spatial data.

    Args:
        coordinates_deg (np.ndarray): set of xy coordinates in degrees
        weights (ty.Optional[np.ndarray], optional): weights (in the range [0,1]) corresponding to each coordinate
        max_distance_km (ty.Union[float, int], optional): max distance to other points to consider part of
            cluster (see DBSCAN(eps=<..>)). Defaults to 750.
        only_core (bool, optional): Use only core samples. Defaults to True.
        min_samples (int): Minimum number of samples in cluster. Defaults to 8.
        cluster_opts (ty.Optional[dict], optional): Additional options passed to sklearn.cluster.DBSCAN. Defaults to None.
        keep_masks (bool): return a tuple with both the clusters (coords) and masks (2d boolean arrays)

    Returns:
        ty.List[np.ndarray]: list of clustered points (in radians)
        or
        ty.Tuple[ty.List[np.ndarray], ty.List[np.ndarray]]]: list of clustered points (in radians) and list
            of boolean masks with the same length as the input coordinates deg.
    """
    cluster_coords, cluster_masks = _build_clusters(
        coordinates_deg,
        weights,
        max_distance_km,
        only_core,
        min_samples,
        cluster_opts,
    )
    if keep_masks:
        return cluster_coords, cluster_masks
    return cluster_coords


def _build_clusters(
    coordinates_deg: np.ndarray,
    weights: ty.Optional[np.ndarray] = None,
    max_distance_km: ty.Union[float, int] = 750,
    only_core: bool = True,
    min_samples: int = int(config['analyze']['clustering_min_neighbors']),
    cluster_opts: ty.Optional[dict] = None,
) -> ty.Tuple[ty.List[np.ndarray], ty.List[np.ndarray]]:
    cluster_opts = cluster_opts or {}
    for class_label, v in dict(algorithm='ball_tree', metric='haversine').items():
        cluster_opts.setdefault(class_label, v)
    cluster_opts['min_samples'] = min_samples

    from sklearn.cluster import DBSCAN

    coordinates_rad = np.radians(coordinates_deg).T

    # TODO use a more up to date version:
    #  https://scikit-learn.org/stable/auto_examples/cluster/plot_hdbscan.html#sphx-glr-auto-examples-cluster-plot-hdbscan-py
    #  https://scikit-learn.org/stable/modules/generated/sklearn.cluster.HDBSCAN.html#sklearn.cluster.HDBSCAN
    # Thanks https://stackoverflow.com/a/38731787/18280620!
    try:
        db_fit = DBSCAN(eps=max_distance_km / 6371.0, **cluster_opts).fit(
            X=coordinates_rad,
            sample_weight=weights,
        )
    except ValueError as e:  # pragma: no cover
        raise ValueError(
            f'With {coordinates_rad.shape} and {getattr(weights, "shape", None)} {coordinates_rad}, {weights}',
        ) from e

    labels = db_fit.labels_

    unique_labels = sorted(set(labels))
    is_core_sample = np.zeros_like(labels, dtype=bool)
    is_core_sample[db_fit.core_sample_indices_] = True

    return_masks = []
    return_coord = []
    for class_label in unique_labels:
        is_noise = class_label == -1
        if is_noise:
            continue

        is_class_member = labels == class_label
        coord_mask = is_class_member
        if only_core:
            coord_mask &= is_core_sample

        masked_points = coordinates_rad[coord_mask]
        return_coord.append(masked_points)
        return_masks.append(coord_mask)

    return return_coord, return_masks


@timed()
def build_cluster_mask(
    global_mask: np.ndarray,
    lat_coord: np.ndarray,
    lon_coord: np.ndarray,
    show_tqdm: ty.Optional[bool] = None,
    max_distance_km: ty.Union[str, float, int] = 'infer',
    **kw,
) -> ty.Tuple[ty.List[np.ndarray], ty.List[np.ndarray]]:
    """Build set of clusters and masks based on the global mask, basically a
    utility wrapper around build_clusters'.

    Args:
        global_mask (np.ndarray): full 2d mask of the data
        lon_coord (np.array): all longitude values
        lat_coord (np.array): all latitude values
        max_distance_km (ty.Union[str, float, int]): find an appropriate distance
            threshold for build_clusters' max_distance_km argument. If nothing is
            provided, make a guess based on the distance between grid cells.
            Defaults to 'infer'.

    Returns:
        ty.List[ty.List[np.ndarray], ty.List[np.ndarray]]: Return two lists, containing the masks, and clusters respectively.
    """
    if max_distance_km == 'infer':
        max_distance_km = infer_max_step_size(lat_coord, lon_coord)
    lat, lon = _check_input(
        global_mask,
        lat_coord,
        lon_coord,
    )
    xy_data = np.array([lat[global_mask], lon[global_mask]])

    if len(xy_data.T) <= 2:
        get_logger().info(f'No data from this mask {xy_data}!')
        return [], []

    masks, clusters = _build_cluster_with_kw(
        lat=lat,
        lon=lon,
        coordinates_deg=xy_data,
        max_distance_km=max_distance_km,
        global_mask=global_mask,
        show_tqdm=show_tqdm,
        **kw,
    )

    return masks, clusters


@timed()
def build_weighted_cluster(
    weights: np.ndarray,
    lat_coord: np.ndarray,
    lon_coord: np.ndarray,
    show_tqdm: ty.Optional[bool] = None,
    threshold: ty.Optional[float] = 0.99,
    max_distance_km: ty.Union[str, float, int] = 'infer',
    **kw,
) -> ty.Tuple[ty.List[np.ndarray], ty.List[np.ndarray]]:
    """Build set of clusters and masks based on the weights (which should be a
    grid)'.

    Args:
        weights (np.ndarray): normalized score data (values in [0,1])
        lon_coord (np.array): all longitude values
        lat_coord (np.array): all latitude values
        max_distance_km (ty.Union[str, float, int]): find an appropriate distance
            threshold for build_clusters' max_distance_km argument. If nothing is
            provided, make a guess based on the distance between grid cells.
            Defaults to 'infer'.
        threshold: float, min value of the passed weights. Defaults to 0.99.

    Returns:
        ty.List[ty.List[np.ndarray], ty.List[np.ndarray]]: Return two lists, containing the masks, and clusters respectively.
    """
    if max_distance_km == 'infer':
        max_distance_km = infer_max_step_size(lat_coord, lon_coord)

    lat, lon = _check_input(weights, lat_coord, lon_coord)
    xy_data = np.array([lat.flatten(), lon.flatten()])

    flat_weights = weights.flatten()
    mask = flat_weights > threshold
    global_mask = weights > threshold
    masks, clusters = _build_cluster_with_kw(
        lat=lat,
        lon=lon,
        coordinates_deg=xy_data[:, mask],
        weights=flat_weights[mask],
        show_tqdm=show_tqdm,
        max_distance_km=max_distance_km,
        global_mask=global_mask,
        **kw,
    )

    return masks, clusters


def _check_input(
    data: np.ndarray,
    lat_coord: np.ndarray,
    lon_coord: np.ndarray,
) -> ty.Tuple[np.ndarray, np.ndarray]:
    """Check for consistency and if we need to convert the lon/lat coordinates
    to a meshgrid."""
    if len(lon_coord.shape) <= 1:
        lon, lat = np.meshgrid(lon_coord, lat_coord)
    else:
        # In an older version, this would have been the default.
        lat, lon = lat_coord, lon_coord

    if data.shape != lon.shape or data.shape != lat.shape:  # pragma: no cover
        message = f'Wrong input {data.shape} != {lon.shape, lat.shape}'
        raise ValueError(message)
    return lat, lon


@deprecated
def _split_to_continous(*a, **kw):
    return _split_to_continuous(*a, **kw)


def _split_to_continuous(
    masks: ty.List,
    **kw,
) -> ty.List[np.ndarray]:
    no_group = -1
    mask_groups = masks_array_to_coninuous_sets(masks, no_group_value=no_group, **kw)
    continous_masks = []
    for grouped_members in mask_groups:
        for group_id in np.unique(grouped_members):
            if group_id == no_group:
                continue
            continous_masks.append(grouped_members == group_id)

    small_first = np.argsort([np.sum(c) for c in continous_masks])
    large_first = small_first[::-1]
    continous_masks = [
        group for i in large_first for k, group in enumerate(continous_masks) if i == k
    ]

    return continous_masks


def _find_lat_lon_values(
    mask_2d: np.ndarray,
    lats: np.ndarray,
    lons: np.ndarray,
) -> np.ndarray:
    lon_coords = lons[mask_2d]
    lat_coords = lats[mask_2d]
    return np.vstack([lat_coords, lon_coords]).T


def _build_cluster_with_kw(
    lat: np.ndarray,
    lon: np.ndarray,
    show_tqdm=None,
    global_mask=None,
    force_continuity: bool = False,
    **cluster_kw,
) -> ty.Tuple[ty.List[np.ndarray], ty.List[np.ndarray]]:
    """Overlapping logic between functions to get the masks and clusters.

    force_continuity (bool): split the masks until each is a continuous
    set
    """

    clusters, sub_masks = build_clusters(**cluster_kw, keep_masks=True)

    if global_mask is None:
        raise ValueError('global_mask is required')
    clusters = [np.rad2deg(cluster) for cluster in clusters]

    if lat.shape != lon.shape:
        raise ValueError(
            f'Got inconsistent input {lat.shape} != {lon.shape}',
        )  # pragma: no cover

    masks: ty.List[np.ndarray] = []
    for sub_mask in sub_masks:
        full_2d_mask = np.zeros_like(global_mask)
        full_2d_mask[global_mask] = sub_mask

        masks.append(np.array(full_2d_mask))

    if force_continuity:
        masks = _split_to_continuous(masks=masks)

        clusters = [_find_lat_lon_values(m, lats=lat, lons=lon) for m in masks]

    return masks, clusters


def infer_max_step_size(
    lat: np.ndarray,
    lon: np.ndarray,
    off_by_factor: ty.Optional[float] = None,
) -> float:
    """Infer the max. distance between two points to be considered as belonging
    to the same cluster.

    There are two methods implemented, preferably, the lon, lat values
    are 1d-arrays, which can be interpreted as a regular grid. If this
    is the case, calculate the distance for each point to it's neighbors
    (also diagonally). Then, the max distance for the clustering can be
    taken as the max. distance to any of the neighboring points.

    Empirically, we found that this distance is not enough, and an
    additional fudge factor is taken into account from version v1.0.3
    onwards, this is taken to be sqrt(2). This is probably not a
    coincidence, but it's not really clear where it's coming from.
    """
    if off_by_factor is None:
        off_by_factor = float(config['analyze']['clustering_fudge_factor'])
    assert len(lat.shape) == 1
    # Simple 1D array
    return off_by_factor * np.max(calculate_distance_map(lat, lon))


def calculate_distance_map(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """For each point in a spanned lat lon grid, calculate the distance to the
    neighboring points."""
    if isinstance(lat, xr.DataArray):  # pragma: no cover
        raise ValueError('Numpy array required')
    return _calculate_distance_map(lat, lon)


@numba.njit
def _calculate_distance_map(
    lat: np.ndarray,
    lon: np.ndarray,
) -> np.ndarray:  # sourcery skip: use-itertools-product
    n_lat = len(lat)
    n_lon = len(lon)
    distances = np.zeros((n_lat, n_lon))

    shift_by_index = np.array(
        [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (1, 1), (1, -1), (-1, 1)],
    )
    neighbors = np.zeros(len(shift_by_index), dtype=np.float64)
    for lon_i in range(n_lon):
        for lat_i in range(n_lat):
            neighbors[:] = 0
            current = (lat[lat_i], lon[lon_i])
            for i, (x, y) in enumerate(shift_by_index):
                alt_lon = np.mod(lon_i + x, n_lon)
                alt_lat = lat_i + y
                if alt_lat == n_lat or alt_lat < 0:
                    continue
                alt_coord = (lat[alt_lat], lon[alt_lon])
                if alt_coord == current:
                    raise ValueError('How can this happen?')  # pragma: no cover
                neighbors[i] = _distance_bf_coord(*current, *alt_coord)
            distances[lat_i][lon_i] = np.max(neighbors)
    return distances


def _distance(coords: np.ndarray, force_math: bool = False) -> float:
    """Wrapper for if geopy is not installed."""
    if not force_math:
        with contextlib.suppress(ImportError):
            from geopy.distance import geodesic

            return geodesic(*coords).km
    if len(coords) != 4:
        coords = np.array([c for cc in coords for c in cc])
    return _distance_bf_coord(*coords)


@numba.njit
def _distance_bf_coord(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    lat1 = radians(lat1)
    lon1 = radians(lon1)
    lat2 = radians(lat2)
    lon2 = radians(lon2)
    return _distance_bf(lat1, lon1, lat2, lon2)


@numba.njit
def _distance_bf(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    # sourcery skip: inline-immediately-returned-variable
    # https://stackoverflow.com/a/19412565/18280620

    # Approximate radius of earth in km
    R = 6373.0

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    distance = R * c

    return distance


@numba.njit
def _nb_clip(
    a: ty.Union[float, int],
    b: ty.Union[float, int],
    c: ty.Union[float, int],
) -> ty.Union[float, int]:
    """Cheap numba alternative to np.clip."""
    x = max(a, b)
    return min(x, c)


@numba.njit
def _all_adjacent_indexes(
    index: np.ndarray,
    len_lon: int,
    len_lat: int,
    add_diagonal: bool = True,
    add_double_lat: bool = False,
    add_double_lon: bool = True,
    add_90NS_bound: bool = True,
) -> np.ndarray:
    """For a given index, return an array of indexes that are adjacent.

    There are several options to add points:
        - add_diagonal: add diagonal elements seen from index
        - add_double_lat: add items that are 2 steps from the index in the lat direction
        - add_double_lon: add items that are 2 steps from the index in the lon direction
        - add_90NS_bound: for items that are at the lat bound, add all lon at the same lat.
    """
    lat, lon = index
    lat_up = _nb_clip(lat + 1, 0, len_lat - 1)
    lat_do = _nb_clip(lat - 1, 0, len_lat - 1)
    lon_up = np.mod(lon + 1, len_lon)
    lon_do = np.mod(lon - 1, len_lon)

    alt = [(lat_up, lon), (lat_do, lon), (lat, lon_up), (lat, lon_do)]

    if add_diagonal:
        alt = alt + [
            (lat_up, lon_up),
            (lat_do, lon_up),
            (lat_up, lon_do),
            (lat_do, lon_do),
        ]
    if add_double_lat:
        lat_double_up = _nb_clip(lat + 2, 0, len_lat - 1)
        lat_double_do = _nb_clip(lat - 2, 0, len_lat - 1)
        alt = alt + [(lat_double_up, lon), (lat_double_do, lon)]
    if add_double_lon:
        lon_double_up = np.mod(lon + 2, len_lon)
        lon_double_do = np.mod(lon - 2, len_lon)
        alt = alt + [(lat, lon_double_up), (lat, lon_double_do)]
    if add_90NS_bound and (lat == len_lat - 1 or lat == 0):
        alt = alt + [(lat, i) for i in range(len_lon)]
    return np.array([a for a in alt if ~np.array_equal(a, index)])


@numba.njit
def _indexes_to_2d_buffer(
    indexes: np.ndarray,
    buffer_2d: np.ndarray,
    result_2d: np.ndarray,
    only_if_val,
) -> None:
    """Fill elements in buffer2d with on indexes if they are not in
    exclude_2d."""
    for index in indexes:
        if result_2d[index[0], index[1]] == only_if_val:
            buffer_2d[index[0], index[1]] = True


def masks_array_to_coninuous_sets(
    masks: ty.List,
    no_group_value: int = -1,
    add_diagonal: bool = True,
    **kw,
) -> ty.List:
    """Call _group_mask_in_continous_sets for a group of masks with the same
    dimensions to reuse buffer arrays."""
    if not masks:
        return []

    len_x, len_y = masks[0].shape

    result_groups = np.ones_like(masks[0], dtype=np.int64) * no_group_value
    check_buffer = np.zeros_like(masks[0], dtype=np.bool_)
    kw_cont_sets = dict(
        len_x=len_x,
        len_y=len_y,
        add_diagonal=add_diagonal,
    )
    kw_cont_sets.update(kw)
    # Warning, do notice that the result_buffer and check_buffer are modified in place! However, _group_mask_in_continous_sets does reset the buffer each time
    # Therefore, we have to copy the result each time! Otherwise that result will be overwritten in the next iteration
    return [
        _group_mask_in_continous_sets(
            mask=mask,
            no_group_value=no_group_value,
            result_buffer=result_groups,
            check_buffer=check_buffer,
            **kw_cont_sets,
        ).copy()
        for mask in masks
    ]


def group_mask_in_continous_sets(mask: np.ndarray, *a, **kw) -> np.ndarray:
    return masks_array_to_coninuous_sets([mask])[0]


@numba.njit
def _group_mask_in_continous_sets(
    mask: np.ndarray,
    no_group_value: int,
    len_x: int,
    len_y: int,
    result_buffer: np.ndarray,
    check_buffer: np.ndarray,
    add_diagonal: bool = True,
    add_double_lat: bool = False,
    add_double_lon: bool = True,
    add_90NS_bound: bool = True,
) -> np.ndarray:
    # resetting the buffer is essential for calling `masks_array_to_coninuous_sets`
    result_buffer[:] = no_group_value
    check_buffer[:] = False
    indexes_to_iterate = np.argwhere(mask)
    group_id = 0

    for index in indexes_to_iterate:
        if result_buffer[index[0], index[1]] != no_group_value:
            continue

        group_id += 1
        check_buffer[:] = False
        adjacent_indexes = _all_adjacent_indexes(
            index,
            add_diagonal=add_diagonal,
            add_double_lat=add_double_lat,
            add_double_lon=add_double_lon,
            add_90NS_bound=add_90NS_bound,
            len_lat=len_x,
            len_lon=len_y,
        )
        _indexes_to_2d_buffer(
            adjacent_indexes,
            check_buffer,
            result_buffer,
            only_if_val=no_group_value,
        )

        included_another_index = True
        while included_another_index:
            included_another_index = False
            for another_index in np.argwhere(check_buffer):
                i, j = another_index

                if not mask[i][j]:
                    continue
                if result_buffer[i][j] == group_id:
                    continue

                included_another_index = True
                result_buffer[i][j] = group_id
                adjacent_indexes = _all_adjacent_indexes(
                    another_index,
                    add_diagonal=add_diagonal,
                    add_double_lat=add_double_lat,
                    add_double_lon=add_double_lon,
                    add_90NS_bound=add_90NS_bound,
                    len_lat=len_x,
                    len_lon=len_y,
                )
                _indexes_to_2d_buffer(
                    adjacent_indexes,
                    check_buffer,
                    result_buffer,
                    only_if_val=no_group_value,
                )
    return result_buffer
