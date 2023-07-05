import numpy as np
from optim_esm_tools.utils import tqdm, timed
import typing as ty
from warnings import warn
import numba
from math import sin, cos, sqrt, atan2, radians


@timed()
def build_clusters(
    coordinates_deg: np.ndarray,
    weights: ty.Optional[np.ndarray] = None,
    max_distance_km: ty.Union[float, int] = 750,
    only_core: bool = True,
    min_samples: int = 10,
    cluster_opts: ty.Optional[dict] = None,
) -> ty.List[np.ndarray]:
    """Build clusters based on a list of coordinates, use halfsine metric for spherical spatiol data

    Args:
        coordinates_deg (np.ndarray): set of xy coordinates in degrees
        max_distance_km (ty.Union[float, int], optional): max distance to other points to consider part of
            cluster (see DBSCAN(eps=<..>)). Defaults to 750.
        only_core (bool, optional): Use only core samples. Defaults to True.
        min_sample (int): Minimum number of samples in cluster. Defaults to 20.
        cluster_opts (ty.Optional[dict], optional): Additional options passed to sklearn.cluster.DBSCAN. Defaults to None.

    Returns:
        ty.List[np.ndarray]: list of clustered points (in radians)
    """
    if cluster_opts is None:
        cluster_opts = dict()
    for class_label, v in dict(algorithm='ball_tree', metric='haversine').items():
        cluster_opts.setdefault(class_label, v)
    cluster_opts['min_samples'] = min_samples

    from sklearn.cluster import DBSCAN
    import numpy as np

    coordinates_rad = np.radians(coordinates_deg).T

    # TODO use a more up to date version:
    #  https://scikit-learn.org/stable/auto_examples/cluster/plot_hdbscan.html#sphx-glr-auto-examples-cluster-plot-hdbscan-py
    #  https://scikit-learn.org/stable/modules/generated/sklearn.cluster.HDBSCAN.html#sklearn.cluster.HDBSCAN
    # Thanks https://stackoverflow.com/a/38731787/18280620!
    try:
        db_fit = DBSCAN(eps=max_distance_km / 6371.0, **cluster_opts).fit(
            X=coordinates_rad, sample_weight=weights
        )
    except ValueError as e:
        raise ValueError(
            f'With {coordinates_rad.shape} and {weights.shape} {coordinates_rad}, {weights}'
        ) from e

    labels = db_fit.labels_

    unique_labels = sorted(set(labels))
    is_core_sample = np.zeros_like(labels, dtype=bool)
    is_core_sample[db_fit.core_sample_indices_] = True

    return_masks = []

    for class_label in unique_labels:
        is_noise = class_label == -1
        if is_noise:
            continue

        is_class_member = labels == class_label
        coord_mask = is_class_member
        if only_core:
            coord_mask &= is_core_sample

        masked_points = coordinates_rad[coord_mask]
        return_masks.append(masked_points)

    return return_masks


@timed()
def build_cluster_mask(
    global_mask: np.ndarray,
    x_coord: np.array,
    y_coord: np.array,
    show_tqdm: bool = False,
    max_distance_km: ty.Union[str, float, int] = 'infer',
    **kw,
) -> ty.Tuple[ty.List[np.ndarray], ty.List[np.ndarray]]:
    """Build set of clusters and masks based on the global mask, basically a utility wrapper arround build_clusters'

    Args:
        global_mask (np.ndarray): full 2d mask of the data
        x_coord (np.array): all x values
        y_coord (np.array): all y values
        max_distance_km (ty.Union[str, float, int]): find an appropriate distance
            threshold for build_clusters' max_distance_km argument. If nothing is
            provided, make a guess based on the distance between grid cells.
            Defaults to 'infer'.
        show_tqdm (bool, optional): use verboose progressbar. Defaults to False.

    Returns:
        ty.List[ty.List[np.ndarray], ty.List[np.ndarray]]: Return two lists, containing the masks, and clusters respectively.
    """

    _check_input(global_mask, x_coord, y_coord)
    xm, ym = np.meshgrid(x_coord, y_coord)
    xy_data = np.array([xm[global_mask.T], ym[global_mask.T]])

    if len(xy_data.T) <= 2:
        warn(f'No data from this mask {xy_data}!')
        return [], []

    if max_distance_km == 'infer':
        max_distance_km = _infer_max_step_size(x_coord, y_coord)

    masks, clusters = _build_cluster_with_kw(
        x_coord,
        y_coord,
        coordinates_deg=xy_data,
        show_tqdm=show_tqdm,
        max_distance_km=max_distance_km,
        **kw,
    )

    return masks, clusters


@timed()
def build_weighted_cluster(
    weights: np.ndarray,
    x_coord: np.array,
    y_coord: np.array,
    show_tqdm: bool = False,
    threshold=0.99,
    max_distance_km: ty.Union[str, float, int] = 'infer',
    **kw,
) -> ty.Tuple[ty.List[np.ndarray], ty.List[np.ndarray]]:
    """Build set of clusters and masks based on the weights (which should be a grid)'

    Args:
        weights (np.ndarray): normalized score data (values in [0,1])
        x_coord (np.array): all x values
        y_coord (np.array): all y values
        max_distance_km (ty.Union[str, float, int]): find an appropriate distance
            threshold for build_clusters' max_distance_km argument. If nothing is
            provided, make a guess based on the distance between grid cells.
            Defaults to 'infer'.
        show_tqdm (bool, optional): use verboose progressbar. Defaults to False.

    Returns:
        ty.List[ty.List[np.ndarray], ty.List[np.ndarray]]: Return two lists, containing the masks, and clusters respectively.
    """

    _check_input(weights, x_coord, y_coord)
    xm, ym = np.meshgrid(x_coord, y_coord)
    xy_data = np.array([xm.flatten(), ym.flatten()])

    if max_distance_km == 'infer':
        max_distance_km = _infer_max_step_size(x_coord, y_coord)

    flat_weights = weights.T.flatten()
    mask = flat_weights > threshold
    masks, clusters = _build_cluster_with_kw(
        x_coord,
        y_coord,
        coordinates_deg=xy_data[:, mask],
        weights=flat_weights[mask],
        show_tqdm=show_tqdm,
        max_distance_km=max_distance_km,
        **kw,
    )

    return masks, clusters


def _check_input(data, x_coord, y_coord):
    if data.shape != (len(x_coord), len(y_coord)):
        message = f'Wrong input {data.shape} != {len(x_coord), len(y_coord)}'
        raise ValueError(message)


def _build_cluster_with_kw(x_coord, y_coord, show_tqdm=False, **cluster_kw):
    masks = []
    clusters = [np.rad2deg(cluster) for cluster in build_clusters(**cluster_kw)]

    for cluster in clusters:
        mask = np.zeros((len(y_coord), len(x_coord)), np.bool_)
        for s_x, s_y in tqdm(cluster, desc='fill_mask', disable=not show_tqdm):
            # This is a bit blunt, but it's fast enough to regain the indexes such that we can build a 2d masked array.
            x_i = np.argwhere(np.isclose(x_coord, s_x))[0]
            y_i = np.argwhere(np.isclose(y_coord, s_y))[0]
            mask[y_i, x_i] = 1
        masks.append(mask)
    return masks, clusters


def _infer_max_step_size(xs, ys):
    ys = ys[ys > 0]
    # coords = [[[xs[0], ys[0]], [xs[0], ys[1]]], [[xs[0], ys[0]], [xs[1], ys[0]]]]
    # Return long:lat
    coords = [[[ys[0], xs[0]], [ys[0], xs[1]]], [[ys[0], xs[0]], [ys[1], xs[0]]]]
    # Return 2x the distance between grid cells
    return 2 * max(_distance(c) for c in coords)


def _distance(coords, force_math=False):
    """Wrapper for if geopy is not installed"""
    if not force_math:
        try:
            import geopy.distance

            return geopy.distance.geodesic(*coords).km
        except (ImportError, ModuleNotFoundError):
            pass
    return _distance_bf_coord(*coords)


@numba.njit
def _distance_bf_coord(lat1, lon1, lat2, lon2):
    lat1 = radians(lat1)
    lon1 = radians(lon1)
    lat2 = radians(lat2)
    lon2 = radians(lon2)
    return _distance_bf(lat1, lon1, lat2, lon2)


@numba.njit
def _distance_bf(lat1, lon1, lat2, lon2):
    # https://stackoverflow.com/a/19412565/18280620

    # Approximate radius of earth in km
    R = 6373.0

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    distance = R * c

    return distance
