import numpy as np
from optim_esm_tools.utils import tqdm
import typing as ty
from warnings import warn


def build_clusters(
    coordinates_deg: np.ndarray,
    max_distance_km: ty.Union[float, int] = 500,
    only_core: bool = True,
    cluster_opts: ty.Optional[dict] = None,
) -> ty.List[np.ndarray]:
    """Build clusters based on a list of coordinates, use halfsine metric for spherical spatiol data

    Args:
        coordinates_deg (np.ndarray): set of xy coordinates in degrees
        max_distance_km (ty.Union[float, int], optional): max distance to other points to consider part of cluster (see DBSCAN(eps=<..>)). Defaults to 500.
        only_core (bool, optional): Use only core samples. Defaults to True.
        cluster_opts (ty.Optional[dict], optional): Additional options passed to sklearn.cluster.DBSCAN. Defaults to None.

    Returns:
        ty.List[np.ndarray]: list of clustered points (in radians)
    """
    if cluster_opts is None:
        cluster_opts = dict()
    for class_label, v in dict(
        min_samples=20, algorithm='ball_tree', metric='haversine'
    ).items():
        cluster_opts.setdefault(class_label, v)

    from sklearn.cluster import DBSCAN
    import numpy as np

    coordinates_rad = np.radians(coordinates_deg).T

    # Thanks https://stackoverflow.com/a/38731787/18280620!
    db_fit = DBSCAN(eps=max_distance_km / 6371.0, **cluster_opts).fit(coordinates_rad)

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


def build_cluster_mask(
    global_mask: np.ndarray,
    x_coord: np.array,
    y_coord: np.array,
    show_tqdm: bool = False,
    **kw,
) -> ty.Tuple[ty.List[np.ndarray], ty.List[np.ndarray]]:
    """Build set of clusters and masks based on the global mask, basically a utility wrapper arround build_clusters

    Args:
        global_mask (np.ndarray): full 2d mask of the data
        x_coord (np.array): all x values
        y_coord (np.array): all y values
        show_tqdm (bool, optional): use verboose progressbar. Defaults to False.

    Returns:
        ty.List[ty.List[np.ndarray], ty.List[np.ndarray]]: Return two lists, containing the masks, and clusters respectively.
    """

    assert global_mask.shape == (len(x_coord), len(y_coord)), (
        global_mask.shape,
        (len(x_coord), len(y_coord)),
    )
    xm, ym = np.meshgrid(x_coord, y_coord)
    xy_data = np.array([xm[global_mask.T], ym[global_mask.T]])
    if len(xy_data.T) <= 2:
        warn(f'No data from this mask {xy_data}!')
        return [], []
    masks = []
    clusters = [np.rad2deg(cluster) for cluster in build_clusters(xy_data, **kw)]

    for cluster in clusters:
        mask = np.zeros((len(y_coord), len(x_coord)), np.bool_)
        for s_x, s_y in tqdm(cluster, desc='fill_mask', disable=not show_tqdm):
            # This is a bit blunt, but it's fast enough to regain the indexes such that we can build a 2d masked array.
            x_i = np.argwhere(np.isclose(x_coord, s_x))[0]
            y_i = np.argwhere(np.isclose(y_coord, s_y))[0]
            mask[y_i, x_i] = 1
        masks.append(mask)

    return masks, clusters
