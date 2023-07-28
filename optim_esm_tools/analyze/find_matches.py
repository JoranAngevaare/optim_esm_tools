import os
import glob
from optim_esm_tools.utils import check_accepts, timed
from optim_esm_tools.config import config, get_logger
from collections import defaultdict


@timed
@check_accepts(
    accepts=dict(
        activity_id=('AerChemMIP', 'ScenarioMIP', 'CMIP', '*'),
        experiment_id=(
            'piControl',
            'historical',
            'ssp119',
            'ssp126',
            'ssp245',
            'ssp370',
            'ssp585',
            '*',
        ),
    )
)
def find_matches(
    base: str,
    activity_id='ScenarioMIP',
    institution_id='*',
    source_id='*',
    experiment_id='ssp585',
    variant_label='*',
    domain='Ayear',
    variable_id='tas',
    grid_label='*',
    version='*',
    max_versions: int = 1,
    max_members: int = 1,
    required_file='merged.nc',
    # Depricated arg
    grid=None,
) -> list:
    """Follow synda folder format to find matches

    Args:
        base (str): where start looking for matches
        activity_id (str, optional): As synda convention. Defaults to 'ScenarioMIP'.
        institution_id (str, optional): As synda convention. Defaults to '*'.
        source_id (str, optional): As synda convention. Defaults to '*'.
        experiment_id (str, optional): As synda convention. Defaults to 'ssp585'.
        variant_label (str, optional): As synda convention. Defaults to '*'.
        domain (str, optional): As synda convention. Defaults to 'Ayear'.
        variable_id (str, optional): As synda convention. Defaults to 'tas'.
        grid_label (str, optional): As synda convention. Defaults to '*'.
        version (str, optional): As synda convention. Defaults to '*'.
        max_versions (int, optional): Max mumber of different versions that match. Defaults to 1.
        max_members (int, optional): Max mumber of different members that match. Defaults to 1.
        required_file (str, optional): Filename to match. Defaults to 'merged.nc'.

    Returns:
        list: of matches corresponding to the query
    """
    if grid is not None:
        get_logger().warning(
            f'grid argument for find_matches is deprecated, use grid_label'
        )
        grid_label = grid
    if max_versions is None:
        max_versions = int(9e9)
    if max_members is None:
        max_members = int(9e9)
    variants = sorted(
        glob.glob(
            os.path.join(
                base,
                activity_id,
                institution_id,
                source_id,
                experiment_id,
                variant_label,
                domain,
                variable_id,
                grid_label,
                version,
            )
        ),
        key=_variant_label_id_and_version,
    )
    seen = dict()
    for candidate in variants:
        folders = candidate.split(os.sep)
        group = folders[-7]
        member = folders[-5]
        version = folders[-1]

        if group not in seen:
            seen[group] = defaultdict(list)
        seen_members = seen[group]
        if len(seen_members) < max_members or member in seen_members:
            if required_file and required_file not in os.listdir(candidate):
                get_logger().warning(f'{required_file} not in {candidate}')
                continue
            if len(seen_members.get(version, [])) == max_versions:
                continue
            if is_excluded(candidate):
                continue
            seen_members[version].append(candidate)

    return [
        folder
        for group_dict in seen.values()
        for versions in group_dict.values()
        for folder in versions
    ]


def _get_head(path):
    log = get_logger()
    if path.endswith(os.sep):
        log.debug(f'Stripping tailing "/" from {path}')
        path = path[: -len(os.sep)]

    if os.path.isfile(path):
        log.debug(f'Splitting file from {path}')
        path = os.path.split(path)[0]
    return path


def is_excluded(path):
    from fnmatch import fnmatch

    path = _get_head(path)

    for excluded in config['CMIP_files']['excluded'].split('\n'):
        if not excluded:
            continue
        folders = excluded.split()

        path_ends_with = os.path.join(*path.split(os.sep)[-len(folders) :])
        match_to = os.path.join(*folders)
        if fnmatch(path_ends_with, match_to):
            return True
    return False


def _variant_label_id_and_version(full_path):
    run_variant_number = None
    grid_version = None
    for folder in full_path.split(os.sep):
        if len(folder):
            if folder[0] == 'r' and run_variant_number is None:
                index = folder.split('i')
                if len(index) == 2:
                    run_variant_number = int(index[0][1:])
            if (
                folder[0] == 'v'
                and len(folder) == len('v20190731')
                and grid_version is None
            ):
                grid_version = int(folder[1:])
    if run_variant_number is None or grid_version is None:
        raise ValueError(
            f'could not find run and version from {full_path} {run_variant_number} {grid_version}'
        )
    return -grid_version, run_variant_number


def folder_to_dict(path):
    path = _get_head(path)
    folders = path.split(os.sep)
    if folders[-1][0] == 'v' and len(folders[-1]) == len('v20190731'):
        return {
            k: folders[-i - 1]
            for i, k in enumerate(config['CMIP_files']['folder_fmt'].split()[::-1])
        }
        # great
    raise NotImplementedError(f'folder {path} does not end with a version')


def base_from_path(path, look_back_extra=0):
    path = _get_head(path)
    return os.path.join(
        os.sep,
        *path.split(os.sep)[
            : -len(config['CMIP_files']['folder_fmt'].split()) - look_back_extra
        ],
    )


def associate_historical(
    data_set=None,
    path=None,
    match_to='piControl',
    activity_id='CMIP',
    look_back_extra=0,
    query_updates=None,
    search_kw=None,
    strict=True,
):
    if data_set is None and path is None:
        raise ValueError(
            'No dataset, no path, can\'t match if I don\'t know what I\'m looking for'
        )
    log = get_logger()
    path = path or data_set.attrs['path']
    base = base_from_path(path, look_back_extra=look_back_extra)
    search = folder_to_dict(path)
    search['activity_id'] = activity_id
    if search['experiment_id'] == match_to:
        message = f'Cannot match {match_to} to itself!'
        if strict:
            raise NotImplementedError(message)
        log.warning(message)
    search['experiment_id'] = match_to
    if search_kw:
        search.update(search_kw)
        print(search)
    if query_updates is None:
        query_updates = [
            dict(),
            dict(variant_label='*'),
            dict(version='*'),
            # can lead to funny behavior as grid differences may cause breaking compares
            dict(grid_label='*'),
        ]

    for try_n, update_query in enumerate(query_updates):
        if try_n:
            message = f'No results after {try_n} try, retying with {update_query}'
            log.info(message)
        search.update(update_query)
        this_try = find_matches(base, **search)
        if this_try:
            return this_try
    message = f'Looked for {search}, in {base} found nothing'
    if strict:
        raise RuntimeError(message)
    log.warning(message)
