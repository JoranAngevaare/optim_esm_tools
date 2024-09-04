import glob
import os
from collections import defaultdict

from optim_esm_tools.config import config
from optim_esm_tools.config import get_logger
from optim_esm_tools.utils import check_accepts
from optim_esm_tools.utils import deprecated
from optim_esm_tools.utils import timed
import xarray as xr
import logging
import typing as ty


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
    ),
)
def find_matches(
    base: str,
    activity_id: str = 'ScenarioMIP',
    institution_id: str = '*',
    source_id: str = '*',
    experiment_id: str = '*',
    variant_label: str = '*',
    domain: str = '*',
    variable_id: str = '*',
    grid_label: str = '*',
    version: str = '*',
    max_versions: int = 1,
    max_members: int = 1,
    required_file: str = 'merged.nc',
    # Deprecated arg
    grid=None,
) -> ty.List[str]:
    """Follow synda folder format to find matches.

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
        max_versions (int, optional): Max number of different versions that match. Defaults to 1.
        max_members (int, optional): Max number of different members that match. Defaults to 1.
        required_file (str, optional): Filename to match. Defaults to 'merged.nc'.

    Returns:
        list: of matches corresponding to the query
    """
    log = get_logger()
    if grid is not None:  # pragma: no cover
        log.error('grid argument for find_matches is deprecated, use grid_label')
        grid_label = grid
    if max_versions is None:
        max_versions = int(9e9)  # pragma: no cover
    if max_members is None:
        max_members = int(9e9)  # pragma: no cover
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
            ),
        ),
        key=_variant_label_id_and_version,
    )
    seen: dict = {}
    for candidate in variants:
        folders = candidate.split(os.sep)
        group = folders[-7]
        version = folders[-1]

        if group not in seen:
            seen[group] = defaultdict(list)
        seen_members = seen[group]

        if (len(seen_members) == max_versions and version not in seen_members) or len(
            seen_members.get(version, []),
        ) == max_members:
            continue  # pragma: no cover
        if required_file and required_file not in os.listdir(
            candidate,
        ):  # pragma: no cover
            log.warning(f'{required_file} not in {candidate}')
            continue
        if is_excluded(candidate):  # pragma: no cover
            log.info(f'{candidate} is excluded')
            continue
        seen_members[version].append(candidate)
    assert all(len(group) <= max_versions for group in seen.values())
    assert all(
        len(members) <= max_members
        for group in seen.values()
        for members in group.values()
    )
    return [
        folder
        for group_dict in seen.values()
        for versions in group_dict.values()
        for folder in versions
    ]


def _get_head(path: str) -> str:
    log = get_logger()
    if path.endswith(os.sep):
        log.debug(f'Stripping tailing "/" from {path}')
        path = path[: -len(os.sep)]

    if os.path.isfile(path):
        log.debug(f'Splitting file from {path}')
        path = os.path.split(path)[0]
    return path


def is_excluded(path: str, exclude_too_short: bool = True) -> bool:
    from fnmatch import fnmatch

    path = _get_head(path)
    exlusion_list = config['CMIP_files']['excluded'].split('\n')
    if exclude_too_short:
        exlusion_list += config['CMIP_files']['too_short'].split('\n')
    for excluded in exlusion_list:
        if not excluded:
            continue
        folders = excluded.split()

        path_ends_with = os.path.join(*path.split(os.sep)[-len(folders) :])
        match_to = os.path.join(*folders)
        if fnmatch(path_ends_with, match_to):
            return True  # pragma: no cover
    return False


def _variant_label_id_and_version(full_path: str) -> ty.Tuple[int, int]:
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
        return int(1e9), int(1e9)  # pragma: no cover
    return -grid_version, run_variant_number


def folder_to_dict(path: str, strict: bool = True) -> ty.Optional[ty.Dict[str, str]]:
    path = _get_head(path)
    folders = path.split(os.sep)
    if folders[-1][0] == 'v' and len(folders[-1]) == len('v20190731'):
        return {
            k: folders[-i - 1]
            for i, k in enumerate(config['CMIP_files']['folder_fmt'].split()[::-1])
        }
        # great
    if strict:
        raise NotImplementedError(
            f'folder {path} does not end with a version',
        )  # pragma: no cover
    return None


def base_from_path(path: str, look_back_extra: int = 0) -> str:
    path = _get_head(path)
    return os.path.join(
        os.sep,
        *path.split(os.sep)[
            : -len(config['CMIP_files']['folder_fmt'].split()) - look_back_extra
        ],
    )


@deprecated
def associate_historical(*a, **kw):
    return associate_parent(*a, **kw)


def _get_search_kw(
    data_set: xr.Dataset,
    keep_keys: tuple = tuple(
        'parent_activity_id parent_experiment_id parent_source_id parent_variant_label'.split(),
    ),
    required_file: str = 'merged.nc',
) -> ty.Dict[str, str]:
    query = {
        k.replace('parent_', ''):
        # Filter out some institutes that ended up adding a bunch of spaces here?!
        data_set.attrs.get(k, '*').replace(' ', '')
        for k in keep_keys
    }
    query.update(dict(required_file=required_file))
    return query


def _check_search_kw(
    search: dict,
    data_set: xr.Dataset,
    log: logging.Logger,
    path: str,
) -> dict:
    if (
        search['source_id'] == 'GISS-E2-1-G'
        and data_set.attrs['source_id'] == 'GISS-E2-2-G'
    ):
        # I'm quite sure there has been some mixup here.
        log.error(f'Hacking GISS-E2-1-G -> GISS-E2-2-G ?!!?!')
        search['source_id'] = 'GISS-E2-2-G'

    if search['source_id'] != data_set.attrs['source_id']:
        log.critical(
            f"Misalignment in source-ids for {path} got {search['source_id']} and {data_set.attrs['source_id']}",
        )

    if search['activity_id'] not in ['ScenarioMIP', 'CMIP']:
        log.warning(
            f"{search['activity_id']} seems invalid for {path}, trying wildcard!",
        )
        search['activity_id'] = '*'

    return search


def _read_dataset(
    data_set: ty.Optional[xr.Dataset],
    required_file: ty.Optional[str],
    path: ty.Optional[str],
) -> xr.Dataset:
    if data_set is None and path is None:
        raise ValueError(
            'No dataset, no path, can\'t match if I don\'t know what I\'m looking for',
        )  # pragma: no cover
    if path is None:
        assert data_set is not None
        path = data_set.attrs['path']

    assert isinstance(path, str) and os.path.exists(path), path
    if data_set is None:
        from optim_esm_tools import load_glob

        assert required_file is not None
        file = (
            os.path.join(path, required_file)
            if not path.endswith(required_file)
            else path
        )
        data_set = load_glob(file)
    return data_set


def associate_parent(
    data_set: ty.Optional[xr.Dataset] = None,
    path: ty.Optional[str] = None,
    match_to: str = 'piControl',
    look_back_extra: int = 0,
    query_updates: ty.Optional[ty.List[dict]] = None,
    search_kw: ty.Optional[dict] = None,
    strict: bool = True,
    required_file: ty.Optional[str] = 'merged.nc',
):

    log = get_logger()

    data_set = _read_dataset(data_set=data_set, required_file=required_file, path=path)
    path = path or data_set.attrs['path']
    base = base_from_path(path, look_back_extra=look_back_extra)
    search = _get_search_kw(data_set, required_file=required_file)
    search = _check_search_kw(search=search, data_set=data_set, log=log, path=path)

    if all(v == '*' for v in search.values()) or search['source_id'] == '*':
        raise ValueError(f'Unclear search for {path} - attributes are missing')

    search.update(dict(variable_id=data_set.attrs['variable_id']))

    if search_kw:
        raise ValueError(f'Not used any more, got {search_kw}')

    if query_updates is None:
        # It's important to match the variant-label last, otherwise we get mismatched simulations
        # from completely different ensamble members. We used to accept such cases, but disabled
        # this later as it gave funky results
        query_updates = [
            {},
            dict(version='*'),
            dict(grid_label='*'),
            dict(variant_label='*'),
        ]

    for try_n, update_query in enumerate(query_updates):
        if try_n:
            if not strict:
                message = f'No results after {try_n} try, retrying with {update_query}'
                log.info(message)
            else:
                break
        search.update(update_query)
        if this_try := find_matches(base, **search):
            if match_to == 'piControl' and search.get('experiment_id') == 'historical':
                log.debug(
                    f'Found historical, but we need to match to PiControl, recursively returning!',
                )
                results = [
                    associate_parent(
                        path=t,
                        match_to=match_to,
                        look_back_extra=look_back_extra,
                        query_updates=query_updates,
                        search_kw=search_kw,
                        strict=strict,
                    )
                    for t in this_try
                ]
                # Return a flat array!
                return [rr for r in results if r for rr in r]

            return this_try
    message = f'Looked for {search}, in {base} found nothing'
    if strict:
        raise RuntimeError(message)
    log.warning(message)  # pragma: no cover
