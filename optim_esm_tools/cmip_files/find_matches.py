import os
import glob
from optim_esm_tools.utils import check_accepts, timed
from optim_esm_tools.config import config, get_logger
from collections import defaultdict


@timed
@check_accepts(
    accepts=dict(
        activity_id=('ScenarioMIP', 'CMIP', '*'),
        experiment_id=(
            'piControl',
            'historical',
            'ssp119',
            'sp126',
            'sp245',
            'ssp370',
            'ssp585',
            '*',
        ),
    )
)
def find_matches(
    base,
    activity_id='ScenarioMIP',
    institution_id='*',
    source_id='*',
    experiment_id='ssp585',
    variant_label='*',
    domain='Ayear',
    variable_id='tas',
    grid='*',
    version='*',
    max_versions=1,
    max_members=1,
):
    """Follow synda folder format to find matches"""
    if max_versions is None:
        max_versions = int(9e9)
    if max_members is None:
        max_members = int(9e9)
    variantes = sorted(
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
                grid,
                version,
            )
        ),
        key=_variant_label_id_and_version,
    )
    seen = dict()
    for candidate in variantes:
        folders = candidate.split(os.sep)
        group = folders[-7]
        member = folders[-5]
        version = folders[-1]

        if group not in seen:
            seen[group] = defaultdict(list)
        seen_members = seen[group]
        if len(seen_members) < max_members or member in seen_members:
            if len(seen_members.get(version, [])) == max_versions:
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
