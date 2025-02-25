import os
import typing as ty

import numba
import numpy as np
import optim_esm_tools as oet
import pandas as pd
import xarray as xr


def should_merge(
    ds1: ty.Union[xr.DataArray, np.ndarray],
    ds2: ty.Union[xr.DataArray, np.ndarray],
    min_frac_overlap: float = 0.5,
    min_border_frac: float = 0.05,
    min_n_adjacent: int = 100,
) -> bool:
    """Based on two 2d-grids (ds1 and ds2) decide if we should merge the grids
    since they either overlap or are adjacent.

    Merge if either is true:
        - The two regions overlap by a fraction of at least <min_frac_overlap>
        - The two regions border, where a fraction of at least <min_border_frac> of the borders are adjacent
        - The two regions border, with at least <min_n_adjacent> grid cells
    """

    if isinstance(ds1, xr.DataArray) and isinstance(ds2, xr.DataArray):
        ar1 = ds1.values
        ar2 = ds2.values
    elif isinstance(ds1, np.ndarray) and isinstance(ds2, np.ndarray):
        ar1 = ds1
        ar2 = ds2
    else:
        raise TypeError(f"Bad type {type(ds1)} {type(ds2)}")

    if _frac_overlap_nb(ar1, ar2) > min_frac_overlap:
        return True
    merge_a = _should_merge_adjacent(
        ar1,
        ar2,
        min_border_frac=min_border_frac,
        min_n_adjacent=min_n_adjacent,
    )
    if merge_a:
        return True

    # Also check the other way around one area is usually larger. For a small area, the fraction of overlap requirement is easier to satisfy.
    merge_b = _should_merge_adjacent(
        ar2,
        ar1,
        min_border_frac=min_border_frac,
        min_n_adjacent=min_n_adjacent,
    )
    if merge_b:
        oet.get_logger().debug(
            f"Got that we should not merge a to b, but b to a should be merged!",
        )
    return merge_b


def _frac_overlap_nb(ar1: np.ndarray, ar2: np.ndarray):
    """Fraction of ar1 and ar2 that overlap."""
    assert ar1.shape == ar2.shape
    return _frac_overlap_numba(ar1, ar2, ar1.shape)


@numba.njit
def _frac_overlap_numba(ar1, ar2, shape):
    x, y = shape
    both = np.int64(0)
    tot_1 = np.int64(0)
    tot_2 = np.int64(0)
    for i in range(x):
        for j in range(y):
            if ar1[i][j]:
                tot_1 += 1
                if ar2[i][j]:
                    tot_2 += 1
                    both += 1
            elif ar2[i][j]:
                tot_2 += 1
    lowest = min(tot_1, tot_2)
    return both / lowest


def _should_merge_adjacent(
    ar1: np.ndarray,
    ar2: np.ndarray,
    min_border_frac=0.05,
    min_n_adjacent=100,
):
    n_ad, n_border = _n_adjacent(ar1, ar2)
    return (n_ad / n_border) > min_border_frac or n_ad > min_n_adjacent


@numba.njit
def _n_adjacent(ar1: np.ndarray, ar2: np.ndarray):
    _n_adjacent = 0
    _n_border = 0
    x, y = ar1.shape

    for i in range(x):
        for j in range(y):
            if ar1[i][j] == False:
                continue

            _this_is_border = False

            left = np.mod(j + 1, y - 1)
            right = j - 1

            up = min(i + 1, x - 1)
            down = max(i - 1, 0)

            for altx, alty in ((up, j), (down, j), (i, left), (i, right)):
                if (altx, alty) == (i, j):
                    # Can raise this error, but for performance disabled, if it's working once, it's always true
                    # if (altx, alty) == (i, left) or (altx, alty) == (i, right):
                    #     raise ValueError((altx, alty) , (i,j))
                    # For up and down, we don't assume it's circular, so here the min and max can crop it to the same coordinate as i,j, let's skip those cases.
                    continue

                if not ar1[altx][alty]:
                    _this_is_border = True
                    if ar2[altx][alty]:
                        _n_adjacent += 1
            if _this_is_border:
                _n_border += 1
    return _n_adjacent, _n_border


class Merger:
    _log = None
    _sorted = False

    def __init__(
        self,
        pass_criteria: ty.Callable,
        summary_calculation: ty.Callable,
        data_sets: ty.Optional[ty.List[xr.Dataset]] = None,
        common_mother: ty.Optional[xr.Dataset] = None,
        common_pi: ty.Optional[xr.Dataset] = None,
        merge_options: ty.Optional[dict] = None,
        merge_method: str = "independent",
    ):
        assert data_sets, 'datasets'
        assert isinstance(pass_criteria, ty.Callable)
        assert isinstance(summary_calculation, ty.Callable)
        self.summary_calculation = summary_calculation
        self.pass_criteria = pass_criteria
        self.data_sets = data_sets

        common_mother, common_pi = self.get_mother_and_pi(common_mother, common_pi)
        self.common_mother = common_mother
        self.common_pi = common_pi
        self.merge_options = merge_options or dict(
            min_frac_overlap=0.5,
            min_border_frac=0.05,
            min_n_adjacent=100,
        )
        self.merge_method = merge_method

    def get_mother_and_pi(
        self,
        common_mother,
        common_pi,
    ) -> ty.Tuple[xr.Dataset, xr.Dataset]:
        common_mother = common_mother or oet.load_glob(self.data_sets[0]['file'])
        common_pi = common_pi or oet.analyze.time_statistics.get_historical_ds(
            common_mother,
        )
        assert isinstance(
            common_mother,
            xr.Dataset,
        ), f"Got wrong type {type(common_mother)}"
        assert isinstance(common_pi, xr.Dataset), f"Got wrong type {type(common_pi)}"
        return common_mother, common_pi

    def set_passing_largest_data_sets_first(self) -> None:
        candidates_info = [
            dict(
                passes=self.pass_criteria(
                    **self.summary_calculation(
                        **self.summary_kw,
                        mask=c['global_mask'],
                    ),
                ),
                cells=int(c['global_mask'].sum()),
                candidate=c,
            )
            for c in self.data_sets
        ]
        _max_number_of_cells = max(c['cells'] for c in candidates_info)

        candidates_sorted = sorted(
            candidates_info,
            key=lambda x: -int(x['passes'] * 2) * _max_number_of_cells
            - int(x['cells']),
        )

        self.data_sets = [c['candidate'] for c in candidates_sorted]
        self._sorted = True

    def merge_datasets(self):
        if not self._sorted:
            self.set_passing_largest_data_sets_first()
        candidates = self.data_sets
        groups = []
        pbar = oet.utils.tqdm(total=len(candidates), desc='Looping candidates')
        while candidates:
            pbar.n = len(self.data_sets) - len(candidates)
            pbar.display()
            self.log.info(pbar)
            doc = self._group_to_first(candidates)

            if not self.pass_criteria(**doc['stats']):
                self.log.info(
                    f'Discarding group {doc["merged"]} because {doc["stats"]} does not pass',
                )
                self.log.debug(f"Discarded doc {doc}")
            else:
                self.log.info(f'Adding {doc["merged"]} because {doc["stats"]} passes')
                doc['ds'].attrs.update(
                    dict(
                        mask_paths=[
                            c.attrs.get('mask_path', 'path=?')
                            for i, c in enumerate(candidates)
                            if i in doc['merged']
                        ],
                    ),
                )
                groups.append(doc)
            candidates = [c for i, c in enumerate(candidates) if i not in doc['merged']]

            if doc.get('force_break', False):
                self.log.warning('Breaking forcefully')
                candidates = []
        pbar.n = pbar.total
        pbar.close()
        pbar.display()
        self.log.info(pbar)
        return groups

    @property
    def summary_kw(self):
        return dict(
            ds_global=self.common_mother,
            ds_pi=self.common_pi,
            field=self.common_mother.attrs['variable_id'],
        )

    def _group_to_first(
        self,
        candidates: ty.List[xr.Dataset],
    ) -> ty.Dict[str, ty.Union[ty.Mapping, xr.Dataset, ty.List]]:
        something_merged = True
        global_masks = {
            i: ds['global_mask'].load().copy() for i, ds in enumerate(candidates)
        }
        current_global_mask: xr.DataArray = global_masks.pop(0)

        merge_to_current: ty.List[int] = []

        summary_kw = self.summary_kw
        first_doc = self.summary_calculation(
            **self.summary_kw,
            mask=candidates[0]['global_mask'],
        )
        if not self.pass_criteria(**first_doc):
            # The first candidate is not passing anything, so we should stop here.
            # Because _set_passing_largest_data_sets_first set the passing sets first, there
            # Should be no reason to continue here.
            self.log.info(
                f"Exhausted passing regions, so next candidates are ignored ({len(candidates)-1} remaining)",
            )
            # We are going to pass one additional argument that allows us to break the overencompasing loop
            return dict(stats=first_doc, ds=candidates[0], merged=[0], force_break=True)
        while something_merged:
            something_merged = False
            for i, ds_alt in global_masks.items():
                if i in merge_to_current:
                    continue
                if should_merge(current_global_mask, ds_alt, **self.merge_options):
                    merge_to_current += [i]
                    current_global_mask = current_global_mask | ds_alt
                    something_merged = True
        self.log.info(f"Merging items {merge_to_current} to [0]")
        if not merge_to_current:
            return dict(stats=first_doc, ds=candidates[0], merged=[0])

        ds_merged = oet.analyze.xarray_tools.mask_to_reduced_dataset(
            self.common_mother.load().copy(),
            oet.analyze.xarray_tools.reverse_name_mask_coords(current_global_mask),
        )
        stat = self.summary_calculation(**summary_kw, mask=current_global_mask)
        if self.pass_criteria(**stat):
            self.log.info(f"After merging everything, we pass")
            return dict(stats=stat, ds=ds_merged, merged=[0] + merge_to_current)

        for it, candidate_i in enumerate(
            oet.utils.tqdm([0] + merge_to_current, desc='checking at least one passes'),
        ):
            m = candidates[candidate_i]['global_mask']
            if self.pass_criteria(
                **self.summary_calculation(**summary_kw, mask=m),
            ):
                if it == 0:
                    self.log.info('First dataset passes, stop check')
                    break
                # Set the passing candidate as first in the list to merge to item 0.
                # Account for [0] being it 0
                it -= 1
                self.log.info(
                    f"Merging {candidate_i} of {merge_to_current} (it = {it})",
                )
                passing_first = merge_to_current[it:] + merge_to_current[:it]
                assert all(np.in1d(passing_first, merge_to_current)) and all(
                    np.in1d(merge_to_current, passing_first),
                ), f"{passing_first} {merge_to_current} are not equal!"
                merge_to_current = passing_first
                self.log.info(
                    f"Changed order of merge_to_current to {merge_to_current}",
                )
                break
        else:
            self.log.warning(
                f"All the datasets that are adjacent to current mask are not passing the criteria. Not trying to merge this group {merge_to_current}",
            )
            return dict(stats=stat, ds=ds_merged, merged=[0] + merge_to_current)

        self.log.warning(
            f"Merging would lead to failed test, going over items one by one",
        )
        if self.merge_method != 'independent':
            raise NotImplementedError(f'{self.merge_method} is not implemented')

        return self._iter_mergable_candidates(
            candidates,
            merge_to_current,
            summary_kw,
        )

    def _iter_mergable_candidates(
        self,
        candidates: ty.List[xr.Dataset],
        merge_to_current: ty.List[int],
        summary_kw: ty.Mapping,
    ) -> ty.Dict[str, ty.Union[ty.Mapping, xr.Dataset, ty.List]]:
        global_masks = {
            i: ds['global_mask'].load().copy() for i, ds in enumerate(candidates)
        }
        current_global_mask = global_masks.pop(0)
        do_merge: ty.List[int] = []

        do_a_merge = True
        it = 0
        pbar = oet.utils.tqdm(total=len(global_masks), desc='merging iteratively')
        while do_a_merge:
            # Keep looping over the candidates until we can loop over all the remaining candidates and conclude that they cannot be merged to the current mask
            it += 1
            pbar.n = len(do_merge)
            self.log.info(pbar)
            self.log.info(f"{it}: Remaining {merge_to_current} done {do_merge}")
            do_a_merge = False
            # Try this many times. We always pop the first element of the remaining candidates.
            for _ in merge_to_current:
                merge = merge_to_current.pop(0)
                ds_alt = global_masks[merge]
                passes_crit = False
                if should_merge(current_global_mask, ds_alt, **self.merge_options):
                    self.log.info(f"Try merging {merge}")
                    candidate_mask = current_global_mask.copy() | ds_alt
                    candidate_ds = oet.analyze.xarray_tools.mask_to_reduced_dataset(
                        self.common_mother.load().copy(),
                        oet.analyze.xarray_tools.reverse_name_mask_coords(
                            candidate_mask,
                        ),
                    )
                    candidate_stat = self.summary_calculation(
                        **summary_kw,
                        mask=candidate_mask,
                    )
                    passes_crit = self.pass_criteria(**candidate_stat)
                else:
                    self.log.info(f"No reason to merge {merge}")
                    continue

                if passes_crit:
                    self.log.info(
                        f"Merging {merge}. Currently do_a_merge={do_a_merge} remaining={merge_to_current}",
                    )
                    # Now break
                    do_a_merge = True
                    current_global_mask = candidate_mask
                    ds_merged = candidate_ds
                    stat = candidate_stat
                    do_merge += [merge]
                    break
                else:
                    # Now let's add i back into the list of things that we could merge
                    merge_to_current += [merge]
                    self.log.info(f"Merge {merge} failed, lead to {candidate_stat}")
        pbar.close()
        pbar.display()
        self.log.info(pbar)
        self.log.info(f"Merging {do_merge} from {merge_to_current}")
        if not do_merge:
            single_stat = self.summary_calculation(
                **summary_kw,
                mask=candidates[0]['global_mask'],
            )
            return dict(stats=single_stat, ds=candidates[0], merged=[0])
        return dict(stats=stat, ds=ds_merged, merged=[0] + do_merge)

    @property
    def log(self):
        self._log = self._log or oet.get_logger()
        return self._log


class MergerCached(Merger):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        del self.summary_calculation

        self._cache = {}
        self._summary_calculation = kw['summary_calculation']
        self.summary_calculation = self.cache_summary

    @staticmethod
    def hash_for_mask(mask: ty.Union[np.ndarray, xr.DataArray]) -> str:
        if isinstance(mask, xr.DataArray):
            return hash(mask.values.tobytes())
        return hash(mask.tobytes())

    def cache_summary(self, mask, *a, **kw) -> str:
        key = self.hash_for_mask(mask)
        if key not in self._cache:
            self.log.warning(f"Loading {key} new. Tot size {len(self._cache.keys())}")
            self._cache[key] = self._summary_calculation(*a, **kw, mask=mask)
        return self._cache[key]

    def set_all_caches_as_false(
        self,
        masks: ty.List[np.ndarray],
        fill_doc: ty.Optional[dict] = None,
    ) -> None:
        """A method to set all masks to not fullfull the summay calculation"""
        if fill_doc is None:
            doc_0 = self._summary_calculation(**self.summary_kw, mask=masks[0])
            fill_doc = {
                k: np.nan if isinstance(v, (float, int, np.number)) else v
                for k, v in doc_0.items()
            }
        for mask in masks:
            key = self.hash_for_mask(mask)
            self._cache[key] = fill_doc
