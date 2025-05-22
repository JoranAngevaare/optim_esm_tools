4.0.0 / 2025-04-24
------------------
## Major (breaking change)
* Remove symmetry test and R-requirement, test minimal install by @JoranAngevaare in https://github.com/JoranAngevaare/optim_esm_tools/pull/245

## Minor
* Further sort candidate regions on lat lon in `Merger` by @JoranAngevaare in https://github.com/JoranAngevaare/optim_esm_tools/pull/238
* Advanced example notebook by @JoranAngevaare in https://github.com/JoranAngevaare/optim_esm_tools/pull/239

## Patch
* Test for `analyze.region_calculation.calculate_norm` by @JoranAngevaare in https://github.com/JoranAngevaare/optim_esm_tools/pull/240
* [pre-commit.ci] pre-commit autoupdate by @pre-commit-ci in https://github.com/JoranAngevaare/optim_esm_tools/pull/243
* Bump actions/setup-python from 5.5.0 to 5.6.0 by @dependabot in https://github.com/JoranAngevaare/optim_esm_tools/pull/242
* Calculate frac for lowess based on non-nan values by @JoranAngevaare in https://github.com/JoranAngevaare/optim_esm_tools/pull/244
* Notes on installation added to README by @JoranAngevaare in https://github.com/JoranAngevaare/optim_esm_tools/pull/246


**Full Changelog**: https://github.com/JoranAngevaare/optim_esm_tools/compare/v4.0.0...v5.0.0

4.0.0 / 2025-04-24
------------------
## Major
* Add running mean and detrending on the fly to save disk by @JoranAngevaare in https://github.com/JoranAngevaare/optim_esm_tools/pull/230

## Minor
* Optimized version of `is_excluded` by @JoranAngevaare in https://github.com/JoranAngevaare/optim_esm_tools/pull/229

## Patch
* Remove old functions by @JoranAngevaare in https://github.com/JoranAngevaare/optim_esm_tools/pull/226
* `setup-miniconda@v3` by @JoranAngevaare in https://github.com/JoranAngevaare/optim_esm_tools/pull/233
* Bump actions/setup-python from 5.2.0 to 5.5.0 by @dependabot in https://github.com/JoranAngevaare/optim_esm_tools/pull/232
* Update example path in `pyproject.toml` by @JoranAngevaare in https://github.com/JoranAngevaare/optim_esm_tools/pull/234
* Only test ubuntu latest by @JoranAngevaare in https://github.com/JoranAngevaare/optim_esm_tools/pull/235

**Full Changelog**: https://github.com/JoranAngevaare/optim_esm_tools/compare/v3.1.0...v4.0.0

3.1.0 / 2025-02-13
------------------
## Minor
* Gridpatcher for holes in interpolated grids by @JoranAngevaare in https://github.com/JoranAngevaare/optim_esm_tools/pull/223

## Documentation
* DiscontinuousGridPatcher demo by @JoranAngevaare in https://github.com/JoranAngevaare/optim_esm_tools/pull/224

## Patch
* Weighted average for `find_max_in_equal_area` by @JoranAngevaare in https://github.com/JoranAngevaare/optim_esm_tools/pull/219
* Allow kw in clustering by @JoranAngevaare in https://github.com/JoranAngevaare/optim_esm_tools/pull/218
* Break merging early for `merge_datasets` by @JoranAngevaare in https://github.com/JoranAngevaare/optim_esm_tools/pull/220

**Full Changelog**: https://github.com/JoranAngevaare/optim_esm_tools/compare/v3.0.1...v3.1.0

3.0.1 / 2025-01-14
------------------
## Bugfixes
* Exit early from candidates without passing properties by @JoranAngevaare in https://github.com/JoranAngevaare/optim_esm_tools/pull/212
* Weighted average in combine variables by @JoranAngevaare in https://github.com/JoranAngevaare/optim_esm_tools/pull/206

## Documentation
* Example notebook by @JoranAngevaare in https://github.com/JoranAngevaare/optim_esm_tools/pull/204

## Patch
* Miniforge installer by @JoranAngevaare in https://github.com/JoranAngevaare/optim_esm_tools/pull/203
* Update rasterio requirement from <=1.3.10 to <=1.4.3 by @dependabot in https://github.com/JoranAngevaare/optim_esm_tools/pull/207
* Bump nbconvert from 7.2.3 to 7.16.4 by @dependabot in https://github.com/JoranAngevaare/optim_esm_tools/pull/211
* [pre-commit.ci] pre-commit autoupdate by @pre-commit-ci in https://github.com/JoranAngevaare/optim_esm_tools/pull/205

**Full Changelog**: https://github.com/JoranAngevaare/optim_esm_tools/compare/v3.0.0...v3.0.1

3.0.0 / 2024-11-27
------------------
## Major
* Region calculation by @JoranAngevaare in https://github.com/JoranAngevaare/optim_esm_tools/pull/201

## Patch
* Merge adjacent candidates by @JoranAngevaare in https://github.com/JoranAngevaare/optim_esm_tools/pull/199
* Add numpy, xarray and other utility functions by @JoranAngevaare in https://github.com/JoranAngevaare/optim_esm_tools/pull/200

**Full Changelog**: https://github.com/JoranAngevaare/optim_esm_tools/compare/v2.1.1...v3.0.0


2.1.0 / 2024-09-30
------------------
## Minor
* Improved typing and path handing in `find_matches.py` by @JoranAngevaare in https://github.com/JoranAngevaare/optim_esm_tools/pull/177
* Speedup `mask_xr_ds` by a factor of `2-4x` using `numba`  when `drop=True` by @JoranAngevaare in https://github.com/JoranAngevaare/optim_esm_tools/pull/182
* Add lowess filter by @JoranAngevaare in https://github.com/JoranAngevaare/optim_esm_tools/pull/186
* Plotting typing and flexibility by @JoranAngevaare in https://github.com/JoranAngevaare/optim_esm_tools/pull/176
* Improve direct region finding by @JoranAngevaare in https://github.com/JoranAngevaare/optim_esm_tools/pull/175
* Increase pre-process flexibility by @JoranAngevaare in https://github.com/JoranAngevaare/optim_esm_tools/pull/180

## Patch
* Clean v2.0.0 by @JoranAngevaare in https://github.com/JoranAngevaare/optim_esm_tools/pull/172
* Skip check-builtin-literals by @JoranAngevaare in https://github.com/JoranAngevaare/optim_esm_tools/pull/178
* Raise NotImplementedError for deprecated syntax by @JoranAngevaare in https://github.com/JoranAngevaare/optim_esm_tools/pull/179
* Disable full synda test by @JoranAngevaare in https://github.com/JoranAngevaare/optim_esm_tools/pull/183
* Pin few python3.8 requirements by @JoranAngevaare in https://github.com/JoranAngevaare/optim_esm_tools/pull/189
* [pre-commit.ci] pre-commit autoupdate by @pre-commit-ci in https://github.com/JoranAngevaare/optim_esm_tools/pull/173
* [pre-commit.ci] pre-commit autoupdate by @pre-commit-ci in https://github.com/JoranAngevaare/optim_esm_tools/pull/181
* [pre-commit.ci] pre-commit autoupdate by @pre-commit-ci in https://github.com/JoranAngevaare/optim_esm_tools/pull/184
* Bump actions/setup-python from 5.0.0 to 5.1.0 by @dependabot in https://github.com/JoranAngevaare/optim_esm_tools/pull/174
* Bump actions/setup-python from 5.1.0 to 5.2.0 by @dependabot in https://github.com/JoranAngevaare/optim_esm_tools/pull/187


**Full Changelog**: https://github.com/JoranAngevaare/optim_esm_tools/compare/v2.0.0...v2.1.0

2.0.0 / 2024-04-22
------------------
This is an intermediate stage and will be finetuned in the following pull requests

## Major
* Work towards version 3 by @JoranAngevaare in https://github.com/JoranAngevaare/optim_esm_tools/pull/171

## Patch
* Suppress scipy warnings in `rank2d` by @JoranAngevaare in https://github.com/JoranAngevaare/optim_esm_tools/pull/164

* [pre-commit.ci] pre-commit autoupdate by @pre-commit-ci in https://github.com/JoranAngevaare/optim_esm_tools/pull/163
* [pre-commit.ci] pre-commit autoupdate by @pre-commit-ci in https://github.com/JoranAngevaare/optim_esm_tools/pull/166
* [pre-commit.ci] pre-commit autoupdate by @pre-commit-ci in https://github.com/JoranAngevaare/optim_esm_tools/pull/168
* [pre-commit.ci] pre-commit autoupdate by @pre-commit-ci in https://github.com/JoranAngevaare/optim_esm_tools/pull/170
* Bump actions/setup-python from 4.5.0 to 4.7.1 by @dependabot in https://github.com/JoranAngevaare/optim_esm_tools/pull/165
* Bump actions/setup-python from 4.7.1 to 5.0.0 by @dependabot in https://github.com/JoranAngevaare/optim_esm_tools/pull/169


**Full Changelog**: https://github.com/JoranAngevaare/optim_esm_tools/compare/v1.7.0...v2.0.0

1.7.0 / 2023-10-30
------------------
## Minor
* Allow different masks by @JoranAngevaare in https://github.com/JoranAngevaare/optim_esm_tools/pull/152
* `IterStartEnd` method for finding region by @JoranAngevaare in https://github.com/JoranAngevaare/optim_esm_tools/pull/161

## Patch
* Query with constant variable_id by @JoranAngevaare in https://github.com/JoranAngevaare/optim_esm_tools/pull/147
* Repeated testing for `rpy_symmetry` tests by @JoranAngevaare in https://github.com/JoranAngevaare/optim_esm_tools/pull/153
* Sanity checks in time merging during pre-processing by @JoranAngevaare in https://github.com/JoranAngevaare/optim_esm_tools/pull/158
* Update config internals `v0.6.0` by @JoranAngevaare in https://github.com/JoranAngevaare/optim_esm_tools/pull/159
* Find method from `folder_to_dict` in `read_ds` by @JoranAngevaare in https://github.com/JoranAngevaare/optim_esm_tools/pull/160

## Package updates
* Bump actions/checkout from 3 to 4 by @dependabot in https://github.com/JoranAngevaare/optim_esm_tools/pull/151
* [pre-commit.ci] pre-commit autoupdate by @pre-commit-ci in https://github.com/JoranAngevaare/optim_esm_tools/pull/149
* [pre-commit.ci] pre-commit autoupdate by @pre-commit-ci in https://github.com/JoranAngevaare/optim_esm_tools/pull/150
* [pre-commit.ci] pre-commit autoupdate by @pre-commit-ci in https://github.com/JoranAngevaare/optim_esm_tools/pull/156
* [pre-commit.ci] pre-commit autoupdate by @pre-commit-ci in https://github.com/JoranAngevaare/optim_esm_tools/pull/162

## New Contributors
* @dependabot made their first contribution in https://github.com/JoranAngevaare/optim_esm_tools/pull/151

**Full Changelog**: https://github.com/JoranAngevaare/optim_esm_tools/compare/v1.6.3...v1.7.0


1.6.3 / 2023-09-13
------------------
* Merge method to deal with offset time series by @JoranAngevaare in https://github.com/JoranAngevaare/optim_esm_tools/pull/144
* Historical data in plotting routine by @JoranAngevaare in https://github.com/JoranAngevaare/optim_esm_tools/pull/145
* [pre-commit.ci] pre-commit autoupdate by @pre-commit-ci in https://github.com/JoranAngevaare/optim_esm_tools/pull/146
* Fix installation method by @JoranAngevaare in https://github.com/JoranAngevaare/optim_esm_tools/pull/148


**Full Changelog**: https://github.com/JoranAngevaare/optim_esm_tools/compare/v1.6.1...v1.6.2


1.6.0 / 2023-09-06
------------------
* Fix the sign issue for good by @JoranAngevaare in https://github.com/JoranAngevaare/optim_esm_tools/pull/143

#### Notes
* Important bugfix in https://github.com/JoranAngevaare/optim_esm_tools/pull/143 which solves an issue with decreasing trends not being treated the same as increasing trends due to a `np.max` call to signed values.

**Full Changelog**: https://github.com/JoranAngevaare/optim_esm_tools/compare/v1.5.0...v1.6.0


1.5.0 / 2023-09-06
------------------
_bugged release, please use >v1.6.0_
#### Minor change
* Iterative masking in `region_finding.py` to gain control of masked areas by @JoranAngevaare in https://github.com/JoranAngevaare/optim_esm_tools/pull/141

#### Patches
* Fix tests for past py39 by @JoranAngevaare in https://github.com/JoranAngevaare/optim_esm_tools/pull/138
* Fix absolute numbers for tipping criteria by @JoranAngevaare in https://github.com/JoranAngevaare/optim_esm_tools/pull/136
* Update merge variables by @JoranAngevaare in https://github.com/JoranAngevaare/optim_esm_tools/pull/137
* Update coverage by @JoranAngevaare in https://github.com/JoranAngevaare/optim_esm_tools/pull/142


#### Notes
* Important bugfix in https://github.com/JoranAngevaare/optim_esm_tools/pull/136 which solves an issue with decreasing trends not being treated the same as increasing trends due to a `np.max` call to signed values.




**Full Changelog**: https://github.com/JoranAngevaare/optim_esm_tools/compare/v1.4.1...v1.5.0

1.4.1 / 2023-08-28
------------------
* Sourcery refactor suggestions by @JoranAngevaare in https://github.com/JoranAngevaare/optim_esm_tools/pull/132
* Fix singularly valued merged datasets by @JoranAngevaare in https://github.com/JoranAngevaare/optim_esm_tools/pull/133
* Loading with historical dataset by @JoranAngevaare in https://github.com/JoranAngevaare/optim_esm_tools/pull/134
* All sub vars to merged dataset by @JoranAngevaare in https://github.com/JoranAngevaare/optim_esm_tools/pull/135


**Full Changelog**: https://github.com/JoranAngevaare/optim_esm_tools/compare/v1.4.0...1.4.1


1.4.0 / 2023-08-24
------------------
#### Minor change
* Add concise dataframe by @JoranAngevaare in https://github.com/JoranAngevaare/optim_esm_tools/pull/129
* Combine variables in coherent dataset by @JoranAngevaare in https://github.com/JoranAngevaare/optim_esm_tools/pull/130
* Direct tipping masking routine by @JoranAngevaare in https://github.com/JoranAngevaare/optim_esm_tools/pull/131

#### Patches
* reverse dict option for masking coords by @JoranAngevaare in https://github.com/JoranAngevaare/optim_esm_tools/pull/127
* Refactor global mask plot by @JoranAngevaare in https://github.com/JoranAngevaare/optim_esm_tools/pull/128


**Full Changelog**: https://github.com/JoranAngevaare/optim_esm_tools/compare/v1.3.2...v1.4.0


1.3.2 / 2023-08-18
------------------
* Refactor time statistics by @JoranAngevaare in https://github.com/JoranAngevaare/optim_esm_tools/pull/126


**Full Changelog**: https://github.com/JoranAngevaare/optim_esm_tools/compare/v1.3.1...v1.3.2


1.3.1 / 2023-08-16
------------------
* Release `v1.3.0` by @JoranAngevaare in https://github.com/JoranAngevaare/optim_esm_tools/pull/120
* Fix publish using `twine` by @JoranAngevaare in https://github.com/JoranAngevaare/optim_esm_tools/pull/125
* Config fixes by @JoranAngevaare in https://github.com/JoranAngevaare/optim_esm_tools/pull/124


**Full Changelog**: https://github.com/JoranAngevaare/optim_esm_tools/compare/v1.2.1...v1.3.1

1.3.0 / 2023-08-16
------------------
#### Minor change
* Setup using `pyproject.toml` by @JoranAngevaare in https://github.com/JoranAngevaare/optim_esm_tools/pull/122
* Ruptures module for breakpoint testing by @JoranAngevaare in https://github.com/JoranAngevaare/optim_esm_tools/pull/119

#### Patches
* Basic cartopy projection function by @JoranAngevaare in https://github.com/JoranAngevaare/optim_esm_tools/pull/114
* Extend square masks on rounded lon/lat by @JoranAngevaare in https://github.com/JoranAngevaare/optim_esm_tools/pull/115
* Remove temporary R install from workflow by @JoranAngevaare in https://github.com/JoranAngevaare/optim_esm_tools/pull/118
* Average over region before calculating statistics by @JoranAngevaare in https://github.com/JoranAngevaare/optim_esm_tools/pull/116
* Fix error message in `_mask_xr_ds` by @JoranAngevaare in https://github.com/JoranAngevaare/optim_esm_tools/pull/121


**Full Changelog**: https://github.com/JoranAngevaare/optim_esm_tools/compare/v1.2.1...v1.3.0

1.2.1 / 2023-08-10
------------------
* Fix logic finding by @JoranAngevaare in https://github.com/JoranAngevaare/optim_esm_tools/pull/112
* Fix threading issues between rpy2 and dask by @JoranAngevaare in https://github.com/JoranAngevaare/optim_esm_tools/pull/113


**Full Changelog**: https://github.com/JoranAngevaare/optim_esm_tools/compare/v1.2.0...v1.2.1


1.2.0 / 2023-08-09
------------------
#### Minor change
Add `rpy_symmetry` for statistical properties
* `rpy_symmetry` for symmetry test using Mira (1999) by @JoranAngevaare in https://github.com/JoranAngevaare/optim_esm_tools/pull/106
#### Patches
* Fix format HISTORY.md by @JoranAngevaare in https://github.com/JoranAngevaare/optim_esm_tools/pull/98
* Cartopy transform argument by @JoranAngevaare in https://github.com/JoranAngevaare/optim_esm_tools/pull/99
* Off by one in find by @JoranAngevaare in https://github.com/JoranAngevaare/optim_esm_tools/pull/100
* Fix where to detrend by @JoranAngevaare in https://github.com/JoranAngevaare/optim_esm_tools/pull/101
* Add excluded datasets to config by @JoranAngevaare in https://github.com/JoranAngevaare/optim_esm_tools/pull/105
* Fix historical fields for time series statistics - max jump  by @JoranAngevaare in https://github.com/JoranAngevaare/optim_esm_tools/pull/107
* Save statistics with mask by @JoranAngevaare in https://github.com/JoranAngevaare/optim_esm_tools/pull/110
* Different approach to coverage by @JoranAngevaare in https://github.com/JoranAngevaare/optim_esm_tools/pull/109
* Yearly statistics by @JoranAngevaare in https://github.com/JoranAngevaare/optim_esm_tools/pull/108
* Increase coverage after #109 by @JoranAngevaare in https://github.com/JoranAngevaare/optim_esm_tools/pull/111


**Full Changelog**: https://github.com/JoranAngevaare/optim_esm_tools/compare/v1.1.0...v1.2.0


1.1.0 / 2023-07-28
------------------
#### Minor change
Add the required tools to analyze time series and their properties
* Statistical tools for time series by @JoranAngevaare in https://github.com/JoranAngevaare/optim_esm_tools/pull/96

#### Patch
* Fix lon/lat for cluster plot and temp folder for preprocess by @JoranAngevaare in https://github.com/JoranAngevaare/optim_esm_tools/pull/93
* Load intake store from intake by @JoranAngevaare in https://github.com/JoranAngevaare/optim_esm_tools/pull/94
* Plotting flexibility by @JoranAngevaare in https://github.com/JoranAngevaare/optim_esm_tools/pull/95
* Minor search tweak by @JoranAngevaare in https://github.com/JoranAngevaare/optim_esm_tools/pull/97


**Full Changelog**: https://github.com/JoranAngevaare/optim_esm_tools/compare/v1.0.3...v1.1.0


1.0.3 / 2023-07-24
------------------
* Fix clustering fudge factor by @JoranAngevaare in https://github.com/JoranAngevaare/optim_esm_tools/pull/91
* Preprocessing running mean fix by @JoranAngevaare in https://github.com/JoranAngevaare/optim_esm_tools/pull/92


**Full Changelog**: https://github.com/JoranAngevaare/optim_esm_tools/compare/v1.0.2...v1.0.3


1.0.2 / 2023-07-20
------------------
* fix /0 by @JoranAngevaare in https://github.com/JoranAngevaare/optim_esm_tools/pull/89
* Cache region maps by @JoranAngevaare in https://github.com/JoranAngevaare/optim_esm_tools/pull/90
* Clustering bugs by @JoranAngevaare in https://github.com/JoranAngevaare/optim_esm_tools/pull/88


**Full Changelog**: https://github.com/JoranAngevaare/optim_esm_tools/compare/v1.0.0...v1.0.1

1.0.1 / 2023-07-20
------------------
*Dummy release*

1.0.0 / 2023-07-18
------------------
#### major change
* Harmonize preprocessing with `cdo` by @JoranAngevaare in https://github.com/JoranAngevaare/optim_esm_tools/pull/81

#### minor changes
* Plotting routine for masked regions by @JoranAngevaare in https://github.com/JoranAngevaare/optim_esm_tools/pull/82
* Plot region mask by @JoranAngevaare in https://github.com/JoranAngevaare/optim_esm_tools/pull/85
* Add plot maker by @JoranAngevaare in https://github.com/JoranAngevaare/optim_esm_tools/pull/83

#### patches
* Raise error for invalid dates by @JoranAngevaare in https://github.com/JoranAngevaare/optim_esm_tools/pull/84
* [pre-commit.ci] pre-commit autoupdate by @pre-commit-ci in https://github.com/JoranAngevaare/optim_esm_tools/pull/86
* Ready for release v1.0.0 by @JoranAngevaare in https://github.com/JoranAngevaare/optim_esm_tools/pull/87

**Full Changelog**: https://github.com/JoranAngevaare/optim_esm_tools/compare/v0.5.0...v1.0.0


0.5.0 / 2023-07-07
------------------
**Last release before refactoring pre-processing**
*Next release will use CDO tools instead of xMip tools to do pre-processing for reliability reasons*

* Lon and lat coords for clustering by @JoranAngevaare in https://github.com/JoranAngevaare/optim_esm_tools/pull/79
* Queryable area field by @JoranAngevaare in https://github.com/JoranAngevaare/optim_esm_tools/pull/80


**Full Changelog**: https://github.com/JoranAngevaare/optim_esm_tools/compare/v0.4.0...v0.5.0

0.4.0 / 2023-07-07
------------------
*  Technical release related to the setup of the package relating to the config

**Full Changelog**: https://github.com/JoranAngevaare/optim_esm_tools/compare/v0.3.0...v0.4.0


0.3.0 / 2023-07-05
------------------
* Remove old setup by @JoranAngevaare in https://github.com/JoranAngevaare/optim_esm_tools/pull/77
* Area calculation  & percentile recombination & weighted clustering by @JoranAngevaare in https://github.com/JoranAngevaare/optim_esm_tools/pull/76


**Full Changelog**: https://github.com/JoranAngevaare/optim_esm_tools/compare/v0.2.2...v0.3.0


0.2.2 / 2023-07-02
------------------
* Remove todo by @JoranAngevaare in https://github.com/JoranAngevaare/optim_esm_tools/pull/67
* small tas tweaks by @JoranAngevaare in https://github.com/JoranAngevaare/optim_esm_tools/pull/68
* Add non-tas support by @JoranAngevaare in https://github.com/JoranAngevaare/optim_esm_tools/pull/69
* Match fuzzy for version last by @JoranAngevaare in https://github.com/JoranAngevaare/optim_esm_tools/pull/70


**Full Changelog**: https://github.com/JoranAngevaare/optim_esm_tools/compare/v0.2.1...v0.2.2


0.2.1 / 2023-06-21
------------------
* Add few badges (#62) by @JoranAngevarae in https://github.com/JoranAngevaare/optim_esm_tools/pull/62
* Fix few codefactor issues (#63) by @JoranAngevarae in https://github.com/JoranAngevaare/optim_esm_tools/pull/63
* Delete workspace directory (#64) by @JoranAngevarae in https://github.com/JoranAngevaare/optim_esm_tools/pull/64
* simplify find_historical (#65) by @JoranAngevarae in https://github.com/JoranAngevaare/optim_esm_tools/pull/65
* Fix finding (#66) by @JoranAngevarae in https://github.com/JoranAngevaare/optim_esm_tools/pull/66

**Full Changelog**: https://github.com/JoranAngevaare/optim_esm_tools/compare/v0.1.1...v0.2.1

0.2.0 / 2023-06-21
------------------
## What's Changed
* Refactor code - next release by @JoranAngevaare in https://github.com/JoranAngevaare/optim_esm_tools/pull/46
* Add new file tools by @JoranAngevaare in https://github.com/JoranAngevaare/optim_esm_tools/pull/48
* Add timing decorator by @JoranAngevaare in https://github.com/JoranAngevaare/optim_esm_tools/pull/49
* Clustering improvements by @JoranAngevaare in https://github.com/JoranAngevaare/optim_esm_tools/pull/50
* Add download routine from pangeo store by @JoranAngevaare in https://github.com/JoranAngevaare/optim_esm_tools/pull/51
* Add more timed functions by @JoranAngevaare in https://github.com/JoranAngevaare/optim_esm_tools/pull/52
* Split `read_ds` by @JoranAngevaare in https://github.com/JoranAngevaare/optim_esm_tools/pull/53
* stop double test by @JoranAngevaare in https://github.com/JoranAngevaare/optim_esm_tools/pull/54
* Condition and results as proper classes by @JoranAngevaare in https://github.com/JoranAngevaare/optim_esm_tools/pull/55
* Add config for timing tool by @JoranAngevaare in https://github.com/JoranAngevaare/optim_esm_tools/pull/56
* Add minimal install test by @JoranAngevaare in https://github.com/JoranAngevaare/optim_esm_tools/pull/57
* Add region finder by @JoranAngevaare in https://github.com/JoranAngevaare/optim_esm_tools/pull/58
* Fix labels by @JoranAngevaare in https://github.com/JoranAngevaare/optim_esm_tools/pull/59
* Fix issue with labelling by @JoranAngevaare in https://github.com/JoranAngevaare/optim_esm_tools/pull/60
* Configuration tweaks by @JoranAngevaare in https://github.com/JoranAngevaare/optim_esm_tools/pull/61


**Full Changelog**: https://github.com/JoranAngevaare/optim_esm_tools/compare/v0.1.1...v0.2.0


0.1.1 / 2023-06-13
------------------
## What's Changed
* Fix classifier by @JoranAngevaare in https://github.com/JoranAngevaare/optim_esm_tools/pull/30
* Fix renaming properly (#30) by @JoranAngevaare in https://github.com/JoranAngevaare/optim_esm_tools/pull/31
* Add abs. value comparison method by @JoranAngevaare in https://github.com/JoranAngevaare/optim_esm_tools/pull/32
* Improved time series plotting by @JoranAngevaare in https://github.com/JoranAngevaare/optim_esm_tools/pull/33
* Add testing for latex by @JoranAngevaare in https://github.com/JoranAngevaare/optim_esm_tools/pull/34
* Add more precommit hooks by @JoranAngevaare in https://github.com/JoranAngevaare/optim_esm_tools/pull/35
* Add black formatter by @JoranAngevaare in https://github.com/JoranAngevaare/optim_esm_tools/pull/36
* Make moving average configurable by @JoranAngevaare in https://github.com/JoranAngevaare/optim_esm_tools/pull/38
* Activate latex test by @JoranAngevaare in https://github.com/JoranAngevaare/optim_esm_tools/pull/37
* add prints to failing tests by @JoranAngevaare in https://github.com/JoranAngevaare/optim_esm_tools/pull/39
* Complete requirements file by @JoranAngevaare in https://github.com/JoranAngevaare/optim_esm_tools/pull/41
* [pre-commit.ci] pre-commit autoupdate by @pre-commit-ci in https://github.com/JoranAngevaare/optim_esm_tools/pull/40
* Change mlp settings by @JoranAngevaare in https://github.com/JoranAngevaare/optim_esm_tools/pull/42
* Fix defaults for t1-t0 by @JoranAngevaare in https://github.com/JoranAngevaare/optim_esm_tools/pull/43
* Add clustering tools by @JoranAngevaare in https://github.com/JoranAngevaare/optim_esm_tools/pull/44

**Full Changelog**: https://github.com/JoranAngevaare/optim_esm_tools/compare/v0.1.0...v0.1.1


0.1.0  / 2023-06-02
------------------
## What's Changed
* Add first blobs by @JoranAngevaare in https://github.com/JoranAngevaare/optim_esm_tools/pull/1
* Debug synda files by @JoranAngevaare in https://github.com/JoranAngevaare/optim_esm_tools/pull/11
* rename time field by @JoranAngevaare in https://github.com/JoranAngevaare/optim_esm_tools/pull/12
* more tools by @JoranAngevaare in https://github.com/JoranAngevaare/optim_esm_tools/pull/13
* Fix setting kws in loading files by @JoranAngevaare in https://github.com/JoranAngevaare/optim_esm_tools/pull/14
* Tools and tests by @JoranAngevaare in https://github.com/JoranAngevaare/optim_esm_tools/pull/15
* Use mamba by @JoranAngevaare in https://github.com/JoranAngevaare/optim_esm_tools/pull/17
* Upgrade CI/CD by @JoranAngevaare in https://github.com/JoranAngevaare/optim_esm_tools/pull/19
* Flexible loading by @JoranAngevaare in https://github.com/JoranAngevaare/optim_esm_tools/pull/16
* Initialize testing by @JoranAngevaare in https://github.com/JoranAngevaare/optim_esm_tools/pull/20
* Add test for synda viewer by @JoranAngevaare in https://github.com/JoranAngevaare/optim_esm_tools/pull/28
* fix lat lon handling by @JoranAngevaare in https://github.com/JoranAngevaare/optim_esm_tools/pull/29

**Full Changelog**: https://github.com/JoranAngevaare/optim_esm_tools/commits/v0.1.0
