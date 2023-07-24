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
**major change**
* Harmonize preprocessing with `cdo` by @JoranAngevaare in https://github.com/JoranAngevaare/optim_esm_tools/pull/81

**minor changes**
* Plotting routine for masked regions by @JoranAngevaare in https://github.com/JoranAngevaare/optim_esm_tools/pull/82
* Plot region mask by @JoranAngevaare in https://github.com/JoranAngevaare/optim_esm_tools/pull/85
* Add plot maker by @JoranAngevaare in https://github.com/JoranAngevaare/optim_esm_tools/pull/83

**patches**
* Raise error for invalid dates by @JoranAngevaare in https://github.com/JoranAngevaare/optim_esm_tools/pull/84
* [pre-commit.ci] pre-commit autoupdate by @pre-commit-ci in https://github.com/JoranAngevaare/optim_esm_tools/pull/86
* Ready for release v1.0.0 by @JoranAngevaare in https://github.com/JoranAngevaare/optim_esm_tools/pull/87

**Full Changelog**: https://github.com/JoranAngevaare/optim_esm_tools/compare/v0.5.0...v1.0.0


0.5.0 / 2023-07-07
------------------
**Last release before refactoring pre-processing**
*Next release will use CDO tools instead of xMip tools to do pre-processing for reliability reasons*

* Lon and lat coords for clustering by @JoranAngevaare in https://github.com/JoranAngevaare/optim_esm_tools/pull/79
* Queriable area field by @JoranAngevaare in https://github.com/JoranAngevaare/optim_esm_tools/pull/80


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
