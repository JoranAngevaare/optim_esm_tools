import operator
import os
import typing as ty
from functools import partial

import numpy as np
import scipy
import xarray as xr

import optim_esm_tools as oet


def calculate_dip_test(
    ds: ty.Optional[xr.Dataset] = None,
    field: ty.Optional[str] = None,
    values: ty.Optional[np.ndarray] = None,
    nan_policy: str = 'omit',
):
    """[citation]:

    Hartigan, P. M. (1985). Computation of the Dip Statistic to Test for Unimodality.
    Journal of the Royal Statistical Society. Series C (Applied Statistics), 34(3),
    320-325.
    Code from:
    https://pypi.org/project/diptest/
    """
    values = _extract_values_from_sym_args(values, ds, field, nan_policy)
    import diptest

    if len(values) < 3:  # pragma: no cover
        # At least 3 samples are needed
        oet.config.get_logger().error('Dataset too short for diptest')
        return None
    _, pval = diptest.diptest(values, boot_pval=False)
    return pval


def calculate_skewtest(
    ds: ty.Optional[xr.Dataset] = None,
    field: ty.Optional[str] = None,
    values: ty.Optional[np.ndarray] = None,
    nan_policy: str = 'omit',
):
    """[citation] R.

    B. D'Agostino, A. J. Belanger and R. B. D'Agostino Jr., "A suggestion for using
    powerful and informative tests of normality", American Statistician 44, pp.
    316-321, 1990.
    Code from:
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.skewtest.html
    """
    values = _extract_values_from_sym_args(values, ds, field, nan_policy)
    if sum(~np.isnan(values)) < 8:  # pragma: no cover
        # At least 8 samples are needed
        oet.config.get_logger().error('Dataset too short for skewtest')
        return None
    return scipy.stats.skewtest(values, nan_policy=nan_policy).pvalue


def _extract_values_from_sym_args(
    values: ty.Optional[np.ndarray] = None,
    ds: ty.Optional[xr.Dataset] = None,
    field: ty.Optional[str] = None,
    nan_policy: str = 'omit',
) -> np.ndarray:
    _ds_args_are_none = ds is None and field is None
    if values is not None:
        if not _ds_args_are_none:
            raise TypeError(
                f'Got both values, dataset and field. Either provide values or ds and field.',
            )
        if not isinstance(values, np.ndarray):
            try:
                values = values.values
            except Exception as error:
                raise TypeError(
                    f'values should be np.ndarray, got {type(values)}',
                ) from error
        if not len(values.shape) == 1:
            raise TypeError(f'values have wrong shape {values.shape}, should be 1D')

    else:
        if _ds_args_are_none:
            raise TypeError('No ds is provided or field is missing!')

        da = ds[field]
        da = da.mean(set(da.dims) - {'time'})
        values = da.values

    if nan_policy == 'omit':
        values = values[~np.isnan(values)]
    else:  # pragma: no cover
        message = 'Not sure how to deal with nans other than omit'
        raise NotImplementedError(message)

    return values


def calculate_symmetry_test(
    ds: ty.Optional[xr.Dataset] = None,
    field: ty.Optional[str] = None,
    values: ty.Optional[np.ndarray] = None,
    nan_policy: str = 'omit',
    test_statistic: str = 'MI',
    n_repeat: int = int(oet.config.config['analyze']['n_repeat_sym_test']),
    _fast_mode: bool = True,
    _fast_above: float = 0.05,
    _fast_min_repeat: int = 2,
    **kw,
) -> np.float64:
    """The function `calculate_symmetry_test` calculates the symmetry test
    statistic for a given dataset and field using the R package `rpy_symmetry`.

    :param values: A numpy array with the values to calculate the diptest on. Should be 1D, and `ds` and
    `field` should be None
    :param ds: An xarray Dataset containing the data
    :type ds: xr.Dataset
    :param field: The `field` parameter is an optional string that specifies the field or variable from
    the dataset (`ds`) that you want to calculate symmetry test for. If `field` is not provided, the
    function will calculate the symmetry test for all variables in the dataset
    :type field: ty.Optional[str]
    :param nan_policy: The `nan_policy` parameter determines how to handle missing values (NaNs) in the
    data. The default value is 'omit', which means that any NaN values will be excluded from the
    calculation, defaults to omit
    :type nan_policy: str (optional)
    :param test_statistic: The `test_statistic` parameter is a string that specifies the test statistic
    to be used in the symmetry test. It determines how the symmetry of the data will be measured,
    defaults to MI
    :type test_statistic: str (optional)
    :param n_repeat: The parameter `n_repeat` specifies the number of times the symmetry test should be
    repeated. The symmetry test in R does give non-deterministic results. As such repeat a test this
    many times and take the average
    :type n_repeat: int
    :param _fast_mode: if `_fast_mode` is activated only run the test `_fast_min_repeat` times if the first try is above `_fast_above`
    :param _fast_above: if `_fast_mode` is activated only run the test `_fast_min_repeat` times if the first try is above `_fast_above`
    :param _fast_min_repeat: if `_fast_mode` is activated only run the test `_fast_min_repeat` times if the first try is above `_fast_above`

    :return: The function `calculate_symmetry_test` returns a `np.float64` value.

    [citation]:
        Mira A (1999) Distribution-free test for symmetry based on Bonferroni's measure.
        J Appl Stat 26(8):959–972. https://doi.org/10.1080/02664769921963
        Code from:
        https://cran.r-project.org/web/packages/symmetry
        Code at:
        https://github.com/JoranAngevaare/rpy_symmetry
    """
    import rpy_symmetry as rsym

    values = _extract_values_from_sym_args(values, ds, field, nan_policy)

    results = [rsym.p_symmetry(values, test_statistic=test_statistic, **kw)]
    if _fast_mode:
        n_repeat = n_repeat - 1 if results[0] < _fast_above else _fast_min_repeat - 1
    for _ in range(n_repeat):
        if len(results) > 3 and np.std(results) <= np.mean(results) / 10:
            break
        results.append(rsym.p_symmetry(values, test_statistic=test_statistic, **kw))
    oet.get_logger().debug(
        f'Evaluated {test_statistic} {len(results)} times: {results}',
    )
    return np.mean(results)


def calculate_n_breaks(
    ds: ty.Optional[xr.Dataset] = None,
    field: ty.Optional[str] = None,
    values: ty.Optional[np.ndarray] = None,
    nan_policy: str = 'omit',
    penalty: ty.Optional[float] = None,
    min_size: ty.Optional[int] = None,
    jump: ty.Optional[int] = None,
    model: ty.Optional[str] = None,
    method: ty.Optional[str] = None,
):
    """[citation] C. Truong, L. Oudre, N. Vayatis. Selective review of offline change point detection
    methods. Signal Processing, 167:107299, 2020.
    Code from:
    https://centre-borelli.github.io/ruptures-docs/
    """
    import ruptures as rpt

    values = _extract_values_from_sym_args(values, ds, field, nan_policy)

    penalty = penalty or float(oet.config.config['analyze']['rpt_penalty'])
    min_size = min_size or int(oet.config.config['analyze']['rpt_min_size'])
    jump = jump or int(oet.config.config['analyze']['rpt_jump'])
    model = model or oet.config.config['analyze']['rpt_model']
    method = method or oet.config.config['analyze']['rpt_method']

    if len(values) < min_size:  # pragma: no cover
        return None

    algorithm = getattr(rpt, method)(model=model, min_size=min_size, jump=jump)
    fit = algorithm.fit(values)

    return len(fit.predict(pen=penalty)) - 1
