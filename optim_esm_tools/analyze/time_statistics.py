import optim_esm_tools as oet
import numpy as np
import xarray as xr
import typing as ty
from functools import partial


class TimeStatistics:
    calculation_kwargs: ty.Mapping = None

    def __init__(self, data_set: xr.Dataset, calculation_kwargs=None) -> None:
        self.data_set = data_set
        self.calculation_kwargs = calculation_kwargs or dict()
        self.functions = self.default_calculations()
        if any(k not in self.functions for k in self.calculation_kwargs):
            bad = set(self.calculation_kwargs.keys()) - set(self.functions.keys())
            message = f'One or more of {bad} are not used by any function'
            raise ValueError(message)

    def default_calculations(self) -> ty.Mapping:
        return dict(
            max_jump=calculate_max_jump_in_std_vs_history,
            p_skewness=calculate_skewtest,
            p_dip=calculate_dip_test,
            p_symmetry=calculate_symmetry_test,
            n_std_global=n_times_global_std,
        )

    def calculate_statistics(self) -> ty.Dict[str, ty.Optional[float]]:
        """
        For a given dataset calculate the statistical properties of the dataset based on three tests:
            1. The max 10-year jump w.r.t. the standard deviation of the piControl run. Based on
                yearly means
            2. The p-value of the "dip test" [1]
            3. The p-value of the Skewness test [2]
            4. The p-value fo the symmetry test [3]
            5. The fraction of the selected regions standard-deviation w.r.t. to the standard
                deviation of the global average standard-deviation. Yearly means

        Citations:
            [1]:
                Hartigan, P. M. (1985). Computation of the Dip Statistic to Test for Unimodality.
                Journal of the Royal Statistical Society. Series C (Applied Statistics), 34(3),
                320-325.
                Code from:
                https://pypi.org/project/diptest/
            [2]:
                R. B. D'Agostino, A. J. Belanger and R. B. D'Agostino Jr., "A suggestion for using
                powerful and informative tests of normality", American Statistician 44, pp.
                316-321, 1990.
                Code from:
                https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.skewtest.html
            [3]:
                Mira A (1999) Distribution-free test for symmetry based on Bonferroni's measure.
                J Appl Stat 26(8):959–972. https://doi.org/10.1080/02664769921963
                Code from:
                https://cran.r-project.org/web/packages/symmetry
                Code at:
                https://github.com/JoranAngevaare/rpy_symmetry

        Returns:
            ty.Dict[ty.Optional[float]]: Mapping of test to result value
        """
        return {
            k: partial(f, **self.calculation_kwargs.get(k, {}))(self.data_set)
            for k, f in self.functions.items()
        }


def n_times_global_std(ds, field='std detrended', average_over=None, **read_kw):
    average_over = average_over or oet.config.config['analyze']['lon_lat_dim'].split(
        ','
    )
    ds_global = oet.load_glob(ds.attrs['file'], **read_kw)
    return float(ds[field].mean(average_over) / ds_global[field].mean(average_over))


def get_mask_from_global_mask(ds, mask_key='global_mask', rename_dict=None):
    """Load the global mask and rename it's dims to the original ones"""
    mapping = oet.analyze.xarray_tools.default_rename_mask_dims_dict()
    inverse_mapping = {v: k for k, v in mapping.items()}
    rename_dict = rename_dict or inverse_mapping
    mask = ds[mask_key].copy()
    mask = mask.rename(rename_dict)
    return mask


def get_historical_ds(ds, _file_name=None, **kw):
    find = oet.analyze.find_matches.associate_historical
    find_kw = oet.utils.filter_keyword_arguments(kw, find, allow_varkw=False)
    read_kw = oet.utils.filter_keyword_arguments(kw, oet.read_ds, allow_varkw=False)
    if _file_name is not None:
        find_kw['search_kw'] = dict(required_file=_file_name)
        read_kw['_file_name'] = _file_name
    try:
        hist_path = oet.analyze.find_matches.associate_historical(
            path=ds.attrs['path'], **find_kw
        )
    except RuntimeError as e:
        print(e)
        return
    read_kw.setdefault('max_time', None)
    read_kw.setdefault('min_time', None)
    hist_ds = oet.read_ds(hist_path[0], **read_kw)
    return hist_ds


def get_values_from_data_set(ds, field, add='_detrend'):
    if field is None:
        field = ds.attrs['variable_id'] + add
    da = ds[field]
    da = da.mean(set(da.dims) - {'time'})
    return da.values


def calculate_dip_test(ds, field=None):
    import diptest

    values = get_values_from_data_set(ds, field, add='')

    _, pval = diptest.diptest(values, boot_pval=False)
    return pval


def calculate_skewtest(ds, field=None, nan_policy='omit'):
    import scipy

    values = get_values_from_data_set(ds, field, add='')
    if sum(~np.isnan(values)) < 8:
        # At least 8 samples are needed
        oet.config.get_logger().error('Dataset too short for skewtest')
        return None
    return scipy.stats.skewtest(values, nan_policy=nan_policy).pvalue


def calculate_symmetry_test(ds, field=None, nan_policy='omit'):
    import rpy_symmetry as rsym

    values = get_values_from_data_set(ds, field, add='')
    if nan_policy == 'omit':
        values = values[~np.isnan(values)]
    else:
        raise NotImplementedError('Not sure how to deal with nans other than omit')
    return rsym.p_symmetry(values)


def calculate_max_jump_in_std_vs_history(
    ds, field='max jump yearly', field_pi_control='std detrended yearly', **kw
):
    ds_hist = get_historical_ds(ds, **kw)
    if ds_hist is None:
        return None
    mask = get_mask_from_global_mask(ds)
    ds_hist_masked = oet.analyze.xarray_tools.mask_xr_ds(ds_hist, mask, drop=True)
    da = ds[field]
    da_hist = ds_hist_masked[field_pi_control]
    assert da.shape == da_hist.shape, f'{da.shape} != {da_hist.shape}'
    cur = da.values
    his = da_hist.values
    isnnan = np.isnan(cur) | np.isnan(his)
    cur = cur[~isnnan]
    his = his[~isnnan]

    return np.mean(cur) / np.mean(his)
