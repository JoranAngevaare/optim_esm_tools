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

    def calculate_statistics(self) -> ty.Dict[str, ty.Optional[float]]:
        """
        For a given dataset calculate the statistical properties of the dataset based on three tests:
            1. The standard deviation w.r.t. the standard

        Returns:
            ty.Dict[ty.Optional[float]]: _description_
        """
        functions = dict(
            n_sigma_historical=calculate_historical_std,
            p_skewness=calculate_skewtest,
            p_dip=calculate_dip_test,
        )
        return {
            k: partial(f, **self.calculation_kwargs.get(k, {}))(self.data_set)
            for k, f in functions.items()
        }


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

    values = get_values_from_data_set(ds, field)

    _, pval = diptest.diptest(values, boot_pval=False)
    return pval


def calculate_skewtest(ds, field=None):
    import scipy

    values = get_values_from_data_set(ds, field, add='')
    return scipy.stats.skewtest(values, nan_policy='omit').pvalue


def calculate_historical_std(ds, field='std detrended', **kw):
    ds_hist = get_historical_ds(ds, **kw)
    if ds_hist is None:
        return None
    mask = get_mask_from_global_mask(ds)
    ds_hist_masked = oet.analyze.xarray_tools.mask_xr_ds(ds_hist, mask, drop=True)
    assert (
        ds[field].shape == ds_hist_masked[field].shape
    ), f'{ds[field].shape} != {ds_hist_masked[field].shape}'
    cur = ds[field].values
    his = ds_hist_masked[field].values
    isnnan = np.isnan(cur) | np.isnan(his)
    cur = cur[~isnnan]
    his = his[~isnnan]

    return np.median(cur / his)
