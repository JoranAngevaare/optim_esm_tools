import numpy as np

import optim_esm_tools as oet


def pass_test(props, thresholds, always_true=('max_jump', 'n_std_global')):
    thresholds = thresholds.copy()
    for k in always_true:
        op, thr = thresholds[k]
        if not op(props.pop(k), thr):
            return False  # pragma: no cover

    for k, v in props.items():
        if k in thresholds:
            operator, thr = thresholds[k]
            t = f'{v} {operator} {thr}'
            if operator(v, thr):
                return True
    return False  # pragma: no cover


def change_global_mask(
    ds,
    lat_i,
    lon_i,
):
    m = ds['global_mask'].copy()
    idx_lon = np.argwhere(m['lon_mask'].values == ds['lon'].values[lon_i])[0][0]
    idx_lat = np.argwhere(m['lat_mask'].values == ds['lat'].values[lat_i])[0][0]
    m[:] = 0
    m[idx_lat, idx_lon] = 1
    ds['global_mask'] = m
    return ds


def direct_test(ds, _ds_global=None, _ds_hist=None, over_ride_thresholds=None):
    ds = ds.copy().load()
    over_ride_thresholds = over_ride_thresholds or {}
    thresholds = oet.analyze.time_statistics.default_thresholds(**over_ride_thresholds)
    ds_global = _ds_global or oet.analyze.time_statistics._get_ds_global(ds, load=True)
    ds_hist = _ds_hist or oet.analyze.time_statistics.get_historical_ds(ds, load=True)
    var = ds.attrs['variable_id']
    index_2d = np.array(
        np.meshgrid(np.arange(len(ds['lat'])), np.arange(len(ds['lon']))),
    ).T
    index_2d = index_2d.reshape(np.prod(index_2d.shape) // 2, 2)
    bool_mask = np.zeros((len(ds['lat']), len(ds['lon'])), dtype=np.bool_)
    masks = {'direct_test': bool_mask.copy()}
    for lat_i, lon_i in oet.utils.tqdm(index_2d):
        ds_sel = change_global_mask(
            ds,
            lat_i,
            lon_i,
        )
        ds_sel = ds_sel.isel(lat=slice(lat_i, lat_i + 1), lon=slice(lon_i, lon_i + 1))
        if float(np.sum(~ds_sel[var].isnull())):
            calculator = oet.analyze.time_statistics.TimeStatistics(
                ds_sel,
                calculation_kwargs=dict(
                    max_jump=dict(_ds_hist=ds_hist),
                    max_jump_yearly=dict(_ds_hist=ds_hist),
                    n_std_global=dict(_ds_global=ds_global),
                ),
            )
            calculator.functions = {
                k: oet.utils.timed(v)
                for k, v in calculator.functions.items()
                if k not in ['max_jump', 'max_jump_yearly', 'p_skewness']
            }
            props = calculator.calculate_statistics()
            for k, v in props.items():
                if v is None:
                    continue  # pragma: no cover
                if k not in masks:
                    masks[k] = bool_mask.copy().astype(type(v))
                    if isinstance(v, float):
                        masks[k][:] = np.nan
                masks[k][lat_i, lon_i] = v

            if res := pass_test(
                props,
                thresholds=thresholds,
                always_true=('n_std_global',),
            ):
                jump = oet.analyze.time_statistics.calculate_max_jump_in_std_history(
                    ds=ds,
                    _ds_hist=ds_hist,
                )
                operator, thr = thresholds['max_jump']
                res = operator(jump, thr)  # type: ignore
                masks['direct_test'][lat_i, lon_i] = res

    for k, mask in masks.items():
        ds[k] = (('lat', 'lon'), mask)
    return ds
