import optim_esm_tools as oet
import numpy as np
import os


def add_area_to_ds(
    ds,
    path=None,
    method=('local', 'pangeo', 'local_fuzzy', 'pangeo_fuzzy'),
    area_field='cell_area',
    compare_field='lat',
    _external_variable=None,
):
    path = path or ds.attrs.get('path')

    if 'external_variables' not in ds.attrs and not _external_variable:
        oet.config.get_logger().warning('This ds has no external variable!')
        for _ext_v in 'areacello', 'areacella':
            try:
                return add_area_to_ds(
                    ds,
                    path,
                    method,
                    area_field,
                    compare_field,
                    _external_variable=_ext_v,
                )
            except NoMatchFoundError:
                pass
        raise NoMatchFoundError('No matches')
    variable_id = ds.attrs.get('external_variables') or _external_variable
    method = oet.utils.to_str_tuple(method)

    for met in method:
        try:
            res = _get_area(
                path,
                method=met,
                ds=ds,
                variable_id=variable_id,
                compare_field=compare_field,
            )
        except NoMatchFoundError:
            continue
        else:
            if res is not None:
                area, attrs = res
                break
    else:
        raise NoMatchFoundError('No mathces for this dataset - no area can be inferred')

    if ds.lon.shape != area.shape:
        raise ValueError(
            f'area_from_path returned wrong shape {ds["compare_field"].shape}, {area.shape}, {path}'
        )
    ds[area_field] = (ds.lon.dims, area, attrs)
    return ds


@oet.utils.check_accepts(
    accepts=dict(
        method=('local', 'pangeo', 'local_fuzzy', 'pangeo_fuzzy', 'brute_force')
    )
)
def _get_area(path, method, **kw):
    from functools import partial

    # Strip source_id, hope that institution_id will somehow work
    fuzzy_kw = dict(match_keys=('grid_label', 'institution_id'))
    func = dict(
        local=area_from_local,
        pangeo=area_pangeo_from_path,
        local_fuzzy=partial(area_from_local, **fuzzy_kw),
        pangeo_fuzzy=partial(area_pangeo_from_path, **fuzzy_kw),
        brute_force=area_brute_force,
    )
    log = oet.config.get_logger()
    if 'fuzzy' in method:
        log.warning('fuzzy matching is risky')
    if method == 'brute_force':
        log.warning('Brute force calculating grid only "works" for regular grids')
    return func[method](path, **kw)


def area_brute_force(path, ds, **kw):
    # ds = oet.analyze.io.load_glob(path)
    # ds = oet.analyze.io.recast(ds)
    kw.pop('target_shape')
    kw.pop('variable_id')
    kw.pop('compare_field')
    return oet.analyze.calculate_metric.calucluate_grid(ds, **kw), dict(
        name='Cell area m$^2$', units='m$^2$'
    )


def area_pangeo_from_path(path, **kw):
    meta = oet.analyze.find_matches.folder_to_dict(path)
    return area_from_pangeo(meta, **kw)


def area_from_local(
    path,
    variable_id=None,
    match_keys=('grid_label', 'source_id', 'institution_id'),
    required_file='merged.nc',
    **kw,
):
    variable_id = variable_id or ['areacella', 'areacello']
    variable_id = oet.utils.to_str_tuple(variable_id)
    meta = oet.analyze.find_matches.folder_to_dict(path)
    base = oet.analyze.find_matches.base_from_path(path)
    search = {k: (v if k in match_keys else '*') for k, v in meta.items()}

    for var in variable_id:
        search.update(
            dict(
                base=base,
                max_versions=None,
                variable_id=var,
                max_members=None,
                # I think this is right, but have to check if we really merge
                required_file=required_file,
            )
        )
        for match in sorted(oet.analyze.find_matches.find_matches(**search)):
            ds = oet.analyze.io.load_glob(os.path.join(match, required_file))
            ds = oet.analyze.io.recast(ds)
            result = return_area_and_attr_if_match(ds, **kw, variable_id=var)
            if result is not None:
                return result
    return None


def area_from_pangeo(
    meta,
    variable_id=None,
    match_keys=('grid_label', 'source_id', 'institution_id'),
    **kw,
):
    log = oet.config.get_logger()

    search = {k: v for k, v in meta.items() if k in match_keys}
    variable_id = variable_id or ['areacella', 'areacello']
    variable_id = oet.utils.to_str_tuple(variable_id)
    for var in variable_id:
        log.info(f'Looking for {var}')
        area = yield_search(search, variable_id=var, **kw)
        if area is not None:
            return area
    raise NoMatchFoundError(f'No matches for {meta}')


def yield_search(search, variable_id, **kw):
    from xmip.utils import google_cmip_col
    from xmip.preprocessing import combined_preprocessing

    col = google_cmip_col()

    d = col.search(**search, variable_id=variable_id)
    if not len(d):
        return None

    kwargs = {
        'xarray_open_kwargs': {'consolidated': True, 'use_cftime': True},
        'aggregate': False,
        'preprocess': combined_preprocessing,
    }
    for member in np.unique(d.df['member_id'].values):
        sub_d = d.search(**search, member_id=member, variable_id=variable_id)
        for ds_candidate in sub_d.to_dataset_dict(**kwargs).values():
            if variable_id not in ds_candidate:
                oet.config.get_logger().warning(
                    f'Search {search} for var {variable_id} did not yield {variable_id}'
                    f'only {ds_candidate.dims}'
                )
                continue
            result = return_area_and_attr_if_match(
                ds_other=ds_candidate, variable_id=variable_id, **kw
            )
            if result is not None:
                return result
    oet.config.get_logger().warning('No matches!')
    return None


def return_area_and_attr_if_match(ds_other, ds, variable_id, compare_field='lat'):
    area = ds_other[variable_id].values.squeeze()
    log = oet.config.get_logger()
    target_shape = ds[compare_field].shape
    if area.shape == target_shape:
        return area, ds_other[variable_id].attrs

    log.info('Retry transposed!')
    area = area.T
    if area.shape == target_shape:
        return area, ds_other[variable_id].attrs

    # Also log later, such that we increase verbosity if exact match also fails
    search_message = f'No match! {area.shape} != {target_shape}'
    log.debug(search_message)

    area = exact_match(
        ds_other, ds, compare_field=compare_field, variable_id=variable_id
    )
    if area is not None:
        log.info(search_message)
        log.info('Exact match found!')
        return area, ds_other[variable_id].attrs
    log.warning(search_message)
    log.warning('Exact match also did not yield results')


def exact_match(
    ds_other, ds, compare_field='lat', variable_id='areacello', do_raise=True
):
    log = oet.config.get_logger()
    need_dims = ds[compare_field].dims
    log.debug(f'Look for {need_dims}')
    if len(need_dims) != 2:
        raise ValueError(
            f'Got {need_dims} for {compare_field}, this needs to be 2 dims!'
        )

    mask_0 = np.in1d(ds_other[need_dims[0]], ds[need_dims[0]])
    mask_1 = np.in1d(ds_other[need_dims[1]], ds[need_dims[1]])

    equal_0 = np.array_equal(
        ds[need_dims[0]].values, ds_other[need_dims[0]].values[mask_0]
    )
    equal_1 = np.array_equal(
        ds[need_dims[1]].values, ds_other[need_dims[1]].values[mask_1]
    )

    if equal_0 + equal_1 != 2:
        if do_raise:
            is_wrong = [d for i, d in enumerate(need_dims) if not [equal_0, equal_1][i]]
            log.warning(
                f'Got inconsistent dimention(s) {is_wrong}. See {ds[compare_field].shape}, {ds_other[variable_id].shape}'
            )
            # raise ValueError(f'Got inconsistent dimention(s) {is_wrong}')
        return

    orig_area = ds_other[variable_id].values
    log.debug(f'Start with {orig_area.shape}. Now squeeze')
    reshaped_area = orig_area.squeeze()
    log.debug(
        f'Squeezed to {reshaped_area.shape}, now mask {need_dims[0]} ({len(mask_0)}) and transpose'
    )
    reshaped_area = reshaped_area[mask_0].T
    log.debug(
        f'Masked to {reshaped_area.shape}, now mask {need_dims[1]} ({len(mask_1)}) and transpose'
    )
    reshaped_area = reshaped_area[mask_1].T
    log.debug(f'Masked to {reshaped_area.shape}')

    if reshaped_area.shape != ds[compare_field].shape:
        if do_raise:
            raise ValueError(
                f'Inconcsitent shapes {reshaped_area.shape, ds[compare_field].shape}'
            )
        return
    return reshaped_area


class NoMatchFoundError(ValueError):
    """Error raised when no match is found"""
