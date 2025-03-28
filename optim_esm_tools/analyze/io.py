import os

import xarray as xr

from optim_esm_tools.utils import add_load_kw
from optim_esm_tools.config import config
from optim_esm_tools.analyze.inferred_variable_field import inferred_fields_to_dataset


@add_load_kw
def load_glob(
    pattern: str,
    add_inferred_fields=None,
    field_kw=None,
    **kw,
) -> xr.Dataset:
    """Load cmip dataset from provided pattern.

    Args:
        pattern (str): Path where to load the data from

    Returns:
        xr.Dataset: loaded from pattern
    """
    if add_inferred_fields is None:
        add_inferred_fields = config['versions']['cmip_handler'] in pattern

    if not os.path.exists(pattern):
        raise FileNotFoundError(f'"{pattern}" does not exists')  # pragma: no cover
    for k, v in dict(
        use_cftime=True,
        concat_dim='time',
        combine='nested',
        data_vars='minimal',
        coords='minimal',
        compat='override',
        decode_times=True,
    ).items():
        kw.setdefault(k, v)
    try:
        ds = xr.open_mfdataset(pattern, **kw)
    except ValueError as e:  # pragma: no cover
        raise ValueError(f'Fatal error while reading {pattern}') from e

    if add_inferred_fields:
        field_kw = field_kw or dict()
        ds = inferred_fields_to_dataset(ds, **field_kw)
    return ds
