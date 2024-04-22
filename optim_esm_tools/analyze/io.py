import os

import xarray as xr

from optim_esm_tools.utils import add_load_kw


@add_load_kw
def load_glob(
    pattern: str,
    **kw,
) -> xr.Dataset:
    """Load cmip dataset from provided pattern.

    Args:
        pattern (str): Path where to load the data from

    Returns:
        xr.Dataset: loaded from pattern
    """
    if not os.path.exists(pattern):
        raise FileNotFoundError(f'{pattern} does not exists')  # pragma: no cover
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
        return xr.open_mfdataset(pattern, **kw)
    except ValueError as e:  # pragma: no cover
        raise ValueError(f'Fatal error while reading {pattern}') from e
