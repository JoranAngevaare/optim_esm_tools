import typing as ty

import matplotlib.pyplot as plt
import xarray as xr

import optim_esm_tools as oet
from optim_esm_tools.config import config
from optim_esm_tools.config import get_logger


def setup_map(
    projection: ty.Optional[str] = None,
    coastlines: bool = True,
    add_features: bool = False,
    add_gridlines: bool = True,
    coast_line_kw: ty.Optional[dict] = None,
    gridline_kw: ty.Optional[dict] = None,
    no_top_labels: bool = True,
    **projection_kwargs,
):
    plt.gcf().add_subplot(
        projection=get_cartopy_projection(projection, **projection_kwargs),
    )
    ax = plt.gca()
    if coastlines:
        coast_line_kw = coast_line_kw or dict()
        ax.coastlines(**coast_line_kw)
    if add_features:
        import cartopy.feature as cfeature

        allowed = 'LAND OCEAN COASTLINE BORDERS LAKES RIVERS'.split()
        for feat in oet.utils.to_str_tuple(add_features):
            assert feat.upper() in allowed, f'{feat} not in {allowed}'
            ax.add_feature(getattr(cfeature, feat.upper()))
    if add_gridlines:
        gridline_kw = gridline_kw or dict(draw_labels=True)
        gl = ax.gridlines(**gridline_kw)
        if no_top_labels:
            gl.top_labels = False


def _show(show):
    if show:
        plt.show()  # pragma: no cover
    else:
        plt.clf()
        plt.close()


def default_variable_labels() -> ty.Dict[str, str]:
    labels = dict(config['variable_label'].items())
    ma = config['analyze']['moving_average_years']
    for k, v in list(labels.items()):
        labels[f'{k}_detrend'] = f'Detrend {v}'
        labels[f'{k}_run_mean_{ma}'] = f'$RM_{{{ma}}}$ {v}'
        labels[f'{k}_detrend_run_mean_{ma}'] = f'Detrend $RM_{{{ma}}}$ {v}'
    return labels


def get_range(var: str) -> ty.List[float]:
    r = (
        dict(oet.config.config['variable_range'].items())
        .get(var, 'None,None')
        .split(',')
    )
    return [(float(l) if l != 'None' else None) for l in r]


def set_y_lim_var(var: str) -> None:
    d, u = get_range(var)
    cd, cu = plt.ylim()
    plt.ylim(
        cd if d is None else min(cd, d),
        cu if u is None else max(cu, u),
    )


def get_unit_da(da: xr.DataArray) -> str:
    return da.attrs.get('units', '?').replace('%', r'\%')


def get_unit(ds: xr.Dataset, var: str) -> str:
    return get_unit_da(ds[var])


def get_cartopy_projection(
    projection: ty.Optional[ty.Any] = None,
    _field: str = 'projection',
    **projection_kwargs,
) -> ty.Any:
    import cartopy.crs as ccrs

    projection = projection or config['cartopy'][_field]
    if not hasattr(ccrs, projection):
        raise ValueError(f'Invalid projection {projection}')  # pragma: no cover
    return getattr(ccrs, projection)(**projection_kwargs)


def get_cartopy_transform(
    projection: ty.Optional[ty.Any] = None,
    **projection_kwargs,
) -> ty.Any:
    return get_cartopy_projection(
        projection=projection,
        _field='transform',
        **projection_kwargs,
    )


def get_xy_lim_for_projection(
    projection: ty.Optional[str] = None,
) -> ty.Tuple[ty.Tuple[float, float], ty.Tuple[float, float]]:
    """Blunt hardcoding for the different projections.

    Calling plt.xlim(0, 360) will have vastly different outcomes
    depending on the projection used. Here we hardcoded some of the more
    common.
    """
    projection = projection or config['cartopy']['projection']
    lims = dict(
        Robinson=(
            (-17005833.33052523, 17005833.33052523),
            (-8625154.6651, 8625154.6651),
        ),
        EqualEarth=(
            (-17243959.06221695, 17243959.06221695),
            (-8392927.59846645, 8392927.598466456),
        ),
        Mollweide=(
            (-18040095.696147293, 18040095.696147293),
            (-9020047.848073646, 9020047.848073646),
        ),
        PlateCarree=((0, 360), (-90, 90)),
    )
    if projection not in lims:
        get_logger().warning(
            f'No hardcoded x/y lims for {projection}, might yield odd figures.',
        )  # pragma: no cover
    return lims.get(projection, ((0, 360), (-90, 90)))


def plot_da(
    da: xr.DataArray,
    projection: ty.Optional[str] = None,
    setup_kw: ty.Optional[dict] = None,
    **kw,
):
    """Simple wrapper for da.plot() with correct transforms and projections."""
    setup_kw = setup_kw or dict()
    setup_map(projection=projection, **setup_kw)
    da.plot(transform=get_cartopy_transform(), **kw)


def get_ylabel(ds: xr.Dataset, var: ty.Optional[str] = None):
    var = var or ds.attrs.get('variable_id', 'var')
    return f'{oet.plotting.plot.default_variable_labels().get(var, var)} [{get_unit(ds, var)}]'
