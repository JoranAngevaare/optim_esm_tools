import matplotlib.pyplot as plt
import optim_esm_tools as oet
from optim_esm_tools.config import config, get_logger
import typing as ty
import xarray as xr


def setup_map(
    projection=None, coastlines=True, add_features=False, **projection_kwargs
):
    plt.gcf().add_subplot(
        projection=get_cartopy_projection(projection, **projection_kwargs)
    )
    ax = plt.gca()
    if coastlines:
        ax.coastlines()
    if add_features:
        import cartopy.feature as cfeature

        allowed = 'LAND OCEAN COASTLINE BORDERS LAKES RIVERS'.split()
        for feat in oet.utils.to_str_tuple(add_features):
            assert feat.upper() in allowed, f'{feat} not in {allowed}'
            ax.add_feature(getattr(cfeature, feat.upper()))

    gl = ax.gridlines(draw_labels=True)
    gl.top_labels = False


def _show(show):
    if show:
        plt.show()  # pragma: no cover
    else:
        plt.clf()
        plt.close()


def default_variable_labels():
    labels = dict(config['variable_label'].items())
    ma = config['analyze']['moving_average_years']
    for k, v in list(labels.items()):
        labels[f'{k}_detrend'] = f'Detrend {v}'
        labels[f'{k}_run_mean_{ma}'] = f'$RM_{{{ma}}}$ {v}'
        labels[f'{k}_detrend_run_mean_{ma}'] = f'Detrend $RM_{{{ma}}}$ {v}'
    return labels


def get_range(var):
    r = (
        dict(oet.config.config['variable_range'].items())
        .get(var, 'None,None')
        .split(',')
    )
    return [(float(l) if l != 'None' else None) for l in r]


def set_y_lim_var(var):
    d, u = get_range(var)
    cd, cu = plt.ylim()
    plt.ylim(
        cd if d is None else min(cd, d),
        cu if u is None else max(cu, u),
    )


def get_unit(ds, var):
    return ds[var].attrs.get('units', '?').replace('%', '\%')


def get_cartopy_projection(projection=None, _field='projection', **projection_kwargs):
    import cartopy.crs as ccrs

    projection = projection or config['cartopy'][_field]
    if not hasattr(ccrs, projection):
        raise ValueError(f'Invalid projection {projection}')  # pragma: no cover
    return getattr(ccrs, projection)(**projection_kwargs)


def get_cartopy_transform(projection=None, **projection_kwargs):
    return get_cartopy_projection(
        projection=projection, _field='transform', **projection_kwargs
    )


def get_xy_lim_for_projection(
    projection=None,
) -> ty.Tuple[ty.Tuple[float, float], ty.Tuple[float, float]]:
    """
    Blunt hardcoding for the different projections. Calling plt.xlim(0, 360) will have vastly
    different outcomes depending on the projection used. Here we hardcoded some of the more common.
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
            f'No hardcoded x/y lims for {projection}, might yield odd figures.'
        )  # pragma: no cover
    return lims.get(projection, ((0, 360), (-90, 90)))


def plot_da(da: xr.DataArray, projection: str = None, **kw):
    """Simple wrapper for da.plot() with correct transforms and projections"""
    setup_map(projection=projection)
    da.plot(transform=get_cartopy_transform(), **kw)
