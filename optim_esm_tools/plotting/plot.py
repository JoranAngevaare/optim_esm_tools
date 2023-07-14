import matplotlib.pyplot as plt
import optim_esm_tools as oet
from optim_esm_tools.config import config


def setup_map():
    plt.gcf().add_subplot(projection=get_cartopy_projection())
    ax = plt.gca()
    ax.coastlines()
    gl = ax.gridlines(draw_labels=True)
    gl.top_labels = False


def _show(show):
    if show:
        plt.show()
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


def get_cartopy_projection():
    import cartopy.crs as ccrs

    projection = config['analyze']['cartopy_projection']
    if not hasattr(ccrs, projection):
        raise ValueError(f'Invalid projection {projection}')
    return getattr(ccrs, projection)()
