import matplotlib.pyplot as plt


def setup_map():
    import cartopy.crs as ccrs

    plt.gcf().add_subplot(
        projection=ccrs.PlateCarree(
            central_longitude=0.0,
        )
    )
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
    from optim_esm_tools.config import config

    labels = dict(config['variable_label'].items())
    ma = config['analyze']['moving_average_years']
    for k, v in list(labels.items()):
        labels[f'{k}_detrend'] = f'Detrend {v}'
        labels[f'{k}_run_mean_{ma}'] = f'$RM_{ma}$ {v}'
        labels[f'{k}_detrend_run_mean_{ma}'] = f'Detrend $RM_{ma}$ {v}'
    return labels
