import os
import typing as ty

import numpy as np
import optim_esm_tools as oet


def legend_kw(**kw) -> dict:
    options = dict(
        bbox_to_anchor=(0.0, 1.02, 1, 0.32),
        loc=3,
        ncol=3,
        mode='expand',
        borderaxespad=0.0,
        frameon=True,
    )
    options.update(kw)
    return options


def save_fig(
    name: str,
    file_types: ty.Tuple[str] = ('png', 'pdf'),
    save_in: ty.Optional[str] = None,
    sub_dir: str = 'figures',
    skip: bool = False,
    remove_space: bool = True,
    **kwargs,
):
    """Save a figure in the figures dir."""
    import matplotlib.pyplot as plt

    save_in = save_in or oet.utils.root_folder

    kwargs.setdefault('dpi', 150)
    kwargs.setdefault('bbox_inches', 'tight')
    if remove_space:
        name = name.replace(' ', '_')
        save_in = save_in.replace(' ', '')
        sub_dir = sub_dir.replace(' ', '')
    if sub_dir is None:
        sub_dir = ''
    for file_type in file_types:
        path = os.path.join(save_in, sub_dir, f'{name}.{file_type}')
        if not os.path.exists(p := os.path.join(save_in, sub_dir)):
            os.makedirs(p, exist_ok=True)
        if skip:
            print(f'Skip save {path}')
            return
        plt.savefig(path, **kwargs)


def get_plt_colors():
    """Get matplotlib colors."""
    import matplotlib.pyplot as plt
    import matplotlib

    my_colors = [matplotlib.colors.to_hex(c) for c in plt.cm.Set1.colors]
    # I don't like the yellowish color
    del my_colors[5]
    return my_colors


def default_plt_params():
    return {
        'axes.grid': True,
        'font.size': 18,
        'axes.titlesize': 20,
        'axes.labelsize': 18,
        'axes.linewidth': 2,
        'xtick.labelsize': 18,
        'ytick.labelsize': 18,
        'ytick.major.size': 8,
        'ytick.minor.size': 4,
        'xtick.major.size': 8,
        'xtick.minor.size': 4,
        'xtick.major.width': 2,
        'xtick.minor.width': 2,
        'ytick.major.width': 2,
        'ytick.minor.width': 2,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'legend.fontsize': 14,
        'figure.facecolor': 'w',
        'figure.figsize': (8, 6),
        'image.cmap': 'viridis',
        'lines.linewidth': 2,
    }


def setup_plt(use_tex: bool = True, custom_cmap_name: str = 'custom_map'):
    """Change the plots to have uniform style defaults."""

    import matplotlib.pyplot as plt
    import matplotlib
    from cycler import cycler

    params = default_plt_params()
    if use_tex:
        params.update(
            {
                'font.family': 'Times New Roman',
            },
        )
    plt.rcParams.update(params)

    custom_cycler = cycler(color=get_plt_colors())
    # Could add cycler(marker=['o', 's', 'v', '^', 'D', 'P', '>', 'x'])

    plt.rcParams.update({'axes.prop_cycle': custom_cycler})
    if use_tex and not os.environ.get('DISABLE_LATEX', False):
        # Allow latex to be disabled from the environment coverage see
        matplotlib.rc('text', usetex=True)

    from matplotlib.colors import ListedColormap
    import matplotlib as mpl

    # Create capped custom map for printing (yellow does not print well)
    custom = ListedColormap(mpl.colormaps['viridis'](np.linspace(0, 0.85, 1000)))
    mpl.colormaps.register(custom, name=custom_cmap_name, force=True)
    setattr(mpl.pyplot.cm, custom_cmap_name, custom)

    custom_cmap_name += '_r'
    custom = ListedColormap(mpl.colormaps['viridis_r'](np.linspace(0.15, 1, 1000)))
    mpl.colormaps.register(custom, name=custom_cmap_name, force=True)
    setattr(mpl.pyplot.cm, custom_cmap_name, custom)
