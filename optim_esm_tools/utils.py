# -*- coding: utf-8 -*-
import os
import socket
import sys
import typing as ty
from collections import defaultdict
from importlib import import_module
from platform import python_version

import warnings
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from git import Repo, InvalidGitRepositoryError

    GIT_INSTALLED = True
except (ImportError, ModuleNotFoundError):
    GIT_INSTALLED = False
from cycler import cycler
import json
from base64 import b32encode
from hashlib import sha1
from collections.abc import Mapping
import sys


# From https://github.com/AxFoundation/strax/blob/136a16975b18ee87500051fd81a90c894d9b58dc/strax/utils.py#L33
if any('jupyter' in arg for arg in sys.argv):
    # In some cases we are not using any notebooks,
    # Taken from 44952863 on stack overflow thanks!
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm


# https://github.com/JoranAngevaare/thesis_plots/blob/d828c08e6f6c9c6926527220a23fd0e61e5d8c60/thesis_plots/main.py

root_folder = os.path.join(os.path.split(os.path.realpath(__file__))[0], '..')


def setup_plt(use_tex=True):
    """Change the plots to have uniform style defaults"""
    params = {
        'axes.grid': False,
        'font.size': 20,
        'axes.titlesize': 22,
        'axes.labelsize': 20,
        'axes.linewidth': 2,
        'xtick.labelsize': 20,
        'ytick.labelsize': 20,
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
        'legend.fontsize': 20,
        'figure.facecolor': 'w',
        'figure.figsize': (10, 8),
        'image.cmap': 'viridis',
        'lines.linewidth': 2,
    }
    plt.rcParams.update(params)
    # from https://github.com/XENONnT/nton/blob/d5d71b2798d74b9632a8846eb2e0f19ab0d1f563/nton/mplconfigs/stylelib/xenonnt.mplstyle
    custom_cycler = cycler(
        color=[
            f'#{c}'
            for c in [
                '000000',
                '0052FF',
                'FF2A2A',
                '4AC124',
                'ffa500',
                '00CFFF',
                'FF6AFF',
                'A54040',
            ]
        ]
    ) + cycler(marker=['o', 's', 'v', '^', 'D', 'P', '>', 'x'])

    plt.rcParams.update({'axes.prop_cycle': custom_cycler})
    if use_tex and not os.environ.get('DISABLE_LATEX', False):
        # Allow latex to be disabled from the environment coverage see #30
        matplotlib.rc('text', usetex=True)

    from matplotlib.colors import ListedColormap
    import matplotlib as mpl

    # Create capped custom map for printing (yellow does not print well)
    register_as = 'custom_map'
    custom = ListedColormap(mpl.colormaps['viridis'](np.linspace(0, 0.85, 1000)))
    mpl.colormaps.register(custom, name=register_as, force=True)
    setattr(mpl.pyplot.cm, register_as, custom)

    register_as += '_r'
    custom = ListedColormap(mpl.colormaps['viridis_r'](np.linspace(0.15, 1, 1000)))
    mpl.colormaps.register(custom, name=register_as, force=True)
    setattr(mpl.pyplot.cm, register_as, custom)


def save_fig(
    name, file_types=('png', 'pdf'), save_in=root_folder, sub_dir='figures', **kwargs
):
    """Save a figure in the figures dir"""
    kwargs.setdefault('dpi', 150)
    kwargs.setdefault('bbox_inches', 'tight')
    if sub_dir is None:
        sub_dir = ''
    for file_type in file_types:
        path = os.path.join(save_in, sub_dir, f'{name}.{file_type}')
        if not os.path.exists(p := os.path.join(save_in, sub_dir)):
            os.makedirs(p, exist_ok=True)
        plt.savefig(path, **kwargs)


def print_versions(
    modules=('optim_esm_tools',),
    print_output=True,
    include_python=True,
    return_string=False,
    include_git=True,
):
    """
    Print versions of modules installed.

    :param modules: Modules to print, should be str, tuple or list. E.g.
        print_versions(modules=('numpy', 'optim_esm_tools',))
    :param return_string: optional. Instead of printing the message,
        return a string
    :param include_git: Include the current branch and latest
        commit hash
    :return: optional, the message that would have been printed
    """
    versions = defaultdict(list)
    if not GIT_INSTALLED and include_git:
        warnings.warn('Git is not installed, maybe try pip install gitpython')
        include_git = False
    if include_python:
        versions['module'] = ['python']
        versions['version'] = [python_version()]
        versions['path'] = [sys.executable]
        versions['git'] = [None]
    for m in to_str_tuple(modules):
        result = _version_info_for_module(m, include_git=include_git)
        if result is None:
            continue
        version, path, git_info = result
        versions['module'].append(m)
        versions['version'].append(version)
        versions['path'].append(path)
        versions['git'].append(git_info)
    df = pd.DataFrame(versions)
    info = f'Host {socket.getfqdn()}\n{df.to_string(index=False)}'
    if print_output:
        print(info)
    return info if return_string else df


def _version_info_for_module(module_name, include_git):
    try:
        mod = import_module(module_name)
    except ImportError:
        print(f'{module_name} is not installed')
        return
    git = None
    version = mod.__dict__.get('__version__', None)
    module_path = mod.__dict__.get('__path__', [None])[0]
    if include_git:
        try:
            repo = Repo(module_path, search_parent_directories=True)
        except InvalidGitRepositoryError:
            # not a git repo
            pass
        else:
            try:
                branch = repo.active_branch
            except TypeError:
                branch = 'unknown'
            try:
                commit_hash = repo.head.object.hexsha
            except TypeError:
                commit_hash = 'unknown'
            git = f'branch:{branch} | {commit_hash[:7]}'
    return version, module_path, git


def to_str_tuple(
    x: ty.Union[str, bytes, list, tuple, pd.Series, np.ndarray]
) -> ty.Tuple[str]:
    """
    Convert any sensible instance to a tuple of strings
    from https://github.com/AxFoundation/strax/blob/d3608efc77acd52e1d5a208c3092b6b45b27a6e2/strax/utils.py#242
    """
    if isinstance(x, (str, bytes)):
        return (x,)
    if isinstance(x, list):
        return tuple(x)
    if isinstance(x, tuple):
        return x
    raise TypeError(f'Expected string or tuple of strings, got {type(x)}')


def string_to_mathrm(string):
    """wrap a string in mathrm mode for latex labels"""
    string = string.replace(' ', '\ ')
    return f'$\mathrm{{{string}}}$'


def legend_kw(**kw):
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
