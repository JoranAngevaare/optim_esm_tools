import inspect
import os
import socket
import sys
import time
import typing as ty
import warnings
from collections import defaultdict
from functools import wraps
from importlib import import_module
from platform import python_version

import numpy as np
import pandas as pd
from immutabledict import immutabledict

try:
    from git import Repo, InvalidGitRepositoryError

    GIT_INSTALLED = True
except ImportError:  # pragma: no cover
    GIT_INSTALLED = False

import sys


# From https://github.com/AxFoundation/strax/blob/136a16975b18ee87500051fd81a90c894d9b58dc/strax/utils.py#L33
if any('jupyter' in arg for arg in sys.argv):
    # In some cases we are not using any notebooks,
    # Taken from 44952863 on stack overflow thanks!
    from tqdm.notebook import tqdm  # pragma: no cover
else:
    from tqdm import tqdm


# https://github.com/JoranAngevaare/thesis_plots/blob/d828c08e6f6c9c6926527220a23fd0e61e5d8c60/thesis_plots/main.py
root_folder = os.path.join(os.path.split(os.path.realpath(__file__))[0], '..')

from optim_esm_tools.plotting.plot_utils import *


def print_versions(
    modules=('optim_esm_tools',),
    print_output=True,
    include_python=True,
    return_string=False,
    include_git=True,
):
    """Print versions of modules installed.

    :param modules: Modules to print, should be str, tuple or list. E.g.
        print_versions(modules=('numpy', 'optim_esm_tools',))
    :param return_string: optional. Instead of printing the message,
        return a string
    :param include_git: Include the current branch and latest commit
        hash
    :return: optional, the message that would have been printed
    """
    versions = defaultdict(list)
    if not GIT_INSTALLED and include_git:  # pragma: no cover
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
            continue  # pragma: no cover
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
            except TypeError:  # pragma: no cover
                branch = 'unknown'
            try:
                commit_hash = repo.head.object.hexsha
            except TypeError:  # pragma: no cover
                commit_hash = 'unknown'
            git = f'branch:{branch} | {commit_hash[:7]}'
    return version, module_path, git


def to_str_tuple(
    x: ty.Union[str, bytes, list, tuple, pd.Series, np.ndarray],
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
    raise TypeError(
        f'Expected string or tuple of strings, got {type(x)}',
    )  # pragma: no cover


def mathrm(string):
    return string_to_mathrm(string)


def string_to_mathrm(string):
    """Wrap a string in mathrm mode for latex labels."""
    string = string.replace(' ', r'\ ')
    return fr'$\mathrm{{{string}}}$'


def filter_keyword_arguments(
    kw: ty.Mapping,
    func: ty.Callable,
    allow_varkw: bool = False,
) -> ty.Mapping:
    """Only pass accepted keyword arguments (from kw) into function "func".

    Args:
        kw (ty.Mapping): kwargs that could go into function func
        func (type): a function
        allow_varkw (bool, optional): If True and the function take kwargs, just return the <kw>
            argument. Defaults to False.

    Returns:
        dict: Filtered keyword arguments
    """
    spec = inspect.getfullargspec(func)
    if allow_varkw and spec.varkw is not None:
        return kw  # pragma: no cover
    return {k: v for k, v in kw.items() if k in spec.args}


def check_accepts(
    accepts: ty.Mapping[str, ty.Iterable] = immutabledict(unit=('absolute', 'std')),
    do_raise: bool = True,
):
    """Wrapper for function if certain kwargs are from a defined list of
    variables.

    Example:
        ```
            @check_accepts(accepts=dict(far=('boo', 'booboo')))
            def bla(far):
                print(far)

            bla(far='boo')  # prints boo
            bla(far='booboo')  # prints booboo
            bla(far=1)  # raises ValueError
        ```


    Args:
        accepts (ty.Mapping[str, ty.Iterable], optional): which kwarg to accept a limited set of options.
            Defaults to immutabledict(unit=('absolute', 'std')).
        do_raise (bool, optional): if False, don't raise an error but just warn. Defaults to True.

    Returns:
        wrapped function
    """

    def somedec_outer(fn):
        @wraps(fn)
        def somedec_inner(*args, **kwargs):
            message = ''
            for k, v in kwargs.items():
                if k in accepts and v not in accepts[k]:
                    message += (
                        f'{k} for {v} but only accepts {accepts[k]}'  # pragma: no cover
                    )
            if do_raise and message:  # pragma: no cover
                raise ValueError(message)
            if message:  # pragma: no cover
                warnings.warn(message)
            response = fn(*args, **kwargs)
            return response

        return somedec_inner

    return somedec_outer


def add_load_kw(func):
    """Add apply `.load` method to the dataset returned by wrapped function."""

    @wraps(func)
    def dep_fun(*args, **kwargs):
        if 'load' not in inspect.signature(func).parameters:
            add_load = kwargs.pop('load', False)
        else:
            add_load = kwargs.get('load', False)
        res = func(*args, **kwargs)
        return res.load() if add_load else res

    return dep_fun


def deprecated(func, message='is deprecated'):
    @wraps(func)
    def dep_fun(*args, **kwargs):
        warnings.warn(
            f'calling {func.__name__} {message}',
            category=DeprecationWarning,
        )
        return func(*args, **kwargs)

    return dep_fun


def _chopped_string(string, max_len):
    string = str(string)
    return string if len(string) < max_len else string[:max_len] + '...'


@check_accepts(accepts=dict(_report=('debug', 'info', 'warning', 'error', 'print')))
def timed(
    *a,
    seconds: ty.Optional[int] = None,
    _report: ty.Optional[str] = None,
    _args_max: int = 20,
    _fmt: str = '.2g',
    _stacklevel: int = 2,
):
    """Time a function and print if it takes more than <seconds>

    Args:
        seconds (int, optional): Defaults to 5.
        _report (str, optional): Method of reporting, either print or use the global logger. Defaults to 'print'.
        _args_max (int, optional): Max number of characters in the message for the args and kwars of the function. Defaults to 20.
        _fmt (str, optional): time format specification. Defaults to '.2g'.
    """
    if seconds is None or _report is None:
        from .config import config

        if seconds is None:
            seconds = float(config['time_tool']['min_seconds'])
        if _report is None:
            _report = config['time_tool']['reporter']

    def somedec_outer(fn):
        @wraps(fn)
        def timed_func(*args, **kwargs):
            t0 = time.time()
            res = fn(*args, **kwargs)
            dt = time.time() - t0
            if dt > seconds:
                hours = '' if dt < 3600 else f' ({dt/3600:{_fmt}} h) '
                message = (
                    f'{fn.__name__} took {dt:{_fmt}} s{hours} (for '
                    f'{_chopped_string(args, _args_max)}, '
                    f'{_chopped_string(kwargs, _args_max)})'
                ).replace('\n', ' ')
                if _report == 'print':
                    print(message)
                else:
                    from .config import get_logger

                    getattr(get_logger(), _report)(message, stacklevel=_stacklevel)
            return res

        return timed_func

    if a and isinstance(a[0], ty.Callable):
        # Decorator that isn't closed
        return somedec_outer(a[0])
    return somedec_outer


@check_accepts(accepts=dict(_report=('debug', 'info', 'warning', 'error', 'print')))
def logged_tqdm(*a, log=None, _report='warning', **kw):
    from .config import get_logger

    log = log or get_logger()
    pbar = tqdm(*a, **kw)
    generator = iter(pbar)
    while True:
        try:
            v = next(generator)
            getattr(log, _report)(pbar, stacklevel=2)
            yield v

        except StopIteration:
            pbar.close()
            getattr(log, _report)(pbar, stacklevel=2)
            return


def scientific_latex_notation(
    value: ty.Union[str, float, int],
    high: float = 1e3,
    low: float = 1e-3,
    precision: str = '.2e',
):
    """convert a string of float-representation to latex-format with exponents"""
    if isinstance(value, float):
        fl = value
    elif isinstance(value, int):
        fl = float(value)
    elif isinstance(value, str):
        value = str(value)

        try:
            fl = float(value)
        except (TypeError, ValueError) as e:
            return value
    else:
        raise TypeError(f'misunderstood {value} ({type(value)})')
    if abs(fl) > high or abs(fl) < low or "e" in str(value):
        fl_s = f"{fl:{precision}}"
        if "e" not in fl_s:
            return value
        a, b = fl_s.split("e+") if "e+" in fl_s else fl_s.split("e-")
        if '-' in fl_s:
            exp = f'-{int(b)}'
        else:
            exp = f'{int(b)}'
        res = f"${a}\\times 10^{{{exp}}}$"

        return res
    return value
