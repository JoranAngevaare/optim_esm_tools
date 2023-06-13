## Depricated module, will remove in future

from ..cmip_files.io import load_glob, recast
from functools import wraps
from warnings import warn


def depricated(func):
    @wraps(func)
    def dep_fun(*args, **kwargs):
        warn(
            f'calling {func.__name__} from optim_esm_tools.synda_files.format_synda is depricated, use optim_esm_tools.cmip_files.io',
            category=DeprecationWarning,
        )
        return func(*args, **kwargs)

    return dep_fun


load_glob = depricated(load_glob)
recast = depricated(recast)
