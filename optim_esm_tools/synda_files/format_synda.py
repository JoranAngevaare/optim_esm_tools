## Depricated module, will remove in future

from ..cmip_files.io import load_glob, recast
from functools import wraps
from warnings import warn
from optim_esm_tools.utils import depricated


load_glob = depricated(
    load_glob,
    'from optim_esm_tools.synda_files.format_synda is depricated, use optim_esm_tools.cmip_files.io',
)
recast = depricated(
    recast,
    'from optim_esm_tools.synda_files.format_synda is depricated, use optim_esm_tools.cmip_files.io',
)
