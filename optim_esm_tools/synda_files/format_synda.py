## Depricated module, will remove in future

from ..analyze.io import load_glob, recast
from optim_esm_tools.utils import deprecated


load_glob = deprecated(
    load_glob,
    'from optim_esm_tools.synda_files.load_glob is depricated, use optim_esm_tools.cmip_files.io'
    '\nWill raise an error in the next release ~(July 2023).',
)
recast = deprecated(
    recast,
    'from optim_esm_tools.synda_files.recast is depricated, use optim_esm_tools.cmip_files.io'
    '\nWill raise an error in the next release ~(July 2023).',
)
